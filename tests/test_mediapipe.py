"""MediaPipe Tasks-API detector backend — ROADMAP Phase 6.

Adds an optional ``mediapipe`` detector backend using the *current* Tasks API
(``mediapipe.tasks.python.vision.FaceDetector``); the legacy ``solutions`` API
was removed in recent builds. MediaPipe stays an optional extra — Haar is the
default, and a missing package or model degrades gracefully to Haar.

These tests are offline by default: the package is faked via ``sys.modules`` and
the model download is mocked, so they run without ``mediapipe`` installed (its
real, network-touching end-to-end detect is the single ``real_ai`` test, which
skips when the package/model are unavailable — mirroring the YuNet convention).
"""

import sys
import types

import numpy as np
import pytest

import facekeep.detector as det_mod
from facekeep.config import FaceKeepConfig
from facekeep.detector import (
    MEDIAPIPE_FILENAME,
    MEDIAPIPE_SHA256,
    MEDIAPIPE_URL,
    HaarDetector,
    MediaPipeDetector,
    create_detector,
)
from facekeep.exceptions import ConfigError, DetectionError, ModelDownloadError
from facekeep.imageio import load
from facekeep.index import settings_fingerprint


# --- Offline-safe wiring (always runs) ------------------------------------- #


def test_mediapipe_url_and_checksum_are_pinned():
    """The model URL is the Google bucket and the SHA-256 is a real digest.

    A regression to a wrong URL would download the wrong file; the pinned
    checksum (enforced in ensure_weights) catches that at download time.
    """
    assert MEDIAPIPE_URL.startswith("https://storage.googleapis.com/mediapipe-models/")
    assert MEDIAPIPE_URL.endswith(MEDIAPIPE_FILENAME)
    assert len(MEDIAPIPE_SHA256) == 64  # a real hex digest, not a placeholder


def test_ensure_model_routes_through_ensure_weights(monkeypatch):
    """``_ensure_model`` delegates to models.ensure_weights with verify args."""
    captured = {}

    def _fake_ensure(url, filename, *, sha256=None, cache_dir=None):
        captured.update(url=url, filename=filename, sha256=sha256)
        return "<verified-local-path>"

    monkeypatch.setattr(det_mod, "ensure_weights", _fake_ensure)

    path = MediaPipeDetector._ensure_model()

    assert path == "<verified-local-path>"
    assert captured["url"] == MEDIAPIPE_URL
    assert captured["filename"] == MEDIAPIPE_FILENAME
    assert captured["sha256"] == MEDIAPIPE_SHA256  # checksum is enforced


def test_ensure_model_translates_download_error_to_detection_error(monkeypatch):
    """A ModelDownloadError surfaces as DetectionError (keeps the Haar fallback)."""
    def _boom(*a, **k):
        raise ModelDownloadError("simulated offline")

    monkeypatch.setattr(det_mod, "ensure_weights", _boom)

    with pytest.raises(DetectionError):
        MediaPipeDetector._ensure_model()


def test_missing_package_raises_detection_error(monkeypatch):
    """Constructing the detector without ``mediapipe`` installed raises cleanly.

    This is the real situation on a default install. We force the import to fail
    (whether or not the package happens to be present) and assert the actionable
    DetectionError that drives the Haar fallback.
    """
    # Ensure the import inside __init__ fails deterministically.
    for name in list(sys.modules):
        if name == "mediapipe" or name.startswith("mediapipe."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "mediapipe", None)  # import -> ImportError

    with pytest.raises(DetectionError):
        MediaPipeDetector()


def test_create_detector_mediapipe_falls_back_to_haar_when_unavailable(monkeypatch):
    """The end-to-end offline contract: mediapipe -> Haar when it can't load."""
    for name in list(sys.modules):
        if name == "mediapipe" or name.startswith("mediapipe."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "mediapipe", None)

    detector = create_detector(backend="mediapipe")
    assert isinstance(detector, HaarDetector)


# --- Detection parsing with a faked mediapipe package ----------------------- #


class _FakeBox:
    def __init__(self, x, y, w, h):
        self.origin_x, self.origin_y, self.width, self.height = x, y, w, h


class _FakeCategory:
    def __init__(self, score):
        self.score = score


class _FakeDetection:
    def __init__(self, box, score):
        self.bounding_box = box
        self.categories = [_FakeCategory(score)]


class _FakeResult:
    def __init__(self, detections):
        self.detections = detections


class _FakeFaceDetector:
    """Stand-in for the Tasks FaceDetector returning fixed detections."""

    def __init__(self, detections):
        self._detections = detections

    def detect(self, mp_image):
        # The detector stores the data array it was handed so the test can
        # assert the BGR->RGB boundary was applied.
        _FakeFaceDetector.last_image = mp_image
        return _FakeResult(self._detections)


class _FakeMpImage:
    def __init__(self, image_format=None, data=None):
        self.image_format = image_format
        self.data = data


def _install_fake_mediapipe(monkeypatch, detections):
    """Inject a minimal fake ``mediapipe`` package into sys.modules.

    Provides exactly the surface ``MediaPipeDetector`` touches: ``mp.Image``,
    ``mp.ImageFormat.SRGB``, ``tasks.python.BaseOptions``, and
    ``tasks.python.vision.{FaceDetectorOptions,FaceDetector}``.
    """
    mp = types.ModuleType("mediapipe")
    mp.Image = _FakeMpImage
    mp.ImageFormat = types.SimpleNamespace(SRGB="SRGB")

    tasks = types.ModuleType("mediapipe.tasks")
    python = types.ModuleType("mediapipe.tasks.python")
    python.BaseOptions = lambda **kw: types.SimpleNamespace(**kw)
    vision = types.ModuleType("mediapipe.tasks.python.vision")
    vision.FaceDetectorOptions = lambda **kw: types.SimpleNamespace(**kw)
    vision.FaceDetector = types.SimpleNamespace(
        create_from_options=lambda options: _FakeFaceDetector(detections)
    )

    mp.tasks = tasks
    tasks.python = python
    python.vision = vision

    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", tasks)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", python)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python.vision", vision)
    monkeypatch.setattr(
        MediaPipeDetector, "_ensure_model", staticmethod(lambda: "<mock-model>")
    )


def test_mediapipe_parses_detection(monkeypatch):
    """A Tasks detection is parsed into a correct FaceRegion (no network)."""
    det = _FakeDetection(_FakeBox(110, 80, 100, 130), score=0.97)
    _install_fake_mediapipe(monkeypatch, [det])
    detector = MediaPipeDetector(confidence=0.6)

    image = np.zeros((320, 320, 3), dtype=np.uint8)
    faces = detector.detect(image)

    assert len(faces) == 1
    f = faces[0]
    assert f.bbox == (110, 80, 210, 210)  # (x, y, x+w, y+h)
    assert f.confidence == pytest.approx(0.97, abs=1e-4)
    px1, py1, px2, py2 = f.padded_bbox
    assert 0 <= px1 < px2 <= 320
    assert 0 <= py1 < py2 <= 320


def test_mediapipe_handles_no_detections(monkeypatch):
    """No detections yields an empty list."""
    _install_fake_mediapipe(monkeypatch, [])
    detector = MediaPipeDetector()
    assert detector.detect(np.zeros((320, 320, 3), dtype=np.uint8)) == []


def test_mediapipe_shares_filter(monkeypatch):
    """The shared geometric filter drops a too-small box (like the other backends)."""
    big = _FakeDetection(_FakeBox(110, 80, 100, 130), score=0.95)
    tiny = _FakeDetection(_FakeBox(5, 5, 4, 5), score=0.95)  # below min_size_ratio
    _install_fake_mediapipe(monkeypatch, [big, tiny])
    detector = MediaPipeDetector(min_size_ratio=0.05)

    faces = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    assert len(faces) == 1  # the tiny one is filtered out
    assert faces[0].bbox == (110, 80, 210, 210)


def test_mediapipe_converts_bgr_to_rgb(monkeypatch):
    """detect() hands MediaPipe an RGB array (BGR channels swapped)."""
    _install_fake_mediapipe(monkeypatch, [])
    detector = MediaPipeDetector()

    # A pure-blue BGR image (B=255) must arrive at MediaPipe as RGB (last chan).
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # blue in BGR
    detector.detect(image)

    handed = _FakeFaceDetector.last_image.data
    assert handed[0, 0, 2] == 255  # blue is now the RGB *last* channel
    assert handed[0, 0, 0] == 0


# --- Config + fingerprint wiring -------------------------------------------- #


def test_config_accepts_mediapipe_backend():
    cfg = FaceKeepConfig()
    cfg.detector.backend = "mediapipe"
    cfg.validate()  # must not raise


def test_config_accepts_aggressive_mediapipe_override():
    cfg = FaceKeepConfig()
    cfg.aggressive.detector_backend = "mediapipe"
    cfg.validate()  # must not raise


def test_config_rejects_unknown_backend():
    cfg = FaceKeepConfig()
    cfg.detector.backend = "bogus"
    with pytest.raises(ConfigError):
        cfg.validate()


def test_fingerprint_busts_on_mediapipe_backend():
    """Switching the detector backend to mediapipe changes the cache fingerprint."""
    haar = FaceKeepConfig()
    mp = FaceKeepConfig()
    mp.detector.backend = "mediapipe"
    assert settings_fingerprint(haar) != settings_fingerprint(mp)


# --- Real end-to-end (opt-in, skips when unavailable) ----------------------- #


@pytest.mark.real_ai
def test_mediapipe_detects_real_face_when_available(corpus_image):
    """End-to-end MediaPipe on a *real* face — skipped if package/model absent.

    Like YuNet, BlazeFace is a DNN and won't fire on the synthetic ellipse
    fixtures, so it needs a real photograph. Offline / no-package / no-corpus
    skips instead of failing, keeping CI green without network access.
    """
    src = corpus_image("obama_portrait.jpg")  # skips if corpus absent
    try:
        detector = MediaPipeDetector(confidence=0.5)
    except DetectionError as e:
        pytest.skip(f"MediaPipe unavailable (offline / not installed): {e}")

    image = load(str(src)).image
    h, w = image.shape[:2]
    faces = detector.detect(image)

    assert len(faces) >= 1
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
