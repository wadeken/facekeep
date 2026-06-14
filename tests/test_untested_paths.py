"""Coverage for previously-untested code paths — ROADMAP Phase 2.

The suite already exercises `.fkeep` round-trip integrity (``test_e2e.py``) and
the aggressive bicubic-fallback restore (``test_error_handling.py``). The two
gaps this file closes are the ones the ROADMAP item names but nothing covered:

1. **Faithful ``auto_tune`` happy path.** ``test_error_handling.py`` only drives
   the auto-tune *failure* branches (metadata re-embed raising, ICC survival).
   The actual binary search — does it pick a quality whose face region meets the
   target, does it respond to the target, does it fall back correctly with no
   faces — had no correctness test. We test ``faithful._auto_tune_quality``
   directly (deterministic; lets us recompute the exact face-region SSIM the
   search uses) rather than only through the written file.

2. **YuNet backend.** The DNN detector had zero coverage. It is covered three
   ways so the important parts run offline in CI and the real model is still
   exercised when present: (a) the documented offline graceful-degradation
   (``create_detector('yunet')`` falls back to Haar when the model can't be
   obtained); (b) the result-row parsing/filtering, with ``cv2.FaceDetectorYN``
   mocked so no model download happens; (c) an end-to-end detect on a real face,
   skipped when the model is unavailable offline.
"""

import numpy as np
import pytest

from facekeep import faithful, metrics
from facekeep.config import FaceKeepConfig
from facekeep.detector import (
    HaarDetector,
    YuNetDetector,
    create_detector,
)
from facekeep.exceptions import DetectionError
from facekeep.imageio import load


# --------------------------------------------------------------------------- #
# A. Faithful auto-tune happy path
# --------------------------------------------------------------------------- #


def _detect_faces(image, cfg):
    """Run the configured detector the way faithful.compress() does."""
    detector = create_detector(
        backend=cfg.detector.backend,
        confidence=cfg.detector.confidence,
        padding=cfg.detector.padding,
        nms_iou=cfg.detector.nms_iou,
        min_size_ratio=cfg.detector.min_size_ratio,
        max_aspect_ratio=cfg.detector.max_aspect_ratio,
    )
    return detector.detect(image)


def test_auto_tune_meets_face_target(face_image):
    """The chosen quality's face region must actually meet the SSIM target.

    This is the correctness assertion missing today: re-decode the auto-tune
    output and recompute the face-region SSIM, asserting it clears the
    configured ``target_value``. (Synthetic-face SSIM saturates somewhat — see
    IMPROVEMENTS.md — so we use a target the search can realistically hit.)
    """
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = True
    cfg.faithful.target_value = 0.95

    loaded = load(str(face_image))
    image = loaded.image
    faces = _detect_faces(image, cfg)
    assert faces, "fixture should yield at least one Haar face"

    from facekeep import encoders

    data, quality_used = faithful._auto_tune_quality(
        image, faces, cfg.faithful, has_faces=True
    )

    bbox = metrics.face_union_bbox(faces, image.shape[:2])
    x1, y1, x2, y2 = bbox
    decoded = encoders.decode(data)
    face_ssim = metrics.ssim(image[y1:y2, x1:x2], decoded[y1:y2, x1:x2])

    # The acceptance criterion: the returned encode's face region clears target.
    assert face_ssim >= cfg.faithful.target_value
    # Quality stays within (or just below) the search bounds. When the target is
    # met even at the lowest tried quality, the search drives `hi` to q-1 and may
    # return 39 — i.e. "the floor was already more than enough" — which is valid.
    assert quality_used <= 95


def test_auto_tune_no_faces_uses_configured_quality(plain_image):
    """With no faces, auto-tune takes the no-bbox branch: use cfg.quality as-is.

    ``plain_image`` is a smooth gradient Haar finds nothing in, so
    ``face_union_bbox`` is None and the search has no acceptance region. The
    documented fallback (faithful.py) is to encode once at the configured
    quality and return it unchanged.
    """
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = True
    cfg.faithful.quality = 63  # distinctive value to prove it's the one used

    loaded = load(str(plain_image))
    image = loaded.image
    faces = _detect_faces(image, cfg)
    assert faces == [], "plain gradient should yield no faces"

    _, quality_used = faithful._auto_tune_quality(
        image, faces, cfg.faithful, has_faces=False
    )
    assert quality_used == cfg.faithful.quality


def test_auto_tune_lower_target_picks_lower_or_equal_quality(face_image):
    """The search must respond to the target: a laxer target never costs more.

    Run the same image with a strict and a lax target; the lax run's chosen
    quality must be <= the strict run's. Catches a search that ignores
    ``target_value`` (e.g. always returns the same quality).
    """
    loaded = load(str(face_image))
    image = loaded.image

    strict = FaceKeepConfig()
    strict.faithful.auto_tune = True
    strict.faithful.target_value = 0.99
    faces = _detect_faces(image, strict)
    assert faces

    _, q_strict = faithful._auto_tune_quality(
        image, faces, strict.faithful, has_faces=True
    )

    lax = FaceKeepConfig()
    lax.faithful.auto_tune = True
    lax.faithful.target_value = 0.80
    _, q_lax = faithful._auto_tune_quality(
        image, faces, lax.faithful, has_faces=True
    )

    assert q_lax <= q_strict


# --------------------------------------------------------------------------- #
# B. YuNet backend
# --------------------------------------------------------------------------- #


def test_yunet_falls_back_to_haar_offline(monkeypatch):
    """When the YuNet model can't be obtained, the factory degrades to Haar.

    This is the documented graceful-degradation contract in create_detector and
    runs everywhere (offline included): we force ``_ensure_model`` to raise the
    same DetectionError a failed download would.
    """
    def _no_model():
        raise DetectionError("simulated offline: cannot download YuNet model")

    monkeypatch.setattr(YuNetDetector, "_ensure_model", staticmethod(_no_model))

    detector = create_detector(backend="yunet")
    assert isinstance(detector, HaarDetector)


class _FakeYuNet:
    """Stand-in for cv2.FaceDetectorYN returning a fixed YuNet result matrix.

    YuNet's ``detect`` returns ``(retval, faces)`` where each row is
    ``[x, y, w, h, <10 landmark coords>, score]`` (15 columns); the detector
    reads columns 0-3 for the box and column 14 for the score.
    """

    def __init__(self, rows):
        self._rows = rows

    def setInputSize(self, size):  # noqa: N802 - mirrors the cv2 API name
        self._size = size

    def detect(self, image):
        if self._rows is None:
            return 1, None
        return 1, np.asarray(self._rows, dtype=np.float32)


def _make_yunet_with(rows, monkeypatch):
    """Build a YuNetDetector whose model load and cv2 detector are mocked."""
    monkeypatch.setattr(
        YuNetDetector, "_ensure_model", staticmethod(lambda: "<mock-model>")
    )
    import facekeep.detector as det_mod

    monkeypatch.setattr(
        det_mod.cv2.FaceDetectorYN,
        "create",
        staticmethod(lambda **kwargs: _FakeYuNet(rows)),
    )
    return YuNetDetector(confidence=0.6)


def test_yunet_parses_result_row(monkeypatch):
    """A YuNet result row is parsed into a correct FaceRegion (no network).

    Exercises YuNet's own parsing path (box from cols 0-3, score from col 14,
    clip to bounds, padding) deterministically, with cv2.FaceDetectorYN mocked.
    """
    # One plausible, comfortably-large face box on a 320x320 image: a 100x130
    # box at (110, 80). Landmarks are irrelevant to parsing; score = 0.97.
    row = [110, 80, 100, 130] + [0.0] * 10 + [0.97]
    detector = _make_yunet_with([row], monkeypatch)

    image = np.zeros((320, 320, 3), dtype=np.uint8)
    faces = detector.detect(image)

    assert len(faces) == 1
    f = faces[0]
    assert f.bbox == (110, 80, 210, 210)  # (x, y, x+w, y+h)
    assert f.confidence == pytest.approx(0.97, abs=1e-4)
    # Padded box stays within image bounds.
    px1, py1, px2, py2 = f.padded_bbox
    assert 0 <= px1 < px2 <= 320
    assert 0 <= py1 < py2 <= 320


def test_yunet_handles_no_detections(monkeypatch):
    """YuNet returning ``None`` for the result matrix yields an empty list."""
    detector = _make_yunet_with(None, monkeypatch)
    faces = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    assert faces == []


@pytest.mark.real_ai
def test_yunet_detects_real_face_when_available(corpus_image):
    """End-to-end YuNet on a *real* face — skipped if model/corpus unavailable.

    This exercises the actual ONNX model, so it uses a real photograph (YuNet is
    a DNN and does not fire on the synthetic ellipse fixtures — only a real face
    proves the model works). Offline (no cached model, no network) or with no
    corpus it skips instead of failing, so CI stays green without network access.
    """
    src = corpus_image("obama_portrait.jpg")  # skips if corpus absent
    try:
        detector = YuNetDetector(confidence=0.6)
    except DetectionError as e:
        pytest.skip(f"YuNet model unavailable (offline): {e}")

    image = load(str(src)).image
    h, w = image.shape[:2]
    faces = detector.detect(image)

    assert len(faces) >= 1
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
