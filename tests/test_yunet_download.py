"""YuNet model download — ROADMAP Phase 6.

The YuNet ONNX model is stored in opencv_zoo via Git LFS, so the old
``raw.githubusercontent.com`` URL served a ~131-byte LFS *pointer* instead of the
model — the size check then raised and ``create_detector("yunet")`` silently fell
back to Haar, so the yunet backend never actually ran. The fix points the URL at
GitHub's LFS resolver (``media.githubusercontent.com/media/...``), verifies a
known SHA-256, and routes the download through ``models.ensure_weights`` (shared
cache, atomic write).

These tests pin that wiring without a network (mocked ``ensure_weights``) and,
when the network/corpus are available, exercise the *real* model end-to-end —
skipping offline, matching the project's offline-graceful convention.
"""

import pytest

import facekeep.detector as det_mod
from facekeep.detector import (
    YUNET_FILENAME,
    YUNET_SHA256,
    YUNET_URL,
    HaarDetector,
    YuNetDetector,
    create_detector,
)
from facekeep.exceptions import DetectionError, ModelDownloadError


# --- Offline-safe wiring (always runs) ------------------------------------- #


def test_yunet_url_is_the_lfs_media_endpoint():
    """The URL must be the LFS resolver, not the raw pointer path.

    A regression to ``raw.githubusercontent.com`` would re-serve the 131-byte LFS
    pointer (too small → DetectionError → silent Haar fallback), the exact bug
    this item fixed. A pinned SHA-256 backs it up at download time.
    """
    assert YUNET_URL.startswith("https://media.githubusercontent.com/media/")
    assert YUNET_URL.endswith(YUNET_FILENAME)
    assert len(YUNET_SHA256) == 64  # a real hex digest, not a placeholder


def test_ensure_model_routes_through_ensure_weights(monkeypatch):
    """``_ensure_model`` delegates to models.ensure_weights with verify args."""
    captured = {}

    def _fake_ensure(url, filename, *, sha256=None, cache_dir=None):
        captured.update(url=url, filename=filename, sha256=sha256)
        return "<verified-local-path>"

    monkeypatch.setattr(det_mod, "ensure_weights", _fake_ensure)

    path = YuNetDetector._ensure_model()

    assert path == "<verified-local-path>"
    assert captured["url"] == YUNET_URL
    assert captured["filename"] == YUNET_FILENAME
    assert captured["sha256"] == YUNET_SHA256  # checksum is enforced


def test_ensure_model_translates_download_error_to_detection_error(monkeypatch):
    """A ModelDownloadError surfaces as DetectionError (keeps the Haar fallback).

    ``ensure_weights`` raises ``ModelDownloadError``; the detector translates it to
    ``DetectionError`` so the existing ``create_detector`` fallback (which catches
    ``DetectionError``) still degrades to Haar offline.
    """
    def _boom(*a, **k):
        raise ModelDownloadError("simulated offline")

    monkeypatch.setattr(det_mod, "ensure_weights", _boom)

    with pytest.raises(DetectionError):
        YuNetDetector._ensure_model()


def test_create_detector_yunet_falls_back_to_haar_when_download_fails(monkeypatch):
    """The end-to-end offline contract: yunet → Haar when the model can't load."""
    def _boom(*a, **k):
        raise ModelDownloadError("simulated offline")

    monkeypatch.setattr(det_mod, "ensure_weights", _boom)

    detector = create_detector(backend="yunet")
    assert isinstance(detector, HaarDetector)


# The genuine model-download end-to-end (real_ai, skips offline) lives in
# tests/test_untested_paths.py::test_yunet_detects_real_face_when_available,
# which now downloads the real model and detects on a real corpus face — the
# behavioural proof that the fixed LFS media URL + checksum actually work.
