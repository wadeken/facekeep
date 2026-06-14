"""Custom-detector plugin hook (ROADMAP backlog).

A third party can register their own ``FaceDetector`` under a new backend name
(`detector.register_detector(name, factory)`) and select it via config
(`detector.backend: <name>`) exactly like a built-in haar/yunet/mediapipe.

These tests pin: registration + selection through ``create_detector``; the
factory receives the standard resolved kwargs; the returned detector is used
as-is (no Haar fallback); the registry discipline (built-in names reserved,
no duplicate registration, non-FaceDetector returns rejected, unregister);
config ``validate()`` accepts a registered backend (shared *and* aggressive
override) and still rejects a truly-unknown one; and the index fingerprint busts
on a custom backend so the incremental cache stays correct.
"""

import numpy as np
import pytest

from facekeep import detector
from facekeep.config import FaceKeepConfig
from facekeep.detector import (
    FaceDetector,
    FaceRegion,
    create_detector,
    is_known_backend,
    known_backends,
    register_detector,
    registered_detectors,
    unregister_detector,
)
from facekeep.exceptions import ConfigError, DetectionError
from facekeep.index import settings_fingerprint


class _RecordingDetector(FaceDetector):
    """A minimal well-behaved custom detector that records the kwargs it got."""

    def __init__(self, confidence=0.6, padding=1.5, nms_iou=0.3,
                 min_size_ratio=0.05, max_aspect_ratio=1.6, roi="face"):
        self._backend = "recording"
        self.confidence = confidence
        self.padding = padding
        self.nms_iou = nms_iou
        self.min_size_ratio = min_size_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.roi = roi
        self.detect_calls = 0

    def detect(self, image):
        self.detect_calls += 1
        return [FaceRegion(id=0, bbox=(1, 2, 3, 4),
                           padded_bbox=(0, 1, 4, 5), confidence=0.9)]


@pytest.fixture
def clean_registry():
    """Snapshot the registry and restore it after the test (global state)."""
    before = dict(detector._DETECTOR_REGISTRY)
    try:
        yield
    finally:
        detector._DETECTOR_REGISTRY.clear()
        detector._DETECTOR_REGISTRY.update(before)


# --- registration + selection ----------------------------------------------

def test_unknown_backend_raises_before_registration():
    with pytest.raises(DetectionError, match="Unknown detector backend"):
        create_detector("recording")


def test_register_and_create(clean_registry):
    register_detector("recording", lambda **kw: _RecordingDetector(**kw))
    assert "recording" in registered_detectors()
    assert is_known_backend("recording")
    assert "recording" in known_backends()

    d = create_detector("recording")
    assert isinstance(d, _RecordingDetector)
    assert d._backend == "recording"
    faces = d.detect(np.zeros((10, 10, 3), np.uint8))
    assert len(faces) == 1 and d.detect_calls == 1


def test_factory_receives_resolved_kwargs(clean_registry):
    """The factory gets the same kwargs create_detector resolves for built-ins."""
    register_detector("recording", lambda **kw: _RecordingDetector(**kw))
    d = create_detector(
        "recording", confidence=0.42, padding=2.0, nms_iou=0.25,
        min_size_ratio=0.07, max_aspect_ratio=2.0, roi="person",
    )
    assert d.confidence == 0.42
    assert d.padding == 2.0
    assert d.nms_iou == 0.25
    assert d.min_size_ratio == 0.07
    assert d.max_aspect_ratio == 2.0
    assert d.roi == "person"


def test_custom_detector_used_as_is_no_haar_fallback(clean_registry):
    """A custom backend owns its degradation — no silent Haar wrapping."""
    register_detector("recording", lambda **kw: _RecordingDetector(**kw))
    d = create_detector("recording")
    assert type(d).__name__ == "_RecordingDetector"  # not HaarDetector


def test_factory_error_surfaces(clean_registry):
    """A factory that raises is not masked as 'use Haar' (it's the author's bug)."""
    def _boom(**kw):
        raise ValueError("factory exploded")

    register_detector("boom", _boom)
    with pytest.raises(ValueError, match="factory exploded"):
        create_detector("boom")


def test_non_facedetector_return_rejected(clean_registry):
    register_detector("bad", lambda **kw: object())
    with pytest.raises(DetectionError, match="not a FaceDetector"):
        create_detector("bad")


# --- registry discipline ----------------------------------------------------

@pytest.mark.parametrize("builtin", ["haar", "yunet", "mediapipe"])
def test_cannot_override_builtin(clean_registry, builtin):
    with pytest.raises(DetectionError, match="reserved"):
        register_detector(builtin, lambda **kw: _RecordingDetector(**kw))


def test_duplicate_registration_rejected(clean_registry):
    register_detector("recording", lambda **kw: _RecordingDetector(**kw))
    with pytest.raises(DetectionError, match="already registered"):
        register_detector("recording", lambda **kw: _RecordingDetector(**kw))


def test_empty_name_rejected(clean_registry):
    with pytest.raises(DetectionError):
        register_detector("", lambda **kw: _RecordingDetector(**kw))


def test_non_callable_factory_rejected(clean_registry):
    with pytest.raises(DetectionError, match="callable"):
        register_detector("recording", "not-callable")


def test_unregister(clean_registry):
    register_detector("recording", lambda **kw: _RecordingDetector(**kw))
    assert is_known_backend("recording")
    unregister_detector("recording")
    assert not is_known_backend("recording")
    unregister_detector("recording")  # idempotent no-op


def test_cannot_unregister_builtin(clean_registry):
    with pytest.raises(DetectionError, match="built-in"):
        unregister_detector("haar")


# --- config validate --------------------------------------------------------

def test_validate_accepts_registered_backend(clean_registry):
    register_detector("recording", lambda **kw: _RecordingDetector(**kw))
    cfg = FaceKeepConfig()
    cfg.detector.backend = "recording"
    cfg.validate()  # must not raise


def test_validate_rejects_unknown_backend():
    cfg = FaceKeepConfig()
    cfg.detector.backend = "definitely-not-registered"
    with pytest.raises(ConfigError, match="Unknown detector backend"):
        cfg.validate()


def test_validate_accepts_registered_aggressive_override(clean_registry):
    register_detector("recording", lambda **kw: _RecordingDetector(**kw))
    cfg = FaceKeepConfig()
    cfg.aggressive.detector_backend = "recording"
    cfg.validate()  # must not raise


def test_validate_rejects_unknown_aggressive_override():
    cfg = FaceKeepConfig()
    cfg.aggressive.detector_backend = "definitely-not-registered"
    with pytest.raises(ConfigError, match="detector_backend"):
        cfg.validate()


def test_validate_aggressive_override_none_still_ok():
    cfg = FaceKeepConfig()
    cfg.aggressive.detector_backend = None
    cfg.validate()


# --- index fingerprint ------------------------------------------------------

def test_custom_backend_busts_index_fingerprint(clean_registry):
    register_detector("recording", lambda **kw: _RecordingDetector(**kw))
    a = FaceKeepConfig()  # default haar
    b = FaceKeepConfig()
    b.detector.backend = "recording"
    assert settings_fingerprint(a) != settings_fingerprint(b)


def test_custom_detector_fingerprint_reflects_settings(clean_registry):
    """A custom detector that sets _backend + filter fields fingerprints correctly."""
    register_detector("recording", lambda **kw: _RecordingDetector(**kw))
    d1 = create_detector("recording", roi="face")
    d2 = create_detector("recording", roi="person")
    assert d1.fingerprint() != d2.fingerprint()  # roi is output-affecting
