"""Aggressive mode protects every face with robust detection — ROADMAP Phase 4.

The worst failure mode in aggressive mode is a small/distant *background* face
that detection missed: it gets downsampled and the AI reconstructs it into
something uncanny on restore — emotionally worse than a soft background. The
guardrail is to detect those faces and keep them at original quality (crop them
out of the downsampled region). This file pins three things:

1. **Aggressive relaxes the small-face filter so a distant face survives.** The
   shared ``min_size_ratio`` (0.05) discards faces whose short side is < 5% of
   the frame — exactly the background faces we must protect. Aggressive lowers it
   (and the confidence floor) so such a face is kept and cropped. These are pure
   parameters (no model download), so they help even on the default Haar.

2. **YuNet is an opt-in upgrade, not the default.** The default
   ``detector_backend`` is ``None`` = inherit the shared ``DetectorConfig`` (Haar,
   offline, zero-download) — the CLAUDE.md offline-first convention: the *default*
   aggressive run must never need the network. Setting ``detector_backend =
   "yunet"`` opts into the higher-recall DNN (it auto-downloads its model and
   falls back to Haar offline — covered in ``test_untested_paths.py``), and the
   compressor must honour that override.

3. **Faithful mode is unchanged.** Its detector still comes from the shared
   ``DetectorConfig`` (default Haar), so the offline default and tighter
   thresholds the rest of the project relies on are untouched.

The detector backend/params are asserted by capturing the args passed to
``create_detector`` (so no network/model is needed); the "small face survives"
behaviour is driven through a mocked detector so it is deterministic offline.
"""

import pytest

import facekeep.aggressive.compressor as compressor_mod
from facekeep.config import AggressiveConfig, DetectorConfig, FaceKeepConfig
from facekeep.detector import FaceRegion, _pad_and_clip
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _FixedDetector:
    """A detector that returns a preset face list, ignoring the image."""

    def __init__(self, faces):
        self._faces = faces

    def detect(self, image):
        return list(self._faces)


def _capture_create_detector(monkeypatch, faces=None):
    """Patch the compressor's ``create_detector`` to capture its kwargs.

    Returns a dict that, after ``compress_photo`` runs, holds the kwargs the
    pipeline passed (``calls["kwargs"]``). The returned detector yields ``faces``
    (default: none), so no real model/network is touched.
    """
    calls = {}

    def _fake(**kwargs):
        calls["kwargs"] = kwargs
        return _FixedDetector(faces or [])

    monkeypatch.setattr(compressor_mod, "create_detector", _fake)
    return calls


# --------------------------------------------------------------------------- #
# A. Config: the override fields, validation, and resolution
# --------------------------------------------------------------------------- #


def test_aggressive_default_is_offline_safe_with_relaxed_small_face_filter():
    """Default: backend inherits (offline Haar), but small-face thresholds relax.

    The default ``detector_backend`` must be None so the default aggressive run
    never needs the network (CLAUDE.md offline-first). The size/confidence floors
    are still relaxed by default — they need no download and protect small faces
    even on Haar.
    """
    agg = AggressiveConfig()
    assert agg.detector_backend is None  # inherit -> offline Haar by default
    # Must be meaningfully smaller than the shared 0.05 so distant faces survive.
    assert agg.detector_min_size_ratio is not None
    assert agg.detector_min_size_ratio < DetectorConfig().min_size_ratio
    # Confidence floor is relaxed below the shared default too.
    assert agg.detector_confidence is not None
    assert agg.detector_confidence < DetectorConfig().confidence


def test_resolved_detector_default_inherits_backend_keeps_relaxed_thresholds():
    """With the default (backend=None), resolve inherits Haar but keeps the
    relaxed confidence/min-size so small faces survive offline."""
    shared = DetectorConfig()  # backend=haar, confidence=0.6, min_size_ratio=0.05
    agg = AggressiveConfig()  # backend=None, confidence=0.5, min_size_ratio=0.02

    resolved = agg.resolved_detector(shared)

    # Backend inherits the offline default...
    assert resolved.backend == "haar"
    # ...but the relaxed small-face overrides still apply...
    assert resolved.confidence == 0.5
    assert resolved.min_size_ratio == 0.02
    # ...inherited fields keep the shared value...
    assert resolved.padding == shared.padding
    assert resolved.nms_iou == shared.nms_iou
    assert resolved.max_aspect_ratio == shared.max_aspect_ratio
    # ...and the shared config object is not mutated.
    assert shared.backend == "haar"
    assert shared.confidence == 0.6
    assert shared.min_size_ratio == 0.05


def test_resolved_detector_applies_explicit_yunet_override():
    """Opting into YuNet overrides the backend; shared config stays untouched."""
    shared = DetectorConfig()  # backend=haar
    agg = AggressiveConfig(detector_backend="yunet")

    resolved = agg.resolved_detector(shared)
    assert resolved.backend == "yunet"
    assert shared.backend == "haar"  # not mutated


def test_resolved_detector_none_inherits_shared():
    """A None override means 'inherit', so the shared value is used verbatim."""
    shared = DetectorConfig(backend="yunet", confidence=0.42, min_size_ratio=0.07)
    agg = AggressiveConfig(
        detector_backend=None,
        detector_confidence=None,
        detector_min_size_ratio=None,
    )
    resolved = agg.resolved_detector(shared)
    assert resolved.backend == "yunet"
    assert resolved.confidence == 0.42
    assert resolved.min_size_ratio == 0.07


def test_validate_rejects_bad_aggressive_backend():
    cfg = FaceKeepConfig()
    cfg.aggressive.detector_backend = "bogus"  # not haar/yunet/mediapipe
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validate_allows_mediapipe_aggressive_backend():
    cfg = FaceKeepConfig()
    cfg.aggressive.detector_backend = "mediapipe"
    cfg.validate()  # mediapipe is a valid opt-in backend


def test_validate_allows_none_aggressive_backend():
    cfg = FaceKeepConfig()
    cfg.aggressive.detector_backend = None  # inherit -> valid
    cfg.validate()


def test_validate_rejects_negative_aggressive_min_size_ratio():
    cfg = FaceKeepConfig()
    cfg.aggressive.detector_min_size_ratio = -0.01
    with pytest.raises(ConfigError):
        cfg.validate()


def test_aggressive_config_yaml_roundtrips_new_fields(tmp_path):
    """The new override fields survive a save/load cycle."""
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"
    cfg.aggressive.detector_backend = "haar"
    cfg.aggressive.detector_confidence = 0.55
    cfg.aggressive.detector_min_size_ratio = 0.03
    path = tmp_path / "cfg.yaml"
    cfg.save(path)

    loaded = FaceKeepConfig.load(path)
    assert loaded.aggressive.detector_backend == "haar"
    assert loaded.aggressive.detector_confidence == 0.55
    assert loaded.aggressive.detector_min_size_ratio == 0.03


# --------------------------------------------------------------------------- #
# B. Compressor wires the resolved detector through
# --------------------------------------------------------------------------- #


def test_compressor_default_requests_offline_haar_with_relaxed_filter(
    face_image, monkeypatch
):
    """aggressive compress_photo defaults to the offline backend + relaxed filter.

    Captures the kwargs the pipeline passes, so no model/network is touched. The
    default must request the inherited (offline Haar) backend — never the network
    — while still passing the relaxed small-face thresholds that protect distant
    faces.
    """
    calls = _capture_create_detector(monkeypatch)
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"  # default detector_backend=None -> inherit haar

    compressor_mod.compress_photo(str(face_image), cfg)

    kw = calls["kwargs"]
    assert kw["backend"] == "haar"  # offline default, no download
    assert kw["confidence"] == 0.5  # relaxed
    assert kw["min_size_ratio"] == 0.02  # relaxed
    # Inherited knobs come from the shared DetectorConfig untouched.
    assert kw["padding"] == DetectorConfig().padding
    assert kw["nms_iou"] == DetectorConfig().nms_iou
    assert kw["max_aspect_ratio"] == DetectorConfig().max_aspect_ratio


def test_compressor_honours_explicit_yunet_optin(face_image, monkeypatch):
    """Opting into YuNet makes the compressor request the yunet backend.

    Asserted via captured kwargs (create_detector is mocked), so no model is
    downloaded — this pins the plumbing, not real YuNet inference. The real
    YuNet model path is exercised (and skipped offline) in test_untested_paths.py.
    """
    calls = _capture_create_detector(monkeypatch)
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"
    cfg.aggressive.detector_backend = "yunet"  # explicit opt-in

    compressor_mod.compress_photo(str(face_image), cfg)
    assert calls["kwargs"]["backend"] == "yunet"


def test_compressor_honours_inherit_yunet_from_shared(face_image, monkeypatch):
    """With the aggressive override None and the *shared* backend set to yunet,
    aggressive inherits yunet (None means 'defer to the shared config')."""
    calls = _capture_create_detector(monkeypatch)
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"
    cfg.detector.backend = "yunet"
    cfg.aggressive.detector_backend = None  # inherit -> yunet

    compressor_mod.compress_photo(str(face_image), cfg)
    assert calls["kwargs"]["backend"] == "yunet"


def test_small_background_face_is_protected(face_image, monkeypatch):
    """A distant face (below the shared size filter) is cropped, not downsampled.

    Drives a mocked detector that reports one small face — short side well under
    5% of the frame, which the shared min_size_ratio (0.05) would discard but the
    aggressive threshold keeps. The face must be returned, a crop extracted at its
    padded bbox, and a matching mask produced. (We bypass the size *filter* here
    by injecting an already-kept FaceRegion; the filter-threshold plumbing is
    asserted separately via the captured kwargs above.)
    """
    from facekeep.imageio import load

    image = load(str(face_image)).image
    h, w = image.shape[:2]

    # A small face: ~3% of the short side (h=1200 -> ~36px), under the 0.05 floor.
    side = int(0.03 * min(h, w))
    cx, cy = w // 2, h // 2
    bbox = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
    small_face = FaceRegion(
        id=0,
        bbox=bbox,
        padded_bbox=_pad_and_clip(bbox, 1.5, w, h),
        confidence=0.55,
    )
    _capture_create_detector(monkeypatch, faces=[small_face])

    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"
    photo = compressor_mod.compress_photo(str(face_image), cfg)

    # The small face survived into the output and was cropped at original res.
    assert len(photo.faces) == 1
    assert len(photo.face_crops) == 1
    assert len(photo.face_masks) == 1
    px1, py1, px2, py2 = small_face.padded_bbox
    assert photo.face_crops[0].shape[:2] == (py2 - py1, px2 - px1)
    assert photo.face_masks[0].shape[:2] == (py2 - py1, px2 - px1)


# --------------------------------------------------------------------------- #
# C. Faithful mode is unaffected
# --------------------------------------------------------------------------- #


def test_faithful_still_uses_shared_detector(face_image, monkeypatch):
    """Faithful mode must NOT pick up the aggressive YuNet override.

    Guards the CLAUDE.md invariant that Haar is the offline default: the
    aggressive override is scoped to aggressive mode only. We capture the kwargs
    faithful.compress passes to its own create_detector.
    """
    import facekeep.faithful as faithful_mod

    calls = {}

    def _fake(**kwargs):
        calls["kwargs"] = kwargs
        return _FixedDetector([])

    monkeypatch.setattr(faithful_mod, "create_detector", _fake)

    cfg = FaceKeepConfig()  # faithful, shared detector default = haar
    # Set an aggressive override that must be ignored by the faithful path.
    cfg.aggressive.detector_backend = "yunet"
    cfg.aggressive.detector_min_size_ratio = 0.02

    # dry_run keeps it cheap (no file written) while still running detection.
    faithful_mod.compress(str(face_image), None, cfg, dry_run=True)

    kw = calls["kwargs"]
    assert kw["backend"] == "haar"  # shared default, NOT the aggressive override
    assert kw["min_size_ratio"] == DetectorConfig().min_size_ratio


# --------------------------------------------------------------------------- #
# D. Index fingerprint reflects the aggressive override
# --------------------------------------------------------------------------- #


def test_fingerprint_changes_with_aggressive_detector_override():
    """Changing an aggressive detector override must bust the incremental cache.

    The fingerprint hashes the *resolved* detector for aggressive mode, so a
    different backend/min-size yields a different fingerprint — otherwise a re-run
    after changing detection would wrongly skip the file as 'unchanged'.
    """
    base = FaceKeepConfig()
    base.mode = "aggressive"  # resolves to backend=haar by default
    fp_default = settings_fingerprint(base)

    # Opting into YuNet must change the fingerprint (different effective backend).
    changed_backend = FaceKeepConfig()
    changed_backend.mode = "aggressive"
    changed_backend.aggressive.detector_backend = "yunet"
    assert settings_fingerprint(changed_backend) != fp_default

    changed_size = FaceKeepConfig()
    changed_size.mode = "aggressive"
    changed_size.aggressive.detector_min_size_ratio = 0.04
    assert settings_fingerprint(changed_size) != fp_default


def test_fingerprint_stable_for_same_aggressive_config():
    """Same aggressive config -> same fingerprint (a hit stays a hit)."""
    a = FaceKeepConfig()
    a.mode = "aggressive"
    b = FaceKeepConfig()
    b.mode = "aggressive"
    assert settings_fingerprint(a) == settings_fingerprint(b)


def test_faithful_fingerprint_unaffected_by_aggressive_override():
    """A faithful run's fingerprint must not move when an aggressive-only field
    changes (the aggressive override can't affect faithful output)."""
    base = FaceKeepConfig()  # faithful
    fp = settings_fingerprint(base)

    other = FaceKeepConfig()  # faithful
    other.aggressive.detector_backend = "haar"
    other.aggressive.detector_min_size_ratio = 0.04
    assert settings_fingerprint(other) == fp
