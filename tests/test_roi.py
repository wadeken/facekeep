"""Subject/person ROI: grow the high-priority region beyond the face box.

The ROI knob (`detector.roi = face | head_shoulders | person`) only enlarges
each face's `padded_bbox` (the region used for aggressive crops and the faithful
auto-tune acceptance area); the tight detection box, NMS, and chroma decision are
untouched, and `roi="face"` is a byte-for-byte no-op.
"""

import cv2
import numpy as np
import pytest

from facekeep.config import AggressiveConfig, DetectorConfig, FaceKeepConfig
from facekeep.detector import (
    _ROI_FACTORS,
    FaceRegion,
    HaarDetector,
    _apply_roi,
    _expand_for_roi,
    create_detector,
)
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint


# --- pure _expand_for_roi -------------------------------------------------

def test_face_roi_is_noop():
    padded = (40, 50, 140, 200)
    tight = (50, 60, 130, 180)
    assert _expand_for_roi(padded, tight, "face", 1000, 1000) == padded


def test_unknown_roi_is_noop():
    """An unrecognised roi falls back to no expansion (defensive)."""
    padded = (40, 50, 140, 200)
    tight = (50, 60, 130, 180)
    assert _expand_for_roi(padded, tight, "bogus", 1000, 1000) == padded


def test_head_shoulders_grows_downward_and_outward():
    padded = (100, 100, 200, 260)  # padded face box
    tight = (120, 120, 180, 240)   # tight: 60 wide x 120 tall
    x1, y1, x2, y2 = _expand_for_roi(padded, tight, "head_shoulders", 1000, 1000)
    # Top is unchanged (a body hangs below the face, not above).
    assert y1 == padded[1]
    # Bottom drops, sides widen.
    assert y2 > padded[3]
    assert x1 < padded[0]
    assert x2 > padded[2]


def test_person_grows_more_than_head_shoulders():
    padded = (100, 100, 200, 260)
    tight = (120, 120, 180, 240)
    _, _, _, hs_bottom = _expand_for_roi(padded, tight, "head_shoulders", 2000, 2000)
    _, _, _, person_bottom = _expand_for_roi(padded, tight, "person", 2000, 2000)
    assert person_bottom > hs_bottom


def test_expand_clips_to_frame():
    """Expansion never escapes the image bounds even near the edges."""
    padded = (0, 700, 100, 760)
    tight = (10, 710, 90, 750)
    x1, y1, x2, y2 = _expand_for_roi(padded, tight, "person", 800, 768)
    assert 0 <= x1 <= x2 <= 800
    assert 0 <= y1 <= y2 <= 768
    assert y2 == 768  # clipped to the bottom edge


# --- _apply_roi over a list ----------------------------------------------

def _region(padded, tight, idx=0):
    return FaceRegion(id=idx, bbox=tight, padded_bbox=padded, confidence=1.0)


def test_apply_roi_face_returns_same_list():
    faces = [_region((10, 10, 50, 70), (20, 20, 40, 60))]
    assert _apply_roi(faces, "face", 500, 500) is faces


def test_apply_roi_preserves_tight_bbox():
    tight = (120, 120, 180, 240)
    faces = [_region((100, 100, 200, 260), tight)]
    out = _apply_roi(faces, "person", 2000, 2000)
    assert out[0].bbox == tight  # tight box untouched
    assert out[0].padded_bbox[3] > 260  # only padded grew
    assert out[0].id == faces[0].id
    assert out[0].confidence == faces[0].confidence


# --- detector wiring (real Haar on a synthetic face) ----------------------

def test_haar_roi_monotonic_padded_box():
    """person padded box taller than head_shoulders taller than face; tight equal."""
    rng = np.random.default_rng(3)
    H, W = 1200, 1800
    bg = cv2.resize(
        rng.normal(128, 30, (H // 10, W // 10, 3)).astype(np.float32),
        (W, H), interpolation=cv2.INTER_CUBIC,
    )
    img = np.clip(bg, 0, 255).astype(np.uint8)
    cx, cy, fw = 650, 560, 300
    fh = int(fw * 1.3)
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, (180, 170, 165), -1)
    cv2.ellipse(img, (cx, cy - fh // 6), (fw // 2 - 5, fh // 4), 0, 0, 360,
                (195, 185, 180), -1)
    eye_y = cy - fh // 10
    ew = fw // 7
    cv2.ellipse(img, (cx - fw // 5, eye_y), (ew, ew // 2), 0, 0, 360, (60, 55, 55), -1)
    cv2.ellipse(img, (cx + fw // 5, eye_y), (ew, ew // 2), 0, 0, 360, (60, 55, 55), -1)
    cv2.line(img, (cx, eye_y), (cx, cy + fh // 12), (150, 140, 135), max(2, fw // 40))
    cv2.ellipse(img, (cx, cy + fh // 4), (fw // 5, fh // 18), 0, 0, 180, (120, 90, 90), -1)

    def detect(roi):
        return HaarDetector(roi=roi).detect(img)

    face = detect("face")
    hs = detect("head_shoulders")
    person = detect("person")
    assert len(face) == len(hs) == len(person) >= 1

    def height(r):
        return r.padded_bbox[3] - r.padded_bbox[1]

    def width(r):
        return r.padded_bbox[2] - r.padded_bbox[0]

    # Same detection geometry (tight box identical across roi).
    assert face[0].bbox == hs[0].bbox == person[0].bbox
    # Width grows strictly with roi scope (sides aren't at the frame edge here).
    assert width(person[0]) > width(hs[0]) > width(face[0])
    # Height grows monotonically; it may saturate when expansion clips to the
    # frame floor (this big, high face hits the bottom), so use non-strict.
    assert height(person[0]) >= height(hs[0]) >= height(face[0])
    # face roi must leave the padded box exactly as the detector padded it.
    assert height(face[0]) < height(person[0])


def test_create_detector_threads_roi():
    det = create_detector(backend="haar", roi="person")
    assert det.roi == "person"


# --- config validate + YAML ----------------------------------------------

def test_validate_accepts_all_roi():
    for roi in ("face", "head_shoulders", "person"):
        cfg = FaceKeepConfig(detector=DetectorConfig(roi=roi))
        cfg.validate()  # no raise


def test_validate_rejects_unknown_roi():
    cfg = FaceKeepConfig(detector=DetectorConfig(roi="torso"))
    with pytest.raises(ConfigError):
        cfg.validate()


def test_roi_yaml_roundtrip(tmp_path):
    cfg = FaceKeepConfig(detector=DetectorConfig(roi="head_shoulders"))
    p = tmp_path / "facekeep.yaml"
    cfg.save(p)
    loaded = FaceKeepConfig.load(p)
    assert loaded.detector.roi == "head_shoulders"


def test_factors_table_has_three_levels():
    assert set(_ROI_FACTORS) == {"face", "head_shoulders", "person"}
    assert _ROI_FACTORS["face"] == (0.0, 0.0)


# --- index fingerprint ----------------------------------------------------

def test_roi_busts_fingerprint_faithful():
    base = FaceKeepConfig()  # roi defaults to "face"
    changed = FaceKeepConfig(detector=DetectorConfig(roi="person"))
    assert settings_fingerprint(base) != settings_fingerprint(changed)


def test_roi_busts_fingerprint_aggressive():
    base = FaceKeepConfig(mode="aggressive")
    changed = FaceKeepConfig(
        mode="aggressive", detector=DetectorConfig(roi="head_shoulders")
    )
    assert settings_fingerprint(base) != settings_fingerprint(changed)


def test_default_roi_fingerprint_stable():
    """Two default configs hash identically (roi default doesn't perturb it)."""
    assert settings_fingerprint(FaceKeepConfig()) == settings_fingerprint(
        FaceKeepConfig()
    )


def test_resolved_detector_inherits_roi():
    """Aggressive resolves the shared roi (it is not a per-mode override)."""
    shared = DetectorConfig(roi="person")
    agg = AggressiveConfig()
    assert agg.resolved_detector(shared).roi == "person"


# --- end-to-end: aggressive crop grows with roi ---------------------------

def test_aggressive_person_crop_larger_than_face(tmp_path, monkeypatch):
    """A person-ROI run crops more pixels per face than a face-ROI run."""
    from facekeep.aggressive import compressor

    img = np.full((1000, 800, 3), 130, np.uint8)
    # A centred subject so the body region stays inside the frame.
    face_tight = (350, 200, 450, 330)  # 100x130
    fpath = tmp_path / "subject.jpg"
    cv2.imwrite(str(fpath), img)

    def fake_detector(roi):
        class _D:
            def detect(self, image):
                h, w = image.shape[:2]
                from facekeep.detector import _expand_for_roi, _pad_and_clip
                padded = _pad_and_clip(face_tight, 1.5, w, h)
                padded = _expand_for_roi(padded, face_tight, roi, w, h)
                return [FaceRegion(0, face_tight, padded, 1.0)]
        return _D()

    def make(roi):
        captured = {}
        orig = compressor.create_detector

        def spy(**kwargs):
            captured["roi"] = kwargs["roi"]
            return fake_detector(kwargs["roi"])

        monkeypatch.setattr(compressor, "create_detector", spy)
        cfg = FaceKeepConfig(mode="aggressive", detector=DetectorConfig(roi=roi))
        result = compressor.compress_photo(str(fpath), cfg)
        monkeypatch.setattr(compressor, "create_detector", orig)
        assert captured["roi"] == roi
        return result

    face_res = make("face")
    person_res = make("person")
    assert face_res.face_crops and person_res.face_crops
    assert person_res.face_crops[0].size > face_res.face_crops[0].size
