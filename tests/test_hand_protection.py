"""Hand protection in aggressive mode — ROADMAP Phase 4 (region-local follow-up).

Hands aren't faces, so the detector never finds them: they ride the aggressive
``bg_scale`` downsample and the AI upscaler smears their thin finger structure on
restore. This feature keeps hands sharp by reusing the *existing* region-local
patch mechanism (region_NNN.* + mask + bbox) — so it changes **no** .fkeep format
or manifest version. It is tiered, mirroring the offline-Haar / opt-in-YuNet split,
because OpenCV ships no hand cascade:

* **C1 (default, offline):** ``_hand_zones_from_faces`` infers a hand-likely band
  below/beside each detected face from body proportions. Zero-download, a
  probabilistic guess.
* **C2 (opt-in):** a constructed MediaPipe ``HandDetector`` returns tight per-hand
  boxes; missing/offline gracefully falls back to C1.

What these tests pin (all offline — detection + any hand detector are mocked):

* C1 geometry: two side bands below a face, frame-clamped, torso-centre excluded,
  empty on no faces / degenerate frame;
* the tier selector ``_hand_regions``: gated by the switches, C2 boxes preferred,
  C2-empty falls back to C1, no detector → C1;
* ``_dedupe_regions`` drops a hand box already covered by a small-face patch;
* end-to-end ``compress_photo``: hands on → hand region patches+masks + ``regions[]``;
  hands off → none; a fake C2 detector's exact boxes are protected;
* the ``.fkeep`` round-trips and ``verify_fkeep`` passes with hand regions present,
  the manifest needs **no hands-specific version bump** (no format change), and
  restore composites them;
* config ``validate()`` + YAML round-trip of the new fields;
* ``settings_fingerprint`` busts on the new aggressive fields and leaves faithful
  untouched.
"""

import zipfile

import cv2
import numpy as np
import pytest

import facekeep.aggressive.compressor as compressor_mod
from facekeep.aggressive.compressor import (
    _c1_hand_zones,
    _dedupe_regions,
    _hand_regions,
    _hand_zones_from_faces,
    _merge_overlapping_boxes,
    compress_photo,
)
from facekeep.aggressive.format import (
    read_fkeep,
    read_fkeep_info,
    verify_fkeep,
    write_fkeep,
)
from facekeep.aggressive.restorer import Restorer
from facekeep.config import AggressiveConfig, FaceKeepConfig
from facekeep.detector import FaceRegion, create_hand_detector
from facekeep.exceptions import ConfigError, DetectionError
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# Builders (shared style with test_region_local.py)
# --------------------------------------------------------------------------- #

def _natural_texture(h=1000, w=1500) -> np.ndarray:
    """Benign 'natural photo' texture (no sharp edges; reads as benign)."""
    rng = np.random.default_rng(3)
    bg = cv2.resize(
        rng.normal(128, 30, (h // 10, w // 10, 3)).astype(np.float32),
        (w, h), interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(bg, 0, 255).astype(np.uint8)


def _write_jpg(tmp_path, name, img) -> str:
    path = tmp_path / name
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(path)


def _face(x1, y1, x2, y2, *, padded=None, conf=0.9) -> FaceRegion:
    return FaceRegion(
        id=0, bbox=(x1, y1, x2, y2),
        padded_bbox=tuple(padded) if padded else (x1, y1, x2, y2),
        confidence=conf,
    )


def _patch_detector(monkeypatch, faces):
    class _Fixed:
        def detect(self, image):
            return list(faces)

    monkeypatch.setattr(compressor_mod, "create_detector", lambda **kw: _Fixed())


# A normal mid-frame face (NOT small): it won't trip the small-face region path,
# so any regions produced are purely the hand zones — isolating this feature.
def _normal_face():
    return _face(700, 300, 820, 460, padded=(640, 240, 880, 520))


class _FakeHandDetector:
    """Stand-in for a constructed C2 detector — returns canned boxes, no model."""

    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = 0

    def detect_hands(self, image):
        self.calls += 1
        return list(self._boxes)


# --------------------------------------------------------------------------- #
# A. C1 geometry (pure, offline)
# --------------------------------------------------------------------------- #

def test_hand_zones_two_side_bands_below_face():
    """A face yields two side bands at chest/waist height, below it."""
    f = _face(200, 200, 232, 242)  # 32x42 face
    zones = _hand_zones_from_faces([f], img_w=1500, img_h=1000)
    assert len(zones) == 2
    for (zx1, zy1, zx2, zy2) in zones:
        assert zy1 >= 242            # below the face bottom
        assert zx2 > zx1 and zy2 > zy1


def test_hand_zones_exclude_torso_centre():
    """The gap between the two bands straddles the face centre (hand, not torso)."""
    f = _face(200, 200, 232, 242)
    cx = (200 + 232) / 2
    left, right = sorted(
        _hand_zones_from_faces([f], 1500, 1000), key=lambda b: b[0]
    )
    assert left[2] < cx < right[0]   # neither band covers the centre column


def test_hand_zones_clamped_to_frame():
    """A face near an edge yields bands clamped to valid coordinates."""
    f = _face(10, 10, 42, 52)
    zones = _hand_zones_from_faces([f], img_w=1500, img_h=1000)
    for (zx1, zy1, zx2, zy2) in zones:
        assert 0 <= zx1 < zx2 <= 1500
        assert 0 <= zy1 < zy2 <= 1000


def test_hand_zones_empty_for_no_faces():
    assert _hand_zones_from_faces([], 1500, 1000) == []


def test_hand_zones_empty_on_degenerate_frame():
    assert _hand_zones_from_faces([_face(10, 10, 42, 52)], 0, 0) == []


# --------------------------------------------------------------------------- #
# B. The tier selector _hand_regions (C1 default / C2 opt-in)
# --------------------------------------------------------------------------- #

def test_hand_regions_c1_when_no_detector():
    cfg = AggressiveConfig()
    img = np.zeros((1000, 1500, 3), np.uint8)
    regions = _hand_regions(cfg, [_normal_face()], img, hand_detector=None)
    assert regions == _hand_zones_from_faces([_normal_face()], 1500, 1000)
    assert regions  # non-empty


def test_hand_regions_c2_boxes_preferred_over_c1():
    cfg = AggressiveConfig()
    img = np.zeros((1000, 1500, 3), np.uint8)
    fake = _FakeHandDetector([(100, 100, 180, 200)])
    regions = _hand_regions(cfg, [_normal_face()], img, hand_detector=fake)
    assert regions == [(100, 100, 180, 200)]  # the detector's boxes, not geometry
    assert fake.calls == 1


def test_hand_regions_c2_empty_falls_back_to_c1():
    cfg = AggressiveConfig()
    img = np.zeros((1000, 1500, 3), np.uint8)
    fake = _FakeHandDetector([])  # detector ran, found nothing
    regions = _hand_regions(cfg, [_normal_face()], img, hand_detector=fake)
    assert regions == _hand_zones_from_faces([_normal_face()], 1500, 1000)
    assert regions  # the geometric guess kicked in


def test_hand_regions_disabled_by_protect_hands_off():
    cfg = AggressiveConfig(protect_hands=False)
    img = np.zeros((1000, 1500, 3), np.uint8)
    assert _hand_regions(cfg, [_normal_face()], img, hand_detector=None) == []


def test_hand_regions_disabled_when_region_local_off():
    cfg = AggressiveConfig(region_local=False)
    img = np.zeros((1000, 1500, 3), np.uint8)
    assert _hand_regions(cfg, [_normal_face()], img, hand_detector=None) == []


def test_hand_regions_disabled_when_content_aware_off():
    cfg = AggressiveConfig(content_aware=False)
    img = np.zeros((1000, 1500, 3), np.uint8)
    assert _hand_regions(cfg, [_normal_face()], img, hand_detector=None) == []


# --------------------------------------------------------------------------- #
# B2. C1 over-coverage guard: merge overlapping bands + cap+bail on group photos
#
# On a dense group/family photo the per-face C1 bands stack up to cover a large
# slice of the frame (mostly torsos/laps with no hands), bloating the .fkeep
# larger than the source. The guard merges the redundant overlapping bands and
# *drops C1 hand protection entirely* once coverage exceeds hand_zone_max_frac.
# Only C1 (the geometric guess) is capped — C2 real detections are trusted.
# --------------------------------------------------------------------------- #

def _row_of_faces(n, *, w=1500, h=1000, fw=100, fh=120, top=300):
    """``n`` evenly-spaced same-height faces across the frame (a group photo)."""
    faces = []
    for i in range(n):
        cx = int((i + 0.5) * w / n)
        x1 = cx - fw // 2
        faces.append(_face(x1, top, x1 + fw, top + fh))
    return faces


def test_merge_unions_overlapping_boxes():
    a = (100, 100, 300, 300)
    b = (150, 150, 350, 350)        # overlaps a at IoU ~0.39 (> 0.2)
    out = _merge_overlapping_boxes([a, b], 0.2)
    assert out == [(100, 100, 350, 350)]  # one box bounding both


def test_merge_leaves_slightly_overlapping_boxes_separate():
    # IoU ~0.10 (below the 0.2 merge threshold) -> kept separate, not unioned.
    a = (100, 100, 300, 300)
    b = (250, 150, 450, 350)
    assert _merge_overlapping_boxes([a, b], 0.2) == [a, b]


def test_merge_keeps_disjoint_boxes():
    a = (0, 0, 100, 100)
    b = (900, 900, 1000, 1000)
    assert _merge_overlapping_boxes([a, b], 0.2) == [a, b]


def test_merge_empty():
    assert _merge_overlapping_boxes([], 0.2) == []


def test_c1_single_face_unchanged_by_guard():
    """A lone face is well under the cap and has no overlapping bands → no change."""
    cfg = AggressiveConfig()
    raw = _hand_zones_from_faces([_normal_face()], 1500, 1000)
    assert _c1_hand_zones(cfg, [_normal_face()], 1500, 1000) == raw
    assert raw  # non-empty (a few-face photo keeps its hand protection)


def test_c1_dense_group_bails_to_no_zones():
    """A dense row of faces blows past the coverage cap → C1 hand zones dropped."""
    faces = _row_of_faces(6)
    raw = _hand_zones_from_faces(faces, 1500, 1000)
    assert len(raw) > 6  # the un-guarded geometry emits many broad bands
    cfg = AggressiveConfig(hand_zone_max_frac=0.10)  # explicit low cap
    assert _c1_hand_zones(cfg, faces, 1500, 1000) == []


def test_c1_bail_is_the_cap_not_the_geometry():
    """The same dense group survives once the cap is lifted (the bail is the cap)."""
    faces = _row_of_faces(6)
    permissive = AggressiveConfig(hand_zone_max_frac=1.0)
    kept = _c1_hand_zones(permissive, faces, 1500, 1000)
    assert kept  # with no real cap the zones survive (merged), proving the bail
    assert sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in kept) > 0


def test_hand_regions_c1_group_photo_bails():
    """The tier selector routes C1 through the guard → group photo gets no zones."""
    cfg = AggressiveConfig(hand_zone_max_frac=0.10)
    img = np.zeros((1000, 1500, 3), np.uint8)
    assert _hand_regions(cfg, _row_of_faces(6), img, hand_detector=None) == []


def test_c2_detections_never_capped():
    """Real C2 boxes covering most of the frame are NOT subject to the C1 cap."""
    cfg = AggressiveConfig()
    img = np.zeros((1000, 1500, 3), np.uint8)
    # Huge boxes (well over hand_zone_max_frac of the frame) — trusted as-is.
    big = [(0, 0, 700, 900), (800, 0, 1500, 900)]
    fake = _FakeHandDetector(big)
    assert _hand_regions(cfg, _row_of_faces(6), img, hand_detector=fake) == big


# --------------------------------------------------------------------------- #
# C. De-dupe against small-face regions
# --------------------------------------------------------------------------- #

def test_dedupe_drops_box_covered_by_primary():
    primary = [(140, 280, 300, 380)]
    extra = [(150, 290, 280, 360), (900, 290, 980, 360)]  # 1st inside, 2nd outside
    out = _dedupe_regions(primary, extra, 1500, 1000)
    assert out == [(140, 280, 300, 380), (900, 290, 980, 360)]


def test_dedupe_keeps_non_overlapping():
    primary = [(0, 0, 50, 50)]
    extra = [(900, 290, 980, 360)]
    assert _dedupe_regions(primary, extra, 1500, 1000) == primary + extra


# --------------------------------------------------------------------------- #
# D. End-to-end compress_photo
# --------------------------------------------------------------------------- #

def test_compress_default_protects_hands(tmp_path, monkeypatch):
    """Default config (hands on, C1): a normal face yields hand region patches."""
    path = _write_jpg(tmp_path, "fam.jpg", _natural_texture())
    _patch_detector(monkeypatch, [_normal_face()])

    photo = compress_photo(path, FaceKeepConfig())  # protect_hands default True
    expected = _hand_zones_from_faces([_normal_face()], 1500, 1000)
    assert photo.regions == expected
    assert len(photo.region_crops) == len(expected)
    assert len(photo.region_masks) == len(expected)
    # Background stays aggressively compressed — hands are local, not a global raise.
    assert photo.effective_bg_scale == 0.25


def test_compress_hands_off_produces_no_regions(tmp_path, monkeypatch):
    path = _write_jpg(tmp_path, "fam.jpg", _natural_texture())
    _patch_detector(monkeypatch, [_normal_face()])

    cfg = FaceKeepConfig()
    cfg.aggressive.protect_hands = False
    photo = compress_photo(path, cfg)
    assert photo.regions == []
    assert photo.region_crops == [] and photo.region_masks == []


def test_compress_uses_c2_detector_boxes(tmp_path, monkeypatch):
    """A passed-in hand detector's tight boxes become the hand regions."""
    path = _write_jpg(tmp_path, "fam.jpg", _natural_texture())
    _patch_detector(monkeypatch, [_normal_face()])

    fake = _FakeHandDetector([(300, 500, 380, 600), (1000, 500, 1080, 600)])
    photo = compress_photo(path, FaceKeepConfig(), hand_detector=fake)
    assert photo.regions == [(300, 500, 380, 600), (1000, 500, 1080, 600)]
    assert fake.calls == 1
    # Each box was extracted to an original-resolution patch (hand_zone_scale 1.0).
    assert photo.region_crops[0].shape[:2] == (100, 80)


def test_compress_hand_zone_scale_downscales_patch(tmp_path, monkeypatch):
    path = _write_jpg(tmp_path, "fam.jpg", _natural_texture())
    _patch_detector(monkeypatch, [_normal_face()])

    fake = _FakeHandDetector([(300, 500, 380, 600)])  # 80x100 box
    cfg = FaceKeepConfig()
    cfg.aggressive.hand_zone_scale = 0.5
    photo = compress_photo(path, cfg, hand_detector=fake)
    assert photo.region_crops[0].shape[:2] == (50, 40)  # 100x80 at 0.5
    assert photo.regions[0] == (300, 500, 380, 600)      # bbox unchanged (full-res)


def test_compress_dedupes_hand_against_small_face(tmp_path, monkeypatch):
    """A hand zone overlapping a small-face patch isn't stored twice."""
    # A small face -> a small-face region; its hand zones sit below it. Force the
    # C2 detector to return a box fully inside the small face's padded box so the
    # dedupe drops it, leaving only the small-face region (no hand duplicate).
    small = _face(200, 200, 232, 242, padded=(160, 160, 280, 300))
    path = _write_jpg(tmp_path, "fam.jpg", _natural_texture())
    _patch_detector(monkeypatch, [small])

    fake = _FakeHandDetector([(180, 180, 260, 280)])  # inside (160,160,280,300)
    photo = compress_photo(path, FaceKeepConfig(), hand_detector=fake)
    # Only the small-face region survives; the contained hand box was de-duped.
    assert photo.regions == [(160, 160, 280, 300)]


# --------------------------------------------------------------------------- #
# E. .fkeep round-trip + verify + restore (reusing the generic region path)
# --------------------------------------------------------------------------- #

def _pack_with_hands(tmp_path, monkeypatch, name="hands.fkeep"):
    path = _write_jpg(tmp_path, "fam.jpg", _natural_texture())
    _patch_detector(monkeypatch, [_normal_face()])
    photo = compress_photo(path, FaceKeepConfig())
    fkeep = tmp_path / name
    write_fkeep(photo, str(fkeep))
    return str(fkeep), photo


def test_fkeep_with_hands_roundtrips_no_version_bump(tmp_path, monkeypatch):
    fkeep, photo = _pack_with_hands(tmp_path, monkeypatch)
    n = len(photo.regions)
    assert n >= 1
    with zipfile.ZipFile(fkeep) as zf:
        names = set(zf.namelist())
    assert f"region_{n - 1:03d}.jpg" in names
    assert f"region_mask_{n - 1:03d}.png" in names

    info = read_fkeep_info(fkeep)
    # NO format/manifest bump for hands: the version is whatever the current
    # schema is (1.9.0 = the high-bit residual bump), never a hands-specific one.
    assert info["version"] == "1.9.0"
    assert len(info["regions"]) == n

    data = read_fkeep(fkeep)
    assert len(data["region_crops"]) == n
    assert len(data["region_masks"]) == n


def test_verify_passes_with_hand_regions(tmp_path, monkeypatch):
    fkeep, photo = _pack_with_hands(tmp_path, monkeypatch)
    rep = verify_fkeep(fkeep)
    assert rep.ok, rep.problems
    assert rep.regions_declared == len(photo.regions)
    assert rep.region_crops_found == len(photo.regions)


def test_restore_composites_hand_patch_sharper_than_upscale(tmp_path, monkeypatch):
    """The restored hand region matches the original better than a pure upscale."""
    img = _natural_texture()
    # Paint sharp, high-frequency detail where a hand patch will be kept.
    hand_box = (300, 500, 380, 600)
    for x in range(hand_box[0], hand_box[2], 4):
        cv2.line(img, (x, hand_box[1]), (x, hand_box[3]), (10, 10, 10), 1)
    path = _write_jpg(tmp_path, "sharp_hand.jpg", img)
    _patch_detector(monkeypatch, [_normal_face()])

    cfg = FaceKeepConfig()
    fake = _FakeHandDetector([hand_box])
    photo = compress_photo(path, cfg, hand_detector=fake)
    assert hand_box in photo.regions
    fkeep = tmp_path / "sharp.fkeep"
    write_fkeep(photo, str(fkeep))

    restored = Restorer(cfg.aggressive).restore(fkeep)  # bicubic (no AI here)
    assert restored.shape[:2] == (photo.original_height, photo.original_width)

    data = read_fkeep(str(fkeep))
    ow, oh = photo.original_width, photo.original_height
    pure = cv2.resize(data["background"], (ow, oh), interpolation=cv2.INTER_CUBIC)
    hx1, hy1, hx2, hy2 = hand_box
    orig = cv2.imread(path)[hy1:hy2, hx1:hx2].astype(np.float32)
    restored_region = restored[hy1:hy2, hx1:hx2].astype(np.float32)
    pure_region = pure[hy1:hy2, hx1:hx2].astype(np.float32)

    err_restored = np.abs(restored_region - orig).mean()
    err_pure = np.abs(pure_region - orig).mean()
    assert err_restored < err_pure  # the kept patch is sharper than the upscale


# --------------------------------------------------------------------------- #
# F. create_hand_detector degradation
# --------------------------------------------------------------------------- #

def test_create_hand_detector_none_is_offline():
    assert create_hand_detector(None) is None
    assert create_hand_detector("anything-else") is None


def test_create_hand_detector_degrades_when_construct_fails(monkeypatch):
    """A backend that can't be built (no package/model) → None → caller uses C1."""
    import facekeep.detector as det

    def _boom(*a, **k):
        raise DetectionError("mediapipe missing")

    monkeypatch.setattr(det, "HandDetector", _boom)
    assert det.create_hand_detector("mediapipe") is None


# --------------------------------------------------------------------------- #
# G. Config validation + fingerprint
# --------------------------------------------------------------------------- #

def test_validate_accepts_known_backends():
    for backend in (None, "mediapipe"):
        cfg = FaceKeepConfig()
        cfg.aggressive.protect_hands_backend = backend
        cfg.validate()  # no raise


def test_validate_rejects_bogus_backend():
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_hands_backend = "yunet"  # no hand Haar/YuNet
    with pytest.raises(ConfigError, match="protect_hands_backend"):
        cfg.validate()


def test_validate_rejects_bad_hand_zone_scale():
    for bad in (0.0, -0.1, 1.5):
        cfg = FaceKeepConfig()
        cfg.aggressive.hand_zone_scale = bad
        with pytest.raises(ConfigError, match="hand_zone_scale"):
            cfg.validate()


def test_validate_rejects_bad_hand_zone_max_frac():
    for bad in (0.0, -0.1, 1.5):
        cfg = FaceKeepConfig()
        cfg.aggressive.hand_zone_max_frac = bad
        with pytest.raises(ConfigError, match="hand_zone_max_frac"):
            cfg.validate()


def test_yaml_roundtrip_hand_fields(tmp_path):
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"
    cfg.aggressive.protect_hands = False
    cfg.aggressive.protect_hands_backend = "mediapipe"
    cfg.aggressive.hand_zone_scale = 0.75
    cfg.aggressive.hand_zone_max_frac = 0.4
    p = tmp_path / "c.yaml"
    cfg.save(p)
    loaded = FaceKeepConfig.load(p)
    assert loaded.aggressive.protect_hands is False
    assert loaded.aggressive.protect_hands_backend == "mediapipe"
    assert loaded.aggressive.hand_zone_scale == 0.75
    assert loaded.aggressive.hand_zone_max_frac == 0.4


def test_fingerprint_busts_on_hand_fields():
    base = FaceKeepConfig()
    base.mode = "aggressive"
    fp0 = settings_fingerprint(base)

    for field, value in (
        ("protect_hands", False),
        ("protect_hands_backend", "mediapipe"),
        ("hand_zone_scale", 0.5),
        ("hand_zone_max_frac", 0.5),
    ):
        cfg = FaceKeepConfig()
        cfg.mode = "aggressive"
        setattr(cfg.aggressive, field, value)
        assert settings_fingerprint(cfg) != fp0, field


def test_fingerprint_faithful_unaffected_by_hand_fields():
    base = FaceKeepConfig()  # faithful
    fp0 = settings_fingerprint(base)
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_hands = False
    cfg.aggressive.hand_zone_scale = 0.5
    assert settings_fingerprint(cfg) == fp0  # faithful fingerprint ignores aggressive


# --------------------------------------------------------------------------- #
# H. C2 detection tuning: downscale-before-detect + recall knobs
#
# The real MediaPipe path can't run offline, so these build a HandDetector via
# __new__ (skipping the mediapipe import in __init__) and inject a fake landmarker.
# They pin the bug fix — that downscaling the *detection input* must NOT shrink the
# returned boxes (landmarks are normalized → boxes are in the ORIGINAL frame).
# --------------------------------------------------------------------------- #

from facekeep.detector import HandDetector  # noqa: E402 - grouped with its tests


class _FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeLandmarker:
    """Records the size of the image it was handed; returns configurable hands.

    Each hand is a normalized ``(x1, y1, x2, y2)`` extent; the landmarker yields two
    corner landmarks per hand (min/max), which is all ``detect_hands`` reads.
    Default is one hand spanning x∈[0.25,0.75], y∈[0.10,0.40] — independent of the
    input pixel size (the point of normalized coords), so existing tests are
    unchanged.
    """

    def __init__(self, hands=((0.25, 0.10, 0.75, 0.40),)):
        self.seen_shape = None
        self._hands = hands

    def detect(self, mp_image):
        self.seen_shape = np.asarray(mp_image).shape[:2]
        hand_landmarks = [
            [_FakeLandmark(x1, y1), _FakeLandmark(x2, y2)]
            for (x1, y1, x2, y2) in self._hands
        ]
        return type("R", (), {"hand_landmarks": hand_landmarks})()


class _FakeMpImage:
    """Minimal stand-in for mediapipe.Image: np.asarray() returns the data."""

    def __init__(self, image_format=None, data=None):
        self._data = data

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)


class _FakeMp:
    Image = _FakeMpImage

    class ImageFormat:
        SRGB = "srgb"


def _hand_detector_with_fake(detect_long_side=1280, padding=1.0, hands=None):
    """A HandDetector wired to the fake landmarker (no mediapipe import)."""
    det = HandDetector.__new__(HandDetector)
    det.confidence = 0.3
    det.num_hands = 6
    det.detect_long_side = detect_long_side
    det.padding = padding
    det._mp = _FakeMp
    det._landmarker = (
        _FakeLandmarker(hands) if hands is not None else _FakeLandmarker()
    )
    return det


def test_detect_hands_boxes_are_in_original_frame_after_downscale():
    """Downscaling the detection input must keep boxes in full-resolution pixels."""
    det = _hand_detector_with_fake(detect_long_side=1280, padding=1.0)
    img = np.zeros((4096, 3072, 3), np.uint8)  # 12 MP, long side 4096 > 1280
    boxes = det.detect_hands(img)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    # Normalized 0.25..0.75 of width 3072, 0.10..0.40 of height 4096 -> ORIGINAL px.
    assert (x1, x2) == (768, 2304)
    assert (y1, y2) == (409, 1638)
    # And the landmarker actually saw a downscaled input (long side ~1280).
    assert max(det._landmarker.seen_shape) <= 1280


def test_detect_hands_no_downscale_when_disabled():
    det = _hand_detector_with_fake(detect_long_side=0, padding=1.0)
    img = np.zeros((4096, 3072, 3), np.uint8)
    det.detect_hands(img)
    assert det._landmarker.seen_shape == (4096, 3072)  # full-res input


def test_detect_hands_small_image_not_upscaled():
    """A frame already under the long-side budget is detected at native size."""
    det = _hand_detector_with_fake(detect_long_side=1280, padding=1.0)
    img = np.zeros((600, 800, 3), np.uint8)  # long side 800 < 1280
    det.detect_hands(img)
    assert det._landmarker.seen_shape == (600, 800)


def test_detect_hands_padding_widens_box():
    """A larger padding expands the kept box around the tight landmark box."""
    tight = _hand_detector_with_fake(detect_long_side=0, padding=1.0).detect_hands(
        np.zeros((1000, 1000, 3), np.uint8)
    )[0]
    padded = _hand_detector_with_fake(detect_long_side=0, padding=1.5).detect_hands(
        np.zeros((1000, 1000, 3), np.uint8)
    )[0]
    tw = tight[2] - tight[0]
    pw = padded[2] - padded[0]
    assert pw > tw


def test_create_hand_detector_threads_tuning_params(monkeypatch):
    """create_hand_detector passes the recall knobs into HandDetector."""
    import facekeep.detector as det

    captured = {}

    def _fake_ctor(**kw):
        captured.update(kw)
        return "sentinel"

    monkeypatch.setattr(det, "HandDetector", _fake_ctor)
    out = det.create_hand_detector(
        "mediapipe", confidence=0.2, num_hands=8,
        detect_long_side=1600, padding=1.4,
    )
    assert out == "sentinel"
    assert captured == dict(
        confidence=0.2, num_hands=8, detect_long_side=1600, padding=1.4
    )


# --- config validation + fingerprint for the C2 tuning knobs ----------------- #

def test_validate_c2_tuning_ranges():
    bad_cases = [
        ("hand_detect_confidence", -0.1),
        ("hand_detect_confidence", 1.1),
        ("hand_detect_max_hands", 0),
        ("hand_detect_long_side", -1),
        ("hand_detect_padding", 0.9),
    ]
    for field, value in bad_cases:
        cfg = FaceKeepConfig()
        setattr(cfg.aggressive, field, value)
        with pytest.raises(ConfigError, match=field):
            cfg.validate()


def test_validate_accepts_c2_tuning_defaults_and_long_side_zero():
    cfg = FaceKeepConfig()
    cfg.validate()  # defaults are valid
    cfg.aggressive.hand_detect_long_side = 0  # 0 = no downscale, allowed
    cfg.validate()


def test_yaml_roundtrip_c2_tuning_fields(tmp_path):
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"
    cfg.aggressive.hand_detect_confidence = 0.2
    cfg.aggressive.hand_detect_max_hands = 8
    cfg.aggressive.hand_detect_long_side = 1600
    cfg.aggressive.hand_detect_padding = 1.4
    p = tmp_path / "c.yaml"
    cfg.save(p)
    loaded = FaceKeepConfig.load(p)
    assert loaded.aggressive.hand_detect_confidence == 0.2
    assert loaded.aggressive.hand_detect_max_hands == 8
    assert loaded.aggressive.hand_detect_long_side == 1600
    assert loaded.aggressive.hand_detect_padding == 1.4


def test_fingerprint_busts_on_c2_tuning_fields():
    base = FaceKeepConfig()
    base.mode = "aggressive"
    fp0 = settings_fingerprint(base)
    for field, value in (
        ("hand_detect_confidence", 0.2),
        ("hand_detect_max_hands", 8),
        ("hand_detect_long_side", 1600),
        ("hand_detect_padding", 1.4),
    ):
        cfg = FaceKeepConfig()
        cfg.mode = "aggressive"
        setattr(cfg.aggressive, field, value)
        assert settings_fingerprint(cfg) != fp0, field


# --------------------------------------------------------------------------- #
# I. De-dupe overlapping hand detections (NMS)
#
# MediaPipe can emit two near-identical boxes for one physical hand (observed on a
# real photo: IoU 0.69). detect_hands NMS-de-duplicates so each hand yields one
# region patch. The occluded-hand miss (a hand wrapped by a held object) is a
# detector limit with no parameter fix — not tested here (it's an accepted limit).
# --------------------------------------------------------------------------- #

from facekeep.detector import _nms_boxes  # noqa: E402 - grouped with its tests


def test_nms_merges_overlapping_keeps_larger():
    # The real duplicate pair from the repro photo (IoU ~0.69).
    a = (1039, 2571, 1166, 2786)  # 127x215
    b = (1031, 2560, 1194, 2766)  # 163x206 (larger)
    out = _nms_boxes([a, b], 0.4)
    assert out == [b]  # one survivor, the larger box


def test_nms_keeps_disjoint_boxes():
    c = (100, 100, 200, 200)
    d = (900, 900, 1000, 1000)
    assert sorted(_nms_boxes([c, d], 0.4)) == sorted([c, d])


def test_nms_identical_collapses_to_one():
    a = (10, 10, 110, 110)
    assert _nms_boxes([a, a], 0.4) == [a]


def test_nms_empty():
    assert _nms_boxes([], 0.4) == []


def test_detect_hands_dedupes_one_physical_hand():
    """Two overlapping detections for one hand → a single returned box."""
    # Two normalized extents that overlap heavily (one hand seen twice).
    hands = [(0.30, 0.30, 0.50, 0.70), (0.31, 0.31, 0.51, 0.71)]
    det = _hand_detector_with_fake(detect_long_side=0, padding=1.0, hands=hands)
    boxes = det.detect_hands(np.zeros((1000, 1000, 3), np.uint8))
    assert len(boxes) == 1


def test_detect_hands_keeps_two_distinct_hands():
    """Two clearly separate hands are both kept (no over-merging)."""
    hands = [(0.05, 0.05, 0.20, 0.40), (0.70, 0.55, 0.90, 0.95)]
    det = _hand_detector_with_fake(detect_long_side=0, padding=1.0, hands=hands)
    boxes = det.detect_hands(np.zeros((1000, 1000, 3), np.uint8))
    assert len(boxes) == 2
