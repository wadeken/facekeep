"""Region-local conservatism — ROADMAP Phase 4 (the per-region scale map item).

This is the deferred *second half* of content-aware conservatism. Where the
whole-image step (tests/test_content_aware.py) raised the single ``bg_scale`` for
the *entire* background when a risky region was found, region-local conservatism
keeps the benign majority of the frame aggressively compressed and protects only
the risky region — by storing it as a near-original-resolution patch in the
``.fkeep`` and compositing it back on restore. It reuses the existing face-crop
mechanism (a region patch is just a non-face crop with its own soft mask).

Scope of *this* item (a deliberate, single ROADMAP step): only the
**small/distant-face** risk is localized (each such face has a clean bbox — the
ROADMAP's worst-failure case). The edge-density/text signal stays whole-image
(still handled by ``_resolve_bg_scale``) unless the opt-in ``protect_text``
localizer is enabled — that follow-up is covered in
tests/test_text_region_local.py. Default is ``region_local=True``.

What these tests pin:

* ``_risky_regions`` returns the small face's padded box, is empty on benign /
  large-face / disabled inputs, and clamps to the frame;
* end-to-end through ``compress_photo`` a small-face photo produces region
  crops+masks and keeps the *whole-image* ``bg_scale`` aggressive (the local
  protection replaces the whole-image raise), while a benign photo produces no
  regions and is byte-unaffected;
* the ``.fkeep`` carries ``region_NNN.*`` + ``region_mask_NNN.png`` + a
  ``regions[]`` manifest array, bumps to v1.3.0, round-trips through
  ``read_fkeep``, and ``verify_fkeep`` passes (and catches a missing region member);
* restore composites the region patches (the output is correct-sized and the
  region area is sharper than a pure upscale);
* **backward compatibility**: a v1.2.0 ``.fkeep`` (no ``regions`` key) still
  reads, verifies, and restores unchanged;
* the new output-affecting fields bust the aggressive index fingerprint and
  leave faithful's untouched; ``validate()`` range-checks ``region_scale``.

Detection is mocked where a deterministic face list is needed, so these tests
need no network/model.
"""

import json
import zipfile

import cv2
import numpy as np
import pytest

import facekeep.aggressive.compressor as compressor_mod
from facekeep.aggressive.compressor import _risky_regions, compress_photo
from facekeep.aggressive.format import (
    read_fkeep,
    read_fkeep_info,
    verify_fkeep,
    write_fkeep,
)
from facekeep.aggressive.restorer import Restorer
from facekeep.config import AggressiveConfig, FaceKeepConfig
from facekeep.detector import FaceRegion
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# Builders (shared style with test_content_aware.py)
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


def _no_hands_config(**aggressive_overrides) -> FaceKeepConfig:
    """A config with hand protection OFF, to isolate the small-face region path.

    Hand protection (default on) also emits region patches (the offline C1 hand
    zones below a face), so these small-face-region tests turn it off to assert
    on *only* the small-face region they're exercising. Hand protection has its
    own coverage in tests/test_hand_protection.py.
    """
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_hands = False
    for k, v in aggressive_overrides.items():
        setattr(cfg.aggressive, k, v)
    return cfg


# A small (~30px) face on a 1000px-short frame = 3% < 4% default -> risky; the
# padded box extends into the surrounding background (the region we protect).
def _small_face():
    return _face(200, 200, 232, 242, padded=(160, 160, 280, 300))


# --------------------------------------------------------------------------- #
# A. The pure region selector
# --------------------------------------------------------------------------- #

def test_risky_regions_returns_small_face_padded_box():
    cfg = AggressiveConfig()
    regions = _risky_regions(cfg, [_small_face()], img_w=1500, img_h=1000)
    assert regions == [(160, 160, 280, 300)]


def test_risky_regions_empty_for_large_face():
    cfg = AggressiveConfig()
    big = _face(400, 300, 700, 690, padded=(340, 240, 760, 750))
    assert _risky_regions(cfg, [big], 1500, 1000) == []


def test_risky_regions_empty_when_region_local_off():
    cfg = AggressiveConfig(region_local=False)
    assert _risky_regions(cfg, [_small_face()], 1500, 1000) == []


def test_risky_regions_empty_when_content_aware_off():
    cfg = AggressiveConfig(content_aware=False)
    assert _risky_regions(cfg, [_small_face()], 1500, 1000) == []


def test_risky_regions_clamps_to_frame():
    """A padded box spilling past the frame is clamped to valid coordinates."""
    cfg = AggressiveConfig()
    f = _face(10, 10, 42, 52, padded=(-40, -40, 90, 120))
    regions = _risky_regions(cfg, [f], img_w=1500, img_h=1000)
    assert regions == [(0, 0, 90, 120)]


def test_risky_regions_empty_on_degenerate_frame():
    cfg = AggressiveConfig()
    assert _risky_regions(cfg, [_small_face()], 0, 0) == []


# --------------------------------------------------------------------------- #
# B. End-to-end compress_photo: regions extracted, whole-image scale kept
# --------------------------------------------------------------------------- #

def test_compress_small_face_keeps_base_scale_and_extracts_region(
    tmp_path, monkeypatch
):
    """The default path: small face -> a sharp region patch, NOT a whole-image raise."""
    path = _write_jpg(tmp_path, "distant.jpg", _natural_texture())
    _patch_detector(monkeypatch, [_small_face()])

    cfg = _no_hands_config()  # region_local on, bg_scale 0.25, hands off (isolate)
    photo = compress_photo(path, cfg)

    # Whole-image scale stays aggressive (local protection replaces the raise).
    assert photo.effective_bg_scale == cfg.aggressive.bg_scale == 0.25
    # One region patch + mask was extracted, covering the clamped padded box.
    assert len(photo.regions) == 1
    assert len(photo.region_crops) == 1
    assert len(photo.region_masks) == 1
    rx1, ry1, rx2, ry2 = photo.regions[0]
    assert (rx2 - rx1, ry2 - ry1) == (120, 140)  # padded box size, in-frame
    assert photo.region_crops[0].shape[:2] == (140, 120)  # region_scale 1.0


def test_compress_benign_produces_no_regions(tmp_path, monkeypatch):
    """A benign photo with a normal face produces zero region patches."""
    img = _natural_texture()
    cv2.ellipse(img, (750, 500), (160, 210), 0, 0, 360, (180, 170, 165), -1)
    path = _write_jpg(tmp_path, "portrait.jpg", img)
    _patch_detector(monkeypatch, [_face(590, 290, 910, 710)])

    cfg = FaceKeepConfig()
    photo = compress_photo(path, cfg)
    assert photo.regions == []
    assert photo.region_crops == [] and photo.region_masks == []
    assert photo.effective_bg_scale == 0.25


def test_region_scale_downscales_stored_patch(tmp_path, monkeypatch):
    """region_scale < 1.0 stores a smaller patch (still composited to the bbox)."""
    path = _write_jpg(tmp_path, "distant.jpg", _natural_texture())
    _patch_detector(monkeypatch, [_small_face()])

    cfg = FaceKeepConfig()
    cfg.aggressive.region_scale = 0.5
    photo = compress_photo(path, cfg)
    # 120x140 padded box at 0.5 -> 60x70 stored patch.
    assert photo.region_crops[0].shape[:2] == (70, 60)
    assert photo.regions[0] == (160, 160, 280, 300)  # bbox unchanged (full-res coords)


# --------------------------------------------------------------------------- #
# C. .fkeep round-trip + manifest + verify
# --------------------------------------------------------------------------- #

def _pack_small_face(tmp_path, monkeypatch, name="rl.fkeep", cfg=None):
    path = _write_jpg(tmp_path, "distant.jpg", _natural_texture())
    _patch_detector(monkeypatch, [_small_face()])
    # Hands off so the packed .fkeep has exactly the one small-face region these
    # round-trip/verify tests assert on (hand zones are tested separately).
    photo = compress_photo(path, cfg or _no_hands_config())
    fkeep = tmp_path / name
    write_fkeep(photo, str(fkeep))
    return str(fkeep)


def test_fkeep_has_region_members_and_manifest(tmp_path, monkeypatch):
    fkeep = _pack_small_face(tmp_path, monkeypatch)
    with zipfile.ZipFile(fkeep) as zf:
        names = set(zf.namelist())
    assert "region_000.jpg" in names  # default face_codec is jpg
    assert "region_mask_000.png" in names

    info = read_fkeep_info(fkeep)
    # regions[] were added at 1.3.0; the current schema is 1.8.0 (the high-bit
    # bit_depth key).
    assert info["version"] == "1.8.0"
    assert len(info["regions"]) == 1
    r = info["regions"][0]
    assert r["id"] == 0
    assert r["bbox"] == [160, 160, 280, 300]
    assert r["scale"] == 1.0


def test_read_fkeep_returns_region_arrays(tmp_path, monkeypatch):
    fkeep = _pack_small_face(tmp_path, monkeypatch)
    data = read_fkeep(fkeep)
    assert len(data["region_crops"]) == 1
    assert len(data["region_masks"]) == 1
    assert data["region_crops"][0].shape[:2] == (140, 120)


def test_verify_passes_on_region_fkeep(tmp_path, monkeypatch):
    fkeep = _pack_small_face(tmp_path, monkeypatch)
    rep = verify_fkeep(fkeep)
    assert rep.ok, rep.problems
    assert rep.regions_declared == 1
    assert rep.region_crops_found == 1
    assert rep.region_masks_found == 1


def test_verify_catches_missing_region_crop(tmp_path, monkeypatch):
    """Dropping a region crop makes verify report inconsistent (not crash)."""
    fkeep = _pack_small_face(tmp_path, monkeypatch)
    # Rewrite the archive without the region crop member.
    with zipfile.ZipFile(fkeep) as zf:
        members = {n: zf.read(n) for n in zf.namelist() if n != "region_000.jpg"}
    with zipfile.ZipFile(fkeep, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, b in members.items():
            zf.writestr(n, b)

    rep = verify_fkeep(fkeep)
    assert not rep.ok
    assert any("region" in p for p in rep.problems)


def test_verify_catches_malformed_region_bbox(tmp_path, monkeypatch):
    fkeep = _pack_small_face(tmp_path, monkeypatch)
    with zipfile.ZipFile(fkeep) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["regions"][0]["bbox"] = [5, 5, 5, 5]  # x2==x1 -> not well-formed
    members["manifest.json"] = json.dumps(manifest)
    with zipfile.ZipFile(fkeep, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, b in members.items():
            zf.writestr(n, b)

    rep = verify_fkeep(fkeep)
    assert not rep.ok
    assert any("region 0" in p and "bbox" in p for p in rep.problems)


# --------------------------------------------------------------------------- #
# D. Restore composites region patches
# --------------------------------------------------------------------------- #

def test_restore_composites_region_patch_sharper_than_upscale(tmp_path, monkeypatch):
    """The restored region area is closer to the original patch than a pure upscale.

    Build a photo whose small-face region carries sharp content the downsample
    would blur away. After restore, that region should match the original crop
    better than the bicubic-upscaled background alone does (the patch was kept).
    """
    img = _natural_texture()
    # Paint sharp, high-frequency detail inside the region we will protect.
    for x in range(160, 280, 6):
        cv2.line(img, (x, 160), (x, 300), (10, 10, 10), 1)
    path = _write_jpg(tmp_path, "sharp_region.jpg", img)
    _patch_detector(monkeypatch, [_small_face()])

    cfg = _no_hands_config()  # isolate the single small-face region
    photo = compress_photo(path, cfg)
    assert len(photo.regions) == 1
    fkeep = tmp_path / "sharp.fkeep"
    write_fkeep(photo, str(fkeep))

    restored = Restorer(cfg.aggressive).restore(fkeep)  # bicubic (no AI here)
    assert restored.shape[:2] == (photo.original_height, photo.original_width)

    # Compare the region area: restored-with-patch vs a pure bicubic upscale.
    data = read_fkeep(str(fkeep))
    ow, oh = photo.original_width, photo.original_height
    pure = cv2.resize(data["background"], (ow, oh), interpolation=cv2.INTER_CUBIC)
    rx1, ry1, rx2, ry2 = photo.regions[0]
    orig = cv2.imread(path)[ry1:ry2, rx1:rx2].astype(np.float32)
    restored_region = restored[ry1:ry2, rx1:rx2].astype(np.float32)
    pure_region = pure[ry1:ry2, rx1:rx2].astype(np.float32)

    err_restored = np.abs(restored_region - orig).mean()
    err_pure = np.abs(pure_region - orig).mean()
    assert err_restored < err_pure  # the kept patch is closer to the original


# --------------------------------------------------------------------------- #
# E. Backward compatibility: a pre-1.3.0 .fkeep (no regions) still works
# --------------------------------------------------------------------------- #

def _make_v120_fkeep(tmp_path, monkeypatch):
    """Pack a normal (no-region) .fkeep and rewrite it to look like v1.2.0."""
    img = _natural_texture()
    cv2.ellipse(img, (750, 500), (160, 210), 0, 0, 360, (180, 170, 165), -1)
    path = _write_jpg(tmp_path, "old.jpg", img)
    _patch_detector(monkeypatch, [_face(590, 290, 910, 710)])
    photo = compress_photo(path, FaceKeepConfig())
    assert photo.regions == []  # benign -> no regions anyway
    fkeep = tmp_path / "old.fkeep"
    write_fkeep(photo, str(fkeep))

    # Strip the regions key + downgrade version to emulate an older writer.
    with zipfile.ZipFile(fkeep) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest.pop("regions", None)
    manifest["version"] = "1.2.0"
    members["manifest.json"] = json.dumps(manifest)
    with zipfile.ZipFile(fkeep, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, b in members.items():
            zf.writestr(n, b)
    return str(fkeep)


def test_old_fkeep_without_regions_reads(tmp_path, monkeypatch):
    fkeep = _make_v120_fkeep(tmp_path, monkeypatch)
    data = read_fkeep(fkeep)
    assert data["region_crops"] == []
    assert data["region_masks"] == []


def test_old_fkeep_without_regions_verifies(tmp_path, monkeypatch):
    fkeep = _make_v120_fkeep(tmp_path, monkeypatch)
    rep = verify_fkeep(fkeep)
    assert rep.ok, rep.problems
    assert rep.regions_declared == 0


def test_old_fkeep_without_regions_restores(tmp_path, monkeypatch):
    fkeep = _make_v120_fkeep(tmp_path, monkeypatch)
    info = read_fkeep_info(fkeep)
    out = Restorer().restore(fkeep)
    assert out.shape[:2] == (info["original"]["height"], info["original"]["width"])


# --------------------------------------------------------------------------- #
# F. Index fingerprint + validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mutate", [
    lambda c: setattr(c.aggressive, "region_local", False),
    lambda c: setattr(c.aggressive, "region_scale", 0.5),
])
def test_fingerprint_busts_on_region_change(mutate):
    base = FaceKeepConfig(mode="aggressive")
    changed = FaceKeepConfig(mode="aggressive")
    mutate(changed)
    assert settings_fingerprint(base) != settings_fingerprint(changed)


def test_faithful_fingerprint_unaffected_by_region_fields():
    base = FaceKeepConfig(mode="faithful")
    changed = FaceKeepConfig(mode="faithful")
    changed.aggressive.region_local = False
    changed.aggressive.region_scale = 0.5
    assert settings_fingerprint(base) == settings_fingerprint(changed)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_validate_rejects_bad_region_scale(value):
    cfg = FaceKeepConfig()
    cfg.aggressive.region_scale = value
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validate_accepts_region_defaults():
    FaceKeepConfig().validate()  # region_local=True, region_scale=1.0


def test_region_fields_yaml_round_trip(tmp_path):
    cfg = FaceKeepConfig()
    cfg.aggressive.region_local = False
    cfg.aggressive.region_scale = 0.75
    p = tmp_path / "facekeep.yaml"
    cfg.save(p)
    loaded = FaceKeepConfig.load(p)
    assert loaded.aggressive.region_local is False
    assert loaded.aggressive.region_scale == 0.75
