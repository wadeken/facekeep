"""Text-region localization — the edge/text follow-up of region-local conservatism.

The whole-image edge signal (tests/test_content_aware.py) only fires when the
*entire frame* is edge-dense, so a small sign/text block in a big photo gets no
protection at all — yet text is exactly what the AI upscale mangles (garbled
glyphs). ``compressor._text_regions`` localizes the signal: a coarse per-tile
scan over the shared edge map marks text-like tiles, merges them into clusters,
and each cluster is stored as a sharp region patch (the same ``region_NNN.*``
mechanism as small faces/hands — no ``.fkeep``/manifest change), while the
benign rest of the frame keeps the aggressive ``bg_scale``. Widespread risk
(document-like content) bails to the whole-image raise as before.

**``protect_text`` is opt-in (default off), unlike the other region
protections.** The discriminator is still the zero-download edge *proxy*, NOT
OCR, and measured on the real corpus it cannot tell text from benign-but-sharp
organic content at tile granularity (fern/ridge clusters trip it on
landscapes). A false patch is only size waste, but silently growing benign
photos' ``.fkeep`` would break the defaults-must-not-change-benign-output rule
— so the default path stays byte-identical and signage-heavy libraries opt in
(the yunet/mediapipe precedent).

What these tests pin:

* the pure ``_text_regions`` localizer: finds a localized sign (opt-in), stays
  empty on benign/smooth content, **bails on document-like content** (the
  economy rule), respects exclude boxes, drops single-tile noise, clamps to the
  frame, is uint16/empty-safe, and is OFF by default and under each gate;
* ``_resolve_bg_scale``'s ``text_locally_handled`` flag suppresses the
  whole-image raise only when patches were actually emitted (and the
  precomputed ``detail_ratio`` is honored);
* end-to-end through ``compress_photo``: a sign photo (opt-in) keeps the
  aggressive ``bg_scale`` *and* emits a patch covering the sign; the default
  config emits nothing (anti-regression); a document photo falls back to the
  whole-image raise; a text cluster already covered by a small-face region is
  not stored twice;
* the ``.fkeep`` round-trips and verifies with a text region, and restore
  composites the patch (the sign area beats a pure upscale);
* the three new output-affecting fields bust the aggressive index fingerprint
  (faithful untouched), ``validate()`` range-checks them, and they survive a
  YAML round-trip.

Detection is mocked where a deterministic face list is needed, so these tests
need no network/model.
"""

import cv2
import numpy as np
import pytest

import facekeep.aggressive.compressor as compressor_mod
from facekeep.aggressive.compressor import (
    _resolve_bg_scale,
    _text_regions,
    compress_photo,
)
from facekeep.aggressive.format import read_fkeep, verify_fkeep, write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.config import AggressiveConfig, FaceKeepConfig
from facekeep.detector import FaceRegion
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# Builders (shared style with test_region_local.py / test_content_aware.py)
# --------------------------------------------------------------------------- #

def _natural_texture(h=1200, w=1800) -> np.ndarray:
    """Benign 'natural photo' texture (no sharp edges; reads as benign)."""
    rng = np.random.default_rng(3)
    bg = cv2.resize(
        rng.normal(128, 30, (h // 10, w // 10, 3)).astype(np.float32),
        (w, h), interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(bg, 0, 255).astype(np.uint8)


def _smooth(h=1200, w=1800) -> np.ndarray:
    yy = np.linspace(60, 200, h)[:, None]
    xx = np.linspace(40, 160, w)[None, :]
    base = (yy + xx) / 2.0
    return np.clip(np.stack([base, base * 0.95, base * 0.9], axis=-1), 0, 255).astype(
        np.uint8
    )


# The sign's text block (dense storefront-style glyphs). The localizer should
# cover *the glyph area*; the blank right margin of the board may stay outside.
SIGN_BOX = (1300, 80, 1720, 400)


def _paint_sign(img: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = SIGN_BOX
    cv2.rectangle(img, (x1, y1), (x2, y2), (245, 245, 245), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 4)
    lines = ["FAMILY MART 24h", "OPEN 0123456789", "COFFEE TEA 35$",
             "WELCOME * SALE", "EXIT 7-11 ATM"]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x1 + 14, y1 + 52 + i * 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 40), 2)
    return img


def _sign_photo() -> np.ndarray:
    """A localized dense sign on a benign background (global edge ratio quiet)."""
    return _paint_sign(_natural_texture())


def _dense_text(h=1200, w=1800) -> np.ndarray:
    """A page of dense text — document-like, risky *everywhere*."""
    img = np.full((h, w, 3), 230, np.uint8)
    y = 30
    while y < h - 10:
        cv2.putText(img, "The quick brown fox jumps 0123456789 " * 2, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 1)
        y += 28
    return img


def _opt_in(**overrides) -> AggressiveConfig:
    return AggressiveConfig(protect_text=True, **overrides)


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


# A normal-size face far from the sign (so neither the no-face fallback nor the
# small-face signal fires, isolating the text path).
def _normal_face():
    return _face(100, 300, 400, 600, padded=(40, 240, 460, 660))


def _patch_detector(monkeypatch, faces):
    class _Fixed:
        def detect(self, image):
            return list(faces)

    monkeypatch.setattr(compressor_mod, "create_detector", lambda **kw: _Fixed())


def _text_config(**aggressive_overrides) -> FaceKeepConfig:
    """Opt-in config with hand protection OFF, isolating the text-region path."""
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_hands = False
    cfg.aggressive.protect_text = True
    for k, v in aggressive_overrides.items():
        setattr(cfg.aggressive, k, v)
    return cfg


def _covers(box, x, y) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


# --------------------------------------------------------------------------- #
# A. The pure localizer
# --------------------------------------------------------------------------- #

def test_text_regions_off_by_default():
    """protect_text defaults to False: even a dense sign yields no regions."""
    cfg = AggressiveConfig()
    assert cfg.protect_text is False
    assert _text_regions(cfg, _sign_photo(), []) == []


def test_text_regions_finds_localized_sign():
    boxes = _text_regions(_opt_in(), _sign_photo(), [])
    assert len(boxes) == 1
    # The box must cover the glyph area (assert on the text centre, not the
    # whole board — the board's blank margin has no edges to find).
    assert _covers(boxes[0], 1450, 240)
    x1, y1, x2, y2 = boxes[0]
    assert (x2 - x1) * (y2 - y1) < 0.3 * 1800 * 1200  # localized, not the frame


def test_text_regions_benign_content_empty():
    assert _text_regions(_opt_in(), _smooth(), []) == []
    assert _text_regions(_opt_in(), _natural_texture(), []) == []


def test_text_regions_document_bails_to_whole_image():
    """Wall-to-wall text is not economical to patch -> [] (whole-image raise).

    Anti-false-green: with the economy cap lifted the same image *does* produce
    clusters, proving the empty default-cap result is the bail-out, not
    blindness.
    """
    doc = _dense_text()
    assert _text_regions(_opt_in(), doc, []) == []
    uncapped = _text_regions(_opt_in(text_region_max_frac=1.0), doc, [])
    assert uncapped, "economy-cap test image produced no clusters at all"


def test_text_regions_exclude_boxes_suppress_cluster():
    """A sign fully inside an exclude box (e.g. a face crop) yields no patch."""
    img = _sign_photo()
    pad = 60  # cover the cluster's outward padding too
    x1, y1, x2, y2 = SIGN_BOX
    exclude = [(x1 - pad, y1 - pad, x2 + pad, y2 + pad)]
    assert _text_regions(_opt_in(), img, exclude) == []


def test_text_regions_single_tile_noise_dropped():
    """One isolated sharp blip (a single risky tile) is noise, not a sign."""
    img = _natural_texture()
    # A tiny dense scribble well inside one 112x75 tile.
    for k in range(0, 40, 4):
        cv2.line(img, (900 + k, 600), (900 + k, 640), (10, 10, 10), 1)
    assert _text_regions(_opt_in(), img, []) == []


def test_text_regions_clamped_to_frame():
    img = _natural_texture()
    # Dense text flush against the top-right corner: the padded cluster bbox
    # would spill past the frame without clamping.
    for i in range(5):
        cv2.putText(img, "EDGE TEXT 0123456789", (1380, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    boxes = _text_regions(_opt_in(), img, [])
    assert boxes
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= 1800
        assert 0 <= y1 < y2 <= 1200


def test_text_regions_uint16_safe():
    img16 = (_sign_photo().astype(np.uint16)) * 257
    boxes = _text_regions(_opt_in(), img16, [])
    assert len(boxes) == 1 and _covers(boxes[0], 1450, 240)


def test_text_regions_empty_image_safe():
    assert _text_regions(_opt_in(), np.zeros((0, 0, 3), np.uint8), []) == []


def test_text_regions_respects_gates():
    img = _sign_photo()
    assert _text_regions(_opt_in(content_aware=False), img, []) == []
    assert _text_regions(_opt_in(region_local=False), img, []) == []


# --------------------------------------------------------------------------- #
# B. The whole-image decision composition
# --------------------------------------------------------------------------- #

def test_resolve_text_locally_handled_suppresses_raise():
    """Patches were emitted -> the edge risk must not *also* raise the scale."""
    cfg = AggressiveConfig()
    scale, reason = _resolve_bg_scale(
        cfg, faces=[], image=_dense_text(), base_scale=cfg.bg_scale,
        text_locally_handled=True,
    )
    assert scale == cfg.bg_scale and reason is None


def test_resolve_without_flag_still_raises():
    """The localizer bailed (flag False) -> the old whole-image raise fires."""
    cfg = AggressiveConfig()
    scale, reason = _resolve_bg_scale(
        cfg, faces=[], image=_dense_text(), base_scale=cfg.bg_scale,
    )
    assert scale == cfg.conservative_bg_scale and reason is not None


def test_resolve_uses_precomputed_detail_ratio():
    """A passed-in ratio is trusted (no recompute): a smooth image 'raises'."""
    cfg = AggressiveConfig()
    scale, reason = _resolve_bg_scale(
        cfg, faces=[], image=_smooth(), base_scale=cfg.bg_scale,
        detail_ratio=0.99,
    )
    assert scale == cfg.conservative_bg_scale and reason is not None


# --------------------------------------------------------------------------- #
# C. End-to-end through compress_photo
# --------------------------------------------------------------------------- #

def test_compress_sign_photo_patches_sign_and_keeps_aggressive_scale(
    tmp_path, monkeypatch
):
    path = _write_jpg(tmp_path, "sign.jpg", _sign_photo())
    _patch_detector(monkeypatch, [_normal_face()])
    cfg = _text_config()

    photo = compress_photo(path, cfg)
    # The benign majority keeps the aggressive scale...
    assert photo.effective_bg_scale == cfg.aggressive.bg_scale
    # ...while the sign got a sharp patch.
    assert len(photo.regions) == 1
    assert _covers(photo.regions[0], 1450, 240)
    assert len(photo.region_crops) == 1 and len(photo.region_masks) == 1

    # The .fkeep round-trips and verifies with the text region.
    fkeep = tmp_path / "sign.fkeep"
    write_fkeep(photo, str(fkeep))
    assert verify_fkeep(str(fkeep)).ok
    data = read_fkeep(str(fkeep))
    assert len(data["manifest"]["regions"]) == 1


def test_compress_default_config_emits_no_text_patch(tmp_path, monkeypatch):
    """Anti-regression: the default path is unchanged by this feature."""
    path = _write_jpg(tmp_path, "sign.jpg", _sign_photo())
    _patch_detector(monkeypatch, [_normal_face()])
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_hands = False  # isolate: no hand zones either

    photo = compress_photo(path, cfg)
    assert photo.regions == []
    # Global edge ratio is quiet on a localized sign -> no whole-image raise.
    assert photo.effective_bg_scale == cfg.aggressive.bg_scale


def test_compress_document_photo_takes_whole_image_raise(tmp_path, monkeypatch):
    """Opt-in + document-like content: bail -> the conservative scale, no patches."""
    path = _write_jpg(tmp_path, "doc.jpg", _dense_text())
    _patch_detector(monkeypatch, [_normal_face()])
    cfg = _text_config()

    photo = compress_photo(path, cfg)
    assert photo.regions == []
    assert photo.effective_bg_scale == cfg.aggressive.conservative_bg_scale


def test_compress_text_cluster_under_small_face_region_not_stored_twice(
    tmp_path, monkeypatch
):
    """A sign already covered by a small-face region patch yields no extra patch."""
    path = _write_jpg(tmp_path, "sign_face.jpg", _sign_photo())
    # A small/distant face *at the sign*, whose padded box covers the whole sign
    # (plus the cluster padding margin).
    small = _face(1480, 200, 1512, 242, padded=(1240, 20, 1780, 460))
    _patch_detector(monkeypatch, [small])
    cfg = _text_config()

    photo = compress_photo(path, cfg)
    assert photo.regions == [(1240, 20, 1780, 460)]  # the small-face region only


def test_restore_composites_text_patch_sharper_than_upscale(tmp_path, monkeypatch):
    """The restored sign area is closer to the original than a pure upscale."""
    path = _write_jpg(tmp_path, "sign.jpg", _sign_photo())
    _patch_detector(monkeypatch, [_normal_face()])
    cfg = _text_config()

    photo = compress_photo(path, cfg)
    assert len(photo.regions) == 1
    fkeep = tmp_path / "sign.fkeep"
    write_fkeep(photo, str(fkeep))

    restored = Restorer(cfg.aggressive).restore(fkeep)  # bicubic (no AI here)
    assert restored.shape[:2] == (photo.original_height, photo.original_width)

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
# D. Fingerprint / validate / YAML
# --------------------------------------------------------------------------- #

def test_fingerprint_busts_on_each_new_field():
    base = FaceKeepConfig()
    base.mode = "aggressive"
    fp0 = settings_fingerprint(base)

    for field, value in [
        ("protect_text", True),
        ("text_region_tile_threshold", 0.2),
        ("text_region_max_frac", 0.5),
    ]:
        cfg = FaceKeepConfig()
        cfg.mode = "aggressive"
        setattr(cfg.aggressive, field, value)
        assert settings_fingerprint(cfg) != fp0, field


def test_fingerprint_faithful_unaffected():
    base = FaceKeepConfig()
    fp0 = settings_fingerprint(base)  # mode defaults to faithful
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_text = True
    cfg.aggressive.text_region_tile_threshold = 0.2
    assert settings_fingerprint(cfg) == fp0


@pytest.mark.parametrize("field,bad", [
    ("text_region_tile_threshold", 0.0),
    ("text_region_tile_threshold", 1.5),
    ("text_region_max_frac", 0.0),
    ("text_region_max_frac", 1.0001),
])
def test_validate_rejects_bad_ranges(field, bad):
    cfg = FaceKeepConfig()
    setattr(cfg.aggressive, field, bad)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validate_accepts_defaults_and_opt_in():
    cfg = FaceKeepConfig()
    cfg.validate()
    cfg.aggressive.protect_text = True
    cfg.validate()


def test_yaml_round_trip(tmp_path):
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_text = True
    cfg.aggressive.text_region_tile_threshold = 0.08
    cfg.aggressive.text_region_max_frac = 0.4
    path = tmp_path / "facekeep.yaml"
    cfg.save(path)
    loaded = FaceKeepConfig.load(path)
    assert loaded.aggressive.protect_text is True
    assert loaded.aggressive.text_region_tile_threshold == 0.08
    assert loaded.aggressive.text_region_max_frac == 0.4
