"""Content-aware conservatism — ROADMAP Phase 4 / quality guardrails.

Aggressive mode downsamples the background and reconstructs it with AI on
restore. That is safe on *benign* content (sky, bokeh, foliage, plain walls) but
mangles content the AI cannot honestly reconstruct: text/signage, fine regular
structure, and small/distant background faces (the worst case — an uncanny AI
face). The guardrail in this item: detect those risky cases and *raise the
global ``bg_scale``* toward a conservative floor (compress the whole background
less), using the same lever the no-face fallback already uses.

Scope (deliberate, see ROADMAP): this is the *whole-image* first step — a risky
photo is compressed less everywhere. Per-region scale maps (reduce the factor
only *where* the risk is) would change the ``.fkeep`` container and are a tracked
follow-up; this item leaves the format byte-for-byte unchanged (it only picks the
existing single ``effective_bg_scale``).

What these tests pin:

* the edge-density heuristic separates benign (smooth/natural-texture) from risky
  (text/tile) content, and its pre-blur keeps benign fine content benign;
* a small/distant face flags a risky background even though it is itself cropped;
* the pure ``_resolve_bg_scale`` decision raises the scale on a risk, never lowers
  a scale already made conservative (composes with the no-face fallback), and is
  fully disabled by ``content_aware=False`` (the escape hatch);
* end-to-end through ``compress_photo`` the effective scale (and thus the
  manifest) reflects the decision, while a benign photo is unchanged
  (anti-regression — defaults-on must not silently alter benign outputs);
* the new output-affecting fields bust the incremental-index fingerprint
  (aggressive only; faithful is untouched);
* ``validate()`` range-checks the new fields and they survive a YAML round-trip.

The heuristic numbers are asserted *relatively* (risky > benign, risky > the
threshold) rather than as magic absolutes, matching the repo's convention so the
tests don't go flaky across OpenCV versions. Detection is mocked where a
deterministic face list is needed, so these tests need no network/model.
"""

import numpy as np
import cv2
import pytest

import facekeep.aggressive.compressor as compressor_mod
from facekeep.aggressive.compressor import (
    _background_detail_ratio,
    _has_risky_background_face,
    _resolve_bg_scale,
    compress_photo,
)
from facekeep.aggressive.format import read_fkeep_info, write_fkeep
from facekeep.config import AggressiveConfig, FaceKeepConfig
from facekeep.detector import FaceRegion
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# Image builders (deterministic; chosen for clear benign/risky separation)
# --------------------------------------------------------------------------- #

def _smooth(h=600, w=900) -> np.ndarray:
    """A smooth two-axis gradient — the canonical benign background."""
    yy = np.linspace(60, 200, h)[:, None]
    xx = np.linspace(40, 160, w)[None, :]
    base = (yy + xx) / 2.0
    return np.clip(np.stack([base, base * 0.95, base * 0.9], axis=-1), 0, 255).astype(
        np.uint8
    )


def _natural_texture(h=600, w=900) -> np.ndarray:
    """Upscaled low-res noise — a benign 'natural photo' texture (no sharp edges).

    This is the same background style the shared ``face_image`` fixture uses; it
    must read as benign (the pre-blur in the heuristic collapses it), or every
    ordinary photo would be treated as risky.
    """
    rng = np.random.default_rng(3)
    bg = cv2.resize(
        rng.normal(128, 30, (h // 10, w // 10, 3)).astype(np.float32),
        (w, h), interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(bg, 0, 255).astype(np.uint8)


def _dense_text(h=600, w=900) -> np.ndarray:
    """A page of dense text — risky (AI mangles text/signage)."""
    img = np.full((h, w, 3), 230, np.uint8)
    y = 30
    while y < h - 10:
        cv2.putText(img, "The quick brown fox jumps 0123456789", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 1)
        y += 28
    return img


def _tile_pattern(h=600, w=900) -> np.ndarray:
    """A regular grid/tile pattern — risky (AI smears regular structure)."""
    img = np.full((h, w, 3), 180, np.uint8)
    for x in range(0, w, 40):
        cv2.line(img, (x, 0), (x, h), (90, 90, 90), 2)
    for y in range(0, h, 25):
        cv2.line(img, (0, y), (w, y), (90, 90, 90), 2)
    return img


def _write_jpg(tmp_path, name, img) -> str:
    path = tmp_path / name
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(path)


def _face(x1, y1, x2, y2, conf=0.9) -> FaceRegion:
    return FaceRegion(id=0, bbox=(x1, y1, x2, y2),
                      padded_bbox=(x1, y1, x2, y2), confidence=conf)


# --------------------------------------------------------------------------- #
# A. The edge-density heuristic
# --------------------------------------------------------------------------- #

def test_detail_ratio_benign_below_risky():
    """Smooth / natural-texture score below text / tile (the core separation)."""
    smooth = _background_detail_ratio(_smooth())
    natural = _background_detail_ratio(_natural_texture())
    text = _background_detail_ratio(_dense_text())
    tile = _background_detail_ratio(_tile_pattern())

    # Benign content is essentially edge-free after the pre-blur.
    assert smooth < 0.01
    assert natural < 0.01
    # Risky content is clearly edge-dense, and strictly above benign.
    assert text > natural and tile > natural
    assert text > 0.02 and tile > 0.02


def test_detail_ratio_crosses_default_threshold():
    """Risky content exceeds the default threshold; benign stays under it.

    Pins that the *shipped default* actually fires on risky content and not on
    benign — i.e. the default is usefully tuned, not just internally separated.
    """
    th = AggressiveConfig().text_edge_threshold
    assert _background_detail_ratio(_dense_text()) > th
    assert _background_detail_ratio(_tile_pattern()) > th
    assert _background_detail_ratio(_smooth()) <= th
    assert _background_detail_ratio(_natural_texture()) <= th


def test_detail_ratio_empty_image_is_zero():
    """A degenerate (zero-size) image returns 0.0, not a crash/NaN."""
    assert _background_detail_ratio(np.zeros((0, 0, 3), np.uint8)) == 0.0


def test_detail_ratio_uint16_safe():
    """A 16-bit source is down-converted for the analysis, not crashed on.

    Detection/analysis runs on 8-bit; the heuristic only picks a scale, never
    touches output pixels, so the down-convert is harmless (cf. the detector).
    """
    text16 = (_dense_text().astype(np.uint16)) * 257
    smooth16 = (_smooth().astype(np.uint16)) * 257
    assert text16.dtype == np.uint16 and smooth16.dtype == np.uint16
    assert _background_detail_ratio(text16) > _background_detail_ratio(smooth16)


# --------------------------------------------------------------------------- #
# B. The small-background-face signal
# --------------------------------------------------------------------------- #

def test_small_face_is_risky():
    """A face whose short side is below small_face_ratio of the frame is risky."""
    # 30px face on a 1000px-short frame = 3% < 4% default -> risky.
    faces = [_face(100, 100, 130, 140)]
    assert _has_risky_background_face(faces, 1500, 1000, small_face_ratio=0.04)


def test_large_face_is_not_risky():
    """A normal portrait-sized face does not flag the background as risky."""
    # 300px face on a 1000px-short frame = 30% -> well above 4%.
    faces = [_face(400, 300, 700, 690)]
    assert not _has_risky_background_face(faces, 1500, 1000, small_face_ratio=0.04)


def test_no_faces_not_risky_and_empty_frame_safe():
    assert not _has_risky_background_face([], 1500, 1000, small_face_ratio=0.04)
    # Degenerate frame short side -> never crashes / divides by zero.
    assert not _has_risky_background_face([_face(0, 0, 5, 5)], 0, 0,
                                          small_face_ratio=0.04)


# --------------------------------------------------------------------------- #
# C. The pure decision: _resolve_bg_scale
# --------------------------------------------------------------------------- #

def test_resolve_benign_keeps_base_scale():
    cfg = AggressiveConfig()  # content_aware=True, conservative=0.5
    scale, reason = _resolve_bg_scale(cfg, faces=[_face(400, 300, 700, 690)],
                                      image=_smooth(), base_scale=0.25)
    assert scale == 0.25 and reason is None


def test_resolve_detailed_raises_to_conservative():
    cfg = AggressiveConfig()
    scale, reason = _resolve_bg_scale(cfg, faces=[], image=_dense_text(),
                                      base_scale=0.25)
    assert scale == cfg.conservative_bg_scale
    assert reason is not None and "detail" in reason


def test_resolve_small_face_raises_to_conservative():
    """With region_local OFF, a small face raises the *whole-image* scale.

    This is the original whole-image behavior; the default (region_local on)
    handles a small face locally instead (a sharp patch) and does NOT raise the
    whole-image scale — that path is covered in tests/test_region_local.py.
    """
    cfg = AggressiveConfig(region_local=False)
    # Small face + smooth bg: the face alone must trip conservatism.
    scale, reason = _resolve_bg_scale(cfg, faces=[_face(100, 100, 130, 140)],
                                      image=_smooth(1000, 1500), base_scale=0.25)
    assert scale == cfg.conservative_bg_scale
    assert reason is not None and "face" in reason


def test_resolve_small_face_with_region_local_keeps_base_scale():
    """Default (region_local on): a small face on a benign bg does NOT raise the
    whole-image scale — it is protected locally instead."""
    cfg = AggressiveConfig()  # region_local=True by default
    scale, reason = _resolve_bg_scale(cfg, faces=[_face(100, 100, 130, 140)],
                                      image=_smooth(1000, 1500), base_scale=0.25)
    assert scale == 0.25 and reason is None


def test_resolve_disabled_always_keeps_base():
    """content_aware=False is a full escape hatch — risk is ignored."""
    cfg = AggressiveConfig(content_aware=False)
    scale, reason = _resolve_bg_scale(cfg, faces=[_face(100, 100, 130, 140)],
                                      image=_dense_text(), base_scale=0.25)
    assert scale == 0.25 and reason is None


def test_resolve_never_lowers_an_already_higher_base():
    """Composes with the no-face fallback: never lower a scale already raised.

    If the caller already resolved a base scale at/above the conservative floor
    (e.g. the no-face conservative branch set 0.5), a content risk must not pull
    it *down* to the floor — and we don't claim a change we didn't make.
    """
    cfg = AggressiveConfig(conservative_bg_scale=0.5)
    scale, reason = _resolve_bg_scale(cfg, faces=[], image=_dense_text(),
                                      base_scale=0.75)
    assert scale == 0.75 and reason is None


# --------------------------------------------------------------------------- #
# D. End-to-end through compress_photo (detector mocked for determinism)
# --------------------------------------------------------------------------- #

def _patch_detector(monkeypatch, faces):
    """Make the compressor's detector return a fixed face list (offline)."""
    class _Fixed:
        def detect(self, image):
            return list(faces)

    monkeypatch.setattr(compressor_mod, "create_detector",
                        lambda **kw: _Fixed())


def test_compress_detailed_background_uses_conservative_scale(tmp_path, monkeypatch):
    """A text-heavy photo (with a normal face) compresses less aggressively.

    The face is normal-sized (so the *detail* signal, not the small-face signal,
    is what trips), and the effective scale lands at the conservative floor.
    """
    img = _dense_text(800, 1000)
    # A normal, large face so only the detail signal can be responsible.
    cv2.ellipse(img, (500, 400), (160, 210), 0, 0, 360, (180, 170, 165), -1)
    path = _write_jpg(tmp_path, "signage.jpg", img)
    _patch_detector(monkeypatch, [_face(340, 190, 660, 610)])

    cfg = FaceKeepConfig()
    photo = compress_photo(path, cfg)
    assert photo.effective_bg_scale == cfg.aggressive.conservative_bg_scale

    # The manifest is the honest, observable surface (restore reads bg_scale).
    fkeep = tmp_path / "out.fkeep"
    write_fkeep(photo, str(fkeep))
    info = read_fkeep_info(str(fkeep))
    assert info["settings"]["bg_scale"] == cfg.aggressive.conservative_bg_scale


def test_compress_benign_background_keeps_base_scale(tmp_path, monkeypatch):
    """A smooth-background photo with a normal face keeps the configured scale."""
    img = _natural_texture(800, 1000)
    cv2.ellipse(img, (500, 400), (160, 210), 0, 0, 360, (180, 170, 165), -1)
    path = _write_jpg(tmp_path, "portrait.jpg", img)
    _patch_detector(monkeypatch, [_face(340, 190, 660, 610)])

    cfg = FaceKeepConfig()  # bg_scale 0.25
    photo = compress_photo(path, cfg)
    assert photo.effective_bg_scale == cfg.aggressive.bg_scale == 0.25


def test_compress_disabled_keeps_base_on_detailed(tmp_path, monkeypatch):
    """With content_aware off, even a text photo uses the configured bg_scale."""
    path = _write_jpg(tmp_path, "text_off.jpg", _dense_text(800, 1000))
    _patch_detector(monkeypatch, [_face(340, 190, 660, 610)])

    cfg = FaceKeepConfig()
    cfg.aggressive.content_aware = False
    photo = compress_photo(path, cfg)
    assert photo.effective_bg_scale == cfg.aggressive.bg_scale == 0.25


def test_compress_small_face_whole_image_raise_when_region_local_off(
    tmp_path, monkeypatch
):
    """With region_local OFF, a small/distant face raises the whole-image scale.

    (The default — region_local on — instead keeps that region sharp locally and
    leaves the whole-image scale alone; that behavior is pinned in
    tests/test_region_local.py.)
    """
    path = _write_jpg(tmp_path, "distant.jpg", _natural_texture(1000, 1500))
    # ~30px face on a 1000px-short frame = 3% < 4% default.
    _patch_detector(monkeypatch, [_face(100, 100, 132, 142)])

    cfg = FaceKeepConfig()
    cfg.aggressive.region_local = False
    photo = compress_photo(path, cfg)
    assert photo.effective_bg_scale == cfg.aggressive.conservative_bg_scale


# --------------------------------------------------------------------------- #
# E. Incremental-index fingerprint (output-affecting -> must bust the cache)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mutate", [
    lambda c: setattr(c.aggressive, "content_aware", False),
    lambda c: setattr(c.aggressive, "conservative_bg_scale", 0.75),
    lambda c: setattr(c.aggressive, "text_edge_threshold", 0.30),
    lambda c: setattr(c.aggressive, "small_face_ratio", 0.10),
])
def test_fingerprint_busts_on_content_aware_change(mutate):
    """Changing any content-aware field changes the aggressive fingerprint."""
    base = FaceKeepConfig(mode="aggressive")
    changed = FaceKeepConfig(mode="aggressive")
    mutate(changed)
    assert settings_fingerprint(base) != settings_fingerprint(changed)


def test_fingerprint_stable_when_unchanged():
    a = FaceKeepConfig(mode="aggressive")
    b = FaceKeepConfig(mode="aggressive")
    assert settings_fingerprint(a) == settings_fingerprint(b)


def test_faithful_fingerprint_unaffected_by_content_aware():
    """The new fields are aggressive-only; faithful's fingerprint must not move."""
    base = FaceKeepConfig(mode="faithful")
    changed = FaceKeepConfig(mode="faithful")
    changed.aggressive.content_aware = False
    changed.aggressive.text_edge_threshold = 0.99
    assert settings_fingerprint(base) == settings_fingerprint(changed)


# --------------------------------------------------------------------------- #
# F. Validation + YAML round-trip
# --------------------------------------------------------------------------- #

def test_validate_accepts_defaults():
    FaceKeepConfig().validate()  # must not raise


@pytest.mark.parametrize("field,value", [
    ("conservative_bg_scale", 0.0),   # below 0.05
    ("conservative_bg_scale", 1.5),   # above 1.0
    ("text_edge_threshold", -0.1),
    ("text_edge_threshold", 1.5),
    ("small_face_ratio", -0.01),
    ("small_face_ratio", 2.0),
])
def test_validate_rejects_out_of_range(field, value):
    cfg = FaceKeepConfig()
    setattr(cfg.aggressive, field, value)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_yaml_round_trip_preserves_fields(tmp_path):
    cfg = FaceKeepConfig()
    cfg.aggressive.content_aware = False
    cfg.aggressive.conservative_bg_scale = 0.6
    cfg.aggressive.text_edge_threshold = 0.08
    cfg.aggressive.small_face_ratio = 0.03
    p = tmp_path / "facekeep.yaml"
    cfg.save(p)

    loaded = FaceKeepConfig.load(p)
    assert loaded.aggressive.content_aware is False
    assert loaded.aggressive.conservative_bg_scale == 0.6
    assert loaded.aggressive.text_edge_threshold == 0.08
    assert loaded.aggressive.small_face_ratio == 0.03
