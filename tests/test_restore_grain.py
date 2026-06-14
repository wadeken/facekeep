"""Grain matching on the reconstructed background — ROADMAP 8.2.

The aggressive composite mixes *real* pixels (face crops / region patches,
carrying natural sensor noise + JPEG texture) with a GAN/bicubic background
that is too smooth ("plastic"), so even a perfectly feathered paste is findable
by texture discontinuity — the biggest visible tell. ``Restorer.restore()``
therefore estimates the grain level from the real crops
(``restorer._estimate_grain_sigma``: MAD x 1.4826 on the luma high-frequency
residual — robust to real edges, which would inflate a std) and adds matched,
seeded, mono grain to the reconstructed background before compositing
(``restorer._apply_grain``). Restore-only: no ``.fkeep``/manifest change and
``aggressive.restore_grain`` is NOT in ``index.settings_fingerprint``. Unlike
8.1's anchor it applies to BOTH the AI and bicubic paths (a bicubic upscale is
just as smooth), so the aggressive corpus lock's bicubic LPIPS moves —
re-baselined deliberately in tests/test_corpus_aggressive_regression.py.

What these tests pin:

* the estimator recovers a known injected sigma within tolerance, is robust to
  a hard synthetic edge (MAD, not std), medians across crops, and returns
  ``None`` for no crops;
* the synthesizer adds grain of the requested strength, identically to all
  three channels (mono/luma — chroma noise looks wrong), deterministically
  (fixed seed), clipping instead of wrapping, and is a no-op for sigma <= 0;
* wired through ``restore()``: with the flag off the output is byte-identical
  to ``preview()`` (the pre-8.2 bicubic composite); a no-crop file restores
  byte-identically with the flag on or off (nothing to mismatch); with crops
  the restored background's grain level matches the crops' (the actual goal);
  region patches feed the estimate when there are no faces; the same
  ``.fkeep`` restores to the same bytes twice (determinism); the AI path gets
  grain too; ``preview()`` is untouched by the flag;
* YAML round-trip keeps the knob; the fingerprint is UNCHANGED by it
  (anti-regression, mirrors test_restore_anchor.py).

Detection is mocked (a fixed large face) so the fixtures are deterministic and
offline; the autouse ``_force_bicubic_restore`` fixture keeps the default path
non-AI, and the fake AI upsampler is injected directly on the instance.
"""

import cv2
import numpy as np

import facekeep.aggressive.compressor as compressor_mod
from facekeep.aggressive.compressor import CompressedPhoto, compress_photo
from facekeep.aggressive.format import read_fkeep, write_fkeep
from facekeep.aggressive.restorer import (
    Restorer,
    _apply_grain,
    _estimate_grain_sigma,
)
from facekeep.config import FaceKeepConfig
from facekeep.detector import FaceRegion
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _smooth_base(h, w, seed=7) -> np.ndarray:
    """A smooth mid-range float32 scene: no clipping headroom issues and almost
    no high-frequency content of its own (so injected noise dominates the
    residual the estimator measures)."""
    rng = np.random.default_rng(seed)
    small = rng.normal(128, 25, (max(2, h // 32), max(2, w // 32), 3))
    img = cv2.resize(small.astype(np.float32), (w, h),
                     interpolation=cv2.INTER_CUBIC)
    return np.clip(img, 50, 205)


def _noisy_crop(h=96, w=96, sigma=2.0, seed=5) -> np.ndarray:
    """A smooth crop carrying mono (luma) Gaussian noise of known strength.

    Mono because the estimator deliberately measures the *luma* residual: with
    per-channel independent noise, luma would only see ~0.67 sigma of it (the
    BT.601 weights), and that is by design — the synthesizer applies mono grain,
    so luma-to-luma is the self-consistent loop.
    """
    rng = np.random.default_rng(seed)
    base = _smooth_base(h, w, seed=seed + 1)
    noise = rng.normal(0, sigma, (h, w)).astype(np.float32)
    return np.clip(base + noise[:, :, None], 0, 255).astype(np.uint8)


class _OneBigFace:
    """A detector stub returning one fixed large face (no Haar variance)."""

    BBOX = (350, 150, 470, 270)
    PADDED = (330, 130, 490, 290)

    def detect(self, image):
        return [FaceRegion(id=0, bbox=self.BBOX, padded_bbox=self.PADDED,
                           confidence=1.0)]


class _PlainBicubicAI:
    """Fake AI upsampler: plain bicubic (no drift) — isolates the grain knob.

    ``enhance`` matches RealESRGANer's signature so ``_upscale_background``
    reports ``used_ai=True`` through it.
    """

    def enhance(self, bg, outscale):
        h, w = bg.shape[:2]
        out = cv2.resize(
            bg, (int(round(w * outscale)), int(round(h * outscale))),
            interpolation=cv2.INTER_CUBIC,
        )
        return out, None


def _make_fkeep_with_face(tmp_path, monkeypatch, noise_sigma=6.0):
    """Compress a noisy one-face photo to a .fkeep.

    The whole original carries sensor-like Gaussian noise, so the stored face
    crop (JPEG q95, near-lossless) keeps it while the background path
    (INTER_AREA quarter-scale + JPEG q85 + upscale on restore) averages it
    away — exactly the real texture mismatch 8.2 fixes.
    """
    h, w = 400, 600
    rng = np.random.default_rng(13)
    img = np.clip(
        _smooth_base(h, w) + rng.normal(0, noise_sigma, (h, w, 3)),
        0, 255,
    ).astype(np.uint8)
    path = tmp_path / "noisy_face.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    monkeypatch.setattr(compressor_mod, "create_detector",
                        lambda **kw: _OneBigFace())
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_hands = False
    photo = compress_photo(str(path), cfg)
    fkeep = tmp_path / "noisy_face.fkeep"
    write_fkeep(photo, str(fkeep))
    return str(fkeep)


def _make_fkeep_no_crops(tmp_path, monkeypatch):
    """A no-face (and no-region) .fkeep: nothing real-pixel to grain-match."""
    class _NoFaces:
        def detect(self, image):
            return []

    h, w = 200, 300
    img = np.clip(_smooth_base(h, w), 0, 255).astype(np.uint8)
    path = tmp_path / "plain.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    monkeypatch.setattr(compressor_mod, "create_detector",
                        lambda **kw: _NoFaces())
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_hands = False
    photo = compress_photo(str(path), cfg)
    fkeep = tmp_path / "plain.fkeep"
    write_fkeep(photo, str(fkeep))
    return str(fkeep)


def _grain_restorer(grain: bool, ai: bool = False) -> Restorer:
    cfg = FaceKeepConfig()
    cfg.aggressive.restore_grain = grain
    r = Restorer(cfg.aggressive)
    if ai:
        r._tried_init = True
        r._upsampler = _PlainBicubicAI()
    return r


# --------------------------------------------------------------------------- #
# A. Estimator: _estimate_grain_sigma
# --------------------------------------------------------------------------- #

def test_estimator_recovers_known_sigma():
    """Injected sigma=2 noise on a smooth crop estimates back ~2.

    The blur-residual high-pass keeps ~95% of white-noise energy, so the
    estimate sits slightly under the injected value — the band is asymmetric
    around 2 accordingly.
    """
    est = _estimate_grain_sigma([_noisy_crop(sigma=2.0)])
    assert est is not None
    assert 1.5 <= est <= 2.3


def test_estimator_is_edge_robust():
    """A hard luminance edge (|step| 130) + sigma=2 noise still estimates ~2.

    This is why the estimator uses MAD x 1.4826 and not a std: the edge's
    residual pixels are sparse outliers, invisible to a median, but a std over
    the same residual is dragged far above the true noise floor.
    """
    h, w, sigma = 96, 96, 2.0
    rng = np.random.default_rng(9)
    base = np.full((h, w, 3), 60.0, dtype=np.float32)
    base[:, w // 2:] = 190.0  # hard vertical edge
    noise = rng.normal(0, sigma, (h, w)).astype(np.float32)
    crop = np.clip(base + noise[:, :, None], 0, 255).astype(np.uint8)

    est = _estimate_grain_sigma([crop])
    assert est is not None
    assert est <= 3.0  # nowhere near the edge magnitude

    # Anti-false-green: a std-based estimate on the same residual IS inflated.
    luma = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = luma - cv2.GaussianBlur(luma, (0, 0), 1.5)
    assert float(residual.std()) > 2.0 * est


def test_estimator_medians_across_crops():
    """One outlier crop (sigma=12) doesn't drag the estimate off the majority."""
    crops = [
        _noisy_crop(sigma=2.0, seed=1),
        _noisy_crop(sigma=2.0, seed=2),
        _noisy_crop(sigma=12.0, seed=3),
    ]
    est = _estimate_grain_sigma(crops)
    assert est is not None
    assert 1.5 <= est <= 2.3


def test_estimator_no_crops_returns_none():
    assert _estimate_grain_sigma([]) is None


# --------------------------------------------------------------------------- #
# B. Synthesizer: _apply_grain
# --------------------------------------------------------------------------- #

def test_apply_grain_strength_shape_dtype():
    """Grain of the requested strength lands on the image (renormalized after
    the soften blur, so the applied std really equals sigma)."""
    bg = np.full((120, 160, 3), 128, dtype=np.uint8)
    out = _apply_grain(bg, 3.0)
    assert out.dtype == np.uint8 and out.shape == bg.shape
    diff = out.astype(np.float32) - 128.0
    assert 2.4 <= float(diff.std()) <= 3.6
    assert abs(float(diff.mean())) < 0.5  # zero-mean: no brightness shift


def test_apply_grain_is_mono_across_channels():
    """The same noise field lands on all three BGR channels (luma grain);
    chroma noise would make the channels diverge."""
    bg = np.full((80, 100, 3), 128, dtype=np.uint8)
    out = _apply_grain(bg, 4.0)
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 1], out[:, :, 2])


def test_apply_grain_is_deterministic():
    bg = np.full((80, 100, 3), 128, dtype=np.uint8)
    assert np.array_equal(_apply_grain(bg, 4.0), _apply_grain(bg, 4.0))


def test_apply_grain_nonpositive_sigma_is_noop():
    bg = np.full((40, 60, 3), 128, dtype=np.uint8)
    assert _apply_grain(bg, 0.0) is bg
    assert _apply_grain(bg, -1.0) is bg


def test_apply_grain_clips_never_wraps():
    """At the dtype boundaries grain clips: a wrap would show as far values."""
    bright = np.full((60, 80, 3), 255, dtype=np.uint8)
    out = _apply_grain(bright, 5.0)
    assert out.min() >= 230  # a wrap would read near 0
    dark = np.zeros((60, 80, 3), dtype=np.uint8)
    out2 = _apply_grain(dark, 5.0)
    assert out2.max() <= 25  # a wrap would read near 255


# --------------------------------------------------------------------------- #
# C. Wiring through Restorer.restore()
# --------------------------------------------------------------------------- #

def test_grain_off_matches_preview_on_bicubic_path(tmp_path, monkeypatch):
    """With the flag off, the bicubic restore is byte-identical to preview()
    (the pre-8.2 composite): the flag really is the only difference."""
    fkeep = _make_fkeep_with_face(tmp_path, monkeypatch)
    out = _grain_restorer(grain=False).restore(fkeep)
    ref = Restorer().preview(fkeep)
    assert np.array_equal(out, ref)


def test_no_crops_restores_identically_with_flag_on(tmp_path, monkeypatch):
    """No faces and no regions -> no real pixels to match -> grain is skipped
    and the flag changes nothing, byte-for-byte."""
    fkeep = _make_fkeep_no_crops(tmp_path, monkeypatch)
    out_on = _grain_restorer(grain=True).restore(fkeep)
    out_off = _grain_restorer(grain=False).restore(fkeep)
    assert np.array_equal(out_on, out_off)


def test_grain_matches_background_texture_to_crops(tmp_path, monkeypatch):
    """THE goal: with grain on, the restored background's grain level matches
    the real crops'; with grain off it is visibly smoother (the tell)."""
    fkeep = _make_fkeep_with_face(tmp_path, monkeypatch)
    crop_sigma = _estimate_grain_sigma(read_fkeep(fkeep)["face_crops"])
    assert crop_sigma is not None and crop_sigma > 2.0  # the fixture has grain

    out_on = _grain_restorer(grain=True).restore(fkeep)
    out_off = _grain_restorer(grain=False).restore(fkeep)

    # A pure-background patch, well away from the padded face box (330..490 x).
    bg_on = out_on[60:340, 30:280]
    bg_off = out_off[60:340, 30:280]
    sigma_on = _estimate_grain_sigma([bg_on])
    sigma_off = _estimate_grain_sigma([bg_off])

    # Anti-false-green: the smooth-background tell really exists without grain.
    assert sigma_off < 0.5 * crop_sigma
    # And grain matching closes it (within a tolerant band — JPEG q85 on the
    # stored bg and the residual band edges cost a little either way).
    assert 0.6 * crop_sigma <= sigma_on <= 1.6 * crop_sigma


def test_region_patches_feed_the_estimate_without_faces(tmp_path):
    """With no face crops the estimator falls back to region patches.

    Built directly as a CompressedPhoto (compress can't easily produce a
    region-without-face file, but the format allows it and restore must handle
    it): one noisy region patch, no faces.
    """
    h, w = 200, 300
    base = np.clip(_smooth_base(h, w), 0, 255).astype(np.uint8)
    patch = _noisy_crop(h=80, w=100, sigma=5.0, seed=4)
    mask = np.full((80, 100), 255, dtype=np.uint8)

    cfg = FaceKeepConfig().aggressive
    photo = CompressedPhoto(
        original_filename="region_only.jpg",
        original_width=w, original_height=h,
        original_size_bytes=12345, original_hash="0" * 64,
        original_orientation=1, exif=None,
        background=cv2.resize(base, (w // 2, h // 2),
                              interpolation=cv2.INTER_AREA),
        face_crops=[], face_masks=[], faces=[],
        thumbnail=cv2.resize(base, (w // 4, h // 4)),
        effective_bg_scale=0.5, config=cfg,
        region_crops=[patch], region_masks=[mask],
        regions=[(100, 60, 200, 140)],
    )
    fkeep = tmp_path / "region_only.fkeep"
    write_fkeep(photo, str(fkeep))

    out_on = _grain_restorer(grain=True).restore(str(fkeep))
    out_off = _grain_restorer(grain=False).restore(str(fkeep))
    # Grain landed (estimated from the region patch), measured off-patch.
    bg_on = out_on[10:50, 10:90]
    bg_off = out_off[10:50, 10:90]
    assert _estimate_grain_sigma([bg_on]) > _estimate_grain_sigma([bg_off]) + 1.0


def test_restore_is_deterministic_with_grain(tmp_path, monkeypatch):
    """Seeded grain: the same .fkeep restores to the same bytes every run."""
    fkeep = _make_fkeep_with_face(tmp_path, monkeypatch)
    out1 = _grain_restorer(grain=True).restore(fkeep)
    out2 = _grain_restorer(grain=True).restore(fkeep)
    assert np.array_equal(out1, out2)


def test_grain_applies_on_ai_path_too(tmp_path, monkeypatch):
    """Unlike 8.1's anchor, grain is NOT gated on used_ai: a GAN background is
    just as smooth, so the AI path gets matched grain as well."""
    fkeep = _make_fkeep_with_face(tmp_path, monkeypatch)
    out_on = _grain_restorer(grain=True, ai=True).restore(fkeep)
    out_off = _grain_restorer(grain=False, ai=True).restore(fkeep)
    bg_on = out_on[60:340, 30:280]
    bg_off = out_off[60:340, 30:280]
    assert _estimate_grain_sigma([bg_on]) > _estimate_grain_sigma([bg_off]) + 1.0


def test_preview_untouched_by_grain_flag(tmp_path, monkeypatch):
    """preview() skips grain unconditionally (speed — the GFPGAN precedent)."""
    fkeep = _make_fkeep_with_face(tmp_path, monkeypatch)
    p_on = _grain_restorer(grain=True).preview(fkeep)
    p_off = _grain_restorer(grain=False).preview(fkeep)
    assert np.array_equal(p_on, p_off)


# --------------------------------------------------------------------------- #
# D. Config: YAML round-trip, fingerprint exemption
# --------------------------------------------------------------------------- #

def test_yaml_roundtrip_preserves_grain_knob(tmp_path):
    cfg = FaceKeepConfig()
    cfg.aggressive.restore_grain = False
    path = tmp_path / "facekeep.yaml"
    cfg.save(path)
    assert FaceKeepConfig.load(path).aggressive.restore_grain is False


def test_grain_knob_not_in_fingerprint():
    """Restore-only knob: must not bust the compress cache (mirrors
    test_restore_anchor.py's guard)."""
    base = FaceKeepConfig()
    base.mode = "aggressive"
    fp_base = settings_fingerprint(base)

    c = FaceKeepConfig()
    c.mode = "aggressive"
    c.aggressive.restore_grain = False
    assert settings_fingerprint(c) == fp_base
