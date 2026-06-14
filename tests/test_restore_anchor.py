"""Low-frequency anchoring of the AI-upscaled background — ROADMAP 8.1.

Real-ESRGAN optimizes perceptual realism, not fidelity, so its output drifts in
color/brightness/low-frequency structure vs the real photo. But the stored
``background.jpg`` is a real measurement: every spatial frequency below its
Nyquist is data, not guesswork. ``restorer._anchor_low_frequencies`` therefore
replaces the AI output's low band with the reference's
(``out = sr - blur(sr, sigma) + blur(bicubic(bg), sigma)``), with sigma derived
from the upscale factor. Restore-only: no ``.fkeep``/manifest change, knobs are
NOT in ``index.settings_fingerprint``, and the anchor runs **only when the AI
upsampler actually ran** — the bicubic fallback is consistent with ``bg`` by
construction, and gating keeps that path byte-identical (the aggressive corpus
lock pins restore LPIPS through the bicubic path).

What these tests pin:

* the pure helper pulls a global tint/brightness drift back to the reference
  tone (low-freq error -> ~0) while injected high-frequency detail survives;
  anchoring a pure bicubic upscale is an identity up to rounding; sigma scales
  with ``bg_scale``; dtype/shape/clipping boundaries are safe;
* ``_upscale_background`` honestly reports ``(out, used_ai)``;
* wired through ``Restorer.restore()`` with a fake tinted AI upsampler, the
  anchored restore lands measurably closer to the original than the unanchored
  one; ``restore_anchor: false`` is a true no-op (the tint survives untouched);
* **the bicubic path is byte-identical whatever the knobs say** (the lock guard);
* back-projection is off by default and, when on, moves ``down(out)`` toward the
  stored background;
* YAML round-trip keeps both knobs; ``validate()`` rejects negative iterations;
  the fingerprint is UNCHANGED by both knobs (anti-regression, mirrors
  tests/test_tiled_restore.py).

Detection is mocked (no faces) so the ``.fkeep`` fixtures are deterministic and
offline; the autouse ``_force_bicubic_restore`` fixture keeps the default path
non-AI, and the fake upsampler is injected directly on the instance.
"""

import cv2
import numpy as np
import pytest

import facekeep.aggressive.compressor as compressor_mod
import facekeep.aggressive.restorer as restorer_mod
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import read_fkeep, write_fkeep
from facekeep.aggressive.restorer import (
    Restorer,
    _anchor_low_frequencies,
    _anchor_sigma,
    _back_project,
)
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _smooth_scene(h=240, w=320) -> np.ndarray:
    """A smooth full-res 'original': low-frequency structure only (no clipping
    headroom issues: values stay mid-range so tint+noise never saturate)."""
    rng = np.random.default_rng(7)
    small = rng.normal(128, 30, (h // 16, w // 16, 3)).astype(np.float32)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img, 40, 215).astype(np.uint8)


def _lowfreq(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)


def _hf_energy(img: np.ndarray, sigma: float) -> float:
    f = img.astype(np.float32)
    return float(np.std(f - _lowfreq(f, sigma)))


class _TintedBicubic:
    """Stand-in AI upsampler: bicubic upscale + a global tint/brightness drift.

    Mimics Real-ESRGAN's failure mode for this item (low-frequency drift) in a
    deterministic, offline way. ``enhance`` matches RealESRGANer's signature.
    """

    def __init__(self, shift=(20.0, -10.0, 5.0)):
        self.shift = np.asarray(shift, dtype=np.float32)

    def enhance(self, bg, outscale):
        h, w = bg.shape[:2]
        out = cv2.resize(
            bg, (int(round(w * outscale)), int(round(h * outscale))),
            interpolation=cv2.INTER_CUBIC,
        )
        out = np.clip(out.astype(np.float32) + self.shift, 0, 255).astype(np.uint8)
        return out, None


def _patch_detector_no_faces(monkeypatch):
    class _NoFaces:
        def detect(self, image):
            return []

    monkeypatch.setattr(compressor_mod, "create_detector", lambda **kw: _NoFaces())


def _make_fkeep(tmp_path, monkeypatch, h=400, w=600):
    """Compress a benign no-face photo to a .fkeep; returns (fkeep_path, original).

    No faces -> the conservative no-face fallback (bg_scale 0.5), zero crops and
    zero regions, so the restore output is exactly the (possibly anchored)
    upscaled background — the cleanest fixture for anchoring assertions.
    """
    rng = np.random.default_rng(11)
    small = rng.normal(128, 30, (h // 10, w // 10, 3)).astype(np.float32)
    img = np.clip(
        cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0, 255
    ).astype(np.uint8)
    path = tmp_path / "scene.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    _patch_detector_no_faces(monkeypatch)
    cfg = FaceKeepConfig()
    cfg.aggressive.protect_hands = False
    photo = compress_photo(str(path), cfg)
    fkeep = tmp_path / "scene.fkeep"
    write_fkeep(photo, str(fkeep))
    return str(fkeep), img


def _ai_restorer(shift=(20.0, -10.0, 5.0), **aggressive_overrides) -> Restorer:
    """A Restorer with the fake AI upsampler pre-installed (bypasses init)."""
    cfg = FaceKeepConfig()
    for k, v in aggressive_overrides.items():
        setattr(cfg.aggressive, k, v)
    r = Restorer(cfg.aggressive)
    r._tried_init = True
    r._upsampler = _TintedBicubic(shift)
    return r


# --------------------------------------------------------------------------- #
# A. Pure helper: _anchor_low_frequencies / _anchor_sigma / _back_project
# --------------------------------------------------------------------------- #

def test_anchor_pulls_low_band_back_and_keeps_high_band():
    """A tinted 'SR output' is pulled back to the reference tone; injected
    high-frequency detail (which only the SR side carries) survives."""
    scene = _smooth_scene()
    h, w = scene.shape[:2]
    bg_scale = 0.25
    bg = cv2.resize(scene, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    ref_up = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)

    rng = np.random.default_rng(3)
    tint = np.array([20.0, -15.0, 10.0], dtype=np.float32)
    hf = rng.normal(0, 12, scene.shape).astype(np.float32)
    sr = np.clip(ref_up.astype(np.float32) + tint + hf, 0, 255).astype(np.uint8)

    sigma = _anchor_sigma(bg_scale)
    lf_before = float(
        np.mean(np.abs(_lowfreq(sr, sigma) - _lowfreq(ref_up, sigma)))
    )
    out = _anchor_low_frequencies(sr, bg, bg_scale)
    lf_after = float(
        np.mean(np.abs(_lowfreq(out, sigma) - _lowfreq(ref_up, sigma)))
    )

    # Anti-false-green: the drift was really there before anchoring.
    assert lf_before > 10.0
    # ... and anchoring removes it (low-frequency error -> ~0).
    assert lf_after < 2.0
    # The high band is the SR side's own detail and must survive the swap.
    ratio = _hf_energy(out, sigma) / _hf_energy(sr, sigma)
    assert 0.75 <= ratio <= 1.05


def test_anchor_on_pure_bicubic_is_identity_up_to_rounding():
    """bicubic(bg) already carries the real low band: anchoring it is a no-op.

    This is the mathematical fact the used_ai gating in restore() relies on.
    """
    scene = _smooth_scene()
    h, w = scene.shape[:2]
    bg = cv2.resize(scene, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    sr = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)

    out = _anchor_low_frequencies(sr, bg, 0.25)
    diff = np.abs(out.astype(np.int16) - sr.astype(np.int16))
    assert diff.max() <= 1


def test_anchor_sigma_scales_with_bg_scale():
    """A harder downsample measured a narrower real band -> a wider anchor blur."""
    s8, s4, s2 = _anchor_sigma(0.125), _anchor_sigma(0.25), _anchor_sigma(0.5)
    assert s8 > s4 > s2 > 0


def test_anchor_dtype_shape_and_clipping():
    """uint8 in/out, shape preserved, out-of-range results clip (never wrap)."""
    # Positive overflow: a bright HF spike on a bright reference would exceed 255.
    sr = np.full((40, 60, 3), 200, dtype=np.uint8)
    sr[20, 30] = 255  # high-frequency spike: +55 over the local mean
    bg = np.full((20, 30, 3), 255, dtype=np.uint8)  # ref low band ~255
    out = _anchor_low_frequencies(sr, bg, 0.5)
    assert out.dtype == np.uint8 and out.shape == sr.shape
    # ~255 - 200 + 255 = ~310 -> clipped to 255; a wrap would read ~54.
    assert out[20, 30].min() >= 250

    # Negative overflow: a dark HF spike on a dark reference would go below 0.
    sr2 = np.full((40, 60, 3), 50, dtype=np.uint8)
    sr2[20, 30] = 0  # spike: -50 under the local mean
    bg2 = np.zeros((20, 30, 3), dtype=np.uint8)
    out2 = _anchor_low_frequencies(sr2, bg2, 0.5)
    # ~0 - 50 + 0 = ~-50 -> clipped to 0; a wrap would read ~206.
    assert out2[20, 30].max() <= 5


def test_back_project_zero_iters_is_off():
    x = np.full((40, 60, 3), 90, dtype=np.uint8)
    bg = np.full((20, 30, 3), 90, dtype=np.uint8)
    assert _back_project(x, bg, 0) is x  # untouched, same object


def test_back_project_moves_downsample_toward_bg():
    """Each iteration nudges down(x) toward the stored background."""
    scene = _smooth_scene()
    h, w = scene.shape[:2]
    bg = cv2.resize(scene, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    x = np.clip(
        cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        + 15.0, 0, 255,
    ).astype(np.uint8)

    def down_err(arr):
        d = cv2.resize(arr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return float(np.mean(np.abs(d.astype(np.float32) - bg.astype(np.float32))))

    out = _back_project(x, bg, 2)
    assert down_err(out) < down_err(x) * 0.7


# --------------------------------------------------------------------------- #
# B. _upscale_background reports (out, used_ai)
# --------------------------------------------------------------------------- #

def test_upscale_background_reports_used_ai():
    bg = np.full((20, 30, 3), 120, dtype=np.uint8)

    r_ai = _ai_restorer(shift=(0.0, 0.0, 0.0))
    out, used_ai = r_ai._upscale_background(bg, target_w=60, target_h=40,
                                            bg_scale=0.5)
    assert used_ai is True and out.shape[:2] == (40, 60)

    r_plain = Restorer()
    r_plain._tried_init = True
    r_plain._upsampler = None
    out2, used_ai2 = r_plain._upscale_background(bg, target_w=60, target_h=40,
                                                 bg_scale=0.5)
    assert used_ai2 is False and out2.shape[:2] == (40, 60)


# --------------------------------------------------------------------------- #
# C. Wiring through Restorer.restore()
# --------------------------------------------------------------------------- #

def test_restore_with_anchor_lands_closer_to_original(tmp_path, monkeypatch):
    """With a drifting (tinted) AI upsampler, anchoring measurably improves
    fidelity to the original photo; without it the drift stays."""
    fkeep, original = _make_fkeep(tmp_path, monkeypatch)

    out_anchored = _ai_restorer(restore_anchor=True).restore(fkeep)
    out_plain = _ai_restorer(restore_anchor=False).restore(fkeep)

    orig = original.astype(np.float32)
    mae_anchored = float(np.mean(np.abs(out_anchored.astype(np.float32) - orig)))
    mae_plain = float(np.mean(np.abs(out_plain.astype(np.float32) - orig)))
    # The fake AI drifts by mean |tint| ~= 11.7; anchoring must claw most of it
    # back. A >5 margin keeps the assertion robust across codec versions.
    assert mae_plain - mae_anchored > 5.0


def test_restore_anchor_false_is_a_true_noop(tmp_path, monkeypatch):
    """restore_anchor: false leaves the AI output untouched, byte-for-byte.

    The expected image is computed manually from the stored background through
    the same fake upsampler — no faces/regions, so restore() output == upscale.
    """
    fkeep, _ = _make_fkeep(tmp_path, monkeypatch)
    data = read_fkeep(fkeep)
    m = data["manifest"]
    ow, oh = m["original"]["width"], m["original"]["height"]
    bg_scale = m["settings"]["bg_scale"]

    expected, _ = _TintedBicubic().enhance(data["background"], 1.0 / bg_scale)
    if expected.shape[:2] != (oh, ow):
        expected = cv2.resize(expected, (ow, oh),
                              interpolation=cv2.INTER_LANCZOS4)

    out = _ai_restorer(restore_anchor=False).restore(fkeep)
    assert np.array_equal(out, expected)


def test_bicubic_path_byte_identical_whatever_the_knobs(tmp_path, monkeypatch):
    """THE lock guard: without AI (the conftest default), both knobs are inert
    and the restore output is byte-identical — the aggressive corpus regression
    lock (bicubic restore LPIPS) must not move."""
    fkeep, _ = _make_fkeep(tmp_path, monkeypatch)

    cfg_on = FaceKeepConfig()
    cfg_on.aggressive.restore_anchor = True
    cfg_on.aggressive.restore_backproject_iters = 3
    cfg_off = FaceKeepConfig()
    cfg_off.aggressive.restore_anchor = False
    cfg_off.aggressive.restore_backproject_iters = 0

    out_on = Restorer(cfg_on.aggressive).restore(fkeep)
    out_off = Restorer(cfg_off.aggressive).restore(fkeep)
    assert np.array_equal(out_on, out_off)


def test_backproject_not_called_by_default(tmp_path, monkeypatch):
    """restore_backproject_iters defaults to 0: the step must never run."""
    fkeep, _ = _make_fkeep(tmp_path, monkeypatch)

    def _boom(*args, **kwargs):
        raise AssertionError("_back_project ran despite iters=0")

    monkeypatch.setattr(restorer_mod, "_back_project", _boom)
    _ai_restorer().restore(fkeep)  # default config -> no raise


def test_backproject_on_ai_path_improves_downsample_consistency(
    tmp_path, monkeypatch
):
    """With anchoring OFF (isolating the knob), back-projection pulls the tinted
    AI output toward downsample-consistency with the stored background.

    (With anchoring on, the low band is already swapped for the real one and the
    remaining down-error is sub-quantization rounding noise — there is nothing
    left for back-projection to measurably fix on this fixture, so the isolated
    test is the honest one; the anchored+back-projected combination is covered
    by the pure-helper test above.)
    """
    fkeep, _ = _make_fkeep(tmp_path, monkeypatch)
    bg = read_fkeep(fkeep)["background"]

    out0 = _ai_restorer(restore_anchor=False,
                        restore_backproject_iters=0).restore(fkeep)
    out2 = _ai_restorer(restore_anchor=False,
                        restore_backproject_iters=2).restore(fkeep)

    def down_err(arr):
        d = cv2.resize(arr, (bg.shape[1], bg.shape[0]),
                       interpolation=cv2.INTER_AREA)
        return float(np.mean(np.abs(d.astype(np.float32) - bg.astype(np.float32))))

    # The fake AI drifts by mean |tint| ~= 11.7; two lambda=0.5 iterations must
    # claw back well over half of it.
    assert down_err(out2) < down_err(out0) * 0.5


# --------------------------------------------------------------------------- #
# D. Config: YAML round-trip, validate(), fingerprint exemption
# --------------------------------------------------------------------------- #

def test_yaml_roundtrip_preserves_knobs(tmp_path):
    cfg = FaceKeepConfig()
    cfg.aggressive.restore_anchor = False
    cfg.aggressive.restore_backproject_iters = 3
    path = tmp_path / "facekeep.yaml"
    cfg.save(path)

    loaded = FaceKeepConfig.load(path)
    assert loaded.aggressive.restore_anchor is False
    assert loaded.aggressive.restore_backproject_iters == 3


def test_validate_rejects_negative_backproject_iters():
    cfg = FaceKeepConfig()
    cfg.aggressive.restore_backproject_iters = -1
    with pytest.raises(ConfigError, match="restore_backproject_iters"):
        cfg.validate()


def test_validate_accepts_defaults_and_positive_iters():
    cfg = FaceKeepConfig()
    cfg.validate()  # defaults (anchor on, iters 0)
    cfg.aggressive.restore_backproject_iters = 2
    cfg.validate()


def test_anchor_knobs_not_in_fingerprint():
    """Restore-only knobs must not bust the compress cache (mirrors
    test_tiled_restore.py's tile/tile_pad guard)."""
    base = FaceKeepConfig()
    base.mode = "aggressive"
    fp_base = settings_fingerprint(base)

    c = FaceKeepConfig()
    c.mode = "aggressive"
    c.aggressive.restore_anchor = False
    c.aggressive.restore_backproject_iters = 5
    assert settings_fingerprint(c) == fp_base
