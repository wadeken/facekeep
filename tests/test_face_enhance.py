"""Tests for GFPGAN face restoration on reconstructed background faces — ROADMAP Phase 4.

Aggressive restore upscales the downsampled background with Real-ESRGAN/bicubic.
A face the *detector missed* at compress time rides along in that background and
the super-resolver tends to melt it into something uncanny. GFPGAN (gated by
`aggressive.face_enhance`) re-synthesizes those faces; only GFPGAN's own detected
face regions are soft-mask blended back, and detected faces are composited as real
crops *on top* afterward, so a real face is never replaced by a hallucination.

These tests never import or download GFPGAN: a fake enhancer is injected on the
`Restorer` (the real `_init_face_enhancer` is the only place `gfpgan` is imported,
and its ImportError path is the graceful-degradation contract). What they pin:

* `face_enhance=False` -> `_enhance_background_faces` is an exact no-op.
* GFPGAN not installed (enhancer stays None) -> no-op, no exception.
* A fake enhancer that changes a face region -> restore blends that region in,
  but a *detected* face crop still wins on top (real pixels, not the fake's).
* The enhancer raising at inference time -> restore degrades to the un-enhanced
  background instead of crashing.
* `preview()` (fast, no AI) never calls the face enhancer.
* `enhance` returning no faces -> background unchanged.
"""

import sys
import types

import numpy as np
import pytest

from facekeep.aggressive import restorer as restorer_module
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import _fkeep_path, read_fkeep, write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import ConfigError, ModelDownloadError
from facekeep.index import settings_fingerprint


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _build_fkeep(face_image, dst_dir, name="photo"):
    """Compress ``face_image`` to a real .fkeep under ``dst_dir``; return its Path."""
    photo = compress_photo(str(face_image), FaceKeepConfig())
    write_fkeep(photo, str(dst_dir / name))
    path = _fkeep_path(str(dst_dir / name))
    assert path.exists()
    return path


class _FakeFaceHelper:
    """Minimal stand-in for facexlib's FaceRestoreHelper.

    The restorer reads ``inverse_affine_matrices`` (populating them via
    ``get_inverse_affine`` if empty) to map each restored 512x512 crop back to its
    place in the frame. Here each face is a plain translate-only affine that pastes
    the crop's top-left at the box origin (no rotation/scale), so a test can assert
    exactly where the restored pixels land.
    """

    def __init__(self):
        self.affine_matrices = []
        self.inverse_affine_matrices = []

    def get_inverse_affine(self, _path=None):
        # In the real helper this inverts affine_matrices; our fake stores the
        # inverse directly in affine_matrices, so just copy them across.
        self.inverse_affine_matrices = list(self.affine_matrices)


class _FakeEnhancer:
    """Stands in for ``gfpgan.GFPGANer`` on the bounded ``paste_back=False`` path.

    ``enhance`` returns a list of solid-color restored face *crops* (not a pasted
    full frame — ``restored_img`` is ``None``, matching ``paste_back=False``) plus a
    ``face_helper`` whose inverse-affine matrices translate each crop to ``box``'s
    origin. The restorer then warps + feathers each crop back itself (the bounded
    paste under test). ``box`` gives the crop size and destination.
    """

    def __init__(self, box=(40, 40, 160, 200), value=255, faces=1, raise_exc=None):
        self.box = box
        self.value = value
        self.faces = faces
        self.raise_exc = raise_exc
        self.calls = 0
        self.upscale = 1
        self.face_helper = _FakeFaceHelper()

    def enhance(self, img, **kwargs):
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        x1, y1, x2, y2 = self.box
        crop_w, crop_h = x2 - x1, y2 - y1
        restored = []
        affines = []
        for _ in range(self.faces):
            crop = np.full((crop_h, crop_w, 3), self.value, np.uint8)
            restored.append(crop)
            # Translate-only inverse affine: crop (0,0) -> frame (x1,y1).
            affines.append(np.array([[1.0, 0.0, x1], [0.0, 1.0, y1]], np.float64))
        if self.face_helper is not None:
            self.face_helper.affine_matrices = affines
            self.face_helper.inverse_affine_matrices = []
        return [], restored, None  # paste_back=False -> restored_img is None


def _restorer_with(enhancer, face_enhance=True):
    cfg = FaceKeepConfig().aggressive
    cfg.face_enhance = face_enhance
    r = Restorer(cfg)
    r._face_enhancer = enhancer
    r._tried_face_init = True  # skip the real (gfpgan-importing) init
    return r


# --------------------------------------------------------------------------- #
# _enhance_background_faces gating
# --------------------------------------------------------------------------- #

def test_enhance_disabled_is_noop():
    """face_enhance=False returns the background byte-for-byte unchanged."""
    enh = _FakeEnhancer()
    r = _restorer_with(enh, face_enhance=False)
    bg = np.full((240, 320, 3), 100, np.uint8)

    out = r._enhance_background_faces(bg)

    assert np.array_equal(out, bg)
    assert enh.calls == 0  # the flag short-circuits before ever calling GFPGAN


def test_enhance_without_gfpgan_is_noop():
    """No GFPGAN installed (enhancer None) -> no-op, no exception."""
    r = _restorer_with(enhancer=None, face_enhance=True)
    bg = np.full((240, 320, 3), 100, np.uint8)

    out = r._enhance_background_faces(bg)

    assert np.array_equal(out, bg)


def test_enhance_blends_restored_face_region():
    """The fake's painted face region shows up in the enhanced background."""
    enh = _FakeEnhancer(box=(40, 40, 160, 200), value=255)
    r = _restorer_with(enh)
    bg = np.full((240, 320, 3), 100, np.uint8)

    out = r._enhance_background_faces(bg)

    assert enh.calls == 1
    # Center of the painted box moved toward the fake's value (255); the soft mask
    # makes it ~full strength at the center.
    assert out[120, 100].mean() > 200
    # Far outside the box is untouched background.
    assert np.array_equal(out[10, 10], bg[10, 10])


def test_enhance_no_faces_found_returns_background():
    """enhance reporting zero restored faces leaves the background unchanged."""
    enh = _FakeEnhancer(faces=0)
    r = _restorer_with(enh)
    bg = np.full((240, 320, 3), 100, np.uint8)

    out = r._enhance_background_faces(bg)

    assert np.array_equal(out, bg)


def test_enhance_inference_error_degrades_gracefully():
    """A RuntimeError from enhance -> un-enhanced background, not a crash."""
    enh = _FakeEnhancer(raise_exc=RuntimeError("CUDA OOM"))
    r = _restorer_with(enh)
    bg = np.full((240, 320, 3), 100, np.uint8)

    out = r._enhance_background_faces(bg)

    assert np.array_equal(out, bg)


# --------------------------------------------------------------------------- #
# bounded-memory paste (ROADMAP backlog: GFPGAN OOM on large frames)
# --------------------------------------------------------------------------- #

def test_enhance_uses_paste_back_false():
    """The bounded path must call enhance(paste_back=False) (no full-frame buffer).

    paste_back=True is what allocates the (H,W,3) float64 frame that OOMs on large
    photos; the whole point of this item is to never request it.
    """
    seen = {}

    class _SpyEnhancer(_FakeEnhancer):
        def enhance(self, img, **kwargs):
            seen.update(kwargs)
            return super().enhance(img, **kwargs)

    r = _restorer_with(_SpyEnhancer())
    r._enhance_background_faces(np.full((240, 320, 3), 100, np.uint8))

    assert seen.get("paste_back") is False


def test_enhance_no_full_frame_buffer_allocated():
    """A restored face is pasted into only its box-sized buffer, not the frame.

    We assert this by effect on a *huge* virtual frame: the fake reports one small
    face, and the paste must succeed without allocating anything frame-sized. We
    monkeypatch cv2.warpAffine to record the output sizes it is asked for — none
    may be the full frame.
    """
    import cv2

    sizes = []
    real_warp = cv2.warpAffine

    def _spy_warp(src, M, dsize, *a, **k):
        sizes.append(dsize)  # (w, h)
        return real_warp(src, M, dsize, *a, **k)

    enh = _FakeEnhancer(box=(50, 60, 130, 180))  # an 80x120 face
    r = _restorer_with(enh)
    bg = np.full((4000, 3000, 3), 100, np.uint8)  # large frame

    cv2.warpAffine = _spy_warp
    try:
        out = r._enhance_background_faces(bg)
    finally:
        cv2.warpAffine = real_warp

    assert sizes, "the bounded paste must warp the face crop"
    # Every warp targets the face's destination box (~80x120), never the frame.
    for w, h in sizes:
        assert w <= 200 and h <= 200, f"warp targeted {w}x{h}, not the bounded box"
    # And the face region was actually restored.
    assert out[120, 90].mean() > 150


def test_enhance_face_outside_frame_skipped():
    """A face whose box maps entirely outside the frame is skipped, not crashed."""
    enh = _FakeEnhancer(box=(500, 500, 580, 620))  # well past a 240x320 frame
    r = _restorer_with(enh)
    bg = np.full((240, 320, 3), 100, np.uint8)

    out = r._enhance_background_faces(bg)

    assert np.array_equal(out, bg)


def test_enhance_missing_face_helper_degrades():
    """No face_helper on the enhancer -> graceful no-op (never an AttributeError)."""
    enh = _FakeEnhancer()
    enh.face_helper = None
    r = _restorer_with(enh)
    bg = np.full((240, 320, 3), 100, np.uint8)

    out = r._enhance_background_faces(bg)

    assert np.array_equal(out, bg)


def test_enhance_affine_count_mismatch_degrades():
    """More restored faces than affine matrices -> graceful no-op, not an IndexError."""
    enh = _FakeEnhancer(faces=2)
    r = _restorer_with(enh)
    # Corrupt the helper so it reports one fewer matrix than restored faces.
    orig_enhance = enh.enhance

    def _bad_enhance(img, **kwargs):
        cropped, restored, ri = orig_enhance(img, **kwargs)
        enh.face_helper.affine_matrices = enh.face_helper.affine_matrices[:1]
        enh.face_helper.inverse_affine_matrices = []
        return cropped, restored, ri

    enh.enhance = _bad_enhance
    bg = np.full((240, 320, 3), 100, np.uint8)

    out = r._enhance_background_faces(bg)

    assert np.array_equal(out, bg)


def test_enhance_paste_triggers_get_inverse_affine():
    """If the helper hasn't computed inverse affines yet, the restorer triggers it."""
    enh = _FakeEnhancer()

    class _LazyHelper(_FakeFaceHelper):
        def __init__(self):
            super().__init__()
            self.get_inverse_called = 0

        def get_inverse_affine(self, _path=None):
            self.get_inverse_called += 1
            super().get_inverse_affine(_path)

    lazy = _LazyHelper()
    enh.face_helper = lazy
    # enhance() populates affine_matrices but leaves inverse_affine_matrices empty,
    # so the restorer must call get_inverse_affine to fill them.
    orig = enh.enhance

    def _enhance(img, **kwargs):
        cropped, restored, ri = orig(img, **kwargs)
        lazy.affine_matrices = [np.array([[1.0, 0, 40], [0, 1.0, 40]], np.float64)]
        lazy.inverse_affine_matrices = []
        return cropped, restored, ri

    enh.enhance = _enhance
    r = _restorer_with(enh)

    r._enhance_background_faces(np.full((240, 320, 3), 100, np.uint8))

    assert lazy.get_inverse_called == 1


# --------------------------------------------------------------------------- #
# restore() wiring
# --------------------------------------------------------------------------- #

def test_restore_calls_face_enhancer(face_image, tmp_path):
    """restore() runs the face enhancer on the upscaled background."""
    path = _build_fkeep(face_image, tmp_path)
    enh = _FakeEnhancer(box=(0, 0, 30, 30))  # corner, away from the detected face
    r = _restorer_with(enh)

    r.restore(str(path))

    assert enh.calls == 1


def test_restore_detected_face_crop_wins_over_enhancer(face_image, tmp_path):
    """A real detected-face crop is composited on top of the GFPGAN result.

    The fake enhancer paints the *whole frame* white; if the detected face crop
    were not composited last, the face region would be white too. It must instead
    carry the real (non-white) crop pixels.
    """
    path = _build_fkeep(face_image, tmp_path)
    data = read_fkeep(str(path))
    assert data["manifest"]["faces"], "fixture must have a detected face"
    fx1, fy1, fx2, fy2 = data["manifest"]["faces"][0]["padded_bbox"]
    cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2

    ow = data["manifest"]["original"]["width"]
    oh = data["manifest"]["original"]["height"]
    enh = _FakeEnhancer(box=(0, 0, ow, oh), value=255)  # paint everything white
    r = _restorer_with(enh)

    result = r.restore(str(path))

    # The detected-face center is the real crop, not the enhancer's white wash.
    assert result[cy, cx].mean() < 250


def test_preview_does_not_call_face_enhancer(face_image, tmp_path):
    """preview() is the fast, no-AI path: it must not invoke GFPGAN."""
    path = _build_fkeep(face_image, tmp_path)
    enh = _FakeEnhancer()
    r = _restorer_with(enh)

    r.preview(str(path))

    assert enh.calls == 0


def test_face_enhance_off_skips_enhancer_in_restore(face_image, tmp_path):
    """With the flag off, restore never touches the enhancer."""
    path = _build_fkeep(face_image, tmp_path)
    enh = _FakeEnhancer()
    r = _restorer_with(enh, face_enhance=False)

    r.restore(str(path))

    assert enh.calls == 0


# --------------------------------------------------------------------------- #
# config default
# --------------------------------------------------------------------------- #

def test_face_enhance_default_on():
    assert FaceKeepConfig().aggressive.face_enhance is True


# --------------------------------------------------------------------------- #
# Face-enhance backends + strength (ROADMAP 8.4)
#
# `face_enhance_backend` selects GFPGAN (default) or CodeFormer (opt-in
# [codeformer] extra) for the missed-background-face safety net;
# `face_enhance_fidelity` is CodeFormer's w dial; `face_enhance_strength`
# lerps every restored face toward the un-enhanced pixels at paste time (both
# backends). All three are restore-only -> never fingerprinted. Like the rest
# of this file, nothing here imports gfpgan/codeformer for real — fakes pin
# the wiring; the real CodeFormer net is exercised only by the opt-in
# @pytest.mark.real_ai test at the bottom.
# --------------------------------------------------------------------------- #

def test_face_enhance_backend_defaults():
    a = FaceKeepConfig().aggressive
    assert a.face_enhance_backend == "gfpgan"
    assert a.face_enhance_fidelity == 0.7
    assert a.face_enhance_strength == 1.0


def test_validate_rejects_unknown_backend():
    cfg = FaceKeepConfig()
    cfg.aggressive.face_enhance_backend = "deepface"
    with pytest.raises(ConfigError, match="face_enhance_backend"):
        cfg.validate()


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_validate_rejects_out_of_range_fidelity(value):
    cfg = FaceKeepConfig()
    cfg.aggressive.face_enhance_fidelity = value
    with pytest.raises(ConfigError, match="face_enhance_fidelity"):
        cfg.validate()


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_validate_rejects_out_of_range_strength(value):
    cfg = FaceKeepConfig()
    cfg.aggressive.face_enhance_strength = value
    with pytest.raises(ConfigError, match="face_enhance_strength"):
        cfg.validate()


def test_yaml_round_trip_face_enhance_knobs(tmp_path):
    cfg = FaceKeepConfig()
    cfg.aggressive.face_enhance_backend = "codeformer"
    cfg.aggressive.face_enhance_fidelity = 0.5
    cfg.aggressive.face_enhance_strength = 0.8
    path = tmp_path / "facekeep.yaml"
    cfg.save(path)

    loaded = FaceKeepConfig.load(path)

    assert loaded.aggressive.face_enhance_backend == "codeformer"
    assert loaded.aggressive.face_enhance_fidelity == 0.5
    assert loaded.aggressive.face_enhance_strength == 0.8


# The conftest's autouse `_force_bicubic_restore` replaces `_init_face_enhancer`
# wholesale (the offline-by-default contract), so tests of the *real* dispatcher
# must restore it first. Captured at module import time, before any fixture runs.
_REAL_INIT_FACE_ENHANCER = Restorer._init_face_enhancer


def test_backend_selection_default_routes_to_gfpgan(monkeypatch):
    monkeypatch.setattr(Restorer, "_init_face_enhancer", _REAL_INIT_FACE_ENHANCER)
    calls = []
    monkeypatch.setattr(
        Restorer, "_init_gfpgan_enhancer", lambda self: calls.append("gfpgan")
    )
    monkeypatch.setattr(
        Restorer, "_init_codeformer_enhancer",
        lambda self: calls.append("codeformer"),
    )
    r = Restorer(FaceKeepConfig().aggressive)

    r._init_face_enhancer()

    assert calls == ["gfpgan"]
    assert r._tried_face_init is True


def test_backend_selection_codeformer_routes_to_codeformer(monkeypatch):
    monkeypatch.setattr(Restorer, "_init_face_enhancer", _REAL_INIT_FACE_ENHANCER)
    calls = []
    monkeypatch.setattr(
        Restorer, "_init_gfpgan_enhancer", lambda self: calls.append("gfpgan")
    )
    monkeypatch.setattr(
        Restorer, "_init_codeformer_enhancer",
        lambda self: calls.append("codeformer"),
    )
    cfg = FaceKeepConfig().aggressive
    cfg.face_enhance_backend = "codeformer"
    r = Restorer(cfg)

    r._init_face_enhancer()

    assert calls == ["codeformer"]


def test_codeformer_init_threads_fidelity_and_verified_weights(monkeypatch, tmp_path):
    """The codeformer init fetches weights via ensure_weights (checksum-verified
    local path, never the URL) and hands the config fidelity to the enhancer."""
    # Dummy packages so the import pre-check passes without codeformer installed.
    monkeypatch.setitem(sys.modules, "codeformer", types.ModuleType("codeformer"))
    monkeypatch.setitem(sys.modules, "facexlib", types.ModuleType("facexlib"))

    fetched = {}
    fake_path = tmp_path / "codeformer.pth"

    def _fake_ensure(url, filename, sha256=None):
        fetched.update(url=url, filename=filename, sha256=sha256)
        return fake_path

    built = {}

    class _SpyEnhancer:
        def __init__(self, model_path, fidelity):
            built.update(model_path=model_path, fidelity=fidelity)

    monkeypatch.setattr(restorer_module, "ensure_weights", _fake_ensure)
    monkeypatch.setattr(restorer_module, "_CodeFormerEnhancer", _SpyEnhancer)

    cfg = FaceKeepConfig().aggressive
    cfg.face_enhance_backend = "codeformer"
    cfg.face_enhance_fidelity = 0.35
    r = Restorer(cfg)

    # Call the (un-patched) codeformer init directly — the conftest fixture
    # replaces only the `_init_face_enhancer` dispatcher, whose routing is
    # covered by the two tests above.
    r._init_codeformer_enhancer()

    url, filename, sha256 = restorer_module._CODEFORMER_WEIGHTS
    assert fetched == {"url": url, "filename": filename, "sha256": sha256}
    assert built["model_path"] == str(fake_path)  # local verified path, not the URL
    assert built["fidelity"] == 0.35
    assert isinstance(r._face_enhancer, _SpyEnhancer)


def test_codeformer_missing_package_degrades_without_download(monkeypatch):
    """codeformer-pip absent -> warn + skip enhancement (enhancer None), and the
    ~360 MB weights download is never attempted. No silent gfpgan fallback."""
    monkeypatch.setitem(sys.modules, "codeformer", None)  # forces ImportError

    def _must_not_download(*a, **k):
        raise AssertionError("ensure_weights must not run when the package is missing")

    monkeypatch.setattr(restorer_module, "ensure_weights", _must_not_download)

    cfg = FaceKeepConfig().aggressive
    cfg.face_enhance_backend = "codeformer"
    r = Restorer(cfg)

    r._init_codeformer_enhancer()  # direct: the dispatcher is conftest-patched
    r._tried_face_init = True

    assert r._face_enhancer is None  # skipped — not swapped for a GFPGANer
    bg = np.full((240, 320, 3), 100, np.uint8)
    assert np.array_equal(r._enhance_background_faces(bg), bg)


def test_codeformer_weights_failure_degrades(monkeypatch):
    """A weights download/checksum failure -> warn + skip, never a crash."""
    monkeypatch.setitem(sys.modules, "codeformer", types.ModuleType("codeformer"))
    monkeypatch.setitem(sys.modules, "facexlib", types.ModuleType("facexlib"))

    def _fail(*a, **k):
        raise ModelDownloadError("offline")

    monkeypatch.setattr(restorer_module, "ensure_weights", _fail)

    cfg = FaceKeepConfig().aggressive
    cfg.face_enhance_backend = "codeformer"
    r = Restorer(cfg)

    r._init_codeformer_enhancer()  # direct: the dispatcher is conftest-patched

    assert r._face_enhancer is None


class _FakeAlignHelper:
    """Stands in for facexlib's FaceRestoreHelper inside _CodeFormerEnhancer:
    'aligns' a preset list of crops and collects the restored ones."""

    def __init__(self, crops):
        self._crops = crops
        self.cropped_faces = []
        self.restored_faces = []

    def clean_all(self):
        self.cropped_faces = []
        self.restored_faces = []

    def read_image(self, img):
        self._img = img

    def get_face_landmarks_5(self, **kwargs):
        pass

    def align_warp_face(self):
        self.cropped_faces = list(self._crops)

    def add_restored_face(self, face):
        self.restored_faces.append(face)


def test_codeformer_enhancer_passes_fidelity_w():
    """enhance() must call the net with w=<config fidelity> and adain=True, and
    our BGR<->tensor math must round-trip exactly (an identity net returns the
    input crop byte-for-byte)."""
    pytest.importorskip("torch")

    calls = []

    class _IdentityNet:
        def __call__(self, t, w=None, adain=None):
            calls.append({"w": w, "adain": adain})
            return (t, None)  # the real net returns a tuple; [0] is the image

    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    helper = _FakeAlignHelper([crop])
    enh = restorer_module._CodeFormerEnhancer(
        model_path=None, fidelity=0.42, net=_IdentityNet(), face_helper=helper
    )

    cropped, restored, restored_img = enh.enhance(
        np.zeros((240, 320, 3), np.uint8)
    )

    assert calls == [{"w": 0.42, "adain": True}]
    assert restored_img is None  # paste_back=False contract
    assert len(restored) == 1
    assert restored[0].dtype == np.uint8
    # uint8 -> [-1,1] RGB tensor -> back must be the identity, or the backend
    # would tint every restored face.
    assert np.array_equal(restored[0], crop)


def test_codeformer_enhancer_failed_face_keeps_unrestored_crop():
    """GFPGAN parity: one face failing inference keeps its un-restored crop
    instead of failing the whole enhancement pass."""
    pytest.importorskip("torch")

    class _ExplodingNet:
        def __call__(self, t, w=None, adain=None):
            raise RuntimeError("CUDA OOM")

    crop = np.full((64, 64, 3), 77, np.uint8)
    helper = _FakeAlignHelper([crop])
    enh = restorer_module._CodeFormerEnhancer(
        model_path=None, fidelity=0.7, net=_ExplodingNet(), face_helper=helper
    )

    _, restored, _ = enh.enhance(np.zeros((240, 320, 3), np.uint8))

    assert len(restored) == 1
    assert np.array_equal(restored[0], crop)


def test_enhance_strength_zero_skips_inference_entirely():
    """strength=0 is a no-op by construction, so the model is never even called."""
    enh = _FakeEnhancer()
    r = _restorer_with(enh)
    r.config.face_enhance_strength = 0.0
    bg = np.full((240, 320, 3), 100, np.uint8)

    out = r._enhance_background_faces(bg)

    assert np.array_equal(out, bg)
    assert enh.calls == 0


def test_enhance_strength_default_is_full_enhancement():
    """The default (1.0) is byte-identical to explicitly requesting full
    strength — i.e. the pre-8.4 behavior (multiplying the mask by exactly 1.0
    is an IEEE identity)."""
    bg = np.full((240, 320, 3), 100, np.uint8)

    out_default = _restorer_with(_FakeEnhancer())._enhance_background_faces(bg.copy())
    r_explicit = _restorer_with(_FakeEnhancer())
    r_explicit.config.face_enhance_strength = 1.0
    out_explicit = r_explicit._enhance_background_faces(bg.copy())

    assert np.array_equal(out_default, out_explicit)
    assert out_default[120, 100].mean() > 200  # and it IS enhanced (anti-false-green)


def test_enhance_strength_scales_the_blend_linearly():
    """strength=0.5 lands each pixel halfway between un-enhanced and fully
    enhanced: (out_half - bg) ~= 0.5 * (out_full - bg)."""
    bg = np.full((240, 320, 3), 100, np.uint8)

    r_full = _restorer_with(_FakeEnhancer())
    out_full = r_full._enhance_background_faces(bg.copy())
    r_half = _restorer_with(_FakeEnhancer())
    r_half.config.face_enhance_strength = 0.5
    out_half = r_half._enhance_background_faces(bg.copy())

    full_delta = float(out_full[120, 100].mean()) - 100.0  # box center: mask ~= 1
    half_delta = float(out_half[120, 100].mean()) - 100.0
    assert full_delta > 100  # the fake paints 255 over 100
    assert abs(half_delta - 0.5 * full_delta) <= 2.0  # the lerp, up to rounding
    # Outside the face box both are untouched background.
    assert np.array_equal(out_full[10, 10], bg[10, 10])
    assert np.array_equal(out_half[10, 10], bg[10, 10])


def test_fingerprint_unchanged_by_face_enhance_knobs():
    """Restore-only knobs must not bust the compress cache (mirrors
    test_restore_anchor.py's anchor-knob guard)."""
    base = FaceKeepConfig()
    base.mode = "aggressive"
    fp_base = settings_fingerprint(base)

    c = FaceKeepConfig()
    c.mode = "aggressive"
    c.aggressive.face_enhance_backend = "codeformer"
    c.aggressive.face_enhance_fidelity = 0.3
    c.aggressive.face_enhance_strength = 0.5
    assert settings_fingerprint(c) == fp_base


@pytest.mark.real_ai
def test_real_codeformer_restores_a_crop():
    """Drive the REAL CodeFormer net (weights via ensure_weights) through our
    enhancer on a degraded 512x512 crop. Skips unless codeformer-pip is
    installed AND the weights are already in the local cache (never pulls
    ~360 MB inside the suite). The align helper is faked so no facexlib
    detection-model download happens either — this pins weights-loading + the
    net forward + our tensor math, not retinaface."""
    pytest.importorskip("codeformer")
    pytest.importorskip("torch")
    from facekeep.models import MODELS_CACHE_DIR, ensure_weights

    url, filename, sha256 = restorer_module._CODEFORMER_WEIGHTS
    if not (MODELS_CACHE_DIR / filename).exists():
        pytest.skip("codeformer.pth not cached (suite never downloads ~360 MB)")
    model_path = str(ensure_weights(url, filename, sha256=sha256))

    # A heavily blurred noisy crop — what an AI-upscaled missed face looks like.
    rng = np.random.default_rng(7)
    import cv2
    crop = rng.integers(60, 200, (512, 512, 3), dtype=np.uint8)
    crop = cv2.GaussianBlur(crop, (31, 31), 8)

    helper = _FakeAlignHelper([crop])
    enh = restorer_module._CodeFormerEnhancer(model_path, 0.7, face_helper=helper)

    _, restored, _ = enh.enhance(np.zeros((512, 512, 3), np.uint8))

    assert len(restored) == 1
    assert restored[0].shape == (512, 512, 3)
    assert restored[0].dtype == np.uint8
    assert not np.array_equal(restored[0], crop)  # the net actually ran
