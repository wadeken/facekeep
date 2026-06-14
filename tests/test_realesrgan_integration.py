"""Real Real-ESRGAN integration — ROADMAP Phase 4 (core mechanics).

Every *other* restore test exercises the **bicubic fallback** (the AI upsampler
is mocked or absent). This file is the one that drives the genuine ``[ai]`` path
end to end: it lets ``Restorer`` import and construct a real ``RealESRGANer``,
download/cache its weights, and run inference on a real ``.fkeep``, asserting the
output is the right shape/dtype and that the AI path (not bicubic) was taken.

Two real-world frictions this item exists to surface and fix are pinned here:

1. **torchvision compatibility.** ``basicsr`` (imported by both Real-ESRGAN and
   GFPGAN) does ``from torchvision.transforms.functional_tensor import
   rgb_to_grayscale`` at import time, but ``functional_tensor`` was removed in
   torchvision 0.17. ``restorer._ensure_torchvision_compat`` shims it; without
   that, installing ``[ai]`` against a current torchvision still silently
   degrades to bicubic. ``test_torchvision_compat_shim`` is the regression lock.

2. **Weights handling.** ``RealESRGANer(model_path=None)`` crashes
   (``AttributeError`` on ``model_path.startswith``); it only downloads/caches
   weights when given an ``https://`` URL. ``_init_upsampler`` now resolves the
   configured model name to its official weights URL.

This whole module **skips** when the ``[ai]`` extra (or its weights) can't be
loaded — offline, not installed, or a download failure — matching the YuNet /
corpus offline convention, so CI stays green without the extra. The real run is
CPU-only and slow, so the test image is deliberately tiny and ``face_enhance`` is
disabled to isolate the Real-ESRGAN path (and avoid a second model download).
"""

import sys

import cv2
import numpy as np
import pytest

from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import write_fkeep
from facekeep.aggressive.restorer import Restorer, _ensure_torchvision_compat
from facekeep.config import FaceKeepConfig


def _realesrgan_or_skip():
    """Apply the compat shim, import Real-ESRGAN, and build a real upsampler.

    Returns the constructed ``Restorer`` (with ``_upsampler`` loaded) or skips the
    test if the AI extra / its weights are unavailable for any reason (not
    installed, offline download failure, corrupt cache). Restoring the real model
    is the point, so a fallback-to-bicubic here is treated as "cannot verify" and
    skipped rather than passed.
    """
    _ensure_torchvision_compat()
    pytest.importorskip("realesrgan", reason="[ai] extra not installed")
    cfg = FaceKeepConfig()
    cfg.aggressive.face_enhance = False  # isolate Real-ESRGAN; no GFPGAN download
    r = Restorer(cfg.aggressive)
    r._init_upsampler()  # downloads/caches weights on first use
    if r._upsampler is None:
        pytest.skip("Real-ESRGAN weights unavailable (offline?) — cannot verify")
    return r, cfg


def _benign_image(h=160, w=240) -> np.ndarray:
    """A small, smooth, faceless image — the fast, deterministic AI-restore case."""
    rng = np.random.default_rng(0)
    bg = cv2.resize(
        rng.normal(128, 30, (h // 4, w // 4, 3)).astype(np.float32),
        (w, h), interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(bg, 0, 255).astype(np.uint8)


def _pack(tmp_path, cfg) -> tuple[str, np.ndarray]:
    img = _benign_image()
    src = tmp_path / "scene.jpg"
    cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    photo = compress_photo(str(src), cfg)
    fk = tmp_path / "scene.fkeep"
    write_fkeep(photo, str(fk))
    return str(fk), img


# --------------------------------------------------------------------------- #
# Compatibility shim (cheap; no model load) — the regression lock for the
# torchvision >= 0.17 breakage that otherwise hides the whole AI path.
# --------------------------------------------------------------------------- #

def test_torchvision_compat_shim():
    pytest.importorskip("torchvision", reason="torchvision not installed")
    _ensure_torchvision_compat()
    # The legacy module BasicSR imports must now be importable, and expose the
    # symbol BasicSR actually pulls from it.
    import importlib

    mod = importlib.import_module("torchvision.transforms.functional_tensor")
    assert hasattr(mod, "rgb_to_grayscale")
    assert "torchvision.transforms.functional_tensor" in sys.modules


def test_compat_shim_does_not_override_existing(monkeypatch):
    """If functional_tensor already exists, the shim is a no-op (never clobbers)."""
    sentinel = object()
    monkeypatch.setitem(
        sys.modules, "torchvision.transforms.functional_tensor", sentinel
    )
    _ensure_torchvision_compat()
    assert sys.modules["torchvision.transforms.functional_tensor"] is sentinel


# --------------------------------------------------------------------------- #
# Real end-to-end Real-ESRGAN restore (skips without the [ai] extra / weights).
# --------------------------------------------------------------------------- #

@pytest.mark.real_ai
def test_real_esrgan_restore_full_resolution(tmp_path):
    """A genuine Real-ESRGAN restore returns the original dimensions, not bicubic."""
    r, cfg = _realesrgan_or_skip()
    fk, original = _pack(tmp_path, cfg)

    out = r.restore(fk)

    # The AI path, not the bicubic fallback, was taken.
    assert r._upsampler is not None
    # outscale = 1 / bg_scale reproduces the original frame exactly.
    assert out.shape == original.shape
    assert out.dtype == np.uint8


@pytest.mark.real_ai
def test_real_esrgan_restore_writes_standard_file(tmp_path):
    """The AI-restored result writes out as a normal, decodable image."""
    r, cfg = _realesrgan_or_skip()
    fk, original = _pack(tmp_path, cfg)

    out_path = tmp_path / "restored.jpg"
    result = r.restore(fk, str(out_path))

    assert out_path.exists() and out_path.stat().st_size > 0
    decoded = cv2.imread(str(out_path))
    assert decoded is not None
    assert decoded.shape == original.shape
    assert result.shape == original.shape


@pytest.mark.real_ai  # exercises the real _init_upsampler, not the no-AI stub
def test_init_upsampler_unknown_model_falls_back(tmp_path):
    """An unrecognized model name degrades to bicubic (no crash, no AI)."""
    cfg = FaceKeepConfig()
    cfg.aggressive.model = "does-not-exist"
    cfg.aggressive.face_enhance = False
    r = Restorer(cfg.aggressive)
    r._init_upsampler()
    assert r._upsampler is None  # unknown model -> bicubic path
