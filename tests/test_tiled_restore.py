"""Tiled restore: tile / tile_pad are exposed config knobs and reach Real-ESRGAN.

Real-ESRGAN tiles its upscale internally so a 24MP+ background never materializes
a full-resolution intermediate at once. Those tile sizes used to be hard-coded
(512 / 10); this verifies they are now ``aggressive.tile`` / ``aggressive.tile_pad``
config knobs that thread through to ``RealESRGANer``, while keeping the old
defaults and staying out of the (compress-only) index fingerprint.

All tests are offline: ``RealESRGANer`` / ``RRDBNet`` and ``ensure_weights`` are
stubbed so no model is constructed and no weights are fetched.
"""

import sys
import types

import pytest

from facekeep import index as index_mod
from facekeep.aggressive.restorer import Restorer
from facekeep.config import AggressiveConfig, ConfigError, FaceKeepConfig

# The conftest autouse `_force_bicubic_restore` fixture replaces
# `Restorer._init_upsampler` with a no-AI stub for every non-`real_ai` test, so
# these tests (which exercise the real `_init_upsampler` against a *stubbed*
# library) must restore the genuine method. Capture it at import, before any
# fixture patches it.
_REAL_INIT_UPSAMPLER = Restorer._init_upsampler


def _stub_realesrgan(monkeypatch):
    """Stub realesrgan/basicsr so ``_init_upsampler`` runs offline.

    Returns a dict the fake ``RealESRGANer`` fills with the kwargs it was
    constructed with, so a test can assert tile/tile_pad reached the library.
    """
    # Undo the conftest no-AI override: run the real init against our stubs.
    monkeypatch.setattr(Restorer, "_init_upsampler", _REAL_INIT_UPSAMPLER)

    captured = {}

    class _FakeRealESRGANer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeRRDBNet:
        def __init__(self, **kwargs):
            pass

    realesrgan_mod = types.ModuleType("realesrgan")
    realesrgan_mod.RealESRGANer = _FakeRealESRGANer
    monkeypatch.setitem(sys.modules, "realesrgan", realesrgan_mod)

    basicsr_mod = types.ModuleType("basicsr")
    archs_mod = types.ModuleType("basicsr.archs")
    rrdb_mod = types.ModuleType("basicsr.archs.rrdbnet_arch")
    rrdb_mod.RRDBNet = _FakeRRDBNet
    monkeypatch.setitem(sys.modules, "basicsr", basicsr_mod)
    monkeypatch.setitem(sys.modules, "basicsr.archs", archs_mod)
    monkeypatch.setitem(sys.modules, "basicsr.archs.rrdbnet_arch", rrdb_mod)

    # No weights download: hand back a fake local path.
    monkeypatch.setattr(
        "facekeep.aggressive.restorer.ensure_weights",
        lambda url, filename, sha256=None: "/fake/cache/" + filename,
    )
    # Compat shim is irrelevant here (torchvision may be absent) — make it a no-op.
    monkeypatch.setattr(
        "facekeep.aggressive.restorer._ensure_torchvision_compat", lambda: None
    )
    return captured


# --------------------------------------------------------------------------- #
# Config: fields, defaults, validation, YAML round-trip
# --------------------------------------------------------------------------- #

def test_defaults_unchanged():
    """Defaults stay 512 / 10 — exposing the knobs must not change behavior."""
    cfg = AggressiveConfig()
    assert cfg.tile == 512
    assert cfg.tile_pad == 10


def test_validate_accepts_zero_and_positive():
    for tile, pad in ((0, 0), (256, 10), (1024, 32)):
        c = FaceKeepConfig()
        c.aggressive.tile = tile
        c.aggressive.tile_pad = pad
        c.validate()  # must not raise


def test_validate_rejects_negative_tile():
    c = FaceKeepConfig()
    c.aggressive.tile = -1
    with pytest.raises(ConfigError, match="tile must be >= 0"):
        c.validate()


def test_validate_rejects_negative_tile_pad():
    c = FaceKeepConfig()
    c.aggressive.tile_pad = -5
    with pytest.raises(ConfigError, match="tile_pad must be >= 0"):
        c.validate()


def test_yaml_round_trip(tmp_path):
    c = FaceKeepConfig()
    c.aggressive.tile = 256
    c.aggressive.tile_pad = 24
    p = tmp_path / "facekeep.yaml"
    c.save(p)
    loaded = FaceKeepConfig.load(p)
    assert loaded.aggressive.tile == 256
    assert loaded.aggressive.tile_pad == 24


# --------------------------------------------------------------------------- #
# restorer: tile/tile_pad reach RealESRGANer
# --------------------------------------------------------------------------- #

def test_custom_tile_reaches_realesrganer(monkeypatch):
    captured = _stub_realesrgan(monkeypatch)
    cfg = AggressiveConfig(tile=256, tile_pad=24)
    r = Restorer(cfg)
    r._init_upsampler()
    assert r._upsampler is not None  # the fake constructed, not the bicubic fallback
    assert captured["tile"] == 256
    assert captured["tile_pad"] == 24


def test_default_tile_reaches_realesrganer(monkeypatch):
    captured = _stub_realesrgan(monkeypatch)
    r = Restorer(AggressiveConfig())
    r._init_upsampler()
    assert captured["tile"] == 512
    assert captured["tile_pad"] == 10


def test_tile_zero_disables_tiling(monkeypatch):
    captured = _stub_realesrgan(monkeypatch)
    r = Restorer(AggressiveConfig(tile=0))
    r._init_upsampler()
    assert captured["tile"] == 0  # 0 = no tiling, passed through as-is


# --------------------------------------------------------------------------- #
# Restore-only: tile/tile_pad are NOT in the compress index fingerprint
# --------------------------------------------------------------------------- #

def test_tile_not_in_fingerprint():
    """tile/tile_pad are restore-only and must not bust the compress cache."""
    base = FaceKeepConfig()
    base.mode = "aggressive"
    fp_base = index_mod.settings_fingerprint(base)

    c = FaceKeepConfig()
    c.mode = "aggressive"
    c.aggressive.tile = 128
    c.aggressive.tile_pad = 1
    assert index_mod.settings_fingerprint(c) == fp_base
