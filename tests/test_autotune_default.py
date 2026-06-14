"""Auto-tune on by default — ROADMAP Phase 5.

Auto-tune is now the faithful-mode default (its precondition — a perceptual
acceptance metric, SSIMULACRA2 — has landed). Users get a visually-lossless
quality without picking a `-q` number. These tests pin:

* The default config: ``auto_tune`` on, ``target_metric`` the perceptual one
  (``ssimulacra2``) with a ~90 (visually-lossless) target on its scale.
* The default compress path actually runs the auto-tune search (not the fixed
  encode) — proven by spying on ``_auto_tune_quality``.
* CLI interaction (``cli._load_config``, no network/encode needed):
    - bare run keeps auto-tune on (default);
    - an explicit ``-q`` disables auto-tune (explicit quality is an override)
      and sets that quality;
    - ``--auto-tune`` together with ``-q`` keeps auto-tune on (the flag wins),
      with ``-q`` as the search's fallback quality;
    - ``--no-auto-tune`` forces it off even without ``-q``.
* Graceful degradation is unchanged: with the perceptual package absent (the
  default install), the search still runs via the SSIM fallback and produces a
  valid, smaller-than-original encode — never a crash.

These use synthetic fixtures and stay offline (no model download): SSIMULACRA2
is pure-Python and, when absent, the SSIM fallback path is exercised instead.
"""

import pytest

from facekeep import encoders, faithful
from facekeep.cli import _load_config
from facekeep.config import FaceKeepConfig

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


# --------------------------------------------------------------------------- #
# default config
# --------------------------------------------------------------------------- #

def test_default_config_auto_tune_on_with_perceptual_metric():
    """The shipped default enables auto-tune against the perceptual metric."""
    cfg = FaceKeepConfig()
    assert cfg.faithful.auto_tune is True
    assert cfg.faithful.target_metric == "ssimulacra2"
    # ~90 on the SSIMULACRA2 scale is the visually-lossless mark (not an SSIM
    # 0-1 threshold). Guard against accidentally shipping an SSIM-scale number.
    assert cfg.faithful.target_value > 1.0
    cfg.validate()  # the default must always validate


# --------------------------------------------------------------------------- #
# default compress path uses the search
# --------------------------------------------------------------------------- #

@requires_avif
def test_default_compress_invokes_auto_tune(face_image, tmp_path, monkeypatch):
    """A default-config compress goes through the auto-tune search, not a fixed encode."""
    calls = {"n": 0}
    real = faithful._auto_tune_quality

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(faithful, "_auto_tune_quality", _spy)

    result = faithful.compress(str(face_image), str(tmp_path / "out"), FaceKeepConfig())
    assert calls["n"] == 1, "default compress did not run the auto-tune search"
    assert result.output_path.exists()
    assert result.compressed_size < result.original_size


@requires_avif
def test_no_auto_tune_skips_the_search(face_image, tmp_path, monkeypatch):
    """auto_tune=False uses the fixed-quality encode (search never called)."""
    def _boom(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("auto-tune search ran with auto_tune=False")

    monkeypatch.setattr(faithful, "_auto_tune_quality", _boom)

    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = False
    cfg.faithful.quality = 70
    result = faithful.compress(str(face_image), str(tmp_path / "out"), cfg)
    assert result.quality_used == 70


# --------------------------------------------------------------------------- #
# CLI / _load_config interaction
# --------------------------------------------------------------------------- #

@pytest.fixture
def _clean_cwd(tmp_path, monkeypatch):
    """Run in a config-free directory so ``_load_config`` sees the built-in
    defaults, not an ambient ``./facekeep.yaml`` (the repo ships one)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _cfg(quality=None, auto_tune=None):
    """Build a config the way the `compress` CLI does, for the faithful knobs."""
    return _load_config(
        config_path=None, mode=None, codec=None, quality=quality, bg_scale=None,
        auto_tune=auto_tune,
    )


def test_cli_bare_run_keeps_auto_tune_on(_clean_cwd):
    cfg = _cfg()
    assert cfg.faithful.auto_tune is True


def test_cli_explicit_quality_disables_auto_tune(_clean_cwd):
    """An explicit -q is a deliberate override: auto-tune off, quality applied."""
    cfg = _cfg(quality=55)
    assert cfg.faithful.auto_tune is False
    assert cfg.faithful.quality == 55


def test_cli_auto_tune_flag_wins_over_quality(_clean_cwd):
    """--auto-tune alongside -q keeps the search on; -q seeds the fallback."""
    cfg = _cfg(quality=55, auto_tune=True)
    assert cfg.faithful.auto_tune is True
    assert cfg.faithful.quality == 55


def test_cli_no_auto_tune_flag_forces_off_without_quality(_clean_cwd):
    cfg = _cfg(auto_tune=False)
    assert cfg.faithful.auto_tune is False
    # quality untouched -> stays the config default
    assert cfg.faithful.quality == FaceKeepConfig().faithful.quality


# --------------------------------------------------------------------------- #
# graceful degradation (perceptual package absent -> SSIM fallback)
# --------------------------------------------------------------------------- #

@requires_avif
def test_default_compress_degrades_to_ssim_when_perceptual_absent(
    face_image, tmp_path, monkeypatch
):
    """Default (ssimulacra2 target) with the package absent still compresses.

    The search falls back to SSIM (threshold re-based) and produces a valid,
    smaller-than-original encode — never a crash. Forced by stubbing
    availability False regardless of whether the package is installed here.
    """
    monkeypatch.setattr(
        faithful.metrics, "ssimulacra2_available", lambda: False
    )
    result = faithful.compress(str(face_image), str(tmp_path / "out"), FaceKeepConfig())
    assert result.output_path.exists()
    assert result.compressed_size < result.original_size
    # decodes back at the original dimensions (no mangling through the search)
    from facekeep.imageio import load

    original = load(str(face_image)).image
    decoded = encoders.decode(result.output_path.read_bytes())
    assert decoded.shape == original.shape
