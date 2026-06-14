"""Tests for the local Gradio GUI (ROADMAP Phase 7).

The GUI is a thin wrapper over the same pipeline the CLI uses, so the bytes it
produces are the CLI's bytes — there is no new fidelity surface to lock. What
*is* worth testing is the wrapper plumbing, all of which is exercised here
without a browser:

* the pure handler (``compress_image``) produces a real output file + valid
  before/after arrays for both modes, offline;
* config building honors the GUI knobs and the preset/mode precedence rules;
* the ``facekeep gui`` command degrades gracefully when gradio (the ``[gui]``
  extra) is absent, and otherwise launches locally with sharing OFF.

``facekeep.gui`` imports fine without gradio (it is imported lazily inside
``build_demo``/``launch``), so every test here runs in the core/dev venv; the
one test that builds the real ``Blocks`` UI ``importorskip``s gradio.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import gui
from facekeep.cli import cli
from facekeep.exceptions import ConfigError, FaceKeepError


# --- pure handler ---------------------------------------------------------

def _assert_display_array(arr):
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3 and arr.shape[2] == 3


def test_compress_image_faithful(face_image, tmp_path):
    out = gui.compress_image(
        str(face_image), "faithful",
        quality=80, auto_tune=False,  # fast + deterministic
        out_dir=str(tmp_path / "out"),
    )
    assert out.output_path.endswith(".avif")
    assert Path(out.output_path).exists()
    _assert_display_array(out.before_rgb)
    _assert_display_array(out.after_rgb)
    # The "after" is the decoded output: same dimensions as the original.
    assert out.before_rgb.shape == out.after_rgb.shape
    assert "faithful" in out.summary


def test_compress_image_aggressive(face_image, tmp_path):
    out = gui.compress_image(
        str(face_image), "aggressive",
        out_dir=str(tmp_path / "out"),
    )
    assert out.output_path.endswith(".fkeep")
    assert Path(out.output_path).exists()
    _assert_display_array(out.before_rgb)
    _assert_display_array(out.after_rgb)
    # The "after" is the bicubic restore preview at full resolution.
    assert out.before_rgb.shape == out.after_rgb.shape
    assert "aggressive" in out.summary


def test_compress_image_no_input_raises():
    with pytest.raises(FaceKeepError):
        gui.compress_image("", "faithful")


# --- pure compare handler -------------------------------------------------

def test_compare_images_faithful(face_image, tmp_path):
    # Produce a real faithful output, then compare the original against it.
    comp = gui.compress_image(
        str(face_image), "faithful",
        quality=80, auto_tune=False,  # fast + deterministic
        out_dir=str(tmp_path / "out"),
    )
    out = gui.compare_images(str(face_image), comp.output_path)
    _assert_display_array(out.before_rgb)
    _assert_display_array(out.after_rgb)
    _assert_display_array(out.diff_rgb)
    # All three views share the original's geometry (the "after" is aligned).
    assert out.before_rgb.shape == out.after_rgb.shape == out.diff_rgb.shape
    assert "SSIM" in out.summary


def test_compare_images_aggressive_fkeep(face_image, tmp_path):
    # A .fkeep is reconstructed on the fly (offline bicubic preview via conftest).
    comp = gui.compress_image(
        str(face_image), "aggressive", out_dir=str(tmp_path / "out"),
    )
    assert comp.output_path.endswith(".fkeep")
    out = gui.compare_images(str(face_image), comp.output_path)
    _assert_display_array(out.before_rgb)
    _assert_display_array(out.after_rgb)
    _assert_display_array(out.diff_rgb)
    assert out.before_rgb.shape == out.after_rgb.shape
    assert "SSIM" in out.summary


def test_compare_images_missing_input_raises(face_image):
    with pytest.raises(FaceKeepError):
        gui.compare_images("", "")  # nothing dropped
    with pytest.raises(FaceKeepError):
        gui.compare_images(str(face_image), "")  # no compressed file


# --- opt-in real AI restore (Compare tab) ---------------------------------

def _tiny_png(path):
    import cv2
    # A small non-flat image so the metrics are well-defined (no inf PSNR).
    cv2.imwrite(str(path), np.tile(np.arange(60, dtype=np.uint8), (40, 1))[..., None]
                .repeat(3, axis=2))
    return str(path)


def _fkeep_stub(tmp_path):
    """A dummy .fkeep on disk so compare_images can stat() it (load_after mocked)."""
    p = tmp_path / "x.fkeep"
    p.write_bytes(b"stub")
    return str(p)


def _spy_load_after(monkeypatch):
    """Patch compare.load_after to record the `preview` flag (no real restore)."""
    from facekeep import compare as compare_mod
    calls = {}

    def fake(path, agg, *, preview):
        calls["preview"] = preview
        kind = "bicubic preview" if preview else "restore"
        return np.zeros((40, 60, 3), np.uint8), kind

    monkeypatch.setattr(compare_mod, "load_after", fake)
    return calls


def test_compare_images_default_uses_preview(tmp_path, monkeypatch):
    calls = _spy_load_after(monkeypatch)
    gui.compare_images(_tiny_png(tmp_path / "o.png"), _fkeep_stub(tmp_path))
    assert calls["preview"] is True  # default: the instant bicubic preview


def test_compare_images_use_ai_requests_real_restore(tmp_path, monkeypatch):
    # [ai] available -> use_ai must take the genuine (preview=False) restore path.
    monkeypatch.setattr("facekeep.aggressive.restorer.realesrgan_available",
                        lambda: True)
    calls = _spy_load_after(monkeypatch)
    out = gui.compare_images(_tiny_png(tmp_path / "o.png"),
                             _fkeep_stub(tmp_path), use_ai=True)
    assert calls["preview"] is False
    assert "not installed" not in out.summary  # no fallback note when available


def test_compare_images_use_ai_falls_back_when_unavailable(tmp_path, monkeypatch):
    # [ai] absent -> honestly fall back to the fast preview and say so, rather
    # than run the same bicubic slowly and mislabel it as AI.
    monkeypatch.setattr("facekeep.aggressive.restorer.realesrgan_available",
                        lambda: False)
    calls = _spy_load_after(monkeypatch)
    out = gui.compare_images(_tiny_png(tmp_path / "o.png"),
                             _fkeep_stub(tmp_path), use_ai=True)
    assert calls["preview"] is True
    assert "not installed" in out.summary


def test_realesrgan_available_returns_bool():
    from facekeep.aggressive import restorer
    assert isinstance(restorer.realesrgan_available(), bool)


# --- config building ------------------------------------------------------

def test_build_config_explicit_quality_disables_autotune():
    cfg = gui._build_config("faithful", quality=80)
    assert cfg.faithful.quality == 80
    assert cfg.faithful.auto_tune is False


def test_build_config_autotune_checkbox_on():
    cfg = gui._build_config("faithful", auto_tune=True)
    assert cfg.faithful.auto_tune is True


def test_build_config_codec_and_bg_scale():
    assert gui._build_config("faithful", codec="jxl").faithful.codec == "jxl"
    assert gui._build_config("aggressive", bg_scale=0.4).aggressive.bg_scale == 0.4


def test_build_config_preset_implies_aggressive():
    cfg = gui._build_config("aggressive", preset="ratio")
    assert cfg.mode == "aggressive"
    assert cfg.aggressive.preset == "ratio"
    assert cfg.aggressive.bg_scale == 0.125  # the ratio preset's expansion


def test_build_config_preset_with_faithful_is_a_loud_error():
    with pytest.raises(ConfigError):
        gui._build_config("faithful", preset="ratio")


def test_build_config_explicit_bg_scale_beats_preset():
    # Mirrors CLI precedence: an explicit value still beats the preset's.
    cfg = gui._build_config("aggressive", preset="ratio", bg_scale=0.3)
    assert cfg.aggressive.bg_scale == 0.3


# --- CLI command ----------------------------------------------------------

def test_cli_gui_missing_gradio_hints_install(monkeypatch):
    def boom(**kwargs):
        raise ImportError("No module named 'gradio'")

    monkeypatch.setattr("facekeep.gui.launch", boom)
    result = CliRunner().invoke(cli, ["gui"])
    assert result.exit_code == 2
    assert "facekeep[gui]" in result.output


def test_launch_passes_only_stable_kwargs(monkeypatch):
    # Regression: launch() must pass only kwargs stable across gradio 4/5/6 to
    # demo.launch() — gradio 6 removed `show_api`, so hardcoding it crashed a
    # real launch (the patched-launch CLI tests can't see this). A fake demo
    # lets us assert the exact kwarg set without a server (or even gradio).
    captured = {}

    class _FakeDemo:
        def launch(self, **kwargs):
            captured.update(kwargs)
            return "launched"

    monkeypatch.setattr(gui, "build_demo", lambda: _FakeDemo())
    gui.launch(host="0.0.0.0", port=9000, share=True, inbrowser=True)
    assert captured == {
        "server_name": "0.0.0.0",
        "server_port": 9000,
        "share": True,
        "inbrowser": True,
    }


def test_cli_gui_launches_locally_without_sharing(monkeypatch):
    calls = {}

    def fake_launch(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("facekeep.gui.launch", fake_launch)
    result = CliRunner().invoke(cli, ["gui", "--port", "12345"])
    assert result.exit_code == 0
    assert calls["host"] == "127.0.0.1"  # local only by default
    assert calls["port"] == 12345
    assert calls["share"] is False  # no public tunnel by default


# --- real Blocks UI (skips without the [gui] extra) -----------------------

def test_build_demo_returns_blocks():
    gr = pytest.importorskip("gradio")
    demo = gui.build_demo()
    assert isinstance(demo, gr.Blocks)
