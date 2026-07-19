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


# --- one-click backup (ROADMAP 11.2) --------------------------------------

@pytest.fixture(autouse=True)
def _isolated_gui_state(monkeypatch, tmp_path):
    """Keep every test away from the real ~/.cache/facekeep/gui_state.json."""
    monkeypatch.setattr(gui, "_GUI_STATE_PATH",
                        tmp_path / "gui_state" / "state.json")


def _photo(path, seed=0):
    """A small photo-like JPEG (smooth noise) that AVIF reliably beats on size."""
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.normal(128, 40, (24, 32, 3)).astype(np.float32),
                     (320, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(path), np.clip(img, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def _drain(gen):
    """Consume a run_backup generator -> (progress ticks, final BackupResult)."""
    items = list(gen)
    assert isinstance(items[-1], gui.BackupResult)
    progress = items[:-1]
    assert all(isinstance(p, gui.BackupProgress) for p in progress)
    return progress, items[-1]


def test_run_backup_folder(tmp_path):
    src = tmp_path / "inbox"
    src.mkdir()
    _photo(src / "a.jpg", seed=1)
    _photo(src / "b.jpg", seed=2)
    dst = tmp_path / "archive"

    progress, res = _drain(gui.run_backup(str(src), str(dst),
                                          report_dir=str(tmp_path / "rep")))
    # Live progress: one tick per file, in order, before each file runs.
    assert [p.done for p in progress] == [0, 1]
    assert all(p.total == 2 and p.kind == "photo" for p in progress)
    # The batch really ran: real outputs in the archive, counted as ok.
    assert res.files == 2 and res.ok == 2 and res.failed == 0
    outputs = list(dst.glob("a.*")) + list(dst.glob("b.*"))
    assert len(outputs) == 2
    # The per-file ledger is the --report machinery's rows (faces filled).
    assert len(res.rows) == 2
    assert all(r.status in ("written", "kept-original") for r in res.rows)
    assert all(r.faces is not None for r in res.rows)
    # The CSV artifact exists and carries the report header + both rows.
    csv_text = Path(res.report_path).read_text(encoding="utf-8")
    assert csv_text.startswith("file,mode,codec")
    assert "a.jpg" in csv_text and "b.jpg" in csv_text
    assert "Backup complete" in res.summary
    assert "not bit-exact" in res.summary  # the guardrail-2 honesty note


def test_run_backup_rerun_skips_unchanged(tmp_path):
    src = tmp_path / "inbox"
    src.mkdir()
    _photo(src / "a.jpg", seed=3)
    dst = tmp_path / "archive"
    _drain(gui.run_backup(str(src), str(dst)))
    _, res = _drain(gui.run_backup(str(src), str(dst)))
    # Second visit: the incremental index skips the unchanged file.
    assert res.unchanged == 1 and res.ok == 0 and res.failed == 0
    assert res.rows[0].status == "cached"


def test_run_backup_failed_file_is_counted_not_fatal(tmp_path):
    src = tmp_path / "inbox"
    src.mkdir()
    _photo(src / "good.jpg", seed=4)
    (src / "broken.jpg").write_bytes(b"not a jpeg at all")
    dst = tmp_path / "archive"
    _, res = _drain(gui.run_backup(str(src), str(dst)))
    assert res.ok == 1 and res.failed == 1
    assert sorted(r.status == "failed" for r in res.rows) == [False, True]


def test_run_backup_video_without_ffmpeg_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("facekeep.video.ffmpeg_available", lambda: False)
    src = tmp_path / "inbox"
    src.mkdir()
    _photo(src / "a.jpg", seed=5)
    (src / "clip.mp4").write_bytes(b"junk")
    dst = tmp_path / "archive"
    progress, res = _drain(gui.run_backup(str(src), str(dst)))
    # Photos first, then the videos — and the video skips with the hint.
    assert [p.kind for p in progress] == ["photo", "video"]
    assert res.files == 2 and res.ok == 1 and res.skipped == 1
    assert "ffmpeg" in res.summary


def test_run_backup_exclude_videos(tmp_path):
    src = tmp_path / "inbox"
    src.mkdir()
    _photo(src / "a.jpg", seed=6)
    (src / "clip.mp4").write_bytes(b"junk")
    _, res = _drain(gui.run_backup(str(src), str(tmp_path / "archive"),
                                   include_videos=False))
    assert res.files == 1  # the video was never gathered


def test_run_backup_refusals(tmp_path):
    d = tmp_path / "same"
    d.mkdir()
    with pytest.raises(FaceKeepError):  # archive == source
        list(gui.run_backup(str(d), str(d)))
    with pytest.raises(FaceKeepError):  # missing source
        list(gui.run_backup(str(tmp_path / "nope"), str(tmp_path / "a")))
    with pytest.raises(FaceKeepError):  # nothing to back up
        list(gui.run_backup(str(d), str(tmp_path / "a")))
    with pytest.raises(FaceKeepError):  # blank folders
        list(gui.run_backup("", ""))


def test_run_backup_persists_last_folders(tmp_path):
    src = tmp_path / "inbox"
    src.mkdir()
    _photo(src / "a.jpg", seed=7)
    dst = tmp_path / "archive"
    _drain(gui.run_backup(str(src), str(dst), lossless=False))
    state = gui.load_gui_state()
    assert state["backup_source"] == str(src)
    assert state["backup_archive"] == str(dst)
    assert state["backup_lossless"] is False


def test_gui_state_roundtrip_and_corruption():
    assert gui.load_gui_state() == {}  # missing file -> empty, no error
    gui.save_gui_state(backup_source="x")
    gui.save_gui_state(backup_lossless=True)  # merges, doesn't clobber
    state = gui.load_gui_state()
    assert state == {"backup_source": "x", "backup_lossless": True}
    gui._GUI_STATE_PATH.write_text("{corrupt", encoding="utf-8")
    assert gui.load_gui_state() == {}  # best-effort: never raises


def test_backup_config_is_faithful_with_lossless_toggle():
    cfg = gui._backup_config(False)
    assert cfg.mode == "faithful" and cfg.faithful.lossless is False
    assert gui._backup_config(True).faithful.lossless is True


def test_rows_to_table_blank_cells():
    from facekeep.report import ReportRow
    table = gui._rows_to_table([
        ReportRow(file="a.jpg", mode="faithful", status="written",
                  codec="avif", original_bytes=2048, output_bytes=1024,
                  ratio=2.0, quality=80, faces=1),
        ReportRow(file="bad.jpg", mode="faithful", status="failed"),
    ])
    assert table[0] == ["a.jpg", "written", "faithful", "avif",
                        "2.0 KB", "1.0 KB", "2.0x", "80", "1"]
    # None -> blank cells (never an invented 0), matching the report contract.
    assert table[1] == ["bad.jpg", "failed", "faithful", "", "", "", "", "", ""]


# --- real Blocks UI (skips without the [gui] extra) -----------------------

def test_build_demo_returns_blocks():
    gr = pytest.importorskip("gradio")
    demo = gui.build_demo()
    assert isinstance(demo, gr.Blocks)
