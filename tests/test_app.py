"""The desktop tray app (ROADMAP 11.3) — tray-free handler tests.

pystray is NOT required here (and deliberately not installed in the test
env's critical path): facekeep.app keeps it lazily imported behind
``run_app``/``_build_menu``, and everything else — the watch controller, the
notification policy, bundled-tool wiring, the startup registry, the state
handlers — is exercised directly, the same browser-free discipline as
tests/test_gui.py. The packaged build's tray/GUI surface is smoke-tested by
``facekeep app --selftest`` (packaging/windows/build.ps1 runs it on the
frozen exe), not here.
"""

import contextlib
import sys
import threading
import types
from pathlib import Path

import pytest

from facekeep import app as app_mod
from facekeep import gui
from facekeep.cli import _watch_cycle_line
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import FaceKeepError


@pytest.fixture(autouse=True)
def _isolated_gui_state(monkeypatch, tmp_path):
    """Keep every test away from the real ~/.cache/facekeep/gui_state.json."""
    monkeypatch.setattr(gui, "_GUI_STATE_PATH",
                        tmp_path / "gui_state" / "state.json")


def _cycle(**over):
    base = {"processed": 0, "ok": 0, "failed": 0, "skipped": 0, "saved": 0,
            "unchanged": 0, "awaiting": 0, "held": 0}
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The extracted watch-cycle line (the 11.1 output contract, unit level)
# ---------------------------------------------------------------------------

def test_watch_cycle_line_idle_and_processed():
    line = _watch_cycle_line(_cycle(unchanged=3, awaiting=1), interval=60.0)
    assert "idle" in line
    assert "3 unchanged" in line
    assert "1 not yet stable (still syncing?)" in line
    assert "next scan in 60s" in line

    line = _watch_cycle_line(
        _cycle(processed=2, ok=1, failed=1, saved=2048, held=2), interval=None)
    assert "processed 2: 1 ok, 1 failed" in line
    assert "saved 2.0 KB" in line
    assert "2 failed/skipped earlier (retried when the file changes)" in line
    assert "next scan" not in line  # --once announces no next scan


# ---------------------------------------------------------------------------
# Bundled-tool wiring (the frozen build's ffmpeg/avifenc plumbing)
# ---------------------------------------------------------------------------

def test_wire_bundled_tools_wires_present_tools_only(tmp_path):
    ff = tmp_path / "tools" / "ffmpeg" / "ffmpeg.exe"
    ff.parent.mkdir(parents=True)
    ff.write_bytes(b"x")
    env = {}
    wired = app_mod.wire_bundled_tools(tmp_path, env)
    assert env["FACEKEEP_FFMPEG"] == str(ff)
    assert wired == {"FACEKEEP_FFMPEG": str(ff)}
    assert "FACEKEEP_AVIFENC" not in env  # nothing bundled -> not wired


def test_wire_bundled_tools_never_overrides_a_user_setting(tmp_path):
    ff = tmp_path / "tools" / "ffmpeg" / "ffmpeg.exe"
    ff.parent.mkdir(parents=True)
    ff.write_bytes(b"x")
    env = {"FACEKEEP_FFMPEG": "user-set"}
    assert app_mod.wire_bundled_tools(tmp_path, env) == {}
    assert env["FACEKEEP_FFMPEG"] == "user-set"


def test_wire_bundled_tools_noop_when_not_frozen():
    env = {}
    assert app_mod.wire_bundled_tools(None, env) == {}
    assert env == {}


# ---------------------------------------------------------------------------
# Start with Windows (registry value; fake winreg, no real HKCU writes)
# ---------------------------------------------------------------------------

def _fake_winreg(store: dict):
    @contextlib.contextmanager
    def OpenKey(root, path, reserved=0, access=0):
        yield (root, path)

    def QueryValueEx(key, name):
        if name in store:
            return (store[name], 1)
        raise FileNotFoundError(name)

    def SetValueEx(key, name, reserved, type_, value):
        store[name] = value

    def DeleteValue(key, name):
        if name not in store:
            raise FileNotFoundError(name)
        del store[name]

    return types.SimpleNamespace(
        HKEY_CURRENT_USER=object(), KEY_SET_VALUE=0x20006, REG_SZ=1,
        OpenKey=OpenKey, QueryValueEx=QueryValueEx, SetValueEx=SetValueEx,
        DeleteValue=DeleteValue)


def test_start_with_windows_roundtrip(monkeypatch):
    store = {}
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(store))
    assert app_mod.get_start_with_windows() is False
    app_mod.set_start_with_windows(True)
    assert store["FaceKeep"] == app_mod.startup_command()
    assert app_mod.get_start_with_windows() is True
    app_mod.set_start_with_windows(False)
    assert app_mod.get_start_with_windows() is False
    app_mod.set_start_with_windows(False)  # idempotent, no error


def test_startup_command_frozen_exe(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Apps\FaceKeep\FaceKeep.exe")
    assert app_mod.startup_command() == r'"C:\Apps\FaceKeep\FaceKeep.exe"'


def test_startup_command_pip_install():
    assert not getattr(sys, "frozen", False)
    cmd = app_mod.startup_command()
    assert cmd.endswith('-m facekeep app')
    assert "python" in cmd.lower()


# ---------------------------------------------------------------------------
# Notification policy
# ---------------------------------------------------------------------------

def test_cycle_notification_is_silent_on_idle():
    assert app_mod.cycle_notification(_cycle(unchanged=42)) is None


def test_cycle_notification_reports_done_and_failed():
    title, msg = app_mod.cycle_notification(_cycle(processed=3, ok=3, saved=2048))
    assert "done" in title
    assert "3 file(s) compressed" in msg
    assert "2.0 KB" in msg

    title, msg = app_mod.cycle_notification(_cycle(processed=2, ok=1, failed=1))
    assert "problem" in title
    assert "1 file(s) failed" in msg
    assert "1 compressed" in msg


# ---------------------------------------------------------------------------
# WatchController (drives an injected cycles factory; real one is 11.1's)
# ---------------------------------------------------------------------------

def test_watch_controller_validates_folders(tmp_path):
    cfg = FaceKeepConfig()
    ctl = app_mod.WatchController(tmp_path / "missing", tmp_path / "out", cfg)
    with pytest.raises(FaceKeepError):
        ctl.start()
    inbox = tmp_path / "in"
    inbox.mkdir()
    ctl = app_mod.WatchController(inbox, inbox, cfg)
    with pytest.raises(FaceKeepError):
        ctl.start()


def test_watch_controller_runs_cycles_and_stops(tmp_path):
    inbox = tmp_path / "in"
    inbox.mkdir()
    archive = tmp_path / "out"
    first = threading.Event()
    seen = []

    def factory(in_p, out_p, config, **kw):
        assert Path(in_p) == inbox and Path(out_p) == archive
        assert kw["no_progress"] is True  # a tray app has no TTY
        yield _cycle(unchanged=1)
        yield _cycle(processed=1, ok=1)  # never consumed: stop() lands first

    def on_cycle(cycle):
        seen.append(cycle)
        first.set()

    ctl = app_mod.WatchController(
        inbox, archive, FaceKeepConfig(), interval=60.0,
        on_cycle=on_cycle, cycles_factory=factory)
    ctl.start()
    assert first.wait(5)
    ctl.stop(timeout=5)
    assert not ctl.running
    assert seen == [_cycle(unchanged=1)]
    assert ctl.last_cycle == _cycle(unchanged=1)
    assert archive.is_dir()  # created by start()


def test_watch_controller_surfaces_a_crash(tmp_path):
    inbox = tmp_path / "in"
    inbox.mkdir()
    errors = []
    done = threading.Event()

    def factory(*a, **kw):
        yield _cycle(unchanged=1)
        raise RuntimeError("boom")

    def on_error(e):
        errors.append(e)
        done.set()

    ctl = app_mod.WatchController(
        inbox, tmp_path / "out", FaceKeepConfig(), interval=0.0,
        on_error=on_error, cycles_factory=factory)
    ctl.start()
    assert done.wait(5)
    ctl.stop(timeout=5)
    assert isinstance(ctl.error, RuntimeError)
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# FaceKeepApp (menu handlers; controller stubbed)
# ---------------------------------------------------------------------------

class _StubController:
    instances: list = []

    def __init__(self, inbox, archive, config, **kw):
        self.inbox, self.archive, self.config, self.kw = inbox, archive, config, kw
        self.videos_note = None
        self.last_cycle = None
        self.error = None
        self._running = False
        _StubController.instances.append(self)

    def start(self):
        self._running = True

    def stop(self, timeout=None):
        self._running = False

    @property
    def running(self):
        return self._running


@pytest.fixture()
def stub_controller(monkeypatch):
    _StubController.instances = []
    monkeypatch.setattr(app_mod, "WatchController", _StubController)
    return _StubController


def test_app_defaults_share_the_gui_backup_folders(tmp_path):
    gui.save_gui_state(backup_source=str(tmp_path / "in"),
                       backup_archive=str(tmp_path / "out"))
    app = app_mod.FaceKeepApp()
    assert app.inbox == str(tmp_path / "in")
    assert app.archive == str(tmp_path / "out")
    assert app.lossless is False
    assert app.include_videos is True
    assert app.watch_enabled is False


def test_start_watch_needs_folders():
    notes = []
    app = app_mod.FaceKeepApp(notify=lambda t, m: notes.append((t, m)))
    assert app.start_watch() is False
    assert app.watching is False
    assert "folder" in notes[-1][1]


def test_start_watch_persists_and_says_the_honesty_note(tmp_path, stub_controller):
    notes = []
    app = app_mod.FaceKeepApp(notify=lambda t, m: notes.append((t, m)))
    app.set_inbox(str(tmp_path / "in"))
    app.set_archive(str(tmp_path / "out"))
    assert app.start_watch() is True
    assert app.watching is True
    # Guardrail 2: the backup-branded flow states what the copy is, and
    # guardrail 1 rides along.
    title, msg = notes[-1]
    assert "watching" in title
    assert "visually lossless, not bit-exact" in msg
    assert "never deleted or modified" in msg
    state = gui.load_gui_state()
    assert state["app_watch"] is True
    assert state["backup_source"] == str(tmp_path / "in")

    app.toggle_watch()
    assert app.watching is False
    assert gui.load_gui_state()["app_watch"] is False


def test_toggle_lossless_restarts_with_the_new_config(tmp_path, stub_controller):
    app = app_mod.FaceKeepApp()
    app.set_inbox(str(tmp_path / "in"))
    app.set_archive(str(tmp_path / "out"))
    assert app.start_watch() is True
    assert stub_controller.instances[-1].config.faithful.lossless is False

    app.toggle_lossless()
    assert gui.load_gui_state()["app_lossless"] is True
    assert app.watching is True  # restarted, not stopped
    assert stub_controller.instances[-1].config.faithful.lossless is True


def test_set_inbox_while_watching_restarts(tmp_path, stub_controller):
    app = app_mod.FaceKeepApp()
    app.set_inbox(str(tmp_path / "a"))
    app.set_archive(str(tmp_path / "out"))
    app.start_watch()
    n = len(stub_controller.instances)
    app.set_inbox(str(tmp_path / "b"))
    assert len(stub_controller.instances) == n + 1
    assert stub_controller.instances[-1].inbox == str(tmp_path / "b")
    assert gui.load_gui_state()["backup_source"] == str(tmp_path / "b")


def test_status_text_transitions(tmp_path, stub_controller):
    app = app_mod.FaceKeepApp()
    assert "Choose folders" in app.status_text()
    app.set_inbox(str(tmp_path / "in"))
    app.set_archive(str(tmp_path / "out"))
    assert app.status_text() == "Paused"
    app.start_watch()
    assert "first scan" in app.status_text()
    app.controller.last_cycle = _cycle(unchanged=2)
    assert app.status_text() == "Watching - idle, 2 unchanged"
    app.controller.last_cycle = _cycle(processed=2, ok=1, failed=1)
    assert app.status_text() == "Watching - last cycle: 1 ok, 1 failed"


# ---------------------------------------------------------------------------
# Graceful degradation without pystray
# ---------------------------------------------------------------------------

def test_run_app_raises_importerror_without_pystray(monkeypatch):
    monkeypatch.setitem(sys.modules, "pystray", None)  # import -> ImportError
    with pytest.raises(ImportError):
        app_mod.run_app([])


def test_cli_app_prints_the_install_hint(monkeypatch):
    from click.testing import CliRunner
    from facekeep import cli as cli_mod

    def _raise(argv=None):
        raise ImportError("No module named 'pystray'")

    monkeypatch.setattr(app_mod, "run_app", _raise)
    result = CliRunner().invoke(cli_mod.cli, ["app"])
    assert result.exit_code == 2
    assert "pip install facekeep[app]" in result.output
