"""FaceKeep desktop tray app (ROADMAP 11.3) — Windows-first.

A system-tray shell over the existing machinery, so a non-Python user gets the
effortless-backup workflow: the tray icon wraps the 11.1 watch loop
(``cli._watch_cycles`` — the same engine, invariants and all; never a second
loop), the default menu entry opens the 11.2 GUI, "Start with Windows" is an
HKCU Run registry value, and done/failed notifications ride the tray icon.

Discipline (the gui.py precedent): ``pystray`` is imported **lazily** inside
the tray-facing functions, so this module and its handlers import without it —
``facekeep app`` prints an install hint (the ``[app]`` extra) instead of a
traceback, and the handlers (:class:`WatchController`,
:func:`cycle_notification`, :func:`wire_bundled_tools`, the startup-registry
helpers) are unit-tested tray-free. The packaged Windows build
(``packaging/windows/``) uses :func:`main` as its entry point; when frozen,
:func:`wire_bundled_tools` points the opt-in external-binary env vars
(``$FACEKEEP_FFMPEG`` / ``$FACEKEEP_AVIFENC``) at binaries bundled beside the
exe — never overriding an explicit user setting, and degrading to the normal
"binary not found" paths when nothing is bundled (offline-first holds).

Guardrails (ROADMAP Phase 11): sources are never deleted or modified; the
guardrail-2 honesty note is raised when watching starts, with a Lossless
toggle in the menu; the watch flow is faithful-only (``gui._backup_config`` —
the Backup tab's config), aggressive stays in the GUI where its trade-off is
explained; torch/[ai] is never bundled.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path

from .cli import (
    _decide_watch_videos,
    _fmt_size,
    _setup_logging,
    _watch_cycles,
    _watch_honesty_note,
)
from .exceptions import FaceKeepError
from .gui import _backup_config, load_gui_state, save_gui_state

logger = logging.getLogger("facekeep.app")

APP_NAME = "FaceKeep"
_DEFAULT_INTERVAL = 60.0

# Start-with-Windows: a per-user Run value (no admin, no shortcut plumbing).
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_RUN_VALUE = "FaceKeep"

# The windowed (no-console) build's stdio target — also the "see the log"
# pointer in notifications. One file per session (truncated at start).
LOG_PATH = Path.home() / ".cache" / "facekeep" / "app.log"

# Bundled external tools (frozen build only): env var -> (subdir, candidate
# binary names). ffprobe rides as ffmpeg's sibling (video._find_ffprobe), and
# avifdec/avifgainmaputil as avifenc's (the encoders pattern), so wiring these
# two vars enables both tool families.
_BUNDLED_TOOLS = {
    "FACEKEEP_FFMPEG": ("ffmpeg", ("ffmpeg.exe", "ffmpeg")),
    "FACEKEEP_AVIFENC": ("libavif", ("avifenc.exe", "avifenc")),
}


# ---------------------------------------------------------------------------
# Frozen-build plumbing (pure, tray-free)
# ---------------------------------------------------------------------------

def wire_bundled_tools(base_dir=None, environ=None) -> dict:
    """Point the external-binary env vars at tools bundled with a frozen build.

    ``base_dir`` defaults to the PyInstaller bundle dir (``sys._MEIPASS``) and
    the function is a no-op in a normal (non-frozen) install. An env var the
    user already set is **never** overridden — the explicit setting wins, the
    same precedence rule as everywhere else. Returns ``{var: path}`` for the
    vars actually wired.
    """
    env = os.environ if environ is None else environ
    if base_dir is None:
        if not getattr(sys, "frozen", False):
            return {}
        base_dir = getattr(sys, "_MEIPASS", Path(sys.executable).parent)
    wired = {}
    tools = Path(base_dir) / "tools"
    for var, (sub, names) in _BUNDLED_TOOLS.items():
        if env.get(var):
            continue
        for name in names:
            cand = tools / sub / name
            if cand.is_file():
                env[var] = str(cand)
                wired[var] = str(cand)
                break
    return wired


def _redirect_stdio():
    """Give a windowed (no-console) build real stdio.

    A PyInstaller windowed app has ``sys.stdout``/``stderr`` = None, which the
    batch machinery's click.echo/logging would crash on or silently lose.
    Redirect both to a per-session log the user can actually read
    (:data:`LOG_PATH`). No-op (returns None) when a console is attached.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return None
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        f = open(LOG_PATH, "w", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        f = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f
    return f


# ---------------------------------------------------------------------------
# Start with Windows (HKCU Run value; pure winreg, no new dependency)
# ---------------------------------------------------------------------------

def startup_command() -> str:
    """The command line to register for start-with-Windows.

    The frozen exe registers itself; a pip install registers
    ``pythonw -m facekeep app`` (pythonw so no console window flashes at
    login; plain python when pythonw isn't beside the interpreter).
    """
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    if pyw.is_file():
        exe = pyw
    return f'"{exe}" -m facekeep app'


def get_start_with_windows() -> bool:
    """True iff the FaceKeep Run value exists (False on non-Windows/any error)."""
    try:
        import winreg
    except ImportError:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            winreg.QueryValueEx(key, _RUN_VALUE)
        return True
    except OSError:
        return False


def set_start_with_windows(enabled: bool) -> None:
    """Create/delete the HKCU Run value. Raises ImportError off-Windows,
    OSError on registry failure — callers surface both as a notification."""
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0,
                        winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, _RUN_VALUE, 0, winreg.REG_SZ,
                              startup_command())
        else:
            try:
                winreg.DeleteValue(key, _RUN_VALUE)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Notifications (pure)
# ---------------------------------------------------------------------------

def cycle_notification(cycle: dict) -> "tuple[str, str] | None":
    """Map one watch-cycle summary to a ``(title, message)``, or None.

    Idle cycles never notify — a background app must not spam; only a cycle
    that actually compressed or failed files reports, failures leading.
    """
    if not cycle.get("processed"):
        return None
    if cycle.get("failed"):
        msg = f"{cycle['failed']} file(s) failed"
        if cycle.get("ok"):
            msg += f"; {cycle['ok']} compressed"
        msg += f" - see {LOG_PATH.name} for details."
        return (f"{APP_NAME}: backup problem", msg)
    msg = f"{cycle.get('ok', 0)} file(s) compressed"
    if cycle.get("saved"):
        msg += f", saved {_fmt_size(cycle['saved'])}"
    if cycle.get("skipped"):
        msg += f" ({cycle['skipped']} skipped)"
    return (f"{APP_NAME}: backup done", msg)


# ---------------------------------------------------------------------------
# The watch thread (tray-free; drives cli._watch_cycles)
# ---------------------------------------------------------------------------

class WatchController:
    """Runs the 11.1 watch-loop engine on a daemon thread.

    A thin pacing shell over :func:`cli._watch_cycles` — the loop invariants
    (metadata-only idle cycles, stability guard, failure memo) all live in the
    engine; this class only owns the thread, the stop event, and the per-cycle
    callbacks. ``stop()`` interrupts the between-cycle wait immediately; a
    cycle already inside a long encode finishes it first (the thread is a
    daemon, so quitting the app never hangs on it).
    """

    def __init__(self, inbox, archive, config, *, interval: float = _DEFAULT_INTERVAL,
                 no_videos: bool = False, settle: float = 2.0, jobs: int = 1,
                 on_cycle=None, on_error=None, cycles_factory=None):
        self.inbox = Path(inbox)
        self.archive = Path(archive)
        self.config = config
        self.interval = interval
        self.settle = settle
        self.jobs = jobs
        self.last_cycle: "dict | None" = None
        self.error: "BaseException | None" = None
        self.videos_note: "str | None" = None
        self._no_videos = no_videos
        self._include_videos = False
        self._on_cycle = on_cycle
        self._on_error = on_error
        self._cycles_factory = cycles_factory or _watch_cycles
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None

    def start(self) -> None:
        """Validate and start watching. Raises FaceKeepError on a bad setup."""
        if self.running:
            return
        if not self.inbox.is_dir():
            raise FaceKeepError(f"Inbox folder does not exist: {self.inbox}")
        try:
            same = self.inbox.resolve() == self.archive.resolve()
        except OSError:
            same = False
        if same:
            raise FaceKeepError(
                "The archive folder must differ from the inbox (outputs "
                "would land beside the sources being watched)."
            )
        self.archive.mkdir(parents=True, exist_ok=True)
        self._include_videos, self.videos_note = _decide_watch_videos(
            self.config, self._no_videos)
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="facekeep-watch",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: "float | None" = 5.0) -> None:
        """Request a stop and wait up to ``timeout`` for the thread to end.

        A thread mid-encode can outlive the join — it stops after the current
        cycle; being a daemon it never blocks process exit.
        """
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            for cycle in self._cycles_factory(
                    self.inbox, self.archive, self.config,
                    include_videos=self._include_videos,
                    settle=self.settle, jobs=self.jobs, no_progress=True):
                self.last_cycle = cycle
                logger.info("watch cycle: %s", cycle)
                if self._on_cycle is not None:
                    try:
                        self._on_cycle(cycle)
                    except Exception:  # a UI hiccup must never kill the loop
                        logger.exception("on_cycle callback failed")
                if self._stop.wait(self.interval):
                    break
        except Exception as e:  # surfaced, never silent — the loop is the app
            self.error = e
            logger.exception("watch loop stopped on an error")
            if self._on_error is not None:
                try:
                    self._on_error(e)
                except Exception:
                    logger.exception("on_error callback failed")


# ---------------------------------------------------------------------------
# App state + menu handlers (tray-free; notify/refresh are injected hooks)
# ---------------------------------------------------------------------------

class FaceKeepApp:
    """Tray-app state and menu handlers, testable without pystray.

    Folders are shared with the GUI Backup tab (the same ``backup_source`` /
    ``backup_archive`` keys in ``gui_state.json``) — configure once, either
    place. The app-only toggles persist under ``app_*`` keys. ``notify`` and
    ``refresh_menu`` are injected by :func:`run_app` (tray) or a test.
    """

    def __init__(self, *, notify=None, refresh_menu=None):
        state = load_gui_state()
        self.inbox: str = state.get("backup_source") or ""
        self.archive: str = state.get("backup_archive") or ""
        self.lossless: bool = bool(state.get("app_lossless", False))
        self.include_videos: bool = bool(state.get("app_include_videos", True))
        self.watch_enabled: bool = bool(state.get("app_watch", False))
        try:
            self.interval: float = float(state.get("app_interval",
                                                   _DEFAULT_INTERVAL))
        except (TypeError, ValueError):
            self.interval = _DEFAULT_INTERVAL
        self.controller: "WatchController | None" = None
        self.notify = notify or (lambda title, message: None)
        self.refresh_menu = refresh_menu or (lambda: None)
        self._gui_url: "str | None" = None
        self._gui_lock = threading.Lock()

    # -- status ------------------------------------------------------------

    @property
    def watching(self) -> bool:
        return (self.watch_enabled and self.controller is not None
                and self.controller.running)

    def status_text(self) -> str:
        """One menu line saying what the app is doing right now."""
        if self.watching:
            c = self.controller.last_cycle
            if c is None:
                return "Watching (first scan...)"
            if c["processed"]:
                s = f"Watching - last cycle: {c['ok']} ok"
                if c["failed"]:
                    s += f", {c['failed']} failed"
                return s
            s = "Watching - idle"
            if c["unchanged"]:
                s += f", {c['unchanged']} unchanged"
            return s
        if self.controller is not None and self.controller.error is not None:
            return f"Watch stopped on an error - see {LOG_PATH.name}"
        if not (self.inbox and self.archive):
            return "Choose folders below to start"
        return "Paused"

    # -- watch lifecycle ---------------------------------------------------

    def start_watch(self) -> bool:
        if self.controller is not None and self.controller.running:
            if self.watch_enabled:
                return True  # already watching
            # a stop() joined out on a long encode; don't start a second
            # loop over the same folders while the old cycle finishes.
            self.notify(APP_NAME, "Still finishing the previous cycle - "
                                  "try again in a moment.")
            return False
        if not (self.inbox and self.archive):
            self.notify(APP_NAME, "Choose an inbox and an archive folder first.")
            return False
        config = _backup_config(self.lossless)
        controller = WatchController(
            self.inbox, self.archive, config,
            interval=self.interval, no_videos=not self.include_videos,
            on_cycle=self._on_cycle, on_error=self._on_error,
        )
        try:
            controller.start()
        except FaceKeepError as e:
            self.notify(APP_NAME, str(e))
            return False
        self.controller = controller
        self.watch_enabled = True
        save_gui_state(app_watch=True)
        # Guardrail 2: a backup-branded flow says what the copy is, once, at
        # start — plus the videos-excluded reason when there is one.
        note = _watch_honesty_note(config)
        parts = [p for p in (controller.videos_note, note) if p]
        parts.append("Sources are never deleted or modified.")
        self.notify(f"{APP_NAME}: watching", " ".join(parts))
        self.refresh_menu()
        return True

    def stop_watch(self) -> None:
        self.watch_enabled = False
        save_gui_state(app_watch=False)
        if self.controller is not None:
            self.controller.stop()
        self.refresh_menu()

    def toggle_watch(self, *_args) -> None:
        if self.watch_enabled:
            self.stop_watch()
        else:
            self.start_watch()

    def _restart_if_watching(self) -> None:
        if self.watching:
            self.stop_watch()
            self.start_watch()

    def _on_cycle(self, cycle: dict) -> None:
        note = cycle_notification(cycle)
        if note is not None:
            self.notify(*note)
        self.refresh_menu()

    def _on_error(self, exc: BaseException) -> None:
        self.notify(f"{APP_NAME}: watch stopped", str(exc))
        self.refresh_menu()

    # -- folders + toggles -------------------------------------------------

    def set_inbox(self, path: str) -> None:
        self.inbox = path
        save_gui_state(backup_source=path)
        self._restart_if_watching()
        self.refresh_menu()

    def set_archive(self, path: str) -> None:
        self.archive = path
        save_gui_state(backup_archive=path)
        self._restart_if_watching()
        self.refresh_menu()

    def choose_inbox(self, *_args) -> None:
        p = _pick_folder("Choose the inbox folder FaceKeep watches")
        if p:
            self.set_inbox(p)

    def choose_archive(self, *_args) -> None:
        p = _pick_folder("Choose the archive folder compressed copies land in")
        if p:
            self.set_archive(p)

    def toggle_lossless(self, *_args) -> None:
        self.lossless = not self.lossless
        save_gui_state(app_lossless=self.lossless)
        self._restart_if_watching()

    def toggle_videos(self, *_args) -> None:
        self.include_videos = not self.include_videos
        save_gui_state(app_include_videos=self.include_videos)
        self._restart_if_watching()

    def toggle_startup(self, *_args) -> None:
        try:
            set_start_with_windows(not get_start_with_windows())
        except (ImportError, OSError) as e:
            self.notify(APP_NAME, f"Could not update start-with-Windows: {e}")

    # -- GUI ---------------------------------------------------------------

    def open_gui(self, *_args) -> None:
        """Open the 11.2 GUI (menu default action). Serves it on first use."""
        threading.Thread(target=self._open_gui, name="facekeep-gui",
                         daemon=True).start()

    def _open_gui(self) -> None:
        with self._gui_lock:
            if self._gui_url:
                webbrowser.open(self._gui_url)
                return
            try:
                from . import gui as gui_mod
                # prevent_thread_lock: serve in the background; inbrowser
                # opens the user's browser once the server is up.
                result = gui_mod.launch(inbrowser=True, prevent_thread_lock=True)
            except ImportError:
                self.notify(APP_NAME,
                            "The GUI needs Gradio - pip install facekeep[app]")
                return
            except Exception as e:
                logger.exception("GUI failed to start")
                self.notify(APP_NAME, f"GUI failed to start: {e}")
                return
            url = None
            if isinstance(result, tuple) and len(result) >= 2:
                url = result[1]
            self._gui_url = url or "http://127.0.0.1:7860"

    # -- quit --------------------------------------------------------------

    def quit_app(self, icon=None, *_args) -> None:
        if self.controller is not None:
            self.controller.stop(timeout=2.0)
        if icon is not None:
            icon.stop()


def _pick_folder(title: str) -> "str | None":
    """Native folder picker via tkinter (ships with CPython on Windows).

    Returns None when cancelled or when tkinter is unavailable — the caller
    just leaves the setting unchanged (folders can also be set from the GUI
    Backup tab, which shares the same persisted keys).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title, mustexist=True)
    finally:
        root.destroy()
    return path or None


# ---------------------------------------------------------------------------
# Tray layer (pystray imported lazily from here down)
# ---------------------------------------------------------------------------

def _tray_image(size: int = 64):
    """Draw the tray icon with PIL (no bundled asset to lose)."""
    from PIL import Image, ImageDraw

    green = (16, 122, 87, 255)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = size // 16
    d.rounded_rectangle([m, m, size - m, size - m], radius=size // 5, fill=green)
    cx, cy = size // 2, size // 2
    r = size * 9 // 32
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 255))
    er = max(2, size // 16)
    for ex in (cx - r // 2, cx + r // 2):
        d.ellipse([ex - er, cy - r // 3 - er, ex + er, cy - r // 3 + er],
                  fill=green)
    d.arc([cx - r // 2, cy - r // 4, cx + r // 2, cy + r * 2 // 3],
          20, 160, fill=green, width=max(2, size // 20))
    return img


def _build_menu(app: FaceKeepApp):
    import pystray

    item = pystray.MenuItem
    return pystray.Menu(
        item("Open FaceKeep GUI", lambda icon, it: app.open_gui(), default=True),
        pystray.Menu.SEPARATOR,
        item(lambda it: app.status_text(), None, enabled=False),
        item("Watch inbox -> archive", lambda icon, it: app.toggle_watch(),
             checked=lambda it: app.watching),
        item("Choose inbox folder...", lambda icon, it: app.choose_inbox()),
        item("Choose archive folder...", lambda icon, it: app.choose_archive()),
        pystray.Menu.SEPARATOR,
        item("Lossless originals (bit-exact, larger)",
             lambda icon, it: app.toggle_lossless(),
             checked=lambda it: app.lossless),
        item("Include videos", lambda icon, it: app.toggle_videos(),
             checked=lambda it: app.include_videos),
        item("Start with Windows", lambda icon, it: app.toggle_startup(),
             checked=lambda it: get_start_with_windows()),
        pystray.Menu.SEPARATOR,
        item("Quit", lambda icon, it: app.quit_app(icon)),
    )


def _selftest() -> int:
    """Headless packaging smoke test (``--selftest``): exercise every bundled
    surface without showing a tray icon, print a report, exit 0/1.

    This is what makes the frozen windowed build verifiable from a script:
    building the Gradio Blocks catches missing bundled assets, building the
    tray menu catches pystray/backend problems, and the tool report shows
    whether the bundled ffmpeg/avifenc were wired.
    """
    failures = []

    def check(name, fn, *, required=True):
        try:
            detail = fn()
            print(f"selftest ok: {name}" + (f" ({detail})" if detail else ""))
        except Exception as e:
            tag = "FAIL" if required else "missing (optional)"
            print(f"selftest {tag}: {name}: {e!r}")
            if required:
                failures.append(name)

    def _core():
        from . import encoders, faithful  # noqa: F401 — import = the check
        codecs = [c for c in ("avif", "jxl", "webp")
                  if encoders.codec_available(c)]
        if not codecs:
            raise RuntimeError("no faithful codec available")
        return "codecs: " + ",".join(codecs)

    def _tray():
        import pystray  # noqa: F401
        _tray_image()
        _build_menu(FaceKeepApp())
        try:
            from importlib.metadata import version
            return f"pystray {version('pystray')}"
        except Exception:
            return "pystray"

    def _gui():
        from . import gui as gui_mod
        gui_mod.build_demo()  # constructs the Blocks; catches missing assets
        import gradio
        return f"gradio {gradio.__version__}"

    def _video():
        from . import video as video_mod
        ff = video_mod.find_ffmpeg()
        if not ff:
            raise RuntimeError("ffmpeg not found (videos will be skipped)")
        return ff

    def _heic():
        import pillow_heif  # noqa: F401
        return None

    check("core pipeline", _core)
    check("tray (pystray)", _tray)
    check("gui (gradio)", _gui)
    check("video (ffmpeg)", _video, required=False)
    check("heic input", _heic, required=False)
    print(f"selftest startup command: {startup_command()}")
    print("selftest " + ("PASS" if not failures else
                         f"FAIL: {', '.join(failures)}"))
    return 0 if not failures else 1


def run_app(argv: "list[str] | None" = None) -> int:
    """Run the tray app (or the ``--selftest`` smoke check). Returns an exit
    code; raises ImportError when pystray (the ``[app]`` extra) is missing so
    the CLI can print the install hint."""
    args = list(sys.argv[1:] if argv is None else argv)
    _redirect_stdio()
    _setup_logging("-v" in args or "--verbose" in args)
    wired = wire_bundled_tools()
    for var, path in wired.items():
        logger.info("bundled tool wired: %s=%s", var, path)
    if "--selftest" in args:
        return _selftest()

    import pystray

    app = FaceKeepApp()
    icon = pystray.Icon(APP_NAME.lower(), _tray_image(), APP_NAME,
                        menu=_build_menu(app))

    def _notify(title: str, message: str) -> None:
        try:
            icon.notify(message, title)
        except Exception:  # a backend without notifications must not crash us
            logger.info("notification: %s - %s", title, message)

    def _refresh() -> None:
        try:
            icon.update_menu()
        except Exception:
            pass

    app.notify = _notify
    app.refresh_menu = _refresh

    def _on_ready(icon_):
        icon_.visible = True
        # Resume watching if it was on when the app last quit (persisted).
        if app.watch_enabled and app.inbox and app.archive:
            app.watch_enabled = False  # start_watch flips it back on
            app.start_watch()

    icon.run(setup=_on_ready)
    return 0


def main() -> None:
    """Entry point for the packaged exe (and ``python -m facekeep app``)."""
    try:
        rc = run_app()
    except ImportError as e:
        print(
            f"The FaceKeep tray app needs its optional dependencies ({e}).\n"
            "  Install them with:  pip install facekeep[app]",
            file=sys.stderr,
        )
        rc = 2
    sys.exit(rc)
