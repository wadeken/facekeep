"""Folder progress bar (`tqdm`) — ROADMAP Phase 3.

The progress bar is a **pure UX layer** over the batch loop: it conveys "X of N
done" while workers run (the gap parallel mode left, since per-file lines print
only after every worker finishes), but it must not change a single byte of the
results. These tests pin exactly that, plus the show/hide gating, rather than
asserting the bar's visual characters (which `tqdm` only draws to a real TTY —
`CliRunner`/pytest are non-TTY, so the bar is suppressed by design there):

1. **Gating.** A bar is shown only for a multi-file run on a TTY; never for a
   single file, never when piped/non-TTY, never with `--no-progress`.
2. **Graceful degradation.** If `tqdm` isn't importable, `_maybe_progress`
   passes the iterable through untouched and the compress still completes.
3. **Wrapping when enabled.** With `tqdm` present and `enabled=True`, the helper
   returns a wrapper that still yields every element (count is correct);
   `enabled=False` returns the original iterable object itself.
4. **No side effects.** Forcing the bar on (monkeypatched `isatty`) produces
   byte-identical `.avif` outputs and an identical `--report` CSV to a run with
   the bar off — the same guarantee the parallel tests pin for `--jobs`.

The codec runs for real, so these share the suite's `requires_avif` guard.
"""

import builtins

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

import facekeep.cli as cli_mod
from facekeep import encoders
from facekeep.cli import _maybe_progress, cli

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


def _make_photo(path, seed: int):
    """Write one compressible synthetic JPEG with a Haar-detectable face."""
    rng = np.random.default_rng(seed)
    H, W = 600, 800
    bg = cv2.resize(
        rng.normal(128, 25, (H // 10, W // 10, 3)).astype(np.float32),
        (W, H), interpolation=cv2.INTER_CUBIC,
    )
    img = np.clip(bg, 0, 255).astype(np.uint8)
    cx, cy, fw = 400, 300, 200
    fh = int(fw * 1.3)
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, (180, 170, 165), -1)
    ew = fw // 7
    cv2.ellipse(img, (cx - fw // 5, cy - fh // 10), (ew, ew // 2), 0, 0, 360,
                (60, 55, 55), -1)
    cv2.ellipse(img, (cx + fw // 5, cy - fh // 10), (ew, ew // 2), 0, 0, 360,
                (60, 55, 55), -1)
    cv2.ellipse(img, (cx, cy + fh // 4), (fw // 5, fh // 18), 0, 0, 180,
                (120, 90, 90), -1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


@pytest.fixture
def photo_dir(tmp_path):
    """A directory of several distinct compressible JPEGs."""
    d = tmp_path / "photos"
    d.mkdir()
    for i in range(4):
        _make_photo(d / f"p{i}.jpg", seed=10 + i)
    return d


def _file_bytes(folder, pattern):
    """Map {name: bytes} for files in `folder` matching glob `pattern`, sorted."""
    return {p.name: p.read_bytes() for p in sorted(folder.glob(pattern))}


# --------------------------------------------------------------------------- #
# _maybe_progress unit behaviour
# --------------------------------------------------------------------------- #

def test_maybe_progress_disabled_returns_same_object():
    """enabled=False must return the *identical* iterable (no wrapping at all)."""
    items = [1, 2, 3]
    assert _maybe_progress(items, total=3, enabled=False) is items


def test_maybe_progress_wraps_and_yields_all_when_enabled():
    """enabled=True wraps in tqdm but still yields every element in order."""
    items = [1, 2, 3, 4]
    wrapped = _maybe_progress(iter(items), total=4, enabled=True)
    assert wrapped is not items
    assert list(wrapped) == items


def test_maybe_progress_passthrough_without_tqdm(monkeypatch):
    """If tqdm can't be imported, the helper degrades to the bare iterable."""
    real_import = builtins.__import__

    def _no_tqdm(name, *args, **kwargs):
        if name == "tqdm" or name.startswith("tqdm."):
            raise ImportError("tqdm intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tqdm)

    items = [1, 2, 3]
    # enabled=True, but tqdm import fails -> fall back to the same iterable.
    assert _maybe_progress(items, total=3, enabled=True) is items


# --------------------------------------------------------------------------- #
# Gating: when is the bar shown?
# --------------------------------------------------------------------------- #

@requires_avif
def test_no_progress_flag_accepted_and_harmless(photo_dir, tmp_path):
    """`--no-progress` is accepted and doesn't change the result set."""
    runner = CliRunner()
    result = runner.invoke(cli, ["compress", str(photo_dir),
                                 "-o", str(tmp_path / "out"), "--no-progress"])
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "out").glob("*.avif"))) == 4
    assert "4/4 ok" in result.output


@requires_avif
def test_no_bar_in_non_tty_by_default(photo_dir, tmp_path):
    """Under CliRunner (non-TTY) the default run draws no bar chars on stdout."""
    runner = CliRunner()
    result = runner.invoke(cli, ["compress", str(photo_dir),
                                 "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    # tqdm renders a filled bar / percentage; none of it should appear.
    assert "█" not in result.output
    assert "it/s" not in result.output
    assert "img/s" not in result.output
    # The real result lines and summary are still there.
    assert "4/4 ok" in result.output


def test_single_file_never_shows_bar(face_image, tmp_path, monkeypatch):
    """Even on a (faked) TTY, a single-file run must not wrap in a bar.

    Asserted by making `_maybe_progress` explode if it is ever called with
    enabled=True — a single file must keep `show_progress` False.
    """
    monkeypatch.setattr(cli_mod, "_stderr_isatty", lambda: True)

    real = cli_mod._maybe_progress

    def _guard(iterable, total, enabled):
        assert not enabled, "single-file run must not enable the progress bar"
        return real(iterable, total, enabled)

    monkeypatch.setattr(cli_mod, "_maybe_progress", _guard)

    runner = CliRunner()
    result = runner.invoke(cli, ["compress", str(face_image),
                                 "-o", str(tmp_path / "one")])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("*.avif")), "the single file should still be written"


@requires_avif
def test_multifile_tty_enables_bar(photo_dir, tmp_path, monkeypatch):
    """A multi-file run on a TTY enables the bar (show_progress True path).

    We don't assert the drawn characters (tqdm's output under capture is
    environment-dependent); we assert the gate fires by capturing the `enabled`
    value `_maybe_progress` is called with.
    """
    monkeypatch.setattr(cli_mod, "_stderr_isatty", lambda: True)

    seen = []
    real = cli_mod._maybe_progress

    def _spy(iterable, total, enabled):
        seen.append(enabled)
        return real(iterable, total, enabled)

    monkeypatch.setattr(cli_mod, "_maybe_progress", _spy)

    runner = CliRunner()
    result = runner.invoke(cli, ["compress", str(photo_dir),
                                 "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert seen and seen[-1] is True, "multi-file TTY run should enable the bar"


@requires_avif
def test_no_progress_overrides_tty(photo_dir, tmp_path, monkeypatch):
    """`--no-progress` keeps the bar off even on a TTY."""
    monkeypatch.setattr(cli_mod, "_stderr_isatty", lambda: True)

    seen = []
    real = cli_mod._maybe_progress

    def _spy(iterable, total, enabled):
        seen.append(enabled)
        return real(iterable, total, enabled)

    monkeypatch.setattr(cli_mod, "_maybe_progress", _spy)

    runner = CliRunner()
    result = runner.invoke(cli, ["compress", str(photo_dir),
                                 "-o", str(tmp_path / "out"), "--no-progress"])
    assert result.exit_code == 0, result.output
    assert seen and all(e is False for e in seen), "--no-progress must disable the bar"


# --------------------------------------------------------------------------- #
# No side effects: bar on vs off produce identical artifacts
# --------------------------------------------------------------------------- #

@requires_avif
def test_progress_does_not_change_outputs(photo_dir, tmp_path, monkeypatch):
    """Bar-on vs bar-off: byte-identical .avif outputs and identical CSV."""
    runner = CliRunner()

    # Bar OFF (default non-TTY).
    off_out = tmp_path / "off"
    off_csv = tmp_path / "off.csv"
    r_off = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(off_out),
                                "--report", str(off_csv)])
    assert r_off.exit_code == 0, r_off.output

    # Bar ON (force a TTY so show_progress is True and tqdm actually wraps).
    monkeypatch.setattr(cli_mod, "_stderr_isatty", lambda: True)
    on_out = tmp_path / "on"
    on_csv = tmp_path / "on.csv"
    r_on = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(on_out),
                               "--report", str(on_csv)])
    assert r_on.exit_code == 0, r_on.output

    off = _file_bytes(off_out, "*.avif")
    on = _file_bytes(on_out, "*.avif")
    assert len(off) == 4
    assert off.keys() == on.keys()
    for name in off:
        assert off[name] == on[name], f"{name} differs bar-on vs bar-off"

    # The report CSV is identical regardless of the bar.
    assert off_csv.read_text(encoding="utf-8") == on_csv.read_text(encoding="utf-8")
