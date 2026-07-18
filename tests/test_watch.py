"""`facekeep watch` — the automation core (ROADMAP 11.1).

Pins the contracts the watch loop adds around the shared batch machinery:

1. **Stat pre-filter in the index**: rows carry the input's size+mtime_ns
   (captured at hash time); ``is_unchanged_stat`` answers "still unchanged?"
   from metadata alone — no hash read — and is deliberately watch-only.
   The schema migration is additive (old DBs keep their rows; a wiped video
   row would cost a minutes-to-hours re-encode).
2. **File-stability guard**: a file is processed only after its size+mtime
   hold still across two consecutive scans, so a mid-sync file is never
   half-read.
3. **--once**: a single pass for external schedulers; exit 1 iff a file failed.
4. **Loop mode**: clean Ctrl-C stop; a failed file is not retried every cycle
   (only when its stat changes).
5. **Guardrail 1**: sources are never deleted or modified.
6. **Live-Photo pair policy** (guardrail 3, measured): a .mov with a same-stem
   photo sibling AND Apple's pairing key is kept verbatim (kept-original);
   a same-stem coincidence without the key encodes normally; the policy knob
   (video.preserve_live_photos) opts out and busts the video fingerprint.

Real-video tests need the external ffmpeg binaries and skip otherwise (the
test_video_cli.py precedent, including the .tools bootstrap).
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import cli as cli_mod, encoders, index as index_mod, video
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig

# Test-side bootstrap only (same as test_video.py): this dev box keeps ffmpeg
# machine-local at .tools/ffmpeg rather than on PATH.
_TOOLS_FFMPEG = (
    Path(__file__).resolve().parents[1]
    / ".tools" / "ffmpeg" / "bin"
    / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
)
if not video.ffmpeg_available() and _TOOLS_FFMPEG.is_file():
    os.environ["FACEKEEP_FFMPEG"] = str(_TOOLS_FFMPEG)

needs_ffmpeg = pytest.mark.skipif(
    not video.ffmpeg_available(),
    reason="ffmpeg/ffprobe not found (set FACEKEEP_FFMPEG, put ffmpeg on PATH, "
    "or place a build at .tools/ffmpeg/bin) — Live-Photo pair tests need them",
)

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


def _small_jpg(path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)
    return path


def _watch_once(inbox: Path, archive: Path, *extra_args) -> "CliRunner.Result":
    return CliRunner().invoke(cli, [
        "watch", str(inbox), "-o", str(archive),
        "--once", "--settle", "0.05", "--no-videos", *extra_args,
    ])


# --------------------------------------------------------------------------- #
# Index: stat columns + the metadata-only quick check
# --------------------------------------------------------------------------- #

def _record(idx, src: Path, out: Path, *, fp="fp1", with_stat=True):
    st = src.stat()
    h = index_mod.hash_file(src)
    idx.record(src, index_mod.IndexRow(
        content_hash=h, settings_fingerprint=fp, mode="faithful",
        codec="avif", quality=70, original_size=st.st_size,
        output_path=str(out), output_size=out.stat().st_size,
        input_size=st.st_size if with_stat else None,
        input_mtime_ns=st.st_mtime_ns if with_stat else None,
    ))
    return st


def test_is_unchanged_stat_hits_and_misses(tmp_path):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"pixels")
    out = tmp_path / "a.avif"
    out.write_bytes(b"encoded")
    db = tmp_path / "idx.sqlite"

    with index_mod.ProcessIndex(db) as idx:
        st = _record(idx, src, out)
        size, mtime = st.st_size, st.st_mtime_ns
        assert idx.is_unchanged_stat(src, size, mtime, "fp1") is not None
        # Any single mismatch is a miss — no hash is ever read here.
        assert idx.is_unchanged_stat(src, size + 1, mtime, "fp1") is None
        assert idx.is_unchanged_stat(src, size, mtime + 1, "fp1") is None
        assert idx.is_unchanged_stat(src, size, mtime, "fp2") is None
        out.unlink()
        assert idx.is_unchanged_stat(src, size, mtime, "fp1") is None


def test_is_unchanged_stat_misses_on_stat_less_rows(tmp_path):
    """A row without stat columns (pre-11.1) never stat-hits — hash fallback."""
    src = tmp_path / "a.jpg"
    src.write_bytes(b"pixels")
    out = tmp_path / "a.avif"
    out.write_bytes(b"encoded")
    db = tmp_path / "idx.sqlite"

    with index_mod.ProcessIndex(db) as idx:
        st = _record(idx, src, out, with_stat=False)
        assert idx.is_unchanged_stat(
            src, st.st_size, st.st_mtime_ns, "fp1") is None
        # The honest full-hash check still works on the same row.
        assert idx.is_unchanged(src, index_mod.hash_file(src), "fp1") is not None
        # update_stat backfills; the quick check then hits.
        idx.update_stat(src, st.st_size, st.st_mtime_ns)
        assert idx.is_unchanged_stat(
            src, st.st_size, st.st_mtime_ns, "fp1") is not None


def test_schema_migration_is_additive_never_wipes(tmp_path):
    """Opening a pre-11.1 DB adds the stat columns IN PLACE, keeping rows.

    A version-bump wipe would silently discard cached video outcomes that cost
    minutes-to-hours each — the exact work the index exists to skip.
    """
    src = tmp_path / "a.jpg"
    src.write_bytes(b"pixels")
    out = tmp_path / "a.avif"
    out.write_bytes(b"encoded")
    db = tmp_path / "idx.sqlite"
    h = index_mod.hash_file(src)

    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE processed (
            abs_path             TEXT PRIMARY KEY,
            content_hash         TEXT NOT NULL,
            settings_fingerprint TEXT NOT NULL,
            mode                 TEXT NOT NULL,
            codec                TEXT,
            quality              INTEGER,
            original_size        INTEGER NOT NULL,
            output_path          TEXT NOT NULL,
            output_size          INTEGER NOT NULL,
            updated_at           TEXT NOT NULL
        )
        """
    )
    conn.execute(f"PRAGMA user_version = {index_mod.SCHEMA_VERSION}")
    conn.execute(
        "INSERT INTO processed VALUES (?,?,?,?,?,?,?,?,?,?)",
        (str(src.resolve()), h, "fp1", "video", "av1", 32,
         6, str(out), 7, "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    with index_mod.ProcessIndex(db) as idx:
        row = idx.lookup(src)
        assert row is not None                      # the old row survived
        assert row.content_hash == h
        assert row.input_size is None               # migrated columns are NULL
        assert idx.is_unchanged(src, h, "fp1") is not None
        assert idx.is_unchanged_stat(src, 6, 123, "fp1") is None


@requires_avif
def test_compress_records_stat_and_refreshes_on_touch(tmp_path):
    """A compress run records stat; a touched-identical file refreshes it."""
    inbox = tmp_path / "in"
    inbox.mkdir()
    src = _small_jpg(inbox / "a.jpg")
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # a pre-existing dir, so -o keeps directory semantics

    runner = CliRunner()
    r1 = runner.invoke(cli, ["compress", str(inbox), "-o", str(out_dir)])
    assert r1.exit_code == 0, r1.output
    db = out_dir / index_mod.INDEX_FILENAME
    with index_mod.ProcessIndex(db) as idx:
        row = idx.lookup(src)
        st = src.stat()
        assert (row.input_size, row.input_mtime_ns) == (st.st_size, st.st_mtime_ns)

    # Touch: same bytes, new mtime — the re-run hash-hits and refreshes stat.
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    r2 = runner.invoke(cli, ["compress", str(inbox), "-o", str(out_dir)])
    assert r2.exit_code == 0, r2.output
    assert "SKIP (unchanged)" in r2.output
    with index_mod.ProcessIndex(db) as idx:
        row = idx.lookup(src)
        assert row.input_mtime_ns == src.stat().st_mtime_ns


# --------------------------------------------------------------------------- #
# The batch runner: only_files restriction
# --------------------------------------------------------------------------- #

@requires_avif
def test_run_batch_only_files_restricts_the_run(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    a = _small_jpg(folder / "a.jpg", seed=1)
    _small_jpg(folder / "b.jpg", seed=2)
    out = tmp_path / "out"
    out.mkdir()

    code, summary = cli_mod._run_batch(
        str(folder), str(out), FaceKeepConfig(),
        only_files=[a], no_videos=True, no_index=True,
    )
    assert code == 0
    assert summary["files"] == 1
    assert summary["ok"] == 1
    assert (out / "a.avif").exists()
    assert not (out / "b.avif").exists()


# --------------------------------------------------------------------------- #
# watch: guards, --once, stat skip, stability, failure memo
# --------------------------------------------------------------------------- #

def test_watch_rejects_archive_equal_to_inbox(tmp_path):
    result = CliRunner().invoke(cli, [
        "watch", str(tmp_path), "-o", str(tmp_path), "--once",
    ])
    assert result.exit_code == 2
    assert "must differ from the inbox" in result.output


@requires_avif
def test_watch_once_processes_then_stat_skips_without_hashing(
    tmp_path, monkeypatch
):
    inbox = tmp_path / "in"
    inbox.mkdir()
    src = _small_jpg(inbox / "pic.jpg")
    archive = tmp_path / "arch"

    r1 = _watch_once(inbox, archive)
    assert r1.exit_code == 0, r1.output
    assert "processed 1: 1 ok" in r1.output
    assert (archive / "pic.avif").exists()
    assert src.exists()  # guardrail 1: the source is untouched
    # The honesty notes appear once per invocation.
    assert "visually lossless, not bit-exact" in r1.output
    assert "never deleted or modified" in r1.output

    # Second pass: the stat pre-filter must skip WITHOUT reading the file —
    # any hash attempt is a test failure.
    def _no_hash(path):
        raise AssertionError(f"hash_file called on {path} in an idle cycle")

    monkeypatch.setattr(index_mod, "hash_file", _no_hash)
    r2 = _watch_once(inbox, archive)
    assert r2.exit_code == 0, r2.output
    assert "idle" in r2.output
    assert "1 unchanged" in r2.output


def test_watch_stability_guard_defers_a_changing_file(tmp_path, monkeypatch):
    """A file whose stat moves between the paired scans is left alone."""
    inbox = tmp_path / "in"
    inbox.mkdir()
    src = _small_jpg(inbox / "pic.jpg")
    archive = tmp_path / "arch"

    real_sleep = cli_mod.time.sleep
    calls = {"n": 0}

    def growing_file_sleep(seconds):
        calls["n"] += 1
        if calls["n"] == 1:  # the settle sleep between the paired scans
            with open(src, "ab") as fh:
                fh.write(b"still-syncing")
        real_sleep(0)

    monkeypatch.setattr(cli_mod.time, "sleep", growing_file_sleep)
    result = _watch_once(inbox, archive)
    assert result.exit_code == 0, result.output
    assert "not yet stable" in result.output
    assert "processed" not in result.output
    assert not (archive / "pic.avif").exists()


def test_watch_once_exit_1_on_a_failed_file(tmp_path):
    inbox = tmp_path / "in"
    inbox.mkdir()
    (inbox / "corrupt.jpg").write_bytes(b"this is not a jpeg")
    archive = tmp_path / "arch"

    result = _watch_once(inbox, archive)
    assert result.exit_code == 1, result.output
    assert "1 failed" in result.output


def test_watch_loop_stops_cleanly_and_never_retries_failures(
    tmp_path, monkeypatch
):
    """Loop mode: Ctrl-C stops cleanly; a failed file waits for a change."""
    inbox = tmp_path / "in"
    inbox.mkdir()
    (inbox / "corrupt.jpg").write_bytes(b"this is not a jpeg")
    archive = tmp_path / "arch"

    calls = {"n": 0}

    def fake_sleep(seconds):
        # 1 = settle (bootstrap), 2 = interval after cycle 1 (the failure),
        # 3 = interval after cycle 2 (the file must have been held, not
        # retried) -> simulate Ctrl-C.
        calls["n"] += 1
        if calls["n"] >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod.time, "sleep", fake_sleep)
    result = CliRunner().invoke(cli, [
        "watch", str(inbox), "-o", str(archive),
        "--interval", "60", "--no-videos",
    ])
    assert result.exit_code == 0, result.output
    assert result.output.count("FAILED") == 1     # attempted exactly once
    assert "failed/skipped earlier" in result.output
    assert "watch: stopped." in result.output


# --------------------------------------------------------------------------- #
# Live-Photo pair policy
# --------------------------------------------------------------------------- #

def test_live_photo_sibling_by_stem(tmp_path):
    mov = tmp_path / "IMG_0001.MOV"
    mov.write_bytes(b"x")
    assert cli_mod._live_photo_sibling(mov) is None
    still = tmp_path / "IMG_0001.HEIC"
    still.write_bytes(b"y")
    assert cli_mod._live_photo_sibling(mov) == still

    # Dotted stems survive (never Path.with_suffix on raw names).
    dotted = tmp_path / "2024.05.20_trip.mov"
    dotted.write_bytes(b"x")
    assert cli_mod._live_photo_sibling(dotted) is None
    (tmp_path / "2024.05.20_trip.jpg").write_bytes(b"y")
    assert cli_mod._live_photo_sibling(dotted) is not None


def _make_clip(dst: Path, *, content_id: "str | None") -> Path:
    """A tiny fat .mov (worth re-encoding), optionally with the pairing key."""
    cmd = [video.find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
           "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-an"]
    if content_id is not None:
        cmd += ["-metadata",
                f"com.apple.quicktime.content.identifier={content_id}",
                "-movflags", "use_metadata_tags"]
    cmd.append(str(dst))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"fixture generation failed: {proc.stderr[-800:]}"
    return dst


def _video_cfg(path: Path, **extra) -> Path:
    lines = ["video:", "  crf: 45", "  preset: 10", "  vmaf_target: null"]
    for key, value in extra.items():
        value = str(value).lower() if isinstance(value, bool) else value
        lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@needs_ffmpeg
def test_probe_reads_the_pairing_key(tmp_path):
    clip = _make_clip(tmp_path / "IMG_1.mov", content_id="AAAA-BBBB")
    assert video.probe_video(clip).content_identifier == "AAAA-BBBB"
    plain = _make_clip(tmp_path / "IMG_2.mov", content_id=None)
    assert video.probe_video(plain).content_identifier is None


@needs_ffmpeg
def test_live_photo_pair_is_kept_verbatim_and_index_recorded(tmp_path):
    src_dir = tmp_path / "roll"
    src_dir.mkdir()
    mov = _make_clip(src_dir / "IMG_0001.mov", content_id="AAAA-BBBB")
    _small_jpg(src_dir / "IMG_0001.jpg")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = _video_cfg(tmp_path / "cfg.yaml")

    runner = CliRunner()
    r1 = runner.invoke(cli, ["compress", str(src_dir), "-o", str(out_dir),
                             "--config", str(cfg)])
    assert r1.exit_code == 0, r1.output
    assert "KEPT ORIGINAL" in r1.output
    assert "Live Photo pair" in r1.output
    kept = out_dir / "IMG_0001.mov"
    assert kept.exists()
    assert kept.read_bytes() == mov.read_bytes()   # byte-identical, not AV1
    assert not (out_dir / "IMG_0001.mp4").exists()

    # The kept-original verdict is index-recorded: a re-run skips it.
    r2 = runner.invoke(cli, ["compress", str(src_dir), "-o", str(out_dir),
                             "--config", str(cfg)])
    assert r2.exit_code == 0, r2.output
    assert "SKIP (unchanged)" in r2.output


@needs_ffmpeg
def test_same_stem_mov_without_pairing_key_encodes_normally(tmp_path):
    src_dir = tmp_path / "roll"
    src_dir.mkdir()
    _make_clip(src_dir / "IMG_0002.mov", content_id=None)
    _small_jpg(src_dir / "IMG_0002.jpg")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = _video_cfg(tmp_path / "cfg.yaml")

    result = CliRunner().invoke(cli, ["compress", str(src_dir), "-o",
                                      str(out_dir), "--config", str(cfg),
                                      "--no-index"])
    assert result.exit_code == 0, result.output
    assert "Live Photo pair" not in result.output
    assert (out_dir / "IMG_0002.mp4").exists()


@needs_ffmpeg
def test_live_photo_policy_opt_out_reencodes(tmp_path):
    src_dir = tmp_path / "roll"
    src_dir.mkdir()
    _make_clip(src_dir / "IMG_0003.mov", content_id="CCCC-DDDD")
    _small_jpg(src_dir / "IMG_0003.jpg")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = _video_cfg(tmp_path / "cfg.yaml", preserve_live_photos=False)

    result = CliRunner().invoke(cli, ["compress", str(src_dir), "-o",
                                      str(out_dir), "--config", str(cfg),
                                      "--no-index"])
    assert result.exit_code == 0, result.output
    assert "Live Photo pair" not in result.output
    assert (out_dir / "IMG_0003.mp4").exists()


@needs_ffmpeg
def test_live_photo_pair_dry_run_would_keep(tmp_path):
    src_dir = tmp_path / "roll"
    src_dir.mkdir()
    _make_clip(src_dir / "IMG_0004.mov", content_id="EEEE-FFFF")
    _small_jpg(src_dir / "IMG_0004.jpg")
    cfg = _video_cfg(tmp_path / "cfg.yaml")

    result = CliRunner().invoke(cli, ["compress", str(src_dir), "--dry-run",
                                      "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "WOULD KEEP ORIGINAL" in result.output
    assert "Live Photo pair" in result.output
    assert not list(src_dir.glob("*.mp4"))  # dry run writes nothing
