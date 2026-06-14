"""Batch parallelism (`--jobs N`) — ROADMAP Phase 3.

`facekeep compress <folder> --jobs N` spreads a folder across N worker
*processes*. The contract these tests pin is that parallelism is an
*implementation detail with no observable effect on results*:

1. **Same outputs.** A parallel run writes the exact same set of output files,
   byte-for-byte, as the serial (`--jobs 1`) run — the codec is deterministic
   and each file is independent.
2. **Same report.** The `--report` CSV is row-for-row identical (same input
   order, same columns/values) regardless of worker count, because results are
   replayed in input order in the parent.
3. **Same summary / counts.** The `N/M ok` batch line matches.
4. **Single file is never pooled.** `--jobs 8` on one file behaves exactly like
   serial (and must not pay pool/pickle cost — asserted by patching the pool).
5. **Error isolation.** One unreadable file fails its own row; the rest still
   succeed. (Matches the serial batch's per-file catch.)
6. **dry-run + jobs.** A parallel dry-run writes nothing and its numbers match a
   serial dry-run.

The codec is invoked for real in worker processes, so these run under the same
`requires_avif` guard as the rest of the suite.
"""

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import encoders
from facekeep.cli import cli

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


def _make_photo(path, seed: int, faces: bool = True):
    """Write one compressible synthetic JPEG (optionally with a Haar face).

    Distinct per seed so the files aren't trivially identical, but smooth/
    compressible so faithful AVIF genuinely beats the JPEG input (not the
    skip-if-larger path).
    """
    rng = np.random.default_rng(seed)
    H, W = 600, 800
    bg = cv2.resize(
        rng.normal(128, 25, (H // 10, W // 10, 3)).astype(np.float32),
        (W, H), interpolation=cv2.INTER_CUBIC,
    )
    img = np.clip(bg, 0, 255).astype(np.uint8)
    if faces:
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
    for i in range(5):
        _make_photo(d / f"p{i}.jpg", seed=10 + i)
    return d


def _file_bytes(folder, pattern):
    """Map {name: bytes} for files in `folder` matching glob `pattern`, sorted."""
    return {p.name: p.read_bytes() for p in sorted(folder.glob(pattern))}


# --------------------------------------------------------------------------- #
# Same outputs, parallel vs serial
# --------------------------------------------------------------------------- #

@requires_avif
def test_parallel_outputs_match_serial(photo_dir, tmp_path):
    """`--jobs 2` produces byte-identical .avif outputs to `--jobs 1`."""
    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"
    runner = CliRunner()

    r1 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(serial_out),
                             "--jobs", "1"])
    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(parallel_out),
                             "--jobs", "2"])

    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output

    serial = _file_bytes(serial_out, "*.avif")
    parallel = _file_bytes(parallel_out, "*.avif")
    assert len(serial) == 5
    assert serial.keys() == parallel.keys()
    for name in serial:
        assert serial[name] == parallel[name], f"{name} differs parallel vs serial"


@requires_avif
def test_parallel_summary_counts_match(photo_dir, tmp_path):
    """The batch summary reports the same N/M ok for serial and parallel."""
    runner = CliRunner()
    r1 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(tmp_path / "a"),
                             "--jobs", "1"])
    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(tmp_path / "b"),
                             "--jobs", "3"])
    assert "5/5 ok" in r1.output
    assert "5/5 ok" in r2.output


@requires_avif
def test_jobs_zero_runs_all(photo_dir, tmp_path):
    """`--jobs 0` (one per CPU) completes and writes every output."""
    runner = CliRunner()
    out = tmp_path / "auto"
    result = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out),
                                 "--jobs", "0"])
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.avif"))) == 5


# --------------------------------------------------------------------------- #
# --report parity
# --------------------------------------------------------------------------- #

@requires_avif
def test_parallel_report_matches_serial(photo_dir, tmp_path):
    """The --report CSV is row-for-row identical between serial and parallel."""
    runner = CliRunner()
    rep1 = tmp_path / "serial.csv"
    rep2 = tmp_path / "parallel.csv"

    runner.invoke(cli, ["compress", str(photo_dir), "-o", str(tmp_path / "s"),
                        "--jobs", "1", "--report", str(rep1)])
    runner.invoke(cli, ["compress", str(photo_dir), "-o", str(tmp_path / "p"),
                        "--jobs", "4", "--report", str(rep2)])

    a = rep1.read_text(encoding="utf-8").splitlines()
    b = rep2.read_text(encoding="utf-8").splitlines()
    # The only column that could differ is output_path's *directory*? No — the
    # report stores only the file name, and rows are emitted in input order, so
    # the two CSVs must be exactly equal.
    assert a == b
    assert len(a) == 1 + 5  # header + 5 files


# --------------------------------------------------------------------------- #
# Single file is never pooled
# --------------------------------------------------------------------------- #

@requires_avif
def test_single_file_does_not_use_pool(face_image, tmp_path, monkeypatch):
    """`--jobs 8` on ONE file must run serially (no ProcessPoolExecutor built)."""
    import facekeep.cli as cli_mod

    def _boom(*a, **k):  # pragma: no cover - must never be called here
        raise AssertionError("ProcessPoolExecutor must not be used for one file")

    monkeypatch.setattr(cli_mod, "ProcessPoolExecutor", _boom)

    runner = CliRunner()
    result = runner.invoke(cli, ["compress", str(face_image),
                                 "-o", str(tmp_path / "one"), "--jobs", "8"])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("*.avif")), "the single file should still be written"


# --------------------------------------------------------------------------- #
# Error isolation
# --------------------------------------------------------------------------- #

@requires_avif
def test_parallel_isolates_a_bad_file(photo_dir, tmp_path):
    """A corrupt file fails its own row; the others still succeed in parallel."""
    bad = photo_dir / "broken.jpg"
    bad.write_bytes(b"this is not a valid JPEG")

    runner = CliRunner()
    out = tmp_path / "mixed"
    rep = tmp_path / "mixed.csv"
    result = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out),
                                 "--jobs", "3", "--report", str(rep)])

    # The batch doesn't abort: the 5 good files are written, summary says 5/6.
    assert "5/6 ok" in result.output
    assert "FAILED" in result.output
    assert len(list(out.glob("*.avif"))) == 5

    text = rep.read_text(encoding="utf-8")
    assert "broken.jpg" in text
    # The broken file's row is marked failed.
    failed_rows = [ln for ln in text.splitlines() if ln.startswith("broken.jpg")]
    assert len(failed_rows) == 1
    assert ",failed," in failed_rows[0]


# --------------------------------------------------------------------------- #
# dry-run + jobs
# --------------------------------------------------------------------------- #

@requires_avif
def test_parallel_dry_run_writes_nothing(photo_dir, tmp_path):
    """A parallel dry-run reports ratios but writes no output file or directory."""
    out = tmp_path / "would"
    runner = CliRunner()
    result = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out),
                                 "--jobs", "3", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "[dry-run]" in result.output
    assert "WOULD WRITE" in result.output
    assert not out.exists()
    assert not list(photo_dir.glob("*.avif"))


@requires_avif
def test_parallel_dry_run_totals_match_serial(photo_dir, tmp_path):
    """Parallel dry-run's summary totals equal the serial dry-run's."""
    runner = CliRunner()
    r1 = runner.invoke(cli, ["compress", str(photo_dir), "--jobs", "1", "--dry-run"])
    r2 = runner.invoke(cli, ["compress", str(photo_dir), "--jobs", "4", "--dry-run"])
    assert r1.exit_code == 0 and r2.exit_code == 0

    def _summary(text):
        return [ln for ln in text.splitlines() if ln.startswith("---")][0]

    assert _summary(r1.output) == _summary(r2.output)


# --------------------------------------------------------------------------- #
# Aggressive mode
# --------------------------------------------------------------------------- #

@requires_avif
def test_parallel_aggressive_outputs_match_serial(photo_dir, tmp_path):
    """Aggressive `--jobs 2` writes the same set of .fkeep files as serial.

    .fkeep packing is deterministic to the second (created_at uses second
    precision), so two runs in the same second produce the same file names; we
    assert the produced *set* matches and each archive verifies, rather than
    byte-equality (the timestamp can legitimately tick between the two runs).
    """
    from facekeep.aggressive.format import verify_fkeep

    serial_out = tmp_path / "s_fk"
    parallel_out = tmp_path / "p_fk"
    runner = CliRunner()

    r1 = runner.invoke(cli, ["compress", str(photo_dir), "-m", "aggressive",
                             "-o", str(serial_out), "--jobs", "1"])
    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-m", "aggressive",
                             "-o", str(parallel_out), "--jobs", "2"])
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output

    serial = {p.name for p in serial_out.glob("*.fkeep")}
    parallel = {p.name for p in parallel_out.glob("*.fkeep")}
    assert len(serial) == 5
    assert serial == parallel
    # Every parallel-produced archive is structurally valid.
    for p in parallel_out.glob("*.fkeep"):
        assert verify_fkeep(str(p)).ok, f"{p.name} did not verify"
