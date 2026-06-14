"""Incremental processing index — ROADMAP Phase 3.

The index lets a *re-run* skip a file that is byte-identical to the last
successful run, processed with the same output-affecting settings, and whose
output still exists. It is a pure speed feature: it must never change the bytes
of any output it *does* write. These tests pin both halves of that contract —
when a skip is correct, and that a skip never corrupts the result set:

* **Cold run** processes everything and populates the DB; **warm run** skips
  everything (asserted by spying on the pipeline: the compressor is *not* called
  on a hit).
* **Cache busting**: editing one input, or changing ``-q`` / ``--codec`` / ``-m``,
  re-processes (only) the affected files; deleting the output forces a re-make
  even on an otherwise-valid hit.
* **Flags**: ``--force`` re-processes all; ``--no-index`` never creates a DB;
  ``--dry-run`` never writes the DB.
* **No observable effect**: a re-processed file (after an edit) is byte-identical
  to its first encode, and skipping works the same under ``--jobs 2`` — the same
  guarantees the parallel/progress tests pin.

The codec runs for real, so these share the suite's ``requires_avif`` guard.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

import facekeep.cli as cli_mod
from facekeep import encoders, index as index_mod
from facekeep.cli import cli

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


def _avifs(folder):
    """{name: bytes} for the .avif files written into `folder`."""
    return {p.name: p.read_bytes() for p in sorted(folder.glob("*.avif"))}


# --------------------------------------------------------------------------- #
# ProcessIndex unit behaviour
# --------------------------------------------------------------------------- #

def test_fingerprint_changes_with_output_affecting_settings():
    """The settings fingerprint must differ when output-affecting fields differ."""
    from facekeep.config import FaceKeepConfig

    base = FaceKeepConfig()
    fp_base = index_mod.settings_fingerprint(base)

    c_q = FaceKeepConfig()
    c_q.faithful.quality = base.faithful.quality + 5
    assert index_mod.settings_fingerprint(c_q) != fp_base

    c_codec = FaceKeepConfig()
    c_codec.faithful.codec = "jxl"
    assert index_mod.settings_fingerprint(c_codec) != fp_base

    c_mode = FaceKeepConfig()
    c_mode.mode = "aggressive"
    assert index_mod.settings_fingerprint(c_mode) != fp_base

    # Same config -> same fingerprint (stable).
    assert index_mod.settings_fingerprint(FaceKeepConfig()) == fp_base


def test_index_roundtrip_and_output_existence(tmp_path):
    """A recorded row is a hit only while hash+fingerprint match and output exists."""
    db = tmp_path / "idx.sqlite"
    src = tmp_path / "a.jpg"
    src.write_bytes(b"hello")
    out = tmp_path / "a.avif"
    out.write_bytes(b"encoded-output")

    h = index_mod.hash_file(src)
    with index_mod.ProcessIndex(db) as idx:
        assert idx.lookup(src) is None
        idx.record(src, index_mod.IndexRow(
            content_hash=h, settings_fingerprint="fp1", mode="faithful",
            codec="avif", quality=70, original_size=5,
            output_path=str(out), output_size=14,
        ))

    with index_mod.ProcessIndex(db) as idx:
        assert idx.is_unchanged(src, h, "fp1") is not None       # full match
        assert idx.is_unchanged(src, "other", "fp1") is None     # content changed
        assert idx.is_unchanged(src, h, "fp2") is None           # settings changed
        out.unlink()
        assert idx.is_unchanged(src, h, "fp1") is None           # output gone


# --------------------------------------------------------------------------- #
# End-to-end: skip on re-run
# --------------------------------------------------------------------------- #

@requires_avif
def test_cold_run_then_warm_run_skips_all(photo_dir, tmp_path):
    """First run encodes all 4 and writes a DB; second run skips all 4."""
    runner = CliRunner()
    out = tmp_path / "out"

    r1 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)])
    assert r1.exit_code == 0, r1.output
    assert len(_avifs(out)) == 4
    assert "4/4 ok" in r1.output
    assert (out / index_mod.INDEX_FILENAME).exists()

    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)])
    assert r2.exit_code == 0, r2.output
    # Every file reports as an unchanged skip; nothing re-encoded.
    assert r2.output.count("SKIP (unchanged)") == 4
    assert "4 unchanged (skipped)" in r2.output


@requires_avif
def test_warm_run_does_not_invoke_pipeline(photo_dir, tmp_path, monkeypatch):
    """On a full cache hit the compressor is never called (real skip, not re-encode)."""
    runner = CliRunner()
    out = tmp_path / "out"
    r1 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)])
    assert r1.exit_code == 0, r1.output

    # Spy: any call to _process_one on the warm run is a failure to skip.
    calls = []
    real = cli_mod._process_one

    def _spy(file_str, *a, **k):
        calls.append(file_str)
        return real(file_str, *a, **k)

    monkeypatch.setattr(cli_mod, "_process_one", _spy)

    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)])
    assert r2.exit_code == 0, r2.output
    assert calls == [], f"unchanged files should not be processed, got {calls}"


@requires_avif
def test_single_file_explicit_output_path_with_index(photo_dir, tmp_path):
    """Single file + explicit ``-o <new-dir>/<name>.avif`` works with the index ON.

    Regression: the default index path used to be ``out_p / INDEX_FILENAME``
    where ``out_p`` is the *output file path* for a single explicit target, so
    ``ProcessIndex`` mkdir'd the output path as a directory and the image write
    crashed with PermissionError (both modes). The index must land in the
    output's parent directory instead — and the warm re-run must still skip.
    """
    runner = CliRunner()
    src = photo_dir / "p0.jpg"
    target = tmp_path / "newdir" / "photo.avif"

    r1 = runner.invoke(cli, ["compress", str(src), "-o", str(target)])
    assert r1.exit_code == 0, r1.output
    assert "FAILED" not in r1.output
    assert target.is_file(), "output target must be a file (the bug made it a dir)"
    assert (target.parent / index_mod.INDEX_FILENAME).exists()

    r2 = runner.invoke(cli, ["compress", str(src), "-o", str(target)])
    assert r2.exit_code == 0, r2.output
    assert "SKIP (unchanged)" in r2.output


@requires_avif
def test_editing_one_file_reprocesses_only_it(photo_dir, tmp_path, monkeypatch):
    """Editing a single input busts only that file's cache entry."""
    runner = CliRunner()
    out = tmp_path / "out"
    assert runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)]).exit_code == 0

    # Change p1's bytes (different image -> different hash).
    _make_photo(photo_dir / "p1.jpg", seed=999)

    processed = []
    real = cli_mod._process_one

    def _spy(file_str, *a, **k):
        processed.append(Path(file_str).name)
        return real(file_str, *a, **k)

    monkeypatch.setattr(cli_mod, "_process_one", _spy)

    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)])
    assert r2.exit_code == 0, r2.output
    assert processed == ["p1.jpg"], processed
    assert r2.output.count("SKIP (unchanged)") == 3
    assert "3 unchanged (skipped)" in r2.output


@requires_avif
def test_changing_quality_busts_cache(photo_dir, tmp_path):
    """A different -q value changes the fingerprint, so nothing is skipped."""
    runner = CliRunner()
    out = tmp_path / "out"
    assert runner.invoke(
        cli, ["compress", str(photo_dir), "-o", str(out), "-q", "70"]
    ).exit_code == 0

    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out), "-q", "50"])
    assert r2.exit_code == 0, r2.output
    assert "SKIP (unchanged)" not in r2.output
    assert "4/4 ok" in r2.output


@requires_avif
def test_deleting_output_forces_remake(photo_dir, tmp_path, monkeypatch):
    """A valid cache row whose output was deleted must re-make that file."""
    runner = CliRunner()
    out = tmp_path / "out"
    assert runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)]).exit_code == 0

    # Delete one output file.
    (out / "p2.avif").unlink()

    processed = []
    real = cli_mod._process_one

    def _spy(file_str, *a, **k):
        processed.append(Path(file_str).name)
        return real(file_str, *a, **k)

    monkeypatch.setattr(cli_mod, "_process_one", _spy)

    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)])
    assert r2.exit_code == 0, r2.output
    assert processed == ["p2.jpg"], processed
    assert (out / "p2.avif").exists()


@requires_avif
def test_force_reprocesses_everything(photo_dir, tmp_path, monkeypatch):
    """--force ignores the cache and processes every file."""
    runner = CliRunner()
    out = tmp_path / "out"
    assert runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)]).exit_code == 0

    processed = []
    real = cli_mod._process_one

    def _spy(file_str, *a, **k):
        processed.append(Path(file_str).name)
        return real(file_str, *a, **k)

    monkeypatch.setattr(cli_mod, "_process_one", _spy)

    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out), "--force"])
    assert r2.exit_code == 0, r2.output
    assert sorted(processed) == ["p0.jpg", "p1.jpg", "p2.jpg", "p3.jpg"]
    assert "SKIP (unchanged)" not in r2.output


@requires_avif
def test_no_index_never_creates_db(photo_dir, tmp_path):
    """--no-index disables the feature: no DB file, no skipping."""
    runner = CliRunner()
    out = tmp_path / "out"
    r1 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out), "--no-index"])
    assert r1.exit_code == 0, r1.output
    assert not (out / index_mod.INDEX_FILENAME).exists()

    # A second --no-index run still processes everything (no cache consulted).
    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out), "--no-index"])
    assert r2.exit_code == 0, r2.output
    assert "SKIP (unchanged)" not in r2.output
    assert not (out / index_mod.INDEX_FILENAME).exists()


@requires_avif
def test_dry_run_does_not_write_index(photo_dir, tmp_path):
    """--dry-run writes nothing, including no index DB; a later real run can't skip."""
    runner = CliRunner()
    out = tmp_path / "out"
    r1 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out), "--dry-run"])
    assert r1.exit_code == 0, r1.output
    assert not (out / index_mod.INDEX_FILENAME).exists()
    assert not list(out.glob("*.avif")) if out.exists() else True

    # The real run after a dry-run still has to do the work (nothing cached).
    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)])
    assert r2.exit_code == 0, r2.output
    assert "SKIP (unchanged)" not in r2.output
    assert "4/4 ok" in r2.output


@requires_avif
def test_skip_is_byte_identical_to_first_encode(photo_dir, tmp_path):
    """Re-running keeps the exact bytes from the first encode (skip == no rewrite)."""
    runner = CliRunner()
    out = tmp_path / "out"
    assert runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)]).exit_code == 0
    first = _avifs(out)
    assert len(first) == 4

    assert runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)]).exit_code == 0
    second = _avifs(out)
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], f"{name} changed on a skip-run"


@requires_avif
def test_skip_works_under_jobs(photo_dir, tmp_path):
    """The cache skip is honored under --jobs too (parent-only DB)."""
    runner = CliRunner()
    out = tmp_path / "out"
    assert runner.invoke(
        cli, ["compress", str(photo_dir), "-o", str(out), "--jobs", "2"]
    ).exit_code == 0
    assert len(_avifs(out)) == 4

    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out), "--jobs", "2"])
    assert r2.exit_code == 0, r2.output
    assert r2.output.count("SKIP (unchanged)") == 4


@requires_avif
def test_report_marks_cached_rows(photo_dir, tmp_path):
    """On a warm run, --report records status=cached rows with cached sizes."""
    runner = CliRunner()
    out = tmp_path / "out"
    assert runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out)]).exit_code == 0

    csv_path = tmp_path / "warm.csv"
    r2 = runner.invoke(cli, ["compress", str(photo_dir), "-o", str(out),
                             "--report", str(csv_path)])
    assert r2.exit_code == 0, r2.output
    text = csv_path.read_text(encoding="utf-8")
    # Four cached rows, and a non-empty original_bytes column on each.
    assert text.count(",cached,") == 4
    lines = [ln for ln in text.splitlines() if ",cached," in ln]
    assert len(lines) == 4
    for ln in lines:
        cells = ln.split(",")
        # original_bytes is column index 3 (file,mode,codec,original_bytes,...).
        assert cells[3].isdigit() and int(cells[3]) > 0
