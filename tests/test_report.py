"""Sidecar verification report tests — ROADMAP Phase 3.

``compress --report out.csv`` writes a per-file CSV ledger. These tests pin the
contract:

1. **One row per file**, with the fixed column set, plus a header.
2. **The SSIM column is honest:** filled only when a score was *actually
   measured* (faithful + ``--verify-thorough``); blank otherwise, and always
   blank for aggressive mode (no reconstruction at compress time).
3. **The report is the artifact:** it is written even under ``--dry-run`` (rows
   marked ``would-write`` / ``would-keep-original``), while no image is produced.
4. **Skip-if-larger and failures are recorded**, so the CSV is a complete ledger.

The ``verify_roundtrip`` return-contract (the source of the SSIM number) is
pinned with a small unit test so the report can't silently start reporting a
fabricated score.
"""

import csv

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import encoders, report
from facekeep.cli import cli
from facekeep.report import FIELDNAMES

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


def _tiny_incompressible_png(path) -> int:
    """Tiny random PNG whose AVIF re-encode is larger (triggers skip-if-larger)."""
    img = np.random.default_rng(1).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path.stat().st_size


def _read_csv(path):
    """Return (fieldnames, list-of-row-dicts)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


# --------------------------------------------------------------------------- #
# write_report / ReportRow unit behaviour
# --------------------------------------------------------------------------- #

def test_write_report_header_and_blank_cells(tmp_path):
    """None fields render as empty cells; the header is the fixed column set."""
    rows = [
        report.ReportRow(file="a.jpg", mode="faithful", status="written",
                         codec="avif", original_bytes=100, output_bytes=40,
                         ratio=2.5, quality=70, faces=1, ssim_downscaled=0.991,
                         output_path="a.avif"),
        # Aggressive-style row: quality + ssim deliberately None -> blank cells.
        report.ReportRow(file="b.jpg", mode="aggressive", status="written",
                         original_bytes=200, output_bytes=20, ratio=10.0,
                         faces=2, output_path="b.fkeep"),
    ]
    out = report.write_report(rows, str(tmp_path / "r.csv"))
    assert out.exists()

    fieldnames, data = _read_csv(out)
    assert fieldnames == FIELDNAMES
    assert len(data) == 2
    assert data[0]["ssim_downscaled"] == "0.9910"
    assert data[0]["ratio"] == "2.5000"
    # Aggressive row: the un-measured columns are blank strings, not "0"/"None".
    assert data[1]["quality"] == ""
    assert data[1]["ssim_downscaled"] == ""
    assert data[1]["codec"] == ""


def test_write_report_creates_parent_dir(tmp_path):
    """A nested report path has its directory created."""
    nested = tmp_path / "deep" / "nested" / "r.csv"
    report.write_report(
        [report.ReportRow(file="x", mode="faithful", status="written")],
        str(nested),
    )
    assert nested.exists()


# --------------------------------------------------------------------------- #
# verify_roundtrip return contract (the SSIM source)
# --------------------------------------------------------------------------- #

@requires_avif
def test_verify_roundtrip_returns_score_only_when_thorough():
    """thorough -> a float in (0,1]; quick -> None (no fabricated score)."""
    img = np.full((64, 64, 3), 120, np.uint8)
    cv2.circle(img, (32, 32), 16, (200, 180, 170), -1)
    data = encoders.encode(img, "avif", quality=80)

    quick = encoders.verify_roundtrip(data, img, thorough=False)
    assert quick is None

    score = encoders.verify_roundtrip(data, img, thorough=True)
    assert isinstance(score, float)
    assert 0.0 < score <= 1.0


# --------------------------------------------------------------------------- #
# CLI: faithful
# --------------------------------------------------------------------------- #

@requires_avif
def test_report_written_faithful(face_image, tmp_path):
    """A real faithful run produces a CSV with one written row."""
    report_csv = tmp_path / "report.csv"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "compress", str(face_image), "-o", str(tmp_path / "out"),
        "--report", str(report_csv),
    ])

    assert result.exit_code == 0, result.output
    assert "Report written ->" in result.output
    assert report_csv.exists()

    fieldnames, data = _read_csv(report_csv)
    assert fieldnames == FIELDNAMES
    assert len(data) == 1
    row = data[0]
    assert row["mode"] == "faithful"
    assert row["codec"] == "avif"
    assert row["status"] == "written"
    assert float(row["ratio"]) > 1.0
    assert int(row["output_bytes"]) > 0
    assert row["output_path"].endswith(".avif")


@requires_avif
def test_report_ssim_filled_only_with_verify_thorough(face_image, tmp_path):
    """The SSIM cell is a real float under --verify-thorough, blank without it.

    This is the honesty pin: the report never invents a fidelity number. When
    the pipeline measured one (thorough round-trip), it appears; when it didn't,
    the cell stays empty.
    """
    runner = CliRunner()

    # With --verify-thorough: the pipeline measures a downscaled SSIM.
    csv_thorough = tmp_path / "thorough.csv"
    r1 = runner.invoke(cli, [
        "compress", str(face_image), "-o", str(tmp_path / "a"),
        "--verify-thorough", "--report", str(csv_thorough),
    ])
    assert r1.exit_code == 0, r1.output
    _, data1 = _read_csv(csv_thorough)
    score = data1[0]["ssim_downscaled"]
    assert score != "", "verify-thorough should fill the SSIM column"
    assert 0.0 < float(score) <= 1.0

    # Without it (quick verify by default): nothing measured -> blank cell.
    csv_quick = tmp_path / "quick.csv"
    r2 = runner.invoke(cli, [
        "compress", str(face_image), "-o", str(tmp_path / "b"),
        "--report", str(csv_quick),
    ])
    assert r2.exit_code == 0, r2.output
    _, data2 = _read_csv(csv_quick)
    assert data2[0]["ssim_downscaled"] == "", "no measurement -> blank, not a number"


@requires_avif
def test_report_records_skip(tmp_path):
    """An already-optimal input is logged as kept-original with ratio 1.0."""
    src = tmp_path / "tiny.png"
    _tiny_incompressible_png(src)
    report_csv = tmp_path / "r.csv"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "compress", str(src), "-o", str(tmp_path / "out"),
        "--report", str(report_csv),
    ])
    assert result.exit_code == 0, result.output

    _, data = _read_csv(report_csv)
    assert len(data) == 1
    assert data[0]["status"] == "kept-original"
    assert float(data[0]["ratio"]) == 1.0


# --------------------------------------------------------------------------- #
# CLI: dry-run interplay
# --------------------------------------------------------------------------- #

@requires_avif
def test_report_dry_run_writes_report_but_no_images(face_image, tmp_path):
    """--dry-run --report: the CSV is written (would-write), no .avif produced."""
    report_csv = tmp_path / "estimate.csv"
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "compress", str(face_image), "-o", str(out_dir / "photo"),
        "--dry-run", "--report", str(report_csv),
    ])
    assert result.exit_code == 0, result.output

    # The report exists and is the only artifact.
    assert report_csv.exists()
    _, data = _read_csv(report_csv)
    assert data[0]["status"] == "would-write"
    assert float(data[0]["ratio"]) > 1.0
    # No image written anywhere (the report itself may live under tmp_path).
    assert not out_dir.exists()
    assert not list(tmp_path.glob("**/*.avif"))


# --------------------------------------------------------------------------- #
# CLI: aggressive
# --------------------------------------------------------------------------- #

@requires_avif
def test_report_aggressive_row(face_image, tmp_path):
    """Aggressive row: mode=aggressive, blank quality + SSIM, .fkeep path."""
    report_csv = tmp_path / "r.csv"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "compress", str(face_image), "-m", "aggressive",
        "-o", str(tmp_path / "out"), "--report", str(report_csv),
    ])
    assert result.exit_code == 0, result.output

    _, data = _read_csv(report_csv)
    assert len(data) == 1
    row = data[0]
    assert row["mode"] == "aggressive"
    assert row["status"] == "written"
    assert row["quality"] == ""            # bg_scale isn't a 0-100 quality
    assert row["ssim_downscaled"] == ""    # not reconstructed at compress time
    assert row["output_path"].endswith(".fkeep")


# --------------------------------------------------------------------------- #
# CLI: batch
# --------------------------------------------------------------------------- #

@requires_avif
def test_report_batch_one_row_per_file(face_image, plain_image, tmp_path):
    """A directory of two images yields a header + exactly two rows."""
    # face_image and plain_image live in separate tmp dirs; copy both into one.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for p in (face_image, plain_image):
        (src_dir / p.name).write_bytes(p.read_bytes())

    report_csv = tmp_path / "batch.csv"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "compress", str(src_dir), "-o", str(tmp_path / "out"),
        "--report", str(report_csv),
    ])
    assert result.exit_code == 0, result.output

    _, data = _read_csv(report_csv)
    assert len(data) == 2
    assert {r["mode"] for r in data} == {"faithful"}
    assert all(r["status"] in {"written", "kept-original"} for r in data)
