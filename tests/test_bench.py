"""Tests for the benchmark harness (ROADMAP backlog: standalone benchmark).

All tests are **offline** and synthetic: they use the conftest ``face_image`` /
``plain_image`` fixtures (no corpus download) and a *fake* LPIPS (no torch /
``[ai]``), so the suite stays zero-download. The autouse ``_force_bicubic_restore``
fixture already pins aggressive restore to the no-AI path, so the aggressive rows
exercise the bicubic proxy here.

What these pin:
  * ``run_benchmark`` produces one row per (file, mode) and measures the right
    columns per mode (faithful: ratio+SSIM; aggressive: ratio+LPIPS).
  * Graceful degradation: LPIPS unavailable -> ``restore_lpips=None`` (blank), no
    crash. A per-file failure is an isolated ``failed`` row, not an abort.
  * Baseline save/load round-trips; a version/format mismatch is a clean error.
  * ``diff_baselines`` computes per-column deltas and handles NEW/GONE rows.
  * ``format_table`` renders, and shows deltas + NEW/GONE against a baseline.
  * The CLI ``bench`` command prints, saves a baseline, diffs, and exit-codes.
"""

import json

import pytest
from click.testing import CliRunner

from facekeep import bench, encoders, metrics
from facekeep.bench import BenchRow
from facekeep.cli import cli
from facekeep.exceptions import FaceKeepError

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


@pytest.fixture
def fake_lpips(monkeypatch):
    """Make ``metrics.lpips_distance`` return a deterministic value (no torch)."""
    monkeypatch.setattr(metrics, "lpips_distance", lambda a, b: 0.1234)


@pytest.fixture
def no_lpips(monkeypatch):
    """Make ``metrics.lpips_distance`` unavailable (the no-[ai] case)."""
    monkeypatch.setattr(metrics, "lpips_distance", lambda a, b: None)


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


@requires_avif
def test_faithful_row_measures_ratio_and_ssim(face_image):
    rows = bench.run_benchmark([face_image], ["faithful"])
    assert len(rows) == 1
    row = rows[0]
    assert row.mode == "faithful"
    assert row.status == "ok"
    assert row.ratio is not None and row.ratio > 0
    assert row.ssim is not None and 0.0 <= row.ssim <= 1.0
    assert row.faces is not None and row.faces >= 1  # synthetic face is detectable
    assert row.original_bytes and row.output_bytes
    # faithful never measures the aggressive restore metric.
    assert row.restore_lpips is None


@requires_avif
def test_aggressive_row_measures_ratio_and_lpips(face_image, fake_lpips):
    rows = bench.run_benchmark([face_image], ["aggressive"])
    assert len(rows) == 1
    row = rows[0]
    assert row.mode == "aggressive"
    assert row.status == "ok"
    assert row.ratio is not None and row.ratio > 0
    assert row.restore_lpips == pytest.approx(0.1234)
    assert row.faces is not None and row.faces >= 1
    # aggressive scores perceptually, not with decoded SSIM.
    assert row.ssim is None


@requires_avif
def test_both_modes_one_row_each_in_order(face_image, fake_lpips):
    rows = bench.run_benchmark([face_image], ["faithful", "aggressive"])
    assert [r.mode for r in rows] == ["faithful", "aggressive"]


@requires_avif
def test_lpips_unavailable_leaves_blank_not_fabricated(face_image, no_lpips):
    """No [ai] -> restore_lpips is None (blank), and the run does not crash."""
    rows = bench.run_benchmark([face_image], ["aggressive"])
    assert rows[0].status == "ok"
    assert rows[0].restore_lpips is None


def test_per_file_failure_is_isolated_row(tmp_path):
    """A file that fails compression becomes a single failed row, not an abort."""
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a real jpeg")
    rows = bench.run_benchmark([bad], ["faithful"])
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].error
    assert rows[0].ratio is None


@requires_avif
def test_multiple_files_each_get_rows(face_image, plain_image, fake_lpips):
    rows = bench.run_benchmark([face_image, plain_image], ["faithful"])
    assert {r.file for r in rows} == {face_image.name, plain_image.name}
    assert all(r.mode == "faithful" for r in rows)


# ---------------------------------------------------------------------------
# baseline save / load
# ---------------------------------------------------------------------------


def test_baseline_round_trip(tmp_path):
    rows = [
        BenchRow(file="a.jpg", mode="faithful", ratio=1.9, ssim=0.98, faces=1),
        BenchRow(file="a.jpg", mode="aggressive", ratio=8.0, restore_lpips=0.12),
    ]
    path = tmp_path / "base.json"
    bench.save_baseline(rows, str(path))
    loaded = bench.load_baseline(str(path))
    assert loaded == rows


def test_baseline_load_rejects_wrong_version(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 999, "rows": []}), encoding="utf-8")
    with pytest.raises(FaceKeepError, match="version"):
        bench.load_baseline(str(path))


def test_baseline_load_rejects_garbage(tmp_path):
    path = tmp_path / "junk.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(FaceKeepError):
        bench.load_baseline(str(path))


def test_baseline_load_tolerates_unknown_keys(tmp_path):
    """A baseline carrying an extra/older key still loads (forward tolerance)."""
    path = tmp_path / "extra.json"
    payload = {
        "version": bench.BASELINE_VERSION,
        "rows": [{"file": "a.jpg", "mode": "faithful", "ratio": 1.5,
                  "some_future_column": 42}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = bench.load_baseline(str(path))
    assert loaded[0].file == "a.jpg"
    assert loaded[0].ratio == 1.5


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def test_diff_computes_numeric_deltas():
    old = [BenchRow(file="a.jpg", mode="faithful", ratio=1.90, ssim=0.980)]
    new = [BenchRow(file="a.jpg", mode="faithful", ratio=1.80, ssim=0.990)]
    diff = bench.diff_baselines(old, new)
    d = diff[("a.jpg", "faithful")]
    assert d["ratio"] == pytest.approx(-0.10)
    assert d["ssim"] == pytest.approx(0.010)


def test_diff_none_when_metric_missing_one_side():
    old = [BenchRow(file="a.jpg", mode="aggressive", ratio=8.0)]  # no lpips
    new = [BenchRow(file="a.jpg", mode="aggressive", ratio=8.0, restore_lpips=0.1)]
    d = bench.diff_baselines(old, new)[("a.jpg", "aggressive")]
    assert d["ratio"] == pytest.approx(0.0)
    assert d["restore_lpips"] is None  # absent on the old side


def test_diff_handles_added_and_removed_rows():
    old = [BenchRow(file="gone.jpg", mode="faithful", ratio=1.0)]
    new = [BenchRow(file="new.jpg", mode="faithful", ratio=2.0)]
    diff = bench.diff_baselines(old, new)
    assert ("gone.jpg", "faithful") in diff
    assert ("new.jpg", "faithful") in diff
    # No common metric, so every delta is None.
    assert all(v is None for v in diff[("gone.jpg", "faithful")].values())


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


def test_format_table_renders_headers_and_rows():
    rows = [BenchRow(file="a.jpg", mode="faithful", ratio=1.9, ssim=0.98, faces=1)]
    out = bench.format_table(rows)
    assert "file" in out and "ratio" in out and "rest.lpips" in out
    assert "a.jpg" in out
    assert "1.900" in out


def test_format_table_blank_for_none():
    rows = [BenchRow(file="a.jpg", mode="aggressive", ratio=8.0)]  # no ssim/lpips
    out = bench.format_table(rows)
    assert "8.000" in out
    # No fabricated SSIM/LPIPS value appears.
    assert "None" not in out


def test_format_table_shows_deltas_against_baseline():
    base = [BenchRow(file="a.jpg", mode="faithful", ratio=1.90, ssim=0.980)]
    rows = [BenchRow(file="a.jpg", mode="faithful", ratio=1.80, ssim=0.990)]
    out = bench.format_table(rows, baseline=base)
    assert "(-0.100)" in out  # ratio delta
    assert "(+0.0100)" in out  # ssim delta


def test_format_table_marks_new_and_gone():
    base = [BenchRow(file="gone.jpg", mode="faithful", ratio=1.0)]
    rows = [BenchRow(file="new.jpg", mode="faithful", ratio=2.0)]
    out = bench.format_table(rows, baseline=base)
    assert "NEW" in out
    assert "GONE" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@requires_avif
def test_cli_bench_prints_table(face_image, fake_lpips):
    runner = CliRunner()
    result = runner.invoke(cli, ["bench", str(face_image), "-m", "faithful"])
    assert result.exit_code == 0, result.output
    assert "ratio" in result.output
    assert face_image.name in result.output


@requires_avif
def test_cli_bench_saves_and_diffs_baseline(face_image, fake_lpips, tmp_path):
    runner = CliRunner()
    base_path = tmp_path / "base.json"
    # First run: save a baseline.
    r1 = runner.invoke(
        cli, ["bench", str(face_image), "-m", "faithful",
              "--save-baseline", str(base_path)]
    )
    assert r1.exit_code == 0, r1.output
    assert base_path.exists()
    assert "Baseline saved" in r1.output

    # Second run: diff against it (table should render without error).
    r2 = runner.invoke(
        cli, ["bench", str(face_image), "-m", "faithful", "--baseline", str(base_path)]
    )
    assert r2.exit_code == 0, r2.output
    assert face_image.name in r2.output


@requires_avif
def test_cli_bench_writes_report(face_image, fake_lpips, tmp_path):
    runner = CliRunner()
    report_path = tmp_path / "bench.csv"
    result = runner.invoke(
        cli, ["bench", str(face_image), "-m", "faithful", "--report", str(report_path)]
    )
    assert result.exit_code == 0, result.output
    assert report_path.exists()
    assert "file,mode" in report_path.read_text(encoding="utf-8")


def test_cli_bench_no_images_exits_2(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["bench", str(tmp_path)])
    assert result.exit_code == 2
    assert "No images" in result.output


def test_cli_bench_failed_file_exits_1(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a real jpeg")
    runner = CliRunner()
    result = runner.invoke(cli, ["bench", str(bad), "-m", "faithful"])
    assert result.exit_code == 1
    assert "failed" in result.output
