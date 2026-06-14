"""Dry-run / estimate-mode tests — ROADMAP Phase 3.

``--dry-run`` must report the *real* projected size/ratio (it runs the actual
encode/pack, then discards the bytes) while writing nothing to disk. These tests
pin both halves of that contract for both modes:

1. **It writes nothing.** No output file, and — when an output directory was
   requested — no directory is created either.
2. **Its numbers match a real run.** Because dry-run goes through the exact same
   encode (faithful) / pack (aggressive) path, the reported ``compressed_size``,
   ratio, quality, and skip decision are byte-for-byte what a real run produces.
   This is the whole value of the feature, so it is asserted as *equality*, not
   "close".

The faithful skip-if-larger decision is exercised too: a dry-run on an
already-optimal input must still report "would keep original" (and still write
nothing), matching what the real run would do.
"""

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import encoders
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import write_fkeep
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig
from facekeep.faithful import compress as faithful_compress

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


def _tiny_incompressible_png(path) -> int:
    """Tiny random PNG whose AVIF re-encode is larger (triggers skip-if-larger)."""
    img = np.random.default_rng(1).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path.stat().st_size


# --------------------------------------------------------------------------- #
# Faithful mode
# --------------------------------------------------------------------------- #

@requires_avif
def test_faithful_dry_run_writes_nothing(face_image, tmp_path):
    """A faithful dry-run reports a real ratio but creates no output file."""
    target = tmp_path / "out"
    res = faithful_compress(str(face_image), str(target), FaceKeepConfig(),
                            dry_run=True)

    assert res.compressed_size > 0
    assert res.ratio > 1.0
    assert not res.skipped
    # The reported path is the one that *would* be written...
    assert res.output_path.suffix == ".avif"
    # ...but nothing exists on disk (the input family.jpg lives in tmp_path; the
    # contract is that no *output* artifact appears).
    assert not res.output_path.exists()
    assert not list(tmp_path.glob("*.avif"))


@requires_avif
def test_faithful_dry_run_numbers_match_real_run(face_image, tmp_path):
    """Dry-run's size/ratio/quality equal the real run's (same encode path)."""
    cfg = FaceKeepConfig()
    cfg.faithful.quality = 75

    dry = faithful_compress(str(face_image), str(tmp_path / "dry"), cfg,
                            dry_run=True)
    real = faithful_compress(str(face_image), str(tmp_path / "real"), cfg,
                             dry_run=False)

    assert real.output_path.exists()  # sanity: the real run did write
    assert dry.compressed_size == real.compressed_size
    assert dry.ratio == real.ratio
    assert dry.quality_used == real.quality_used
    assert dry.faces_detected == real.faces_detected
    assert dry.skipped == real.skipped
    # Both resolve the same codec extension (the stems differ only because this
    # test deliberately wrote them to different targets).
    assert dry.output_path.suffix == real.output_path.suffix == ".avif"


@requires_avif
def test_faithful_dry_run_reports_skip_without_writing(tmp_path):
    """On an already-optimal input, dry-run says 'would keep original', no write.

    The skip-if-larger guard must be reflected faithfully in the estimate: the
    result is marked skipped with ratio 1.0 and the original's extension, yet no
    file is created (the real run would *copy* the original here).
    """
    src = tmp_path / "tiny.png"
    in_size = _tiny_incompressible_png(src)

    # Resolve output into a separate dir so we can prove nothing new is written.
    out_dir = tmp_path / "estimates"
    res = faithful_compress(str(src), str(out_dir / "tiny"), FaceKeepConfig(),
                            dry_run=True)

    assert res.skipped is True
    assert res.compressed_size == in_size
    assert res.ratio == 1.0
    assert res.output_path.suffix == ".png"  # kept the source extension
    assert not res.output_path.exists()
    assert not out_dir.exists()  # dry-run created no directory


# --------------------------------------------------------------------------- #
# Aggressive mode
# --------------------------------------------------------------------------- #

@requires_avif
def test_aggressive_dry_run_writes_nothing(face_image, tmp_path):
    """write_fkeep(dry_run=True) returns the real archive size but writes no file."""
    photo = compress_photo(str(face_image), FaceKeepConfig())
    target = tmp_path / "out"

    size = write_fkeep(photo, str(target), dry_run=True)

    assert size > 0
    # No .fkeep produced (the input family.jpg legitimately lives in tmp_path).
    assert not (tmp_path / "out.fkeep").exists()
    assert not list(tmp_path.glob("*.fkeep"))


@requires_avif
def test_aggressive_dry_run_size_matches_real_write(face_image, tmp_path):
    """The dry-run size equals the real .fkeep file size (same packing path).

    The only field that can differ between two packs is the manifest
    ``created_at`` timestamp. It is written at *second* precision
    (``isoformat(timespec="seconds")``), a genuinely fixed-width ISO-8601 string,
    so two packs in the same second are byte-identical and the packed size is
    stable — which is exactly what the estimate promises. (Microsecond precision
    used to make this off-by-a-byte and flaky.)
    """
    photo = compress_photo(str(face_image), FaceKeepConfig())

    dry_size = write_fkeep(photo, str(tmp_path / "dry"), dry_run=True)
    real_size = write_fkeep(photo, str(tmp_path / "real"), dry_run=False)

    assert (tmp_path / "real.fkeep").exists()
    assert dry_size == real_size


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@requires_avif
def test_cli_dry_run_leaves_no_output(face_image, tmp_path):
    """`compress --dry-run` prints a projected ratio and writes no .avif."""
    out_dir = tmp_path / "cli_out"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "compress", str(face_image), "-o", str(out_dir / "photo"), "--dry-run",
    ])

    assert result.exit_code == 0, result.output
    assert "WOULD WRITE" in result.output
    assert "x," in result.output  # a ratio like "2.8x," is reported
    # No output produced anywhere.
    assert not out_dir.exists()
    assert not list(tmp_path.glob("**/*.avif"))


@requires_avif
def test_cli_without_dry_run_does_write(face_image, tmp_path):
    """Anti-false-green: the same command *without* --dry-run produces the file."""
    target = tmp_path / "photo"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "compress", str(face_image), "-o", str(target),
    ])

    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert list(tmp_path.glob("*.avif")), "real run should have written an .avif"
