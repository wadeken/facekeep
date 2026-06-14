"""Tests for `facekeep restore` as the "restore to a standard file" path — ROADMAP Phase 4.

Restore is the escape hatch from the proprietary `.fkeep` container: a .fkeep
must never be a dead end, so `restore` brings every photo back as a *standard*
image — a universal `.jpg` (default) or a real `.avif`/`.jxl` — and a whole
folder can be un-fkeep'd in one command.

What these tests pin:

* Default output is `<stem>_restored.jpg`, decodable at the original size.
* `--format avif` writes a *real* AVIF (decodes via `encoders.decode`, not just a
  file with the right name — OpenCV can't write AVIF here, so this proves it went
  through the faithful codec), round-tripping the original dimensions. Skips when
  the avif codec isn't installed (graceful, like the corpus/YuNet-offline tests).
* EXIF survives a restore-to-JPEG (re-embedded from the stored `exif.bin`).
* Batch: a folder of `.fkeep`s yields one output each and an `N/N restored`
  summary.
* A requested format whose codec is missing fails fast (exit 2) with an
  actionable hint, *before* writing anything — not a mid-batch crash.
* `--preview` (bicubic, no AI) works and honors the format.
* Pointing `restore` at a non-`.fkeep` reports cleanly.

Aggressive mode packs the .fkeep with OpenCV (JPEG/PNG), so building the input
needs no AVIF/JXL codec — only the bundled Haar detector (offline, no download).
The avif-output assertions are guarded on `codec_available("avif")`.
"""

import io

import cv2
import numpy as np
import piexif
import pytest
from click.testing import CliRunner
from PIL import Image

from facekeep import encoders
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import _fkeep_path, write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig

AVIF = pytest.mark.skipif(
    not encoders.codec_available("avif"),
    reason="avif codec not installed",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _build_fkeep(face_image, dst_dir, name="photo"):
    """Compress ``face_image`` to a real .fkeep under ``dst_dir``; return its Path."""
    photo = compress_photo(str(face_image), FaceKeepConfig())
    write_fkeep(photo, str(dst_dir / name))
    path = _fkeep_path(str(dst_dir / name))
    assert path.exists()
    return path


def _exif_jpeg(tmp_path):
    """A small JPEG carrying a real EXIF block (orientation tag), as a Path."""
    img = np.full((240, 320, 3), 200, np.uint8)
    cv2.rectangle(img, (130, 90), (190, 160), (170, 160, 155), -1)  # a blob
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    exif = piexif.dump({"0th": {piexif.ImageIFD.Orientation: 1}})
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=92, exif=exif)
    path = tmp_path / "with_exif.jpg"
    path.write_bytes(buf.getvalue())
    return path


# --------------------------------------------------------------------------- #
# default JPEG restore
# --------------------------------------------------------------------------- #

def test_restore_default_jpg(face_image, tmp_path):
    """`restore -o out.jpg` writes a decodable JPEG at the original size."""
    fkeep = _build_fkeep(face_image, tmp_path)
    out = tmp_path / "out.jpg"
    res = CliRunner().invoke(cli, ["restore", str(fkeep), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists()
    img = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert img is not None
    orig = cv2.imread(str(face_image), cv2.IMREAD_COLOR)
    assert img.shape == orig.shape


def test_restore_auto_target_name(face_image, tmp_path):
    """With no -o, the output is `<stem>_restored.jpg` next to the input."""
    fkeep = _build_fkeep(face_image, tmp_path, name="2024.05.20_trip")
    res = CliRunner().invoke(cli, ["restore", str(fkeep)])
    assert res.exit_code == 0, res.output
    # Dotted stem preserved, not mangled by suffix replacement.
    expected = tmp_path / "2024.05.20_trip_restored.jpg"
    assert expected.exists()


# --------------------------------------------------------------------------- #
# AVIF restore — a *real* AVIF via the faithful codec
# --------------------------------------------------------------------------- #

@AVIF
def test_restore_format_avif_is_real_avif(face_image, tmp_path):
    """`--format avif` writes a genuine AVIF that decodes at the original size."""
    fkeep = _build_fkeep(face_image, tmp_path)
    out = tmp_path / "out.avif"
    res = CliRunner().invoke(
        cli, ["restore", str(fkeep), "-o", str(out), "--format", "avif"]
    )
    assert res.exit_code == 0, res.output
    assert out.exists() and out.stat().st_size > 0
    # The proof it isn't a misnamed JPEG: it decodes through the AVIF codec.
    decoded = encoders.decode(out.read_bytes())
    orig = cv2.imread(str(face_image), cv2.IMREAD_COLOR)
    assert decoded.shape == orig.shape


@AVIF
def test_restore_explicit_avif_extension_overrides_format(face_image, tmp_path):
    """A `-o *.avif` path is honored even without `--format` (extension wins)."""
    fkeep = _build_fkeep(face_image, tmp_path)
    out = tmp_path / "explicit.avif"
    res = CliRunner().invoke(cli, ["restore", str(fkeep), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists()
    # Real AVIF, not an OpenCV write of a .avif-named file.
    assert encoders.decode(out.read_bytes()) is not None


@AVIF
def test_restore_avif_quality_changes_size(face_image, tmp_path):
    """Lower `--quality` yields a smaller AVIF (the knob reaches the encoder)."""
    fkeep = _build_fkeep(face_image, tmp_path)
    lo = tmp_path / "lo.avif"
    hi = tmp_path / "hi.avif"
    r = CliRunner()
    assert r.invoke(cli, ["restore", str(fkeep), "-o", str(lo),
                          "-f", "avif", "-q", "30"]).exit_code == 0
    assert r.invoke(cli, ["restore", str(fkeep), "-o", str(hi),
                          "-f", "avif", "-q", "90"]).exit_code == 0
    assert lo.stat().st_size < hi.stat().st_size


# --------------------------------------------------------------------------- #
# EXIF survives restore-to-JPEG
# --------------------------------------------------------------------------- #

def test_restore_jpeg_preserves_exif(tmp_path):
    """EXIF stored in the .fkeep is re-embedded into a restored JPEG."""
    src = _exif_jpeg(tmp_path)
    fkeep = _build_fkeep(src, tmp_path, name="exif_src")
    out = tmp_path / "restored.jpg"
    res = CliRunner().invoke(cli, ["restore", str(fkeep), "-o", str(out)])
    assert res.exit_code == 0, res.output
    # The restored JPEG carries an EXIF block again.
    exif = piexif.load(str(out))
    assert exif["0th"]  # non-empty -> EXIF round-tripped through the .fkeep


# --------------------------------------------------------------------------- #
# batch: un-fkeep a whole folder
# --------------------------------------------------------------------------- #

def test_restore_folder_batch_summary(face_image, tmp_path):
    """Restoring a folder produces one output per .fkeep and an N/N summary."""
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for i in range(3):
        _build_fkeep(face_image, src_dir, name=f"p{i}")
    out_dir = tmp_path / "out"

    res = CliRunner().invoke(cli, ["restore", str(src_dir), "-o", str(out_dir)])
    assert res.exit_code == 0, res.output
    produced = sorted(p.name for p in out_dir.glob("*_restored.jpg"))
    assert produced == ["p0_restored.jpg", "p1_restored.jpg", "p2_restored.jpg"]
    assert "3/3 restored" in res.output


# --------------------------------------------------------------------------- #
# missing-codec guard: fail fast, before writing
# --------------------------------------------------------------------------- #

def test_restore_missing_codec_fails_fast(face_image, tmp_path, monkeypatch):
    """`--format avif` with no avif codec exits 2 with a hint and writes nothing."""
    fkeep = _build_fkeep(face_image, tmp_path)
    out = tmp_path / "out.avif"

    real = encoders.codec_available
    monkeypatch.setattr(
        encoders, "codec_available",
        lambda c: False if c == "avif" else real(c),
    )

    res = CliRunner().invoke(
        cli, ["restore", str(fkeep), "-o", str(out), "--format", "avif"]
    )
    assert res.exit_code == 2
    assert "avif codec is not installed" in res.output
    assert not out.exists()  # nothing written


# --------------------------------------------------------------------------- #
# preview (bicubic, no AI)
# --------------------------------------------------------------------------- #

def test_restore_preview_jpg(face_image, tmp_path):
    """`--preview` restores via bicubic and writes the requested file."""
    fkeep = _build_fkeep(face_image, tmp_path)
    out = tmp_path / "prev.jpg"
    res = CliRunner().invoke(cli, ["restore", str(fkeep), "-o", str(out), "--preview"])
    assert res.exit_code == 0, res.output
    assert out.exists()
    assert "[preview]" in res.output


# --------------------------------------------------------------------------- #
# non-.fkeep input
# --------------------------------------------------------------------------- #

def test_restore_non_fkeep_input_reports(face_image, tmp_path):
    """Pointing `restore` at a plain image (no .fkeep) reports and exits non-zero."""
    res = CliRunner().invoke(cli, ["restore", str(face_image)])
    assert res.exit_code == 1
    assert "No .fkeep files found" in res.output


# --------------------------------------------------------------------------- #
# Restorer API: format routing is independent of the CLI
# --------------------------------------------------------------------------- #

@AVIF
def test_restorer_write_routes_avif_through_codec(face_image, tmp_path):
    """Restorer.restore(out.avif) goes through the codec, not cv2.imwrite."""
    fkeep = _build_fkeep(face_image, tmp_path)
    out = tmp_path / "api.avif"
    restorer = Restorer(FaceKeepConfig().aggressive)
    arr = restorer.restore(str(fkeep), str(out), quality=55)
    assert out.exists()
    # The returned array is BGR full-res; the file decodes as a real AVIF.
    assert arr.ndim == 3 and arr.shape[2] == 3
    assert encoders.decode(out.read_bytes()).shape == arr.shape
