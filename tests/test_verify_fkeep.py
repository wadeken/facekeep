"""Tests for `facekeep verify file.fkeep` — ROADMAP Phase 4.

`verify_fkeep` is a *structural-integrity* check of a .fkeep container: it
confirms the archive opens, the manifest parses, every entry the manifest
promises (background, each face crop + mask, thumbnail) is present and
decodable, the counts line up, and the dimensions are sane.

The honesty contract these tests pin:

* A readable-but-inconsistent file returns ``ok=False`` with the specifics in
  ``problems`` (no exception); only a truly unopenable file raises
  ``FormatError``.
* The manifest stores the *original input file's* SHA-256, but the original
  pixels are gone, so the hash cannot be recomputed from the .fkeep alone:
  ``hash_match`` stays ``None`` unless the caller supplies the original to match
  against — never a fabricated pass.

Aggressive mode packs the .fkeep with OpenCV (JPEG/PNG), so these tests need no
AVIF/JXL codec — only the bundled Haar detector (offline, no download).
"""

import json
import zipfile

import pytest
from click.testing import CliRunner

from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import (
    _fkeep_path,
    read_fkeep_info,
    verify_fkeep,
    write_fkeep,
)
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import FormatError


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _build_fkeep(face_image, tmp_path, name="photo"):
    """Compress ``face_image`` to a real .fkeep; return its Path."""
    photo = compress_photo(str(face_image), FaceKeepConfig())
    write_fkeep(photo, str(tmp_path / name))
    path = _fkeep_path(str(tmp_path / name))
    assert path.exists()
    return path


def _rewrite_without(src_fkeep, dst_fkeep, drop_names):
    """Copy a .fkeep to dst, omitting any archive members in ``drop_names``."""
    with zipfile.ZipFile(src_fkeep, "r") as zin, \
            zipfile.ZipFile(dst_fkeep, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            if item in drop_names:
                continue
            zout.writestr(item, zin.read(item))


def _rewrite_with_manifest(src_fkeep, dst_fkeep, mutate):
    """Copy a .fkeep to dst with ``mutate(manifest_dict)`` applied to the manifest."""
    with zipfile.ZipFile(src_fkeep, "r") as zin:
        members = {n: zin.read(n) for n in zin.namelist()}
    manifest = json.loads(members["manifest.json"])
    mutate(manifest)
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with zipfile.ZipFile(dst_fkeep, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)


# --------------------------------------------------------------------------- #
# verify_fkeep: happy path
# --------------------------------------------------------------------------- #

def test_verify_good_fkeep_ok(face_image, tmp_path):
    """A freshly written .fkeep verifies clean and self-consistent."""
    fkeep = _build_fkeep(face_image, tmp_path)
    rep = verify_fkeep(str(fkeep))

    assert rep.ok, rep.problems
    assert rep.problems == []
    assert rep.faces_declared >= 1
    assert rep.crops_found == rep.faces_declared
    assert rep.masks_found == rep.faces_declared
    assert rep.thumbnail_ok
    # Background is non-empty and no larger than the declared original.
    assert rep.background_size is not None
    bw, bh = rep.background_size
    assert bw > 0 and bh > 0
    if rep.original_size is not None:
        assert bw <= rep.original_size[0] and bh <= rep.original_size[1]
    # No original passed -> we do not invent a hash verdict.
    assert rep.stored_hash  # the manifest records one
    assert rep.hash_match is None


# --------------------------------------------------------------------------- #
# verify_fkeep: --original hash matching
# --------------------------------------------------------------------------- #

def test_verify_original_match(face_image, tmp_path):
    """Passing the real source file makes hash_match True."""
    fkeep = _build_fkeep(face_image, tmp_path)
    rep = verify_fkeep(str(fkeep), original_path=str(face_image))
    assert rep.hash_match is True
    assert rep.ok, rep.problems
    # The stored hash equals the actual file hash.
    import hashlib
    actual = hashlib.sha256(face_image.read_bytes()).hexdigest()
    assert rep.stored_hash == actual


def test_verify_original_mismatch(face_image, plain_image, tmp_path):
    """A different original file makes hash_match False and the report not ok."""
    fkeep = _build_fkeep(face_image, tmp_path)
    rep = verify_fkeep(str(fkeep), original_path=str(plain_image))
    assert rep.hash_match is False
    assert rep.ok is False
    assert any("hash mismatch" in p for p in rep.problems)


# --------------------------------------------------------------------------- #
# verify_fkeep: corruption / inconsistency detection
# --------------------------------------------------------------------------- #

def test_verify_detects_missing_face_crop(face_image, tmp_path):
    """Dropping a face crop is reported (and the same file untouched is ok)."""
    good = _build_fkeep(face_image, tmp_path, name="good")
    # Anti-false-green: the untouched file verifies clean.
    assert verify_fkeep(str(good)).ok

    bad = tmp_path / "bad.fkeep"
    _rewrite_without(good, bad, {"face_000.jpg", "face_000.png"})

    rep = verify_fkeep(str(bad))
    assert rep.ok is False
    assert rep.crops_found < rep.faces_declared
    assert any("face crop 000" in p for p in rep.problems)


def test_verify_detects_missing_mask(face_image, tmp_path):
    """Dropping a face mask is reported."""
    good = _build_fkeep(face_image, tmp_path, name="good")
    bad = tmp_path / "bad.fkeep"
    _rewrite_without(good, bad, {"face_mask_000.png"})

    rep = verify_fkeep(str(bad))
    assert rep.ok is False
    assert rep.masks_found < rep.faces_declared
    assert any("face mask" in p for p in rep.problems)


def test_verify_detects_missing_background(face_image, tmp_path):
    """A .fkeep without a background is reported (not crashed)."""
    good = _build_fkeep(face_image, tmp_path, name="good")
    bad = tmp_path / "bad.fkeep"
    _rewrite_without(good, bad, {"background.jpg"})

    rep = verify_fkeep(str(bad))
    assert rep.ok is False
    assert rep.background_size is None
    assert any("background" in p for p in rep.problems)


def test_verify_detects_inconsistent_face_count(face_image, tmp_path):
    """A manifest claiming more faces than there are crops is caught."""
    good = _build_fkeep(face_image, tmp_path, name="good")
    declared = len(read_fkeep_info(str(good))["faces"])

    bad = tmp_path / "bad.fkeep"

    def _add_phantom_face(manifest):
        # Append a phantom face entry with no corresponding crop/mask members.
        manifest["faces"].append({
            "id": declared,
            "bbox": [0, 0, 10, 10],
            "padded_bbox": [0, 0, 12, 12],
            "confidence": 0.9,
        })

    _rewrite_with_manifest(good, bad, _add_phantom_face)

    rep = verify_fkeep(str(bad))
    assert rep.ok is False
    assert rep.faces_declared == declared + 1
    assert rep.crops_found == declared  # the phantom has no crop
    assert any("count" in p for p in rep.problems)


def test_verify_detects_malformed_bbox(face_image, tmp_path):
    """A face with a degenerate bbox is reported."""
    good = _build_fkeep(face_image, tmp_path, name="good")
    bad = tmp_path / "bad.fkeep"

    def _break_bbox(manifest):
        manifest["faces"][0]["bbox"] = [10, 10, 5, 5]  # x2<x1, y2<y1

    _rewrite_with_manifest(good, bad, _break_bbox)

    rep = verify_fkeep(str(bad))
    assert rep.ok is False
    assert any("bbox" in p for p in rep.problems)


def test_verify_not_a_zip_raises_formaterror(tmp_path):
    """A plain text file with a .fkeep name raises FormatError, not a crash."""
    bogus = tmp_path / "bogus.fkeep"
    bogus.write_text("this is not a zip archive")
    with pytest.raises(FormatError):
        verify_fkeep(str(bogus))


def test_verify_background_larger_than_original(face_image, tmp_path):
    """A background bigger than the declared original is flagged."""
    good = _build_fkeep(face_image, tmp_path, name="good")
    bad = tmp_path / "bad.fkeep"

    def _shrink_declared_original(manifest):
        manifest["original"]["width"] = 1
        manifest["original"]["height"] = 1

    _rewrite_with_manifest(good, bad, _shrink_declared_original)

    rep = verify_fkeep(str(bad))
    assert rep.ok is False
    assert any("larger than the declared original" in p for p in rep.problems)


# --------------------------------------------------------------------------- #
# CLI: facekeep verify
# --------------------------------------------------------------------------- #

def test_cli_verify_good(face_image, tmp_path):
    """`verify` on a good file exits 0 and prints OK."""
    fkeep = _build_fkeep(face_image, tmp_path)
    result = CliRunner().invoke(cli, ["verify", str(fkeep)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_cli_verify_corrupt_exits_nonzero(face_image, tmp_path):
    """`verify` on a corrupt file exits non-zero and prints FAILED."""
    good = _build_fkeep(face_image, tmp_path, name="good")
    bad = tmp_path / "bad.fkeep"
    _rewrite_without(good, bad, {"background.jpg"})

    result = CliRunner().invoke(cli, ["verify", str(bad)])
    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_cli_verify_original_match(face_image, tmp_path):
    """`--original` with the real source reports MATCHES and exits 0."""
    fkeep = _build_fkeep(face_image, tmp_path)
    result = CliRunner().invoke(
        cli, ["verify", str(fkeep), "--original", str(face_image)]
    )
    assert result.exit_code == 0, result.output
    assert "MATCHES" in result.output


def test_cli_verify_original_mismatch_exits_nonzero(face_image, plain_image, tmp_path):
    """`--original` with a different file exits non-zero."""
    fkeep = _build_fkeep(face_image, tmp_path)
    result = CliRunner().invoke(
        cli, ["verify", str(fkeep), "--original", str(plain_image)]
    )
    assert result.exit_code == 1
    assert "MISMATCH" in result.output


def test_cli_verify_non_fkeep_extension(tmp_path):
    """Pointing `verify` at a non-.fkeep file gives a clear message, exit 2."""
    not_fkeep = tmp_path / "image.avif"
    not_fkeep.write_bytes(b"\x00\x01\x02")  # must exist for click's exists=True
    result = CliRunner().invoke(cli, ["verify", str(not_fkeep)])
    assert result.exit_code == 2
    assert "not a .fkeep" in result.output


# --------------------------------------------------------------------------- #
# CLI: status marks degrade to ASCII on legacy consoles (cp950 etc.)
# --------------------------------------------------------------------------- #

def test_marks_ascii_when_stdout_cannot_encode_glyphs(monkeypatch):
    """A stdout whose codepage can't encode the check marks gets ASCII ones."""
    import sys

    import facekeep.cli as cli_mod

    class _Cp950Stream:
        encoding = "cp950"

    monkeypatch.setattr(sys, "stdout", _Cp950Stream())
    assert cli_mod._marks() == ("OK", "x")


def test_marks_glyphs_on_capable_or_encodingless_stdout(monkeypatch):
    """UTF-8 (or an encoding-less StringIO, as test runners use) keeps glyphs."""
    import sys

    import facekeep.cli as cli_mod

    class _Utf8Stream:
        encoding = "utf-8"

    monkeypatch.setattr(sys, "stdout", _Utf8Stream())
    assert cli_mod._marks() == ("✓", "✗")

    class _NoEncodingStream:
        pass

    monkeypatch.setattr(sys, "stdout", _NoEncodingStream())
    assert cli_mod._marks() == ("✓", "✗")


def test_cli_verify_survives_cp950_console(face_image, tmp_path, monkeypatch):
    """The reported crash: `verify` on a genuinely cp950-encoded stdout.

    Before the fix this raised UnicodeEncodeError on the first check-mark line
    (after printing "OK") and exited 1 despite a successful verification. Drive
    the command callback against a real cp950 TextIOWrapper so the encode path
    is exercised for real, not simulated.
    """
    import io
    import sys

    from facekeep.cli import verify as verify_cmd

    fkeep = _build_fkeep(face_image, tmp_path)
    buf = io.BytesIO()
    cp950_stdout = io.TextIOWrapper(buf, encoding="cp950")
    monkeypatch.setattr(sys, "stdout", cp950_stdout)

    with pytest.raises(SystemExit) as exc:
        verify_cmd.callback(fkeep_path=str(fkeep), original_path=None)
    assert exc.value.code == 0

    cp950_stdout.flush()
    out = buf.getvalue().decode("cp950")
    assert ": OK" in out
    assert "faces:" in out  # the line that used to crash printed through
