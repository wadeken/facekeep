"""Background stored as AVIF/JXL instead of JPEG q85 — ROADMAP Phase 8.3.

Aggressive mode can store the downsampled background as AVIF or JXL (4:2:0)
instead of the hard-coded cv2 JPEG: JPEG block artifacts in the background are
exactly what the SR upscaler amplifies into false texture on restore, and the
modern codecs are smaller at equal quality. This is opt-in via
``AggressiveConfig.bg_codec`` and isolated to ``format.py``; the default stays
``jpg`` so existing ``.fkeep`` files and the default output bytes are unchanged.

What these tests pin:

* the chosen codec is what actually lands in the archive
  (``background.avif`` / ``.jxl``) and it **round-trips** through ``read_fkeep``
  back to a BGR background — important because the bundled OpenCV build cannot
  decode the AVIF/JXL that pillow writes, so the reader must route those two
  through the faithful-mode codec (Pillow), not ``cv2.imdecode``;
* the default path is byte-identical to the old hard-coded JPEG
  (anti-regression), the manifest records ``settings.bg_codec`` and bumps the
  schema to 1.5.0;
* a downgraded v1.4.0 file (no ``bg_codec`` key) still reads/verifies/restores
  (tolerant-by-structure backward compatibility);
* ``verify_fkeep`` accepts each codec, still catches a missing background, and
  reports (not crashes on) an undecodable avif background;
* ``settings_fingerprint`` busts on ``bg_codec`` (aggressive only);
  ``validate()`` rejects an unknown codec; YAML round-trips; dry-run size
  parity holds (the shared ``_write_archive`` rule).

**Honest size note (the face_codec lesson).** AVIF/JXL win on smooth content;
AVIF can *lose* to JPEG on noisy/artifact-laden content (AV1-intra's worst
case). So the strict size assertions here use a smooth synthetic background —
JXL broadly, AVIF scoped to its good case — and that is also why the shipped
default stays ``jpg``.
"""

import json
import zipfile

import cv2
import numpy as np
import pytest

from facekeep import encoders, metrics
from facekeep.aggressive.compressor import CompressedPhoto, compress_photo
from facekeep.aggressive.format import (
    _fkeep_path,
    _jpg,
    read_fkeep,
    read_fkeep_info,
    verify_fkeep,
    write_fkeep,
)
from facekeep.aggressive.restorer import Restorer
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import ConfigError, FormatError
from facekeep.index import settings_fingerprint

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)
requires_jxl = pytest.mark.skipif(
    not encoders.codec_available("jxl"), reason="JXL encoder not installed"
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _smooth_background(w: int = 960, h: int = 640) -> np.ndarray:
    """A smooth, codec-friendly synthetic background (no per-pixel noise).

    Soft gradients lightly blurred — the regime where AVIF/JXL beat JPEG.
    Deliberately *not* random noise, which is AV1-intra's worst case and would
    invert the size relationship (see the module docstring).
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.zeros((h, w, 3), np.uint8)
    bg[..., 0] = (120 + 60 * np.sin(xx / 180) + 20 * np.cos(yy / 140)).clip(0, 255)
    bg[..., 1] = (110 + 50 * np.sin((xx + yy) / 220)).clip(0, 255)
    bg[..., 2] = (90 + 40 * np.cos(xx / 260) + 30 * np.sin(yy / 170)).clip(0, 255)
    return cv2.GaussianBlur(bg, (0, 0), 2.0)


def _photo_with_bg(bg: np.ndarray, codec: str) -> CompressedPhoto:
    """Build a zero-face CompressedPhoto around ``bg`` with the given bg_codec.

    Bypasses the detector so the background content is controlled and
    deterministic (the codec size/round-trip behavior is what's under test).
    The declared original is 4x the background (bg_scale 0.25), keeping
    verify_fkeep's background-not-larger-than-original check honest.
    """
    h, w = bg.shape[:2]
    cfg = FaceKeepConfig()
    cfg.aggressive.bg_codec = codec
    return CompressedPhoto(
        original_filename="p.jpg", original_width=w * 4, original_height=h * 4,
        original_size_bytes=999, original_hash="0" * 64, original_orientation=1,
        exif=None,
        background=bg.copy(),
        face_crops=[], face_masks=[], faces=[],
        thumbnail=np.full((256, 256, 3), 128, np.uint8),
        effective_bg_scale=0.25, config=cfg.aggressive,
    )


def _bg_member_and_bytes(fkeep_path) -> tuple:
    """Return (member_name, byte_len) of the background member in a .fkeep."""
    with zipfile.ZipFile(fkeep_path) as zf:
        name = next(n for n in zf.namelist() if n.startswith("background."))
        return name, len(zf.read(name))


def _write(photo, tmp_path, name):
    p = _fkeep_path(str(tmp_path / name))
    write_fkeep(photo, str(tmp_path / name))
    assert p.exists()
    return p


def _rewrite_members(fkeep, *, drop=(), replace=None, edit_manifest=None):
    """Rewrite a .fkeep in place: drop/replace members and/or edit the manifest."""
    with zipfile.ZipFile(fkeep) as zf:
        members = {n: zf.read(n) for n in zf.namelist() if n not in set(drop)}
    if replace:
        members.update(replace)
    if edit_manifest is not None:
        manifest = json.loads(members["manifest.json"])
        edit_manifest(manifest)
        members["manifest.json"] = json.dumps(manifest)
    with zipfile.ZipFile(fkeep, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, b in members.items():
            zf.writestr(n, b)
    return str(fkeep)


# --------------------------------------------------------------------------- #
# the chosen codec lands in the archive + round-trips
# --------------------------------------------------------------------------- #

@requires_avif
def test_avif_bg_member_and_roundtrip(tmp_path):
    """bg_codec='avif' stores background.avif and it decodes back to the bg."""
    bg = _smooth_background()
    fkeep = _write(_photo_with_bg(bg, "avif"), tmp_path, "avif")

    name, _ = _bg_member_and_bytes(fkeep)
    assert name == "background.avif"

    data = read_fkeep(str(fkeep))
    out = data["background"]
    assert out.shape == bg.shape and out.dtype == np.uint8
    # q85 4:2:0 on smooth content is comfortably high-fidelity.
    assert metrics.ssim(bg, out) > 0.95


@requires_jxl
def test_jxl_bg_member_and_roundtrip(tmp_path):
    """bg_codec='jxl' stores background.jxl and it decodes back to the bg."""
    bg = _smooth_background()
    fkeep = _write(_photo_with_bg(bg, "jxl"), tmp_path, "jxl")

    name, _ = _bg_member_and_bytes(fkeep)
    assert name == "background.jxl"

    data = read_fkeep(str(fkeep))
    out = data["background"]
    assert out.shape == bg.shape and out.dtype == np.uint8
    assert metrics.ssim(bg, out) > 0.95


# --------------------------------------------------------------------------- #
# size: the win (content-dependent — asserted honestly)
# --------------------------------------------------------------------------- #

@requires_jxl
def test_jxl_bg_smaller_than_jpeg(tmp_path):
    """On smooth content, the JXL background is strictly smaller than JPEG."""
    bg = _smooth_background()
    jpg = _write(_photo_with_bg(bg, "jpg"), tmp_path, "as_jpg")
    jxl = _write(_photo_with_bg(bg, "jxl"), tmp_path, "as_jxl")

    _, jpg_bytes = _bg_member_and_bytes(jpg)
    _, jxl_bytes = _bg_member_and_bytes(jxl)
    assert jxl_bytes < jpg_bytes, (jxl_bytes, jpg_bytes)


@requires_avif
def test_avif_bg_smaller_than_jpeg_on_smooth_content(tmp_path):
    """On clean smooth content (AVIF's good case) the AVIF background wins too.

    Scoped to smooth content on purpose — AVIF can lose to JPEG on a noisy
    background (its worst case), which is why the default stays jpg and the
    broad strict assertion above uses JXL.
    """
    bg = _smooth_background()
    jpg = _write(_photo_with_bg(bg, "jpg"), tmp_path, "s_jpg")
    avif = _write(_photo_with_bg(bg, "avif"), tmp_path, "s_avif")

    _, jpg_bytes = _bg_member_and_bytes(jpg)
    _, avif_bytes = _bg_member_and_bytes(avif)
    assert avif_bytes < jpg_bytes, (avif_bytes, jpg_bytes)


# --------------------------------------------------------------------------- #
# backward compatibility: the default path is unchanged, old files still work
# --------------------------------------------------------------------------- #

def test_default_bg_is_jpg_and_manifest_records_codec(tmp_path):
    """The default config still stores background.jpg; the manifest says so.

    The schema version is the *current* writer schema (1.7.0 = the preset
    bump), not bg_codec's own 1.5.0.
    """
    bg = _smooth_background(480, 320)
    fkeep = _write(_photo_with_bg(bg, "jpg"), tmp_path, "default")

    name, _ = _bg_member_and_bytes(fkeep)
    assert name == "background.jpg"
    info = read_fkeep_info(str(fkeep))
    assert info["settings"]["bg_codec"] == "jpg"
    assert info["version"] == "1.7.0"


def test_default_bg_bytes_byte_identical_to_cv2_jpeg(tmp_path):
    """The default jpg path produces byte-for-byte the old cv2 JPEG encode."""
    bg = _smooth_background(480, 320)
    photo = _photo_with_bg(bg, "jpg")
    fkeep = _write(photo, tmp_path, "bytes")

    with zipfile.ZipFile(fkeep) as zf:
        member = zf.read("background.jpg")
    assert member == _jpg(bg, photo.config.bg_quality)


def test_v140_file_without_bg_codec_reads_verifies_restores(tmp_path):
    """A downgraded v1.4.0 manifest (no bg_codec key) still works end-to-end."""
    bg = _smooth_background(240, 160)
    fkeep = _write(_photo_with_bg(bg, "jpg"), tmp_path, "old")

    def _downgrade(manifest):
        manifest["version"] = "1.4.0"
        manifest["settings"].pop("bg_codec", None)

    _rewrite_members(fkeep, edit_manifest=_downgrade)

    data = read_fkeep(str(fkeep))
    assert data["background"].shape == bg.shape

    rep = verify_fkeep(str(fkeep))
    assert rep.ok, rep.problems

    out = Restorer().restore(str(fkeep))
    assert out.shape[:2] == (bg.shape[0] * 4, bg.shape[1] * 4)


# --------------------------------------------------------------------------- #
# verify_fkeep: accepts each codec, catches missing/undecodable backgrounds
# --------------------------------------------------------------------------- #

@requires_avif
def test_verify_accepts_avif_bg_from_real_compress(face_image, tmp_path):
    """A real compress with bg_codec='avif' verifies clean (avif member)."""
    cfg = FaceKeepConfig()
    cfg.aggressive.bg_codec = "avif"
    photo = compress_photo(str(face_image), cfg)
    fkeep = _write(photo, tmp_path, "avif_real")

    with zipfile.ZipFile(fkeep) as zf:
        assert "background.avif" in zf.namelist()
    rep = verify_fkeep(str(fkeep))
    assert rep.ok, rep.problems
    assert rep.background_size is not None


@requires_jxl
def test_verify_accepts_jxl_bg(tmp_path):
    bg = _smooth_background(480, 320)
    fkeep = _write(_photo_with_bg(bg, "jxl"), tmp_path, "jxl_v")
    rep = verify_fkeep(str(fkeep))
    assert rep.ok, rep.problems
    assert rep.background_size == (480, 320)


@requires_avif
def test_verify_catches_missing_avif_bg(tmp_path):
    """Dropping the background.avif member is reported, not crashed on."""
    bg = _smooth_background(240, 160)
    fkeep = _write(_photo_with_bg(bg, "avif"), tmp_path, "miss")
    _rewrite_members(fkeep, drop=("background.avif",))

    rep = verify_fkeep(str(fkeep))
    assert rep.ok is False
    assert any("background" in p for p in rep.problems)


@requires_avif
def test_verify_reports_undecodable_avif_bg(tmp_path):
    """Garbage avif bytes are a *problem* (EncodingError caught), not a crash."""
    bg = _smooth_background(240, 160)
    fkeep = _write(_photo_with_bg(bg, "avif"), tmp_path, "garb")
    _rewrite_members(fkeep, replace={"background.avif": b"not an avif"})

    rep = verify_fkeep(str(fkeep))
    assert rep.ok is False
    assert any("does not decode" in p for p in rep.problems)


def test_read_fkeep_missing_background_raises_format_error(tmp_path):
    """read_fkeep on a file with no background member raises FormatError."""
    bg = _smooth_background(240, 160)
    fkeep = _write(_photo_with_bg(bg, "jpg"), tmp_path, "nobg")
    _rewrite_members(fkeep, drop=("background.jpg",))

    with pytest.raises(FormatError, match="background"):
        read_fkeep(str(fkeep))


# --------------------------------------------------------------------------- #
# restore end-to-end + dry-run parity
# --------------------------------------------------------------------------- #

@requires_avif
def test_restore_from_avif_bg(tmp_path):
    """An avif-bg .fkeep restores (bicubic) to the declared original size."""
    bg = _smooth_background(240, 160)
    photo = _photo_with_bg(bg, "avif")
    fkeep = _write(photo, tmp_path, "rest")

    out = Restorer().restore(str(fkeep))
    assert out.shape[:2] == (photo.original_height, photo.original_width)
    assert out.dtype == np.uint8


@requires_avif
def test_dry_run_size_parity_avif_bg(tmp_path):
    """Dry-run size equals the real written size with an avif background."""
    photo = _photo_with_bg(_smooth_background(240, 160), "avif")
    est = write_fkeep(photo, str(tmp_path / "x"), dry_run=True)
    real = write_fkeep(photo, str(tmp_path / "x"))
    assert est == real


# --------------------------------------------------------------------------- #
# config validation + YAML + fingerprint
# --------------------------------------------------------------------------- #

def test_validate_rejects_unknown_bg_codec():
    cfg = FaceKeepConfig()
    cfg.aggressive.bg_codec = "webp"
    with pytest.raises(ConfigError, match="bg_codec"):
        cfg.validate()


@pytest.mark.parametrize("codec", ["jpg", "avif", "jxl"])
def test_validate_accepts_known_bg_codecs(codec):
    cfg = FaceKeepConfig()
    cfg.aggressive.bg_codec = codec
    cfg.validate()  # must not raise


def test_yaml_roundtrip_bg_codec(tmp_path):
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"
    cfg.aggressive.bg_codec = "jxl"
    p = tmp_path / "c.yaml"
    cfg.save(p)
    loaded = FaceKeepConfig.load(p)
    assert loaded.aggressive.bg_codec == "jxl"


def test_fingerprint_busts_on_bg_codec():
    base = FaceKeepConfig(mode="aggressive")
    changed = FaceKeepConfig(mode="aggressive")
    changed.aggressive.bg_codec = "avif"
    assert settings_fingerprint(base) != settings_fingerprint(changed)


def test_faithful_fingerprint_ignores_bg_codec():
    base = FaceKeepConfig()  # faithful
    changed = FaceKeepConfig()
    changed.aggressive.bg_codec = "avif"
    assert settings_fingerprint(base) == settings_fingerprint(changed)
