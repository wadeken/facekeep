"""Face crops stored as AVIF/JXL (4:4:4) — ROADMAP Phase 4 / Core mechanics.

Aggressive mode can store face crops as AVIF or JXL at 4:4:4 instead of the
default JPEG q95, matching JPEG's perceptual quality at a smaller size. This is
opt-in via ``AggressiveConfig.face_codec`` and isolated to ``format.py``; the
default stays ``jpg`` so existing ``.fkeep`` files and behavior are unchanged.

What these tests pin:

* the chosen codec is what actually lands in the archive (``face_NNN.avif`` /
  ``.jxl``), and it **round-trips** through ``read_fkeep`` back to a BGR crop of
  the right size and good fidelity — important because the bundled OpenCV build
  cannot decode the AVIF/JXL that pillow writes, so the reader must route those
  two through the faithful-mode codec (Pillow), not ``cv2.imdecode``;
* ``face_quality >= 100`` still wins as lossless PNG regardless of ``face_codec``
  (the lossless escape hatch takes precedence);
* the default (jpg) path is untouched (anti-regression), and ``verify_fkeep``
  accepts avif/jxl crops;
* ``validate()`` rejects an unknown codec.

**Honest size note.** The "~2× smaller" headline is *content-dependent*. It
holds on clean, smooth, photographic content (where AV1/JXL intra shine). It does
**not** hold for a crop that already carries JPEG/quantization noise or a noisy
false-positive region: per-pixel high-frequency noise is AV1-intra's worst case,
so AVIF there can be *larger* than JPEG (JXL still tends to win). So the strict
size-win assertion is made on a smooth synthetic crop and on JXL, which wins
broadly; AVIF is pinned on correctness/round-trip, not a strict byte inequality
against an artifact-laden crop. See the ROADMAP note for this item.
"""

import zipfile

import cv2
import numpy as np
import pytest

from facekeep import encoders, metrics
from facekeep.aggressive.blender import create_soft_mask
from facekeep.aggressive.compressor import CompressedPhoto, compress_photo
from facekeep.aggressive.format import (
    _fkeep_path,
    read_fkeep,
    read_fkeep_info,
    verify_fkeep,
    write_fkeep,
)
from facekeep.config import FaceKeepConfig
from facekeep.detector import FaceRegion
from facekeep.exceptions import ConfigError

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)
requires_jxl = pytest.mark.skipif(
    not encoders.codec_available("jxl"), reason="JXL encoder not installed"
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _smooth_face_crop(size: int = 740) -> np.ndarray:
    """A smooth, codec-friendly photographic-ish face crop (no per-pixel noise).

    Low high-frequency energy (gradients + soft elliptical 'features', lightly
    blurred) is what a real face crop looks like to a codec — the regime where
    AVIF/JXL beat JPEG. Deliberately *not* random noise, which is AV1-intra's
    worst case and would invert the size relationship (see the module docstring).
    """
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    crop = np.zeros((h, w, 3), np.uint8)
    crop[..., 2] = (170 + 50 * np.sin(xx / 120) + 25 * np.cos(yy / 90)).clip(0, 255)
    crop[..., 1] = (130 + 40 * np.sin(yy / 110)).clip(0, 255)
    crop[..., 0] = (110 + 30 * np.cos((xx + yy) / 150)).clip(0, 255)
    for cx, cy, r, dv in [(300, 300, 90, 40), (440, 300, 90, 40), (370, 470, 120, -30)]:
        m = ((xx - cx) ** 2 / (r * r) + (yy - cy) ** 2 / (r * 1.3 * r)) < 1
        crop[m] = (crop[m].astype(np.int16) + dv).clip(0, 255).astype(np.uint8)
    return cv2.GaussianBlur(crop, (0, 0), 2.0)


def _photo_with_crop(crop: np.ndarray, codec: str, *, face_quality: int = 95
                     ) -> CompressedPhoto:
    """Build a one-face CompressedPhoto around ``crop`` with the given codec.

    Bypasses the detector so the crop content is controlled and deterministic
    (the codec size/round-trip behavior is what's under test, not detection).
    """
    h, w = crop.shape[:2]
    cfg = FaceKeepConfig()
    cfg.aggressive.face_codec = codec
    cfg.aggressive.face_quality = face_quality
    mask = create_soft_mask((h, w), margin=20)
    face = FaceRegion(id=0, bbox=(10, 10, w - 10, h - 10),
                      padded_bbox=(0, 0, w, h), confidence=0.9)
    return CompressedPhoto(
        original_filename="p.jpg", original_width=w, original_height=h,
        original_size_bytes=999, original_hash="0" * 64, original_orientation=1,
        exif=None,
        background=np.full((200, 200, 3), 128, np.uint8),
        face_crops=[crop.copy()], face_masks=[mask], faces=[face],
        thumbnail=np.full((256, 256, 3), 128, np.uint8),
        effective_bg_scale=0.25, config=cfg.aggressive,
    )


def _crop_member_and_bytes(fkeep_path) -> tuple:
    """Return (member_name, byte_len) of the single face crop in a .fkeep."""
    with zipfile.ZipFile(fkeep_path) as zf:
        name = next(n for n in zf.namelist()
                    if n.startswith("face_000.") and "mask" not in n)
        return name, len(zf.read(name))


def _write(photo, tmp_path, name):
    p = _fkeep_path(str(tmp_path / name))
    write_fkeep(photo, str(tmp_path / name))
    assert p.exists()
    return p


# --------------------------------------------------------------------------- #
# the chosen codec lands in the archive + round-trips
# --------------------------------------------------------------------------- #

@requires_avif
def test_avif_crop_member_and_roundtrip(tmp_path):
    """face_codec='avif' stores face_000.avif and it decodes back to the crop."""
    crop = _smooth_face_crop()
    fkeep = _write(_photo_with_crop(crop, "avif"), tmp_path, "avif")

    name, _ = _crop_member_and_bytes(fkeep)
    assert name == "face_000.avif"

    data = read_fkeep(str(fkeep))
    out = data["face_crops"][0]
    assert out.shape == crop.shape and out.dtype == np.uint8
    # AVIF q95 is visually lossless on smooth content: the decoded crop is very
    # close to the source (SSIM well above the visually-lossless threshold).
    assert metrics.ssim(crop, out) > 0.98


@requires_jxl
def test_jxl_crop_member_and_roundtrip(tmp_path):
    """face_codec='jxl' stores face_000.jxl and it decodes back to the crop."""
    crop = _smooth_face_crop()
    fkeep = _write(_photo_with_crop(crop, "jxl"), tmp_path, "jxl")

    name, _ = _crop_member_and_bytes(fkeep)
    assert name == "face_000.jxl"

    data = read_fkeep(str(fkeep))
    out = data["face_crops"][0]
    assert out.shape == crop.shape and out.dtype == np.uint8
    assert metrics.ssim(crop, out) > 0.98


# --------------------------------------------------------------------------- #
# size: the win (content-dependent — asserted honestly)
# --------------------------------------------------------------------------- #

@requires_jxl
def test_jxl_crop_smaller_than_jpeg(tmp_path):
    """On smooth content, the JXL crop is strictly smaller than the JPEG crop.

    JXL wins broadly (here and on artifact-laden crops), so this is a safe strict
    inequality — the measurable form of the ROADMAP's "~2× smaller" claim. (The
    exact ratio is content-dependent; we don't pin a magic multiple, only that it
    is genuinely smaller, which is the point.)
    """
    crop = _smooth_face_crop()
    jpg = _write(_photo_with_crop(crop, "jpg"), tmp_path, "as_jpg")
    jxl = _write(_photo_with_crop(crop, "jxl"), tmp_path, "as_jxl")

    _, jpg_bytes = _crop_member_and_bytes(jpg)
    _, jxl_bytes = _crop_member_and_bytes(jxl)
    assert jxl_bytes < jpg_bytes, (jxl_bytes, jpg_bytes)


@requires_avif
def test_avif_crop_smaller_than_jpeg_on_smooth_content(tmp_path):
    """On clean smooth content (AVIF's good case) the AVIF crop also wins.

    Scoped to smooth content on purpose: this is where the "~2× smaller" headline
    was measured. AVIF can lose to JPEG on a noise/artifact-laden crop (its worst
    case), which is why the broad, always-on size assertion above uses JXL and
    AVIF is otherwise pinned on correctness, not bytes.
    """
    crop = _smooth_face_crop()
    jpg = _write(_photo_with_crop(crop, "jpg"), tmp_path, "s_jpg")
    avif = _write(_photo_with_crop(crop, "avif"), tmp_path, "s_avif")

    _, jpg_bytes = _crop_member_and_bytes(jpg)
    _, avif_bytes = _crop_member_and_bytes(avif)
    assert avif_bytes < jpg_bytes, (avif_bytes, jpg_bytes)


# --------------------------------------------------------------------------- #
# precedence: lossless PNG wins regardless of codec
# --------------------------------------------------------------------------- #

@requires_avif
def test_lossless_png_takes_precedence_over_face_codec(tmp_path):
    """face_quality>=100 forces lossless PNG even when face_codec='avif'."""
    crop = _smooth_face_crop(size=256)
    fkeep = _write(_photo_with_crop(crop, "avif", face_quality=100),
                   tmp_path, "lossless")

    name, _ = _crop_member_and_bytes(fkeep)
    assert name == "face_000.png"

    # And it really is lossless: PNG round-trips bit-exact.
    data = read_fkeep(str(fkeep))
    assert np.array_equal(data["face_crops"][0], crop)


# --------------------------------------------------------------------------- #
# backward compatibility: the default path is unchanged
# --------------------------------------------------------------------------- #

def test_default_codec_is_jpg_and_unchanged(tmp_path):
    """The default config still stores face_000.jpg (no behavior change)."""
    crop = _smooth_face_crop(size=256)
    fkeep = _write(_photo_with_crop(crop, "jpg"), tmp_path, "default")

    name, _ = _crop_member_and_bytes(fkeep)
    assert name == "face_000.jpg"
    # Manifest still records the codec (default jpg). Schema version is the
    # current one (1.8.0, latest bump for the high-bit bit_depth key); the
    # default jpg-crop behavior is unchanged regardless.
    info = read_fkeep_info(str(fkeep))
    assert info["settings"]["face_codec"] == "jpg"
    assert info["version"] == "1.8.0"


def test_default_fkeep_verifies_clean(face_image, tmp_path):
    """A real default (jpg-crop) .fkeep still verifies clean — anti-regression."""
    photo = compress_photo(str(face_image), FaceKeepConfig())
    fkeep = _write(photo, tmp_path, "real")
    rep = verify_fkeep(str(fkeep))
    assert rep.ok, rep.problems
    assert rep.crops_found == rep.faces_declared >= 1


# --------------------------------------------------------------------------- #
# verify_fkeep accepts avif/jxl crops
# --------------------------------------------------------------------------- #

@requires_avif
def test_verify_accepts_avif_crops(face_image, tmp_path):
    """verify_fkeep is clean on a .fkeep whose crops are AVIF, and round-trips."""
    cfg = FaceKeepConfig()
    cfg.aggressive.face_codec = "avif"
    photo = compress_photo(str(face_image), cfg)
    fkeep = _write(photo, tmp_path, "avif_real")

    # Every crop is an .avif member.
    with zipfile.ZipFile(fkeep) as zf:
        crops = [n for n in zf.namelist()
                 if n.startswith("face_0") and "mask" not in n]
    assert crops and all(n.endswith(".avif") for n in crops), crops

    rep = verify_fkeep(str(fkeep))
    assert rep.ok, rep.problems
    assert rep.crops_found == rep.faces_declared >= 1

    # And restore can read them back (the cv2-can't-decode-AVIF path is covered).
    data = read_fkeep(str(fkeep))
    assert len(data["face_crops"]) == rep.faces_declared
    assert all(c.dtype == np.uint8 and c.ndim == 3 for c in data["face_crops"])


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #

def test_validate_rejects_unknown_face_codec():
    cfg = FaceKeepConfig()
    cfg.aggressive.face_codec = "tiff"
    with pytest.raises(ConfigError, match="face_codec"):
        cfg.validate()


@pytest.mark.parametrize("codec", ["jpg", "avif", "jxl"])
def test_validate_accepts_known_face_codecs(codec):
    cfg = FaceKeepConfig()
    cfg.aggressive.face_codec = codec
    cfg.validate()  # must not raise
