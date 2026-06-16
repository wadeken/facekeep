"""Aggressive-mode high-bit (HDR) handling.

Step A (the prereq cleanup): the ``.fkeep`` is an 8-bit container, so a uint16
source (e.g. a 10/12-bit HDR HEIC, which ``imageio.load`` decodes as uint16) must
be down-converted to 8-bit *cleanly* — the ``/257`` rounding
``compressor._to_uint8`` does — for every stored pixel member (face/region crops,
background, thumbnail, residual). Previously a uint16 member rode OpenCV's
``imencode`` CV_8U fallback, which is not a clean down-convert. An 8-bit source
must stay byte-identical (``pixels is image``).

(Step B will add genuine high-bit crop/region storage behind an opt-in knob; its
tests, ``avifenc``/``avifdec``-gated like ``test_bit_depth.py``, live here too.)
"""

import logging
import zipfile

import cv2
import numpy as np
import pytest

from facekeep import encoders, imageio
from facekeep.aggressive.compressor import _to_uint8, compress_photo
from facekeep.aggressive.format import read_fkeep, read_fkeep_info, write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint


def _write_uint16(path, img8):
    """Write a 16-bit PNG whose clean /257 down-convert is exactly ``img8``.

    ``x * 257`` maps 0..255 onto the full 0..65535 range, and ``_to_uint8``
    inverts it exactly (``round(x*257/257) == x``), so the loaded uint16 frame's
    clean down-convert is the original 8-bit image — a deterministic reference.
    """
    img16 = (img8.astype(np.uint16) * 257)
    assert cv2.imwrite(str(path), img16)
    return path


def test_uint16_stored_members_are_clean_8bit(face_image, tmp_path, caplog):
    img8 = cv2.imread(str(face_image))  # 8-bit BGR
    png16 = _write_uint16(tmp_path / "face16.png", img8)

    # Sanity: imageio.load really hands the pipeline a uint16 frame.
    loaded = imageio.load(str(png16))
    assert loaded.image.dtype == np.uint16
    ref8 = _to_uint8(loaded.image)
    assert ref8.dtype == np.uint8

    with caplog.at_level(logging.WARNING, logger="facekeep.aggressive.compressor"):
        photo = compress_photo(str(png16), FaceKeepConfig())

    # Honest about the down-convert (mirrors faithful mode's loud warning).
    assert any("8-bit container" in r.getMessage() for r in caplog.records)

    # Every stored pixel member is 8-bit and equals the clean /257 reference —
    # i.e. built from _to_uint8(image), not OpenCV's CV_8U imencode fallback.
    assert photo.background.dtype == np.uint8
    bh, bw = photo.background.shape[:2]
    assert np.array_equal(
        photo.background, cv2.resize(ref8, (bw, bh), interpolation=cv2.INTER_AREA)
    )

    assert photo.thumbnail.dtype == np.uint8
    th, tw = photo.thumbnail.shape[:2]
    assert np.array_equal(
        photo.thumbnail, cv2.resize(ref8, (tw, th), interpolation=cv2.INTER_AREA)
    )

    assert len(photo.face_crops) >= 1
    for face, crop in zip(photo.faces, photo.face_crops):
        x1, y1, x2, y2 = face.padded_bbox
        assert crop.dtype == np.uint8
        assert np.array_equal(crop, ref8[y1:y2, x1:x2])


def test_uint16_residual_built_from_clean_8bit(face_image, tmp_path):
    img8 = cv2.imread(str(face_image))
    png16 = _write_uint16(tmp_path / "face16.png", img8)
    ref8 = _to_uint8(imageio.load(str(png16)).image)

    cfg = FaceKeepConfig()
    cfg.aggressive.residual = True
    photo = compress_photo(str(png16), cfg)

    # The residual is computed against the same clean 8-bit rendering as the
    # background it corrects (not the CV_8U fallback).
    assert photo.original_image is not None
    assert photo.original_image.dtype == np.uint8
    assert np.array_equal(photo.original_image, ref8)


def test_8bit_source_unchanged(face_image):
    """An 8-bit input is byte-identical to the pre-change path (``pixels is image``)."""
    loaded = imageio.load(str(face_image))
    assert loaded.image.dtype == np.uint8

    photo = compress_photo(str(face_image), FaceKeepConfig())

    # The old code resized/sliced ``image`` directly; with an 8-bit source
    # _to_uint8 is a no-op, so the members must equal those direct ops.
    assert photo.background.dtype == np.uint8
    bh, bw = photo.background.shape[:2]
    assert np.array_equal(
        photo.background,
        cv2.resize(loaded.image, (bw, bh), interpolation=cv2.INTER_AREA),
    )
    for face, crop in zip(photo.faces, photo.face_crops):
        x1, y1, x2, y2 = face.padded_bbox
        assert np.array_equal(crop, loaded.image[y1:y2, x1:x2])


# --- Step B: opt-in high-bit (10/12-bit) crop/region storage ---------------- #
#
# output_bit_depth in (10, 12) + face_codec=avif stores the *real-pixel* members
# (face crops + region patches) at true high bit depth via avifenc, so HDR
# survives the .fkeep round-trip. Background/thumbnail/residual stay 8-bit. The
# round-trip tests need BOTH avifenc (encode) and avifdec (decode), so they skip
# without the binaries — like test_bit_depth.py's high-bit tests.

hdr_tools_required = pytest.mark.skipif(
    not (encoders.avifenc_available() and encoders.avifdec_available()),
    reason="avifenc+avifdec not found (set FACEKEEP_AVIFENC or put them on PATH)",
)


def _distinct_levels(img):
    """Distinct luma levels — an 8-bit image caps at 256, true high-bit exceeds it."""
    return int(np.unique(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).size)


def _highbit_source(face_image, tmp_path):
    """A uint16 PNG with a Haar-detectable face AND genuine >256-level detail.

    ``face_image`` is an 8-bit render, so ``*257`` alone yields only 256 levels; a
    fine horizontal ramp (sub-257 steps) adds true >8-bit detail to preserve. The
    /257 down-convert still recovers the detectable face, so detection works.
    """
    img8 = cv2.imread(str(face_image))
    _, W = img8.shape[:2]
    fine = np.linspace(0, 255, W).astype(np.int32)[None, :, None]  # 0..255 sub-257 steps
    img16 = np.clip(img8.astype(np.int32) * 257 + fine - 128, 0, 65535).astype(np.uint16)
    path = tmp_path / "hdr16.png"
    assert cv2.imwrite(str(path), img16)
    return path


def _highbit_cfg():
    cfg = FaceKeepConfig()
    cfg.aggressive.output_bit_depth = 10
    cfg.aggressive.face_codec = "avif"  # high-bit storage is gated on AVIF crops
    return cfg


@pytest.mark.parametrize("ok", [8, 10, 12])
def test_validate_accepts_output_bit_depth(ok):
    cfg = FaceKeepConfig()
    cfg.aggressive.output_bit_depth = ok
    cfg.validate()  # no raise


@pytest.mark.parametrize("bad", [0, 9, 11, 16, -1])
def test_validate_rejects_bad_output_bit_depth(bad):
    cfg = FaceKeepConfig()
    cfg.aggressive.output_bit_depth = bad
    with pytest.raises(ConfigError, match="output_bit_depth"):
        cfg.validate()


def test_fingerprint_busts_on_output_bit_depth():
    base = FaceKeepConfig()
    base.mode = "aggressive"
    hi = FaceKeepConfig()
    hi.mode = "aggressive"
    hi.aggressive.output_bit_depth = 10
    assert settings_fingerprint(base) != settings_fingerprint(hi)
    # Faithful fingerprint must NOT move (aggressive-only knob).
    assert settings_fingerprint(FaceKeepConfig()) == settings_fingerprint(FaceKeepConfig())


def test_default_8bit_container_has_no_bit_depth_key(face_image, tmp_path):
    """The default (output_bit_depth=8) stores 8-bit crops and writes no key."""
    png16 = _write_uint16(tmp_path / "f16.png", cv2.imread(str(face_image)))
    photo = compress_photo(str(png16), FaceKeepConfig())  # output_bit_depth=8
    assert all(c.dtype == np.uint8 for c in photo.face_crops)

    fk = tmp_path / "o.fkeep"
    write_fkeep(photo, str(fk))
    info = read_fkeep_info(str(fk))
    assert info["version"] == "1.8.0"
    assert "bit_depth" not in info["settings"]  # absent on an 8-bit container


def test_highbit_ignored_when_face_codec_is_jpg(face_image, tmp_path):
    """High-bit is gated on AVIF crops; output_bit_depth=10 + jpg stays 8-bit."""
    png16 = _write_uint16(tmp_path / "f16.png", cv2.imread(str(face_image)))
    cfg = FaceKeepConfig()
    cfg.aggressive.output_bit_depth = 10  # but face_codec stays "jpg"
    photo = compress_photo(str(png16), cfg)
    assert all(c.dtype == np.uint8 for c in photo.face_crops)

    fk = tmp_path / "o.fkeep"
    write_fkeep(photo, str(fk))
    assert "bit_depth" not in read_fkeep_info(str(fk))["settings"]


def test_highbit_falls_back_to_8bit_without_avifenc(face_image, tmp_path, caplog, monkeypatch):
    """Requested but avifenc absent -> crops stored 8-bit + warn (offline-first)."""
    monkeypatch.setattr(encoders, "avifenc_available", lambda: False)
    png16 = _write_uint16(tmp_path / "f16.png", cv2.imread(str(face_image)))
    photo = compress_photo(str(png16), _highbit_cfg())
    # The compressor keeps crops uint16 (it doesn't probe avifenc); the encode
    # layer does the fallback + the warning.
    fk = tmp_path / "o.fkeep"
    with caplog.at_level(logging.WARNING, logger="facekeep.aggressive.format"):
        write_fkeep(photo, str(fk))
    assert any("avifenc is unavailable" in r.getMessage() for r in caplog.records)
    info = read_fkeep_info(str(fk))
    assert "bit_depth" not in info["settings"]  # stored 8-bit -> no key
    assert read_fkeep(str(fk))["face_crops"][0].dtype == np.uint8


@hdr_tools_required
def test_highbit_aggressive_roundtrip_preserves_hdr(face_image, tmp_path):
    """End-to-end: a uint16 source keeps >8-bit face detail through .fkeep + restore."""
    src = _highbit_source(face_image, tmp_path)
    cfg = _highbit_cfg()
    photo = compress_photo(str(src), cfg)
    assert photo.face_crops, "need a detected face for the high-bit crop"
    assert photo.face_crops[0].dtype == np.uint16, "compressor keeps crops uint16"

    fk = tmp_path / "hdr.fkeep"
    write_fkeep(photo, str(fk))

    info = read_fkeep_info(str(fk))
    assert info["version"] == "1.8.0"
    assert info["settings"]["bit_depth"] == 10
    with zipfile.ZipFile(str(fk)) as zf:
        assert "face_000.avif" in zf.namelist()

    # Read back: the face crop decodes uint16 with genuinely >256 levels (an
    # 8-bit crop caps at 256) — HDR survived storage.
    crop = read_fkeep(str(fk))["face_crops"][0]
    assert crop.dtype == np.uint16
    assert _distinct_levels(crop) > 256

    # Restore to a real .avif and decode it high-bit: HDR survives end-to-end.
    out = tmp_path / "restored.avif"
    Restorer(cfg.aggressive).restore(str(fk), str(out))
    assert out.exists()
    restored16 = encoders.decode_highbit_avif(out.read_bytes())
    assert restored16.dtype == np.uint16
    x1, y1, x2, y2 = info["faces"][0]["padded_bbox"]
    assert _distinct_levels(restored16[y1:y2, x1:x2]) > 256


@hdr_tools_required
def test_highbit_restore_to_jpg_warns_and_downconverts(face_image, tmp_path, caplog):
    """Restoring a high-bit .fkeep to the default .jpg warns and writes 8-bit."""
    src = _highbit_source(face_image, tmp_path)
    cfg = _highbit_cfg()
    fk = tmp_path / "hdr.fkeep"
    write_fkeep(compress_photo(str(src), cfg), str(fk))

    out = tmp_path / "restored.jpg"
    with caplog.at_level(logging.WARNING, logger="facekeep.aggressive.restorer"):
        Restorer(cfg.aggressive).restore(str(fk), str(out))
    assert out.exists()
    assert any("rounded down to 8-bit" in r.getMessage() for r in caplog.records)
    assert cv2.imread(str(out)).dtype == np.uint8  # a normal 8-bit JPEG
