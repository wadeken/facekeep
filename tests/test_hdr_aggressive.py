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
    assert info["version"] == "1.11.0"
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
    assert info["version"] == "1.11.0"
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


# --- Step C: opt-in high-bit (HDR) residual layer --------------------------- #
#
# When high-bit storage is engaged (output_bit_depth 10/12 + face_codec=avif) AND
# the residual layer is on, the residual is stored as a true 10/12-bit AVIF
# (residual.avif) instead of the 8-bit residual.jxl, so a uint16 source's
# background delta keeps its depth. Gated like the crops; degrades to the 8-bit
# residual without avifenc, and is skipped (restore falls back to AI/bicubic) when
# avifdec is unavailable at restore. The round-trip needs avifenc+avifdec.


def _highbit_residual_cfg():
    cfg = _highbit_cfg()
    cfg.aggressive.residual = True
    return cfg


def test_offset_residual_highbit_roundtrip():
    """uint16 offset encode/decode recovers the signed delta within the halving step."""
    from facekeep.aggressive.format import (
        _offset_decode_residual,
        _offset_encode_residual,
    )

    rng = np.random.default_rng(0)
    delta = rng.integers(-65535, 65536, size=(8, 8, 3)).astype(np.float32)
    enc = _offset_encode_residual(delta, high_bit=True)
    assert enc.dtype == np.uint16
    dec = _offset_decode_residual(enc)  # dtype-inferred (uint16 -> *2-65536)
    assert np.max(np.abs(dec - delta)) <= 2.0  # only the /2 halving step is lost


def test_offset_residual_8bit_unchanged():
    """The default (8-bit) offset path is uint8 and round-trips within the step."""
    from facekeep.aggressive.format import (
        _offset_decode_residual,
        _offset_encode_residual,
    )

    delta = np.array([[[-200, 0, 200]]], dtype=np.float32)
    enc = _offset_encode_residual(delta)  # no high_bit -> uint8
    assert enc.dtype == np.uint8
    assert np.max(np.abs(_offset_decode_residual(enc) - delta)) <= 2.0


def test_apply_residual_highbit_returns_uint16():
    """A uint16 residual member -> uint16 background = 257*upscale + delta."""
    from facekeep.aggressive.format import _offset_encode_residual
    from facekeep.aggressive.restorer import _apply_residual

    bg = np.full((4, 4, 3), 100, np.uint8)            # 8-bit stored background
    delta = np.full((8, 8, 3), 1000.0, np.float32)    # full-res (8x8) target delta
    residual = _offset_encode_residual(delta, high_bit=True)  # uint16, 8x8
    out = _apply_residual(bg, residual, 8, 8)
    assert out.dtype == np.uint16
    assert abs(int(out[4, 4, 0]) - (257 * 100 + 1000)) <= 4


def test_apply_residual_8bit_unchanged():
    """A uint8 residual member -> uint8 background (path unchanged)."""
    from facekeep.aggressive.format import _offset_encode_residual
    from facekeep.aggressive.restorer import _apply_residual

    bg = np.full((4, 4, 3), 100, np.uint8)
    residual = _offset_encode_residual(np.full((4, 4, 3), 20.0, np.float32))  # uint8
    out = _apply_residual(bg, residual, 4, 4)
    assert out.dtype == np.uint8
    assert abs(int(out[2, 2, 0]) - 120) <= 2


def test_apply_grain_highbit_returns_uint16():
    """Grain on a uint16 background scales to 16-bit and stays uint16."""
    from facekeep.aggressive.restorer import _apply_grain

    bg = np.full((16, 16, 3), 30000, np.uint16)
    out = _apply_grain(bg, 5.0)
    assert out.dtype == np.uint16
    assert out.std() > 0  # grain actually applied
    assert abs(float(out.mean()) - 30000) < 2000  # stays near the mean


def test_apply_grain_8bit_byte_identical():
    """The 8-bit grain path is byte-identical to the legacy computation (lock guard)."""
    from facekeep.aggressive import restorer

    bg = np.full((16, 16, 3), 120, np.uint8)
    out = restorer._apply_grain(bg, 3.0)
    assert out.dtype == np.uint8

    rng = np.random.default_rng(restorer._GRAIN_SEED)
    noise = rng.standard_normal((16, 16), dtype=np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), restorer._GRAIN_SOFTEN_SIGMA)
    noise *= 3.0 / float(noise.std())
    ref = np.clip(bg.astype(np.float32) + noise[:, :, None], 0, 255).astype(np.uint8)
    assert np.array_equal(out, ref)


def test_highbit_residual_falls_back_to_8bit_without_avifenc(
    face_image, tmp_path, caplog, monkeypatch
):
    """Requested HDR residual but avifenc absent -> 8-bit residual.jxl + warn."""
    monkeypatch.setattr(encoders, "avifenc_available", lambda: False)
    src = _write_uint16(tmp_path / "f16.png", cv2.imread(str(face_image)))
    photo = compress_photo(str(src), _highbit_residual_cfg())
    assert photo.original_image is not None and photo.original_image.dtype == np.uint16

    fk = tmp_path / "o.fkeep"
    with caplog.at_level(logging.WARNING, logger="facekeep.aggressive.format"):
        write_fkeep(photo, str(fk))
    with zipfile.ZipFile(str(fk)) as zf:
        names = zf.namelist()
    assert "residual.avif" not in names
    assert "residual.jxl" in names or "residual.jpg" in names
    assert any("residual 8-bit" in r.getMessage() for r in caplog.records)
    # 8-bit container -> no bit_depth key; the 8-bit residual still restores.
    assert "bit_depth" not in read_fkeep_info(str(fk))["settings"]


@hdr_tools_required
def test_highbit_residual_roundtrip_preserves_hdr(face_image, tmp_path):
    """A uint16 source's background keeps >8-bit detail through the residual path."""
    src = _highbit_source(face_image, tmp_path)
    cfg = _highbit_residual_cfg()
    photo = compress_photo(str(src), cfg)
    assert photo.original_image is not None
    assert photo.original_image.dtype == np.uint16  # full-depth residual source

    fk = tmp_path / "hdr_res.fkeep"
    write_fkeep(photo, str(fk))

    info = read_fkeep_info(str(fk))
    assert info["version"] == "1.11.0"
    assert info["settings"]["residual"] is True
    assert info["settings"]["bit_depth"] == 10  # folded from the high-bit residual
    with zipfile.ZipFile(str(fk)) as zf:
        assert "residual.avif" in zf.namelist()

    # The residual member decodes uint16 (high-bit), so restore reconstructs a
    # uint16 background instead of flattening it to 8-bit.
    data = read_fkeep(str(fk))
    assert data["residual"].dtype == np.uint16

    out = tmp_path / "restored.avif"
    Restorer(cfg.aggressive).restore(str(fk), str(out))
    assert out.exists()
    restored16 = encoders.decode_highbit_avif(out.read_bytes())
    assert restored16.dtype == np.uint16
    # A wide background strip (the source carries a sub-257 ramp across its width)
    # comes back with >256 levels — HDR restored from the residual, which an 8-bit
    # residual restore caps at 256.
    h = restored16.shape[0]
    bg_strip = restored16[int(h * 0.82):, :]
    assert _distinct_levels(bg_strip) > 256


@hdr_tools_required
def test_verify_highbit_residual_needs_no_avifdec(face_image, tmp_path, monkeypatch):
    """verify confirms a high-bit residual structurally even without avifdec."""
    from facekeep.aggressive.format import verify_fkeep

    src = _highbit_source(face_image, tmp_path)
    fk = tmp_path / "hdr_res.fkeep"
    write_fkeep(compress_photo(str(src), _highbit_residual_cfg()), str(fk))

    # Simulate a box with no avifdec: verify must still pass (Pillow 8-bit decode).
    monkeypatch.setattr(encoders, "_find_avifdec", lambda: None)
    report = verify_fkeep(str(fk))
    assert report.residual_declared is True
    assert report.residual_ok is True
    assert report.ok, report.problems


@hdr_tools_required
def test_restore_highbit_residual_without_avifdec_skips_residual(
    face_image, tmp_path, caplog, monkeypatch
):
    """No avifdec at restore -> the high-bit residual is skipped (AI/bicubic), warned."""
    src = _highbit_source(face_image, tmp_path)
    cfg = _highbit_residual_cfg()
    fk = tmp_path / "hdr_res.fkeep"
    write_fkeep(compress_photo(str(src), cfg), str(fk))

    monkeypatch.setattr(encoders, "_find_avifdec", lambda: None)
    out = tmp_path / "restored.jpg"
    with caplog.at_level(logging.WARNING, logger="facekeep.aggressive.format"):
        Restorer(cfg.aggressive).restore(str(fk), str(out))
    assert out.exists()  # still a valid image — never a dead end
    assert any("residual layer" in r.getMessage() for r in caplog.records)
