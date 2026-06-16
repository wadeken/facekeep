"""True high-bit (HDR) HEIC decode (ROADMAP Phase 7 HEIC follow-up).

Recent iPhones shoot 10-bit HDR HEIC. The HEIC load path used to decode 8-bit
RGB (the PIL HEIF plugin flattens HDR), so high-bit detail was lost before the
encoder ever saw it. ``imageio.load`` now decodes HEIC via
``pillow_heif.open_heif(convert_hdr_to_8bit=False)`` (see ``imageio._decode_heif``),
yielding genuine ``uint16`` samples with ``source_bit_depth=16`` that feed the
existing ``avifenc`` 10/12-bit AVIF output path.

Gating (two layers, matching the project's offline/optional conventions):
  * ``pillow_heif`` must be importable (the ``[heic]`` extra) — else the whole
    module skips, since the fixtures are *written* with ``pillow_heif`` too;
  * the end-to-end no-banding round-trip additionally needs the external
    ``avifenc`` binary, so it skips without it (same convention as
    ``tests/test_bit_depth.py``; set ``FACEKEEP_AVIFENC`` or put it on PATH).

All HEIC decoding goes through ``open_heif`` (never PIL ``Image.open``): opening
the *same* HEIC with both APIs in one process segfaults inside libheif, so the
tests likewise only ever load HEIC through ``imageio.load`` (open_heif) and write
fixtures through ``pillow_heif.from_bytes`` — never ``Image.open`` on a HEIC.
"""

import base64
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import piexif
import pytest

from facekeep import encoders, faithful, imageio
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import UnsupportedInputError

pillow_heif = pytest.importorskip("pillow_heif")

# A compact, valid Display-P3 ICC profile (the same asset as tests/test_color.py),
# used to prove the source ICC survives the open_heif high-bit decode path
# (color CORE-GOAL — a wide-gamut HEIC must not lose its profile).
_DISPLAY_P3_ICC_B64 = (
    "AAABzGxjbXMEQAAAbW50clJHQiBYWVogAAAAAAAAAAAAAAAAYWNzcAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAPbWAAEAAAAA0y0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAJZGVzYwAAAPAAAAAyY3BydAAAASQAAAA4d3RwdAAAAVwAAAAUclhZ"
    "WgAAAXAAAAAUZ1hZWgAAAYQAAAAUYlhZWgAAAZgAAAAUclRSQwAAAawAAAAgZ1RSQwAAAawAAAAg"
    "YlRSQwAAAawAAAAgbWx1YwAAAAAAAAABAAAADGVuVVMAAAAWAAAAGABEAGkAcwBwAGwAYQB5ACAA"
    "UAAzAAAAAG1sdWMAAAAAAAAAAQAAAAxlblVTAAAAHAAAABgAUAB1AGIAbABpAGMAIABEAG8AbQBh"
    "AGkAbgAAWFlaIAAAAAAAAPbWAAEAAAAA0y1YWVogAAAAAAAAg94AAD2+////u1hZWiAAAAAAAABK"
    "vgAAsTYAAAq5WFlaIAAAAAAAACg7AAARCwAAyLlwYXJhAAAAAAADAAAAAmZmAADypwAADVkAABPQ"
    "AAAKWw=="
)
DISPLAY_P3_ICC = base64.b64decode(_DISPLAY_P3_ICC_B64)

avifenc_required = pytest.mark.skipif(
    not encoders.avifenc_available(),
    reason="avifenc binary not found (set FACEKEEP_AVIFENC or put it on PATH)",
)


def _write_heic(path, rgb, *, mode, exif=None, icc=None, quality=-1):
    """Write a HEIC fixture from an ndarray via pillow_heif.from_bytes.

    ``mode`` is ``"RGB;16"`` for a uint16 (10-bit HDR) source or ``"RGB"`` for a
    uint8 source. ``quality=-1`` requests lossless so the fixture's high-bit
    detail isn't quantized away before we even load it.
    """
    h, w = rgb.shape[:2]
    kw = {"quality": quality}
    if exif is not None:
        kw["exif"] = exif
    if icc is not None:
        kw["icc_profile"] = icc
    pillow_heif.from_bytes(mode=mode, size=(w, h), data=rgb.tobytes()).save(str(path), **kw)


def _gray_ramp16(w=512, h=64):
    """A smooth 16-bit gray gradient (R=G=B) — the classic banding probe."""
    ramp = np.linspace(0, 65535, w).astype(np.uint16)[None, :].repeat(h, 0)
    return np.stack([ramp, ramp, ramp], axis=-1)


def test_highbit_heic_loads_uint16(tmp_path):
    """A 10-bit HDR HEIC loads as uint16 BGR with source_bit_depth == 16."""
    src = tmp_path / "hdr.heic"
    ramp = np.linspace(0, 65535, 512).astype(np.uint16)[None, :].repeat(64, 0)
    # distinct channels so a channel swap would be visible: R=ramp (BGR index 2)
    rgb16 = np.stack([ramp, ramp // 2, ramp // 3], axis=-1)
    _write_heic(src, rgb16, mode="RGB;16")

    loaded = imageio.load(str(src))

    assert loaded.source_bit_depth == 16
    assert loaded.image.dtype == np.uint16
    assert loaded.image.ndim == 3 and loaded.image.shape[2] == 3
    assert (loaded.width, loaded.height) == (512, 64)
    # The red channel (BGR index 2) is the full 512-step ramp; an 8-bit decode
    # would cap it at <=256 distinct levels. >256 proves real high-bit survived.
    assert len(np.unique(loaded.image[0, :, 2])) > 256


def test_8bit_heic_loads_uint8(tmp_path):
    """An ordinary 8-bit HEIC still loads as uint8 / depth 8 (reroute no-regression).

    HEIC now decodes through open_heif for *all* depths; an 8-bit source must
    still come back uint8 with source_bit_depth 8, exactly like the old path.
    """
    src = tmp_path / "lo.heic"
    rgb8 = np.dstack([
        np.full((96, 128), 200, np.uint8),
        np.full((96, 128), 100, np.uint8),
        np.full((96, 128), 50, np.uint8),
    ])
    _write_heic(src, rgb8, mode="RGB", quality=90)

    loaded = imageio.load(str(src))

    assert loaded.source_bit_depth == 8
    assert loaded.image.dtype == np.uint8
    assert loaded.image.shape[2] == 3


def test_highbit_heic_orientation_upright(tmp_path):
    """A high-bit HEIC with orientation 6 loads upright, high-bit, tag normalized.

    open_heif pre-rotates the pixels but leaves the EXIF orientation tag set, so
    this guards that we apply NO further rotation (no double-rotate) AND keep the
    uint16 data through the orientation step AND normalize the carried tag to 1.
    """
    W, H = 80, 40  # landscape; orientation 6 => upright 40x80 portrait
    rgb16 = np.zeros((H, W, 3), np.uint16)
    rgb16[:, : W // 2] = (65535, 0, 0)   # left half red (RGB)
    rgb16[:, W // 2:] = (0, 0, 65535)    # right half blue
    exif6 = piexif.dump({"0th": {piexif.ImageIFD.Orientation: 6}})
    src = tmp_path / "o6.heic"
    _write_heic(src, rgb16, mode="RGB;16", exif=exif6)

    loaded = imageio.load(str(src))

    assert (loaded.width, loaded.height) == (40, 80), "not rotated upright"
    assert loaded.image.dtype == np.uint16, "high-bit must survive orientation"
    img = loaded.image  # BGR, upright
    top = img[:5].reshape(-1, 3).mean(0)      # BGR means
    bottom = img[-5:].reshape(-1, 3).mean(0)
    # Orientation 6 puts the original LEFT (red) stripe at the TOP.
    assert top[2] > top[0], "top should be red (R>B) — wrong/no rotation"
    assert bottom[0] > bottom[2], "bottom should be blue (B>R)"
    # Carried EXIF orientation normalized to 1 so a re-embed never double-rotates.
    assert piexif.load(loaded.exif)["0th"].get(piexif.ImageIFD.Orientation, 1) == 1


def test_highbit_heic_preserves_icc(tmp_path):
    """A wide-gamut (P3) high-bit HEIC keeps its ICC profile through open_heif."""
    src = tmp_path / "p3.heic"
    _write_heic(src, _gray_ramp16(256, 64), mode="RGB;16", icc=DISPLAY_P3_ICC)

    loaded = imageio.load(str(src))

    assert loaded.source_bit_depth == 16
    assert loaded.icc == DISPLAY_P3_ICC


def test_corrupt_heic_raises_not_crashes(tmp_path):
    """A garbage .heic surfaces UnsupportedInputError (graceful), never a crash."""
    bad = tmp_path / "bad.heic"
    bad.write_bytes(b"this is not a valid heic file")
    with pytest.raises(UnsupportedInputError):
        imageio.load(str(bad))


def test_highbit_heic_8bit_fallback_without_avifenc(tmp_path, caplog, monkeypatch):
    """Without avifenc, a high-bit HEIC still decodes uint16 but rounds down to
    8-bit at the encode boundary with the standard loud warning.

    This pins the Option-A / offline-first behavior: the high-bit *decode* is
    unconditional (mirrors the 16-bit PNG path), and the loss only happens at the
    encode boundary — warned, never silent — when there is no high-bit output.
    """
    monkeypatch.setattr(encoders, "avifenc_available", lambda: False)
    src = tmp_path / "hdr.heic"
    _write_heic(src, _gray_ramp16(256, 64), mode="RGB;16")

    loaded = imageio.load(str(src))
    # Decode is unconditional — the uint16 data is carried regardless of avifenc.
    assert loaded.source_bit_depth == 16 and loaded.image.dtype == np.uint16

    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = False
    cfg.faithful.quality = 90
    cfg.faithful.skip_if_larger = False  # force the encode (a tiny ramp may not shrink)
    out_path = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger="facekeep.encoders"):
        result = faithful.compress(str(src), str(out_path), cfg)

    assert any("16-bit" in r.message for r in caplog.records), (
        "expected a high-bit down-convert warning; the loss must not be silent"
    )
    assert not result.skipped
    decoded = encoders.decode(result.output_path.read_bytes())
    assert decoded.dtype == np.uint8  # flattened to 8-bit at the encode boundary


@avifenc_required
def test_highbit_heic_no_banding_roundtrip(tmp_path):
    """CORE-GOAL acceptance: a 10-bit HDR HEIC must not band on faithful round-trip.

    The headline test for this item, mirroring tests/test_bit_depth.py's gradient
    check but with a HEIC source. With avifenc present, faithful.compress should
    produce a true 10-bit AVIF; decoding it back to 16-bit via avifdec must show
    far more than the 8-bit cap of 256 distinct levels.
    """
    src = tmp_path / "grad.heic"
    rgb16 = _gray_ramp16(1024, 64)
    _write_heic(src, rgb16, mode="RGB;16")

    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = False
    cfg.faithful.quality = 95
    cfg.faithful.skip_if_larger = False
    out_path = tmp_path / "out"
    result = faithful.compress(str(src), str(out_path), cfg)

    assert not result.skipped
    assert result.output_path.suffix == ".avif"

    # Decode the AVIF back to true 16-bit via avifdec (Pillow would decode 8-bit
    # and hide the win); avifdec sits next to avifenc.
    avifenc = encoders._find_avifenc()
    avifdec = str(Path(avifenc).with_name(
        "avifdec.exe" if avifenc.endswith(".exe") else "avifdec"
    ))
    dec_png = tmp_path / "dec.png"
    proc = subprocess.run(
        [avifdec, "-d", "16", str(result.output_path), str(dec_png)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dec_png.exists():
        pytest.skip(f"avifdec unavailable/failed: {proc.stderr[:120]}")

    decoded16 = cv2.imread(str(dec_png), cv2.IMREAD_UNCHANGED)
    assert decoded16.dtype == np.uint16, "expected a true 16-bit decode"
    ref_levels = len(np.unique(rgb16[0, :, 0]))
    out_levels = len(np.unique(decoded16[0, :, 0]))
    assert out_levels > 256, f"only {out_levels} levels — still 8-bit banding"
    assert out_levels >= ref_levels * 0.4
