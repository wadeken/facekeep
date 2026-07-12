"""Ultra HDR JPEG restore output (ROADMAP 9.3).

``encoders.encode_gainmap_jpeg`` authors a backward-compatible HDR JPEG with
pure Pillow + a hand-built MPF index — no external binary: the primary frame
is the normal SDR encode (EXIF + real ICC), the gain map rides as the MPF
second frame (the stored ``gainmap.jpg`` bytes verbatim when given as bytes),
and the full Google Ultra HDR XMP (GContainer directory on the primary + Adobe
hdrgm on the gain-map frame) is what real viewers key on — the user-validated
9.3.0 spike showed Chrome renders exactly this flavor as HDR (MPF + a
frame-XMP alone is not enough).

Current Pillow deliberately refuses to open Ultra HDR files as MPO (the
``hdrgm:Version`` sniff), so ``imageio._read_jpeg_gain_map`` grew an MPF-index
fallback (``_parse_mpf_index``) — the self-round-trip tests here cover both
that fallback and the writer. Everything is offline/synthetic; the restore-side
integration (jpg carries the map, png warns, authoring failure degrades to
SDR) lives in test_gain_map_fkeep.py.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facekeep import encoders, imageio
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import EncodingError

APPLE_GAINMAP_XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/" '
    b'xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/">'
    b"<apdi:AuxiliaryImageType>urn:com:apple:photo:2020:aux:hdrgainmap"
    b"</apdi:AuxiliaryImageType></x:xmpmeta>"
)

BASE = np.tile(
    np.linspace(20, 235, 96, dtype=np.uint8)[None, :, None], (64, 1, 3)
)
GAIN_MAP = np.tile(np.linspace(0, 200, 48, dtype=np.uint8), (32, 1))


def _gain_map_jpeg_bytes() -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", GAIN_MAP, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


# ------------------------------------------------------------------- writer


def test_roundtrip_via_facekeep_loader(tmp_path):
    """The written Ultra HDR JPEG self-round-trips through imageio.load —
    including the MPF-index fallback (Pillow opens these as plain JPEG)."""
    gm_jpeg = _gain_map_jpeg_bytes()
    data = encoders.encode_gainmap_jpeg(BASE, gm_jpeg)
    out = tmp_path / "ultra.jpg"
    out.write_bytes(data)

    with Image.open(out) as pil:
        assert pil.format == "JPEG"  # the Pillow Ultra HDR sniff — not MPO

    loaded = imageio.load(str(out))
    assert loaded.gain_map is not None
    assert loaded.gain_map_meta["source"] == "jpeg-mpf"
    # The bytes carrier rides verbatim: decoding the same JPEG twice is
    # byte-equal, no generation loss.
    import cv2

    stored = cv2.imdecode(np.frombuffer(gm_jpeg, np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(loaded.gain_map, stored)


def test_array_carrier_roundtrip(tmp_path):
    """A decoded 2-D array is accepted too (re-encoded q90 grayscale)."""
    data = encoders.encode_gainmap_jpeg(BASE, GAIN_MAP)
    out = tmp_path / "ultra.jpg"
    out.write_bytes(data)

    loaded = imageio.load(str(out))
    assert loaded.gain_map is not None
    assert loaded.gain_map.shape == GAIN_MAP.shape
    # One JPEG q90 generation: close, not exact.
    diff = np.abs(loaded.gain_map.astype(int) - GAIN_MAP.astype(int))
    assert float(diff.mean()) < 4.0


def test_hdrgm_metadata_tracks_headroom(tmp_path):
    """gain_map_headroom lands in the hdrgm XMP (GainMapMax/HDRCapacityMax)."""
    data = encoders.encode_gainmap_jpeg(BASE, _gain_map_jpeg_bytes(), headroom=2.5)
    assert b'hdrgm:GainMapMax="2.5"' in data
    assert b'hdrgm:HDRCapacityMax="2.5"' in data
    assert b'hdrgm:Version="1.0"' in data


def test_primary_carries_exif_and_icc(tmp_path):
    """EXIF and the real ICC profile ride the primary frame (SDR-compatible)."""
    import piexif

    exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"FaceKeepTest"}})
    icc = b"\x00" * 60 + b"Display P3 test profile padding padding"
    data = encoders.encode_gainmap_jpeg(
        BASE, _gain_map_jpeg_bytes(), exif=exif, icc=icc
    )
    with Image.open(io.BytesIO(data)) as pil:
        assert pil.info.get("exif")
        assert pil.info.get("icc_profile") == icc


def test_item_length_matches_appended_frame(tmp_path):
    """The GContainer Item:Length equals the real appended gain-map frame size
    (how Ultra HDR readers locate the map — must be exact)."""
    import re

    data = encoders.encode_gainmap_jpeg(BASE, _gain_map_jpeg_bytes())
    m = re.search(rb'Item:Length="(\d+)"', data)
    assert m, "no Item:Length in the primary XMP"
    length = int(m.group(1))
    # The gain-map frame is the trailing `length` bytes and is its own JPEG.
    frame = data[-length:]
    assert frame[:2] == b"\xff\xd8"
    assert data[: len(data) - length].endswith(b"\xff\xd9")


def test_mpf_index_parses(tmp_path):
    """The hand-built MPF APP2 parses back (our own reader helper) with the
    exact primary/gain-map sizes."""
    gm_jpeg = _gain_map_jpeg_bytes()
    data = encoders.encode_gainmap_jpeg(BASE, gm_jpeg)
    out = tmp_path / "ultra.jpg"
    out.write_bytes(data)

    with Image.open(out) as pil:
        mp = pil.info.get("mp")
        mp_abs = pil.info.get("mpoffset")
    assert mp and mp_abs
    entries = imageio._parse_mpf_index(bytes(mp))
    assert len(entries) == 2
    (p_size, p_off), (g_size, g_off) = entries
    assert p_off == 0
    assert p_size + g_size == len(data)
    assert mp_abs + g_off == p_size  # second frame starts where primary ends


def test_non_jpeg_gain_map_bytes_raise():
    with pytest.raises(EncodingError):
        encoders.encode_gainmap_jpeg(BASE, b"not a jpeg at all")


def test_uint16_base_rounds_down(tmp_path):
    """A uint16 base is rounded down — the gain-map base is 8-bit by design."""
    base16 = (BASE.astype(np.uint16)) * 257
    data = encoders.encode_gainmap_jpeg(base16, _gain_map_jpeg_bytes())
    out = tmp_path / "ultra.jpg"
    out.write_bytes(data)
    loaded = imageio.load(str(out))
    assert loaded.image.dtype == np.uint8
    assert loaded.gain_map is not None


# ------------------------------------------------------------- integration


def _write_apple_style_mpo(path: Path) -> None:
    gm = Image.fromarray(GAIN_MAP, "L")
    gm.encoderinfo = {"xmp": APPLE_GAINMAP_XMP}
    Image.fromarray(BASE, "RGB").save(
        str(path), format="MPO", save_all=True, append_images=[gm]
    )


def test_full_circle_recompress_recaptures_gain_map(tmp_path):
    """HDR source -> .fkeep -> restore .jpg (Ultra HDR) -> compress AGAIN:
    the restored file's gain map is re-extracted and re-stored — the loop
    never drops HDR."""
    src = tmp_path / "hdr.jpg"
    _write_apple_style_mpo(src)
    cfg = FaceKeepConfig(mode="aggressive")

    fkeep1 = tmp_path / "first.fkeep"
    write_fkeep(compress_photo(str(src), cfg), str(fkeep1))
    restored = tmp_path / "restored.jpg"
    Restorer(cfg.aggressive).restore(str(fkeep1), str(restored))

    fkeep2 = tmp_path / "second.fkeep"
    write_fkeep(compress_photo(str(restored), cfg), str(fkeep2))
    with zipfile.ZipFile(fkeep2) as zf:
        assert "gainmap.jpg" in set(zf.namelist())


def test_gain_map_less_restore_stays_plain(tmp_path):
    """A .fkeep without a gain map restores to a plain single-frame JPEG —
    no MPF, no hdrgm markers, byte-for-byte the pre-9.3 write path."""
    import cv2

    src = tmp_path / "plain.jpg"
    cv2.imwrite(str(src), np.full((64, 96, 3), 120, np.uint8))
    cfg = FaceKeepConfig(mode="aggressive")
    fkeep = tmp_path / "plain.fkeep"
    write_fkeep(compress_photo(str(src), cfg), str(fkeep))

    out = tmp_path / "restored.jpg"
    Restorer(cfg.aggressive).restore(str(fkeep), str(out))
    raw = out.read_bytes()
    assert b"hdrgm" not in raw
    assert b"MPF\x00" not in raw
    with Image.open(out) as pil:
        assert pil.format == "JPEG"
        assert getattr(pil, "n_frames", 1) == 1
