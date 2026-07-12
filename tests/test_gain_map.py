"""HDR gain-map carry-through at load (ROADMAP Phase 9.1 / Stage 0).

A modern iPhone HDR still is an 8-bit Display-P3 base plus an Apple HDR *gain
map* (HEIC: the ``…aux:hdrgainmap`` auxiliary image; JPEG: an MPF second frame
whose XMP names the gain map) — not a 10/12-bit deep-color image. ``imageio.load``
used to drop the gain map entirely; it now carries it on ``LoadedImage.gain_map``
(+ ``gain_map_meta``), upright and aligned with the base image. Nothing consumes
it yet, so a load with a gain map present changes no other field.

Test layers, matching the repo's gating conventions:

- **Always-on:** synthetic fixtures. A JPEG-MPF gain-map fixture is authored
  with Pillow (``format="MPO"``, XMP on the appended frame via ``encoderinfo``);
  HEIC no-gain-map fixtures via ``pillow_heif.from_bytes`` (pillow_heif cannot
  *author* aux images, so HEIC extraction is covered by the real-asset layer).
- **Real-asset-gated:** the two local iPhone sample photos
  (``assets/IMG_3457.HEIC`` / ``.JPG``) — not in the repo, so these skip
  elsewhere, like the corpus tests.

HEIC is only ever loaded through ``imageio.load`` (open_heif) — never PIL
``Image.open`` — per the dual-API segfault rule in CLAUDE.md.
"""

from pathlib import Path

import cv2
import numpy as np
import piexif
import pytest
from PIL import Image

from facekeep import imageio

ASSETS = Path(__file__).resolve().parent.parent / "assets"
REAL_HEIC = ASSETS / "IMG_3457.HEIC"
REAL_JPG = ASSETS / "IMG_3457.JPG"

APPLE_GAINMAP_XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description rdf:about="" '
    b'xmlns:apdi="http://ns.apple.com/pixeldatainfo/1.0/" '
    b'xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/">'
    b"<apdi:AuxiliaryImageType>urn:com:apple:photo:2020:aux:hdrgainmap"
    b"</apdi:AuxiliaryImageType>"
    b"<HDRGainMap:HDRGainMapVersion>131072</HDRGainMap:HDRGainMapVersion>"
    b"</rdf:Description></rdf:RDF></x:xmpmeta>"
)


def _write_mpo(path, *, base_size=(64, 48), gm_array=None, frame_xmp=APPLE_GAINMAP_XMP,
               exif=None):
    """Author a two-frame MPO JPEG: an RGB base + a mode-L second frame.

    ``frame_xmp`` lands on the second frame via its ``encoderinfo`` (verified:
    Pillow's MPO saver carries per-frame encoderinfo XMP). ``exif`` (raw bytes)
    goes on the base frame.
    """
    if gm_array is None:
        gm_array = np.full((base_size[1] // 2, base_size[0] // 2), 128, np.uint8)
    base = Image.new("RGB", base_size, (200, 100, 50))
    gm = Image.fromarray(gm_array, "L")
    if frame_xmp is not None:
        gm.encoderinfo = {"xmp": frame_xmp}
    kw = {}
    if exif is not None:
        kw["exif"] = exif
    base.save(str(path), format="MPO", save_all=True, append_images=[gm], **kw)


# ---------------------------------------------------------------- always-on


def test_plain_jpeg_has_no_gain_map(tmp_path):
    """A normal single-frame JPEG loads with gain_map/meta None (the default)."""
    src = tmp_path / "plain.jpg"
    cv2.imwrite(str(src), np.full((48, 64, 3), 120, np.uint8))

    loaded = imageio.load(str(src))

    assert loaded.gain_map is None
    assert loaded.gain_map_meta is None


def test_plain_png_has_no_gain_map(tmp_path):
    """Non-JPEG OpenCV formats never even probe for MPF."""
    src = tmp_path / "plain.png"
    cv2.imwrite(str(src), np.full((48, 64, 3), 120, np.uint8))

    loaded = imageio.load(str(src))

    assert loaded.gain_map is None
    assert loaded.gain_map_meta is None


def test_mpo_jpeg_gain_map_extracted(tmp_path):
    """The MPF second frame is extracted when its XMP names a gain map."""
    src = tmp_path / "hdr.jpg"
    gm = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32) % 200
    _write_mpo(src, base_size=(64, 48), gm_array=gm)

    loaded = imageio.load(str(src))

    assert loaded.gain_map is not None
    assert loaded.gain_map.dtype == np.uint8
    assert loaded.gain_map.shape == (24, 32)
    # JPEG is lossy; the flat-ish ramp should survive approximately.
    assert abs(float(loaded.gain_map.mean()) - float(gm.mean())) < 5.0
    assert loaded.gain_map_meta["source"] == "jpeg-mpf"
    assert loaded.gain_map_meta["frame_index"] == 1
    assert b"hdrgainmap" in loaded.gain_map_meta["xmp"].lower()
    # The base image itself is unaffected.
    assert (loaded.height, loaded.width) == (48, 64)
    assert loaded.image.shape == (48, 64, 3)


def test_mpo_without_gainmap_xmp_is_ignored(tmp_path):
    """A multi-frame MPO whose frames don't name a gain map (e.g. a stereo
    pair) must not be misread as HDR."""
    src = tmp_path / "stereo.jpg"
    _write_mpo(src, frame_xmp=None)

    loaded = imageio.load(str(src))

    assert loaded.gain_map is None
    assert loaded.gain_map_meta is None


def test_mpo_gain_map_rotates_with_base(tmp_path):
    """EXIF orientation is applied to the gain map exactly like the base, so
    the two stay aligned (the MPF frame is stored un-rotated like the base)."""
    src = tmp_path / "rot.jpg"
    gm = np.zeros((24, 32), np.uint8)
    gm[:, :16] = 255  # left half bright, in stored (landscape) coordinates
    exif6 = piexif.dump({"0th": {piexif.ImageIFD.Orientation: 6}})
    _write_mpo(src, base_size=(64, 48), gm_array=gm, exif=exif6)

    loaded = imageio.load(str(src))

    # Orientation 6 = rotate 90 CW: base 64x48 -> 48x64 upright.
    assert (loaded.width, loaded.height) == (48, 64)
    assert loaded.gain_map is not None
    assert loaded.gain_map.shape == (32, 24)  # rotated with the base
    # The stored-left bright half becomes the top half after 90 CW.
    top, bottom = loaded.gain_map[:16], loaded.gain_map[16:]
    assert float(top.mean()) > 200 > 50 > float(bottom.mean())


def test_synthetic_heic_has_no_gain_map(tmp_path):
    """A pillow_heif-authored HEIC (no aux images) loads with gain_map None."""
    pillow_heif = pytest.importorskip("pillow_heif")
    rgb = np.dstack([
        np.full((48, 64), 200, np.uint8),
        np.full((48, 64), 100, np.uint8),
        np.full((48, 64), 50, np.uint8),
    ])
    src = tmp_path / "plain.heic"
    pillow_heif.from_bytes(mode="RGB", size=(64, 48), data=rgb.tobytes()).save(str(src))

    loaded = imageio.load(str(src))

    assert loaded.gain_map is None
    assert loaded.gain_map_meta is None
    assert loaded.image.shape == (48, 64, 3)


# ---------------------------------------------------------- real-asset-gated


@pytest.mark.skipif(not REAL_HEIC.exists(), reason="local iPhone sample not present")
def test_real_iphone_heic_gain_map():
    """The real iPhone HDR HEIC yields its aux gain map, aligned with the base."""
    loaded = imageio.load(str(REAL_HEIC))

    assert loaded.gain_map is not None
    assert loaded.gain_map.ndim == 2
    assert loaded.gain_map.dtype == np.uint8
    assert loaded.gain_map_meta["source"] == "heic-aux"
    assert "hdrgainmap" in loaded.gain_map_meta["urn"].lower()
    # Half-resolution and same aspect as the (upright) base.
    gh, gw = loaded.gain_map.shape
    assert abs(gh / loaded.height - 0.5) < 0.01
    assert abs(gw / loaded.width - 0.5) < 0.01


@pytest.mark.skipif(not REAL_JPG.exists(), reason="local iPhone sample not present")
def test_real_iphone_jpeg_gain_map():
    """The real iPhone HDR JPEG yields its MPF gain map, upright like the base."""
    loaded = imageio.load(str(REAL_JPG))

    assert loaded.gain_map is not None
    assert loaded.gain_map.ndim == 2
    assert loaded.gain_map_meta["source"] == "jpeg-mpf"
    assert b"hdrgainmap" in loaded.gain_map_meta["xmp"].lower()
    # This photo is portrait (orientation 6 applied): the gain map must have
    # been rotated with the base — same aspect, half resolution.
    assert loaded.height > loaded.width
    gh, gw = loaded.gain_map.shape
    assert gh > gw
    assert abs(gh / loaded.height - 0.5) < 0.01
    assert abs(gw / loaded.width - 0.5) < 0.01
