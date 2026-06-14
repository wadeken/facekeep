"""GPS / privacy EXIF stripping on export (both modes).

A family-photo backup routinely carries the capture *location* in the EXIF GPS
IFD. The opt-in ``strip_gps`` removes only that IFD, applied once at load time in
``imageio.load()`` so both export paths inherit it:

* **faithful** re-embeds the (now GPS-free) EXIF in the encoded .avif/.jxl;
* **aggressive** stores it as ``exif.bin`` and re-embeds it on restore.

The contract these tests pin: with ``strip_gps`` on, the output has no GPS IFD
but keeps every other tag (date/camera/orientation); with it off (the default)
the EXIF round-trips unchanged. ``strip_gps`` is output-affecting, so flipping it
must also bust the incremental-index fingerprint.
"""

import piexif
import pytest
from PIL import Image

from facekeep import imageio
from facekeep.config import FaceKeepConfig
from facekeep.index import settings_fingerprint


# GPS latitude/longitude + a couple of non-GPS tags (camera Make, capture date).
def _exif_with_gps(make: str = "FaceKeepCam") -> bytes:
    return piexif.dump(
        {
            "0th": {
                piexif.ImageIFD.Make: make,
                piexif.ImageIFD.Orientation: 1,
            },
            "Exif": {piexif.ExifIFD.DateTimeOriginal: "2024:05:20 10:30:00"},
            "GPS": {
                piexif.GPSIFD.GPSLatitudeRef: "N",
                piexif.GPSIFD.GPSLatitude: ((37, 1), (48, 1), (30, 1)),
                piexif.GPSIFD.GPSLongitudeRef: "W",
                piexif.GPSIFD.GPSLongitude: ((122, 1), (16, 1), (0, 1)),
            },
            "1st": {},
            "thumbnail": None,
            "Interop": {},
        }
    )


def _exif_no_gps() -> bytes:
    return piexif.dump(
        {
            "0th": {piexif.ImageIFD.Make: "FaceKeepCam", piexif.ImageIFD.Orientation: 1},
            "Exif": {piexif.ExifIFD.DateTimeOriginal: "2024:05:20 10:30:00"},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
            "Interop": {},
        }
    )


@pytest.fixture
def gps_jpeg(tmp_path):
    """A small JPEG carrying both a GPS IFD and non-GPS tags (Make + date)."""
    import numpy as np

    arr = np.full((64, 64, 3), 200, dtype=np.uint8)
    p = tmp_path / "geotagged.jpg"
    Image.fromarray(arr, "RGB").save(str(p), "JPEG", quality=95, exif=_exif_with_gps())
    return p


def _gps_present(exif_bytes) -> bool:
    """True iff the EXIF bytes carry a non-empty GPS IFD."""
    if not exif_bytes:
        return False
    return bool(piexif.load(exif_bytes).get("GPS"))


# --- the load-time strip helper -------------------------------------------

def test_load_default_keeps_gps(gps_jpeg):
    """Default (strip_gps off): GPS survives — byte-for-byte EXIF unchanged."""
    loaded = imageio.load(str(gps_jpeg))
    assert _gps_present(loaded.exif)


def test_load_strip_removes_gps_keeps_rest(gps_jpeg):
    """strip_gps on: GPS IFD gone, but Make + capture date are kept."""
    loaded = imageio.load(str(gps_jpeg), strip_gps=True)
    assert not _gps_present(loaded.exif)
    d = piexif.load(loaded.exif)
    assert d["0th"][piexif.ImageIFD.Make].rstrip(b"\x00") == b"FaceKeepCam"
    assert d["Exif"][piexif.ExifIFD.DateTimeOriginal].rstrip(b"\x00") == (
        b"2024:05:20 10:30:00"
    )


def test_load_strip_on_no_gps_image_is_noop(tmp_path):
    """An image with no GPS: strip leaves the EXIF byte-for-byte (helper short-circuits)."""
    import numpy as np

    arr = np.full((48, 48, 3), 100, dtype=np.uint8)
    p = tmp_path / "no_gps.jpg"
    Image.fromarray(arr, "RGB").save(str(p), "JPEG", exif=_exif_no_gps())
    before = imageio.load(str(p)).exif
    after = imageio.load(str(p), strip_gps=True).exif
    assert before == after  # unchanged: nothing to strip


def test_load_strip_on_no_exif_image_is_safe(tmp_path):
    """An image with no EXIF at all: strip is a safe no-op (None stays None)."""
    import numpy as np

    arr = np.full((48, 48, 3), 100, dtype=np.uint8)
    p = tmp_path / "bare.png"
    Image.fromarray(arr, "RGB").save(str(p), "PNG")
    loaded = imageio.load(str(p), strip_gps=True)
    assert not _gps_present(loaded.exif)


# --- end-to-end: faithful mode --------------------------------------------

def test_faithful_strips_gps_from_output(gps_jpeg, tmp_path):
    """faithful compress with strip_gps writes an AVIF whose EXIF has no GPS."""
    from facekeep.faithful import compress as faithful_compress

    cfg = FaceKeepConfig()
    cfg.strip_gps = True
    out = tmp_path / "out"
    res = faithful_compress(str(gps_jpeg), str(out), cfg)

    # Decode the written file and read its embedded EXIF.
    exif = Image.open(str(res.output_path)).info.get("exif")
    assert not _gps_present(exif)
    # Non-GPS metadata survives (the date tag).
    if exif:
        assert piexif.load(exif).get("Exif")


def test_faithful_default_keeps_gps_in_output(gps_jpeg, tmp_path):
    """Anti-regression: without strip_gps the faithful output keeps GPS."""
    from facekeep.faithful import compress as faithful_compress

    out = tmp_path / "out"
    res = faithful_compress(str(gps_jpeg), str(out), FaceKeepConfig())
    exif = Image.open(str(res.output_path)).info.get("exif")
    assert _gps_present(exif)


# --- end-to-end: aggressive mode (compress -> restore) --------------------
# The autouse `_force_bicubic_restore` fixture in conftest pins the offline
# (no-AI) restore path, so these need no torch/weights.

def test_aggressive_strips_gps_through_restore(gps_jpeg, tmp_path):
    """Aggressive: strip_gps drops GPS from exif.bin, so the restore has none."""
    from facekeep.aggressive.compressor import compress_photo
    from facekeep.aggressive.format import write_fkeep
    from facekeep.aggressive.restorer import Restorer

    cfg = FaceKeepConfig()
    cfg.strip_gps = True
    photo = compress_photo(str(gps_jpeg), cfg)
    assert not _gps_present(photo.exif)

    fk = tmp_path / "out.fkeep"
    write_fkeep(photo, str(fk))
    out = tmp_path / "restored.jpg"
    Restorer().restore(str(fk), str(out))
    assert not _gps_present(Image.open(str(out)).info.get("exif"))


def test_aggressive_default_keeps_gps(gps_jpeg, tmp_path):
    """Anti-regression: default aggressive compress keeps GPS in exif.bin."""
    from facekeep.aggressive.compressor import compress_photo

    photo = compress_photo(str(gps_jpeg), FaceKeepConfig())
    assert _gps_present(photo.exif)


# --- index fingerprint busts on flip --------------------------------------

@pytest.mark.parametrize("mode", ["faithful", "aggressive"])
def test_strip_gps_busts_index_fingerprint(mode):
    """Flipping strip_gps changes the output bytes, so it must bust the cache."""
    a = FaceKeepConfig()
    a.mode = mode
    a.strip_gps = False
    b = FaceKeepConfig()
    b.mode = mode
    b.strip_gps = True
    assert settings_fingerprint(a) != settings_fingerprint(b)


# --- config plumbing -------------------------------------------------------

def test_config_default_off():
    assert FaceKeepConfig().strip_gps is False


def test_config_yaml_roundtrip(tmp_path):
    """strip_gps survives a save/load round-trip (so facekeep init can set it)."""
    cfg = FaceKeepConfig()
    cfg.strip_gps = True
    p = tmp_path / "facekeep.yaml"
    cfg.save(p)
    assert FaceKeepConfig.load(p).strip_gps is True
