"""hdrgm gain-map parameter fidelity (ROADMAP 9.4).

An Android Ultra HDR JPEG declares its gain-map application math in the
gain-map frame's Adobe hdrgm XMP (GainMapMin/Max, Gamma, OffsetSDR/HDR,
HDRCapacityMin/Max, BaseRenditionIsHDR). Before 9.4 those values were dropped:
restore re-wrote the fixed Apple-semantics values (Gamma=1, Min=0,
Max=headroom, zero offsets), so e.g. a ``GainMapMax=4.5`` Android photo would
restore with a wrong HDR brightness scale. Now:

* ``imageio.parse_hdrgm_xmp`` parses the params off the frame XMP at load
  (attribute *and* element form, per-channel ``rdf:Seq`` values, rationals),
  attaching them as ``gain_map_meta["hdrgm"]``;
* aggressive compress stores them as the optional manifest key
  ``gain_map_params`` (manifest 1.11.0 — absent = the Apple defaults, so every
  existing ``.fkeep`` restores unchanged);
* restore re-emits them verbatim in the Ultra HDR JPEG's hdrgm XMP
  (``encoders._hdrgm_xmp``) and uses them in the AVIF path's boost math
  (``encoders._apply_gain_map``).

Everything here is offline/synthetic — the "Android source" is authored with
FaceKeep's own Ultra HDR writer carrying custom params. Real-Android-photo
verification is 9.4's separate acceptance step (needs the physical phone).
"""

import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from facekeep import encoders, imageio
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.config import FaceKeepConfig

BASE = np.tile(
    np.linspace(20, 235, 96, dtype=np.uint8)[None, :, None], (64, 1, 3)
)
GAIN_MAP = np.tile(np.linspace(0, 200, 48, dtype=np.uint8), (32, 1))

# Android-flavor params: values a Pixel-style writer plausibly declares, all
# deliberately different from the Apple defaults (Max 3.0 / Gamma 1 / offsets 0).
ANDROID_PARAMS = {
    "gain_map_min": 0.0,
    "gain_map_max": 4.5,
    "gamma": 1.2,
    "offset_sdr": 0.015625,
    "offset_hdr": 0.015625,
    "hdr_capacity_min": 0.0,
    "hdr_capacity_max": 4.5,
    "base_rendition_is_hdr": False,
}


def _gain_map_jpeg_bytes() -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", GAIN_MAP, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _android_style_source(path) -> None:
    """Author a synthetic 'Android Ultra HDR' JPEG: our own writer carrying
    non-Apple hdrgm params (same MPF/XMP shape as a real Pixel photo)."""
    path.write_bytes(
        encoders.encode_gainmap_jpeg(
            BASE, _gain_map_jpeg_bytes(), gain_map_params=ANDROID_PARAMS
        )
    )


# ------------------------------------------------------------------- parser


def _xmp(attrs: str = "", body: str = "") -> bytes:
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"'
        f' hdrgm:Version="1.0"{attrs}'
        + (f">{body}</rdf:Description>" if body else "/>")
        + "</rdf:RDF></x:xmpmeta>"
    ).encode()


def test_parse_attribute_form():
    params = imageio.parse_hdrgm_xmp(_xmp(
        ' hdrgm:GainMapMin="0.5" hdrgm:GainMapMax="4.5" hdrgm:Gamma="1.2"'
        ' hdrgm:OffsetSDR="0.001" hdrgm:OffsetHDR="0.002"'
        ' hdrgm:HDRCapacityMin="0.1" hdrgm:HDRCapacityMax="4.0"'
        ' hdrgm:BaseRenditionIsHDR="False"'
    ))
    assert params == {
        "gain_map_min": 0.5,
        "gain_map_max": 4.5,
        "gamma": 1.2,
        "offset_sdr": 0.001,
        "offset_hdr": 0.002,
        "hdr_capacity_min": 0.1,
        "hdr_capacity_max": 4.0,
        "base_rendition_is_hdr": False,
    }


def test_parse_fills_spec_defaults():
    """Absent attributes take the Adobe spec defaults (offsets 1/64, Min 0,
    Gamma 1; HDRCapacityMax falls back to GainMapMax)."""
    params = imageio.parse_hdrgm_xmp(_xmp(' hdrgm:GainMapMax="2.0"'))
    assert params["gain_map_min"] == 0.0
    assert params["gamma"] == 1.0
    assert params["offset_sdr"] == pytest.approx(1 / 64)
    assert params["offset_hdr"] == pytest.approx(1 / 64)
    assert params["hdr_capacity_min"] == 0.0
    assert params["hdr_capacity_max"] == 2.0
    assert params["base_rendition_is_hdr"] is False


def test_parse_element_form_and_per_channel_seq():
    """Element form + a per-channel rdf:Seq (seen in the wild) parse; the Seq
    stays a 3-list in the spec's RGB order."""
    params = imageio.parse_hdrgm_xmp(_xmp(
        ' hdrgm:BaseRenditionIsHDR="True"',
        "<hdrgm:GainMapMax><rdf:Seq>"
        "<rdf:li>4.0</rdf:li><rdf:li>4.1</rdf:li><rdf:li>4.2</rdf:li>"
        "</rdf:Seq></hdrgm:GainMapMax>"
        "<hdrgm:Gamma>1.5</hdrgm:Gamma>",
    ))
    assert params["gain_map_max"] == [4.0, 4.1, 4.2]
    assert params["gamma"] == 1.5
    # HDRCapacityMax default = max over the per-channel GainMapMax.
    assert params["hdr_capacity_max"] == 4.2
    assert params["base_rendition_is_hdr"] is True


def test_parse_rational_values():
    params = imageio.parse_hdrgm_xmp(_xmp(' hdrgm:GainMapMax="9/4"'))
    assert params["gain_map_max"] == pytest.approx(2.25)


def test_parse_rejects_no_gain_map_max():
    """GainMapMax is the one required attribute; a packet without it (e.g. an
    Apple frame XMP, which only *names* the map) yields None."""
    assert imageio.parse_hdrgm_xmp(_xmp(' hdrgm:Gamma="1.0"')) is None
    apple = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/" '
        b'xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/">'
        b"<apdi:AuxiliaryImageType>urn:com:apple:photo:2020:aux:hdrgainmap"
        b"</apdi:AuxiliaryImageType></x:xmpmeta>"
    )
    assert imageio.parse_hdrgm_xmp(apple) is None


def test_parse_best_effort_on_garbage():
    assert imageio.parse_hdrgm_xmp(b"") is None
    assert imageio.parse_hdrgm_xmp(b"not xml at all") is None
    assert imageio.parse_hdrgm_xmp(b"<x:xmpmeta broken") is None
    # A nonsensical Gamma (<= 0) falls back to None rather than dividing by 0.
    assert imageio.parse_hdrgm_xmp(
        _xmp(' hdrgm:GainMapMax="3.0" hdrgm:Gamma="0"')
    ) is None


# ------------------------------------------------------------------- writer


def test_default_hdrgm_xmp_unchanged():
    """params=None emits byte-for-byte the pre-9.4 Apple-defaults packet —
    Apple/HEIC sources and every existing .fkeep keep their exact output."""
    expected = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"'
        ' hdrgm:Version="1.0"'
        ' hdrgm:GainMapMin="0.0" hdrgm:GainMapMax="3.0"'
        ' hdrgm:Gamma="1.0"'
        ' hdrgm:OffsetSDR="0.0" hdrgm:OffsetHDR="0.0"'
        ' hdrgm:HDRCapacityMin="0.0" hdrgm:HDRCapacityMax="3.0"'
        ' hdrgm:BaseRenditionIsHDR="False"/>'
        "</rdf:RDF></x:xmpmeta>"
    ).encode()
    assert encoders._hdrgm_xmp(3.0) == expected
    assert encoders._hdrgm_xmp(3.0, None) == expected


def test_params_xmp_roundtrips_through_parser():
    """Writer -> parser is the identity on the param dict (scalar form)."""
    packet = encoders._hdrgm_xmp(3.0, ANDROID_PARAMS)
    assert imageio.parse_hdrgm_xmp(packet) == ANDROID_PARAMS


def test_per_channel_params_roundtrip():
    params = dict(ANDROID_PARAMS, gain_map_max=[4.0, 4.1, 4.2])
    packet = encoders._hdrgm_xmp(3.0, params)
    assert b"<rdf:Seq>" in packet  # per-channel values take the element form
    assert imageio.parse_hdrgm_xmp(packet) == params


# --------------------------------------------------------------- boost math


def test_apply_gain_map_default_params_match_apple_expression():
    gain = np.random.default_rng(0).random((8, 8, 1)).astype(np.float32)
    base = np.random.default_rng(1).random((8, 8, 3)).astype(np.float32)
    apple = base * (2.0 ** (3.0 * gain))
    assert np.array_equal(encoders._apply_gain_map(base, gain, 3.0, None), apple)
    equivalent = {
        "gain_map_min": 0.0, "gain_map_max": 3.0, "gamma": 1.0,
        "offset_sdr": 0.0, "offset_hdr": 0.0,
    }
    assert np.allclose(
        encoders._apply_gain_map(base, gain, 3.0, equivalent), apple, atol=1e-5
    )


def test_apply_gain_map_uses_source_math():
    """Non-default params change the alternate the documented way."""
    gain = np.full((4, 4, 1), 0.5, np.float32)
    base = np.full((4, 4, 3), 0.25, np.float32)
    params = {
        "gain_map_min": 0.5, "gain_map_max": 4.5, "gamma": 2.0,
        "offset_sdr": 0.01, "offset_hdr": 0.02,
    }
    log_boost = 0.5 + (4.5 - 0.5) * (0.5 ** (1 / 2.0))
    expected = (0.25 + 0.01) * (2.0 ** log_boost) - 0.02
    out = encoders._apply_gain_map(base, gain, 3.0, params)
    assert out == pytest.approx(np.full((4, 4, 3), expected), abs=1e-4)


def test_apply_gain_map_per_channel_broadcast():
    """Per-channel (RGB-ordered) params broadcast onto the BGR base."""
    gain = np.ones((2, 2, 1), np.float32)
    base = np.ones((2, 2, 3), np.float32)
    params = {
        "gain_map_min": 0.0, "gain_map_max": [1.0, 2.0, 3.0],  # R, G, B
        "gamma": 1.0, "offset_sdr": 0.0, "offset_hdr": 0.0,
    }
    out = encoders._apply_gain_map(base, gain, 3.0, params)
    # Internal order is BGR: channel 0 gets the *B* max (3.0) -> 2^3.
    assert out[0, 0].tolist() == [8.0, 4.0, 2.0]


def test_apply_gain_map_base_rendition_is_hdr_falls_back():
    """The documented approximation: an HDR base rendition (unseen from phone
    writers) warns and uses the default headroom boost."""
    gain = np.random.default_rng(2).random((4, 4, 1)).astype(np.float32)
    base = np.random.default_rng(3).random((4, 4, 3)).astype(np.float32)
    params = dict(ANDROID_PARAMS, base_rendition_is_hdr=True)
    apple = base * (2.0 ** (3.0 * gain))
    assert np.array_equal(
        encoders._apply_gain_map(base, gain, 3.0, params), apple
    )


# ------------------------------------------------------------- integration


def test_load_extracts_android_params(tmp_path):
    src = tmp_path / "android.jpg"
    _android_style_source(src)
    loaded = imageio.load(str(src))
    assert loaded.gain_map is not None
    assert loaded.gain_map_meta["hdrgm"] == ANDROID_PARAMS


def test_fkeep_records_params_and_restore_reemits(tmp_path):
    """Android source -> .fkeep manifest gain_map_params (1.11.0) -> restored
    Ultra HDR JPEG declares the SOURCE math, not the fixed Apple values."""
    src = tmp_path / "android.jpg"
    _android_style_source(src)
    cfg = FaceKeepConfig(mode="aggressive")

    fkeep = tmp_path / "android.fkeep"
    write_fkeep(compress_photo(str(src), cfg), str(fkeep))
    with zipfile.ZipFile(fkeep) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["version"] == "1.11.0"
    assert manifest["gain_map_preserved"] is True
    assert manifest["gain_map_params"] == ANDROID_PARAMS

    out = tmp_path / "restored.jpg"
    Restorer(cfg.aggressive).restore(str(fkeep), str(out))
    raw = out.read_bytes()
    assert b'hdrgm:GainMapMax="4.5"' in raw
    assert b'hdrgm:Gamma="1.2"' in raw
    assert b'hdrgm:GainMapMax="3.0"' not in raw  # not the fixed default
    # The restored file round-trips: loading it re-extracts the same params.
    assert imageio.load(str(out)).gain_map_meta["hdrgm"] == ANDROID_PARAMS


def test_apple_source_manifest_has_no_params_key(tmp_path):
    """An Apple-style source (frame XMP without hdrgm attrs) stores NO
    gain_map_params key, and restore keeps the fixed Apple-default XMP —
    the pre-9.4 behavior, byte-compatible."""
    src = tmp_path / "apple.jpg"
    gm = Image.fromarray(GAIN_MAP, "L")
    gm.encoderinfo = {
        "xmp": (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/" '
            b'xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/">'
            b"<apdi:AuxiliaryImageType>urn:com:apple:photo:2020:aux:hdrgainmap"
            b"</apdi:AuxiliaryImageType></x:xmpmeta>"
        )
    }
    Image.fromarray(BASE, "RGB").save(
        str(src), format="MPO", save_all=True, append_images=[gm]
    )
    cfg = FaceKeepConfig(mode="aggressive")

    fkeep = tmp_path / "apple.fkeep"
    write_fkeep(compress_photo(str(src), cfg), str(fkeep))
    with zipfile.ZipFile(fkeep) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["gain_map_preserved"] is True
    assert "gain_map_params" not in manifest

    out = tmp_path / "restored.jpg"
    Restorer(cfg.aggressive).restore(str(fkeep), str(out))
    raw = out.read_bytes()
    assert b'hdrgm:GainMapMax="3.0"' in raw  # the Apple-default headroom
    assert b'hdrgm:Gamma="1.0"' in raw
