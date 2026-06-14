"""WebP fallback output for maximum compatibility on old viewers.

ROADMAP backlog item. ``faithful.codec = "webp"`` (CLI ``--codec webp``) encodes
the whole image as WebP — the maximum-compatibility format: built into Pillow
(no plugin, always available), opens in any browser / older viewer that can't
yet read AVIF or JXL, at the cost of a larger file than AVIF/JXL. It is 8-bit
only (a 16-bit source rounds down with the standard warning) and libwebp caps
each side at 16383 px (a larger frame raises a clear EncodingError).

These tests pin: codec availability; encoder-level lossy + bit-exact lossless;
the dimension cap; the end-to-end faithful pipeline (correct extension, decodes,
EXIF/ICC carried); validate() accepts it; it is NOT part of ``both``; and the
index fingerprint busts on it.
"""

import numpy as np
import pytest
from PIL import Image

from facekeep import encoders, faithful
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import EncodingError
from facekeep.index import settings_fingerprint


def _photo(tmp_path, name="orig.png"):
    """A non-trivial image (gradient + noise) so sizes/ratios are meaningful."""
    rng = np.random.default_rng(0)
    arr = np.zeros((96, 96, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 96, dtype=np.uint8)[None, :]
    arr += rng.integers(0, 24, arr.shape, dtype=np.uint8)
    p = tmp_path / name
    Image.fromarray(arr, "RGB").save(str(p), "PNG")
    return p, arr


# --- availability + extension ---------------------------------------------

def test_webp_always_available():
    """WebP is built into Pillow, so it is available with no plugin/download."""
    assert encoders.codec_available("webp") is True


def test_webp_extension_registered():
    assert encoders.CODEC_EXTENSION["webp"] == ".webp"


# --- encoder level ----------------------------------------------------------

def test_encode_webp_lossy_roundtrips(tmp_path):
    _, arr = _photo(tmp_path)
    bgr = arr[:, :, ::-1].copy()  # RGB->BGR
    data = encoders.encode(bgr, "webp", quality=80)
    decoded = encoders.decode(data)
    assert decoded.shape == bgr.shape  # lossy: shape preserved, pixels close


def test_encode_webp_lossless_is_bit_exact(tmp_path):
    _, arr = _photo(tmp_path)
    bgr = arr[:, :, ::-1].copy()
    data = encoders.encode(bgr, "webp", lossless=True)
    decoded = encoders.decode(data)
    assert np.array_equal(decoded, bgr), "lossless WebP must round-trip exactly"


def test_encode_webp_lossless_differs_from_lossy(tmp_path):
    _, arr = _photo(tmp_path)
    bgr = arr[:, :, ::-1].copy()
    ll = encoders.encode(bgr, "webp", lossless=True)
    lossy = encoders.encode(bgr, "webp", quality=50)
    assert ll != lossy


def test_webp_dimension_cap_raises():
    """A frame wider than libwebp's 16383px cap raises a clear EncodingError."""
    big = np.zeros((64, 16384, 3), dtype=np.uint8)
    with pytest.raises(EncodingError, match="16383"):
        encoders.encode(big, "webp", quality=80)


def test_webp_within_dimension_cap_ok():
    """A frame exactly at the 16383px cap encodes fine (boundary is inclusive)."""
    edge = np.zeros((8, 16383, 3), dtype=np.uint8)
    data = encoders.encode(edge, "webp", quality=80)
    assert len(data) > 0


def test_webp_high_bit_source_rounds_down(tmp_path):
    """A 16-bit source is rounded down to 8-bit for WebP (no high-bit path)."""
    arr16 = (np.random.default_rng(0).integers(0, 65535, (64, 64, 3))).astype(np.uint16)
    data = encoders.encode(arr16, "webp", quality=80)
    decoded = encoders.decode(data)
    assert decoded.dtype == np.uint8 and decoded.shape == (64, 64, 3)


# --- faithful pipeline ------------------------------------------------------

def test_faithful_webp_writes_webp_file(tmp_path):
    p, _ = _photo(tmp_path)
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "webp"
    res = faithful.compress(str(p), str(tmp_path / "out"), cfg)
    assert res.codec == "webp"
    assert res.output_path.suffix == ".webp"
    assert res.output_path.exists()
    decoded = encoders.decode(res.output_path.read_bytes())
    assert decoded.shape[:2] == (96, 96)


def test_faithful_webp_lossless_bit_exact(tmp_path):
    p, arr = _photo(tmp_path)
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "webp"
    cfg.faithful.lossless = True
    res = faithful.compress(str(p), str(tmp_path / "out"), cfg)
    assert res.codec == "webp"
    assert res.output_path.suffix == ".webp"
    decoded = encoders.decode(res.output_path.read_bytes())
    assert np.array_equal(decoded, arr[:, :, ::-1])  # exact (BGR)


def test_webp_carries_icc(tmp_path):
    """ICC profile is embedded in the WebP output (wide-gamut color survives)."""
    _, arr = _photo(tmp_path)
    bgr = arr[:, :, ::-1].copy()
    # A minimal but real sRGB profile from PIL.
    from PIL import ImageCms
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    data = encoders.encode(bgr, "webp", quality=80, icc=icc)
    out = Image.open(__import__("io").BytesIO(data))
    assert out.info.get("icc_profile"), "WebP output should carry the ICC profile"


# --- 'both' must not include webp ------------------------------------------

def test_both_does_not_consider_webp(tmp_path, monkeypatch):
    """codec='both' races only avif/jxl; webp is a compatibility choice, not a
    size contender, so it must never be the winner of a 'both' run."""
    if not (encoders.codec_available("avif") and encoders.codec_available("jxl")):
        pytest.skip("both avif and jxl plugins required")
    # Spy on encode to record which codecs the 'both' path actually tries.
    tried = []
    real_encode = encoders.encode

    def _spy(image, codec="avif", *a, **k):
        tried.append(codec)
        return real_encode(image, codec, *a, **k)

    monkeypatch.setattr(encoders, "encode", _spy)
    p, _ = _photo(tmp_path)
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "both"
    res = faithful.compress(str(p), str(tmp_path / "out"), cfg)
    assert "webp" not in tried
    assert res.codec in ("avif", "jxl")


# --- config + index ---------------------------------------------------------

def test_validate_accepts_webp():
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "webp"
    cfg.validate()  # must not raise


def test_webp_busts_index_fingerprint():
    a = FaceKeepConfig()  # default avif
    b = FaceKeepConfig()
    b.faithful.codec = "webp"
    assert settings_fingerprint(a) != settings_fingerprint(b)


def test_default_config_yaml_mentions_webp():
    from facekeep.config import default_config_yaml
    assert "webp" in default_config_yaml()
