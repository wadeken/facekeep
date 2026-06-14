"""Mathematically-lossless faithful output (archival of irreplaceable originals).

`faithful.lossless` (CLI `--lossless`) encodes the whole image bit-exact, ignoring
quality/auto-tune. JXL is lossless natively (verified bit-exact here). The bundled
pillow-avif has NO lossless path, so lossless AVIF needs the external `avifenc -l`
CLI; without it the encode honestly falls back to lossless JXL (so the user always
gets a genuinely lossless file). It is output-affecting (busts the index cache) and
opts out of skip-if-larger (a lossless file is expected to be larger than a lossy
original — keeping the original would defeat the purpose).
"""

import numpy as np
import pytest
from PIL import Image

from facekeep import encoders
from facekeep.config import FaceKeepConfig
from facekeep.faithful import compress as faithful_compress
from facekeep.index import settings_fingerprint

jxl = pytest.importorskip("pillow_jxl")  # noqa: F841 - lossless path needs JXL


def _photo(tmp_path, name="orig.png"):
    """A non-trivial image (gradient + noise) so lossless != trivially small."""
    rng = np.random.default_rng(0)
    arr = np.zeros((96, 96, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 96, dtype=np.uint8)[None, :]
    arr += rng.integers(0, 24, arr.shape, dtype=np.uint8)
    p = tmp_path / name
    Image.fromarray(arr, "RGB").save(str(p), "PNG")
    return p, arr


# --- encoder level: lossless JXL is bit-exact ------------------------------

def test_encode_lossless_jxl_is_bit_exact(tmp_path):
    _, arr = _photo(tmp_path)
    bgr = arr[:, :, ::-1].copy()  # RGB->BGR
    data = encoders.encode(bgr, "jxl", lossless=True)
    decoded = encoders.decode(data)  # BGR back
    assert np.array_equal(decoded, bgr), "lossless JXL must round-trip exactly"


def test_encode_lossless_jxl_differs_from_lossy(tmp_path):
    """Lossless and lossy produce different bytes (sanity: lossless really fires)."""
    _, arr = _photo(tmp_path)
    bgr = arr[:, :, ::-1].copy()
    ll = encoders.encode(bgr, "jxl", lossless=True)
    lossy = encoders.encode(bgr, "jxl", quality=70)
    assert ll != lossy


# --- encode() AVIF routing -------------------------------------------------

def test_encode_lossless_avif_without_avifenc_raises(tmp_path, monkeypatch):
    """encode(lossless, avif) with no avifenc raises (faithful catches + redirects)."""
    if not encoders.codec_available("avif"):
        pytest.skip("avif plugin not available")
    monkeypatch.setattr(encoders, "_find_avifenc", lambda: None)
    _, arr = _photo(tmp_path)
    bgr = arr[:, :, ::-1].copy()
    with pytest.raises(encoders.EncodingError):
        encoders.encode(bgr, "avif", lossless=True)


# --- faithful pipeline -----------------------------------------------------

def test_faithful_lossless_jxl_roundtrips_exact(tmp_path):
    p, arr = _photo(tmp_path)
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "jxl"
    cfg.faithful.lossless = True
    res = faithful_compress(str(p), str(tmp_path / "out"), cfg)
    assert res.codec == "jxl"
    assert res.output_path.suffix == ".jxl"
    decoded = encoders.decode(res.output_path.read_bytes())
    assert np.array_equal(decoded, arr[:, :, ::-1])  # exact (BGR)


def test_faithful_lossless_avif_redirects_to_jxl_without_avifenc(tmp_path, monkeypatch):
    """Lossless AVIF without avifenc falls back to a genuinely lossless JXL file."""
    if not encoders.codec_available("avif"):
        pytest.skip("avif plugin not available")
    monkeypatch.setattr(encoders, "_find_avifenc", lambda: None)
    p, arr = _photo(tmp_path)
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "avif"
    cfg.faithful.lossless = True
    res = faithful_compress(str(p), str(tmp_path / "out"), cfg)
    # Redirected to JXL, and the file is bit-exact.
    assert res.codec == "jxl"
    assert res.output_path.suffix == ".jxl"
    decoded = encoders.decode(res.output_path.read_bytes())
    assert np.array_equal(decoded, arr[:, :, ::-1])


def test_faithful_lossless_avif_uses_avifenc_when_present(tmp_path, monkeypatch):
    """When avifenc is present, lossless AVIF stays AVIF (no redirect).

    We stub the avifenc encode itself (the binary isn't installed on this box),
    so this asserts the *routing*: avifenc available => codec stays avif and the
    avifenc lossless helper is the path taken.
    """
    if not encoders.codec_available("avif"):
        pytest.skip("avif plugin not available")
    monkeypatch.setattr(encoders, "_find_avifenc", lambda: "/fake/avifenc")
    sentinel = b"FAKE-LOSSLESS-AVIF-BYTES"
    called = {}

    def _fake_ll_avif(image_bgr, *, exif=None, icc=None):
        called["yes"] = True
        return sentinel

    monkeypatch.setattr(encoders, "encode_lossless_avif", _fake_ll_avif)
    p, _ = _photo(tmp_path)
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "avif"
    cfg.faithful.lossless = True
    cfg.faithful.verify = False  # the sentinel bytes aren't a decodable image
    res = faithful_compress(str(p), str(tmp_path / "out"), cfg)
    assert called.get("yes") is True
    assert res.codec == "avif"
    assert res.output_path.read_bytes() == sentinel


def test_lossless_bypasses_skip_if_larger(tmp_path):
    """A lossless file larger than the (already-small) input is still written."""
    # Save the source as a high-quality JPEG so it's small; lossless will be bigger.
    _, arr = _photo(tmp_path)
    src = tmp_path / "small.jpg"
    Image.fromarray(arr, "RGB").save(str(src), "JPEG", quality=60)
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "jxl"
    cfg.faithful.lossless = True
    assert cfg.faithful.skip_if_larger is True  # default on
    res = faithful_compress(str(src), str(tmp_path / "out"), cfg)
    assert res.skipped is False  # NOT kept-original despite being larger
    assert res.output_path.suffix == ".jxl"


def test_lossless_ignores_auto_tune(tmp_path):
    """Lossless on + auto_tune on: lossless wins, output is bit-exact (no tuning)."""
    p, arr = _photo(tmp_path)
    cfg = FaceKeepConfig()
    cfg.faithful.codec = "jxl"
    cfg.faithful.lossless = True
    cfg.faithful.auto_tune = True  # should be ignored
    res = faithful_compress(str(p), str(tmp_path / "out"), cfg)
    decoded = encoders.decode(res.output_path.read_bytes())
    assert np.array_equal(decoded, arr[:, :, ::-1])


# --- index fingerprint + config plumbing -----------------------------------

def test_lossless_busts_index_fingerprint():
    a = FaceKeepConfig()
    b = FaceKeepConfig()
    b.faithful.lossless = True
    assert settings_fingerprint(a) != settings_fingerprint(b)


def test_config_default_lossless_off():
    assert FaceKeepConfig().faithful.lossless is False
