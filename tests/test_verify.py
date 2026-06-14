"""Output round-trip verification — ROADMAP Phase 1.

Faithful mode otherwise writes the encoded file without ever confirming it can
be decoded, so a corrupt encode would pass silently. These tests cover the
safety net:

- a quick default check (decode + dimension match),
- the thorough ``--verify`` path (downscaled-SSIM floor),
- and the fail-loud behaviour on garbage bytes / size mismatch.
"""

import cv2
import numpy as np
import pytest

from facekeep import encoders, metrics
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import EncodingError
from facekeep.faithful import compress as faithful_compress

avif_only = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


def _photo_bgr(H=240, W=320, seed=0):
    """A small textured BGR image that survives lossy encoding with high SSIM."""
    rng = np.random.default_rng(seed)
    base = cv2.resize(
        rng.normal(128, 40, (H // 8, W // 8, 3)).astype(np.float32),
        (W, H), interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(base, 0, 255).astype(np.uint8)


# --- downscaled_ssim sanity --------------------------------------------------

def test_downscaled_ssim_identical_is_one():
    img = _photo_bgr()
    assert metrics.downscaled_ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_downscaled_ssim_drops_on_different_images():
    a = _photo_bgr(seed=1)
    b = _photo_bgr(seed=2)  # unrelated noise -> structurally very different
    assert metrics.downscaled_ssim(a, b) < 0.5


def test_downscaled_ssim_resizes_mismatched_geometry():
    """Different-sized inputs are matched (b -> a) so SSIM is still defined."""
    a = _photo_bgr(H=240, W=320, seed=3)
    b = cv2.resize(a, (160, 120))  # same content, half size
    # Same content at a different scale stays structurally similar.
    assert metrics.downscaled_ssim(a, b) > 0.8


# --- verify_roundtrip: pass paths --------------------------------------------

@avif_only
def test_verify_roundtrip_passes_on_good_encode():
    img = _photo_bgr()
    data = encoders.encode(img, codec="avif", quality=80)
    # Both quick and thorough should pass on a legitimate encode (no raise).
    encoders.verify_roundtrip(data, img)
    encoders.verify_roundtrip(data, img, thorough=True)


@avif_only
def test_verify_roundtrip_passes_thorough_on_low_quality():
    """A low-quality (but valid) encode must NOT trip the thorough floor.

    The floor detects a blown-up encoder, not lossy quality; a legitimate
    low-quality encode of textured content still stays well above it.
    """
    img = _photo_bgr()
    data = encoders.encode(img, codec="avif", quality=40)
    encoders.verify_roundtrip(data, img, thorough=True)


# --- verify_roundtrip: fail paths --------------------------------------------

def test_verify_roundtrip_raises_on_garbage_bytes():
    img = _photo_bgr()
    with pytest.raises(EncodingError):
        encoders.verify_roundtrip(b"not an image at all", img)


def test_verify_roundtrip_raises_on_truncated_bytes():
    img = _photo_bgr()
    with pytest.raises(EncodingError):
        # A few leading bytes of a plausible header, then nothing decodable.
        encoders.verify_roundtrip(b"\x00\x00\x00\x20ftypavif", img)


@avif_only
def test_verify_roundtrip_raises_on_size_mismatch():
    """A valid encode of one size, verified against a differently-sized source."""
    small = _photo_bgr(H=120, W=160)
    data = encoders.encode(small, codec="avif", quality=80)
    big = _photo_bgr(H=240, W=320)
    with pytest.raises(EncodingError, match="dimensions"):
        encoders.verify_roundtrip(data, big)


@avif_only
def test_verify_roundtrip_thorough_raises_on_unlike_output():
    """Thorough mode rejects output that decodes at the right size but is wrong.

    Encode solid black, then verify against a textured source of the same size:
    dimensions match, but the downscaled SSIM is far below the floor.
    """
    H, W = 200, 200
    black = np.zeros((H, W, 3), np.uint8)
    data = encoders.encode(black, codec="avif", quality=80)
    textured = _photo_bgr(H=H, W=W, seed=7)
    # Quick check passes (sizes match); thorough catches the structural mismatch.
    encoders.verify_roundtrip(data, textured)  # no raise
    with pytest.raises(EncodingError, match="SSIM"):
        encoders.verify_roundtrip(data, textured, thorough=True)


# --- end-to-end through faithful.compress ------------------------------------

@avif_only
def test_faithful_compress_verifies_by_default(tmp_path):
    """The default pipeline runs verification and produces a real output file."""
    img = _photo_bgr()
    src = tmp_path / "photo.png"
    cv2.imwrite(str(src), img)

    config = FaceKeepConfig()
    assert config.faithful.verify is True  # on by default
    res = faithful_compress(str(src), str(tmp_path / "out"), config)

    assert res.output_path.exists()
    assert res.compressed_size > 0


@avif_only
def test_faithful_compress_thorough_succeeds(tmp_path):
    img = _photo_bgr()
    src = tmp_path / "photo.png"
    cv2.imwrite(str(src), img)

    config = FaceKeepConfig()
    config.faithful.verify = True
    config.faithful.verify_thorough = True
    res = faithful_compress(str(src), str(tmp_path / "out"), config)

    assert res.output_path.exists()


@avif_only
def test_faithful_compress_uint16_source_verifies(tmp_path):
    """A 16-bit source still passes verification (the path is dtype-safe).

    The encoder rounds 16-bit down to 8-bit (warned, tested elsewhere); the
    decoded 8-bit output is compared against the uint16 source via the
    dtype-safe metrics, so verification must not crash or false-fail.
    """
    ramp = np.linspace(0, 65535, 320).astype(np.uint16)[None, :].repeat(240, 0)
    img16 = np.stack([ramp, ramp, ramp], axis=-1)
    src = tmp_path / "grad.png"
    cv2.imwrite(str(src), img16)

    config = FaceKeepConfig()
    config.faithful.verify_thorough = True
    res = faithful_compress(str(src), str(tmp_path / "out"), config)

    assert res.output_path.exists()
