"""High bit-depth (10/12/16-bit) handling — ROADMAP Phase 1 [CORE-GOAL].

Scope of what is implemented (and what is deliberately deferred):

The internal pipeline now carries 16-bit (uint16) sources through load /
detection / metrics without crashing or silently truncating. At the *encode*
boundary the bundled Pillow codec plugins have no high-bit path (verified:
pillow-jxl raises on non-8-bit modes; pillow-avif silently down-converts), so
high-bit input is rounded down to 8-bit with a loud warning instead of being
silently truncated.

True 10/12-bit *output* (which is what actually prevents banding on smooth
gradients) needs the external ``avifenc`` CLI. That path is now implemented
(ROADMAP Phase 1): when ``avifenc`` is located (``$FACEKEEP_AVIFENC`` or PATH)
a genuinely uint16 AVIF source is encoded at true 10-bit, and the banding
acceptance check below PASSES. Without the binary the pipeline still falls back
to the warned 8-bit round-down (offline-first / graceful degradation), and the
high-bit tests skip — so a default install is unaffected.
"""

import logging

import cv2
import numpy as np
import pytest

from facekeep import encoders, faithful, imageio, metrics
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint


def _write_gradient_png16(path, H=256, W=512):
    """A smooth 16-bit RGB horizontal gradient (the classic banding probe)."""
    ramp = np.linspace(0, 65535, W).astype(np.uint16)[None, :].repeat(H, axis=0)
    rgb16 = np.stack([ramp, ramp, ramp], axis=-1)  # BGR == RGB for gray ramp
    cv2.imwrite(str(path), rgb16)
    return rgb16


def test_load_preserves_16bit_depth(tmp_path):
    """A 16-bit PNG loads as uint16 BGR with source_bit_depth == 16."""
    src = tmp_path / "grad.png"
    _write_gradient_png16(src)

    loaded = imageio.load(str(src))

    assert loaded.source_bit_depth == 16
    assert loaded.image.dtype == np.uint16
    assert loaded.image.ndim == 3 and loaded.image.shape[2] == 3


def test_load_8bit_unchanged(tmp_path):
    """An ordinary 8-bit image still loads as uint8 / depth 8 (no regression)."""
    src = tmp_path / "plain.png"
    cv2.imwrite(str(src), np.full((64, 64, 3), 100, np.uint8))

    loaded = imageio.load(str(src))

    assert loaded.source_bit_depth == 8
    assert loaded.image.dtype == np.uint8


@pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)
def test_encode_uint16_warns_and_downconverts(tmp_path, caplog, monkeypatch):
    """Encoding a uint16 image must WARN (not silently truncate) and still work.

    This pins the 8-bit *fallback* path, so avifenc is forced unavailable (the
    default-install behavior) — otherwise on a machine that *has* avifenc the
    high-bit path would skip the warning. The high-bit path has its own test.
    """
    monkeypatch.setattr(encoders, "avifenc_available", lambda: False)
    src = tmp_path / "grad.png"
    rgb16 = _write_gradient_png16(src)
    loaded = imageio.load(str(src))
    assert loaded.image.dtype == np.uint16

    with caplog.at_level(logging.WARNING, logger="facekeep.encoders"):
        # bit_depth defaults to 8 here, but pass 16 to prove the fallback fires
        # when avifenc is absent even though high-bit was requested.
        data = encoders.encode(loaded.image, codec="avif", quality=90, bit_depth=16)

    assert any("16-bit" in r.message for r in caplog.records), (
        "expected a high-bit down-convert warning; the loss must not be silent"
    )
    # Output is still a valid, decodable 8-bit image of the right size.
    decoded = encoders.decode(data)
    assert decoded.shape == rgb16.shape
    assert decoded.dtype == np.uint8


@pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)
def test_faithful_compress_uint16_source_warns_and_succeeds(tmp_path, caplog, monkeypatch):
    """A 16-bit source must flow through the *whole* faithful pipeline safely.

    The lower-level `encoders.encode` down-convert is covered above; this is the
    end-to-end CORE-GOAL guard that the silent-truncation fix holds through
    `faithful.compress()` itself — i.e. the uint16 array survives load,
    detection, the down-convert-with-warning at the encode boundary, the output
    round-trip *verification* (which must not misjudge a uint16 source against
    its 8-bit decoded output), and the file write. Without dtype-aware handling
    any of those stages could crash or silently corrupt a high-bit photo.

    avifenc is forced unavailable so this pins the 8-bit fallback (the
    default-install behavior); the high-bit pipeline path has its own test.
    """
    monkeypatch.setattr(encoders, "avifenc_available", lambda: False)
    src = tmp_path / "grad.png"
    rgb16 = _write_gradient_png16(src, H=128, W=256)
    out_path = tmp_path / "out"

    with caplog.at_level(logging.WARNING, logger="facekeep.encoders"):
        result = faithful.compress(str(src), str(out_path), FaceKeepConfig())

    # The high-bit loss is reported (CORE-GOAL: never silent), not swallowed.
    assert any("16-bit" in r.message for r in caplog.records), (
        "expected a high-bit down-convert warning from the faithful pipeline"
    )
    # The pipeline produced a real encode (not the skip-if-larger keep-original
    # path) and verify_roundtrip did not false-fail on the uint16-vs-8-bit check.
    assert not result.skipped
    assert result.output_path.suffix == ".avif"

    decoded = encoders.decode(result.output_path.read_bytes())
    assert decoded.shape == rgb16.shape  # dimensions preserved
    assert decoded.dtype == np.uint8  # flattened to 8-bit at the encode boundary


def test_metrics_dtype_safe_uint16_vs_uint8():
    """SSIM/PSNR must be meaningful when comparing uint16 vs its 8-bit version.

    Without data_range normalization, skimage assumes the wrong range for a
    uint16 array and the scores become garbage. A uint16 image vs the same
    content rounded to 8-bit should score very high (near-identical content),
    not collapse.
    """
    ramp = np.linspace(0, 65535, 512).astype(np.uint16)[None, :].repeat(64, 0)
    img16 = np.stack([ramp, ramp, ramp], axis=-1)
    img8 = np.round(img16.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)

    report = metrics.compare(img16, img8)

    assert report.overall_ssim > 0.99
    assert report.overall_psnr > 45.0


def test_metrics_identical_uint16_is_perfect():
    """Identical uint16 inputs score a perfect SSIM (data_range handled)."""
    ramp = np.linspace(0, 65535, 256).astype(np.uint16)[None, :].repeat(64, 0)
    img16 = np.stack([ramp, ramp, ramp], axis=-1)

    assert metrics.ssim(img16, img16) == pytest.approx(1.0, abs=1e-6)


# --- True high-bit (10-bit) AVIF output via the avifenc CLI ------------------
#
# These exercise the real high-bit path (ROADMAP Phase 1, now unblocked). They
# need the external ``avifenc`` binary, so they SKIP when it can't be located —
# the same offline/optional convention as the real-AI / real-model tests. To run
# them locally, put ``avifenc`` on PATH or set ``FACEKEEP_AVIFENC`` (e.g. the
# repo's ``.tools/avifenc/avifenc.exe`` fetched from the libavif release).

avifenc_required = pytest.mark.skipif(
    not encoders.avifenc_available(),
    reason="avifenc binary not found (set FACEKEEP_AVIFENC or put it on PATH)",
)


def test_find_avifenc_env_takes_precedence(tmp_path, monkeypatch):
    """``$FACEKEEP_AVIFENC`` (a real file) wins over PATH; a bad value is ignored."""
    fake = tmp_path / "myavifenc.exe"
    fake.write_bytes(b"binary")
    monkeypatch.setenv("FACEKEEP_AVIFENC", str(fake))
    assert encoders._find_avifenc() == str(fake)
    assert encoders.avifenc_available() is True

    # A non-existent env value falls through to PATH (shutil.which), not an error.
    monkeypatch.setenv("FACEKEEP_AVIFENC", str(tmp_path / "nope.exe"))
    monkeypatch.setattr(encoders.shutil, "which", lambda name: "/usr/bin/avifenc")
    assert encoders._find_avifenc() == "/usr/bin/avifenc"

    # Neither env nor PATH -> None (graceful, not an exception).
    monkeypatch.delenv("FACEKEEP_AVIFENC", raising=False)
    monkeypatch.setattr(encoders.shutil, "which", lambda name: None)
    assert encoders._find_avifenc() is None
    assert encoders.avifenc_available() is False


def test_highbit_encode_raises_without_binary(monkeypatch):
    """``encode_highbit_avif`` raises EncodingError when avifenc is absent.

    (faithful's ``encode`` catches this and falls back to 8-bit; here we pin the
    raise so the fallback contract is well-defined.)
    """
    monkeypatch.setattr(encoders, "_find_avifenc", lambda: None)
    img = np.zeros((16, 16, 3), np.uint16)
    with pytest.raises(encoders.EncodingError, match="avifenc not found"):
        encoders.encode_highbit_avif(img, bit_depth=10)


def test_highbit_encode_rejects_bad_depth():
    img = np.zeros((16, 16, 3), np.uint16)
    with pytest.raises(encoders.EncodingError, match="Unsupported high-bit depth"):
        encoders.encode_highbit_avif(img, bit_depth=9)


def test_encode_falls_back_to_8bit_on_avifenc_failure(tmp_path, caplog, monkeypatch):
    """If avifenc is 'available' but the encode raises, ``encode`` degrades to 8-bit.

    This is the graceful-degradation guard: a broken/incompatible binary must not
    fail the faithful pipeline — it falls back to the warned 8-bit round-down.
    """
    monkeypatch.setattr(encoders, "avifenc_available", lambda: True)

    def _boom(*a, **k):
        raise encoders.EncodingError("simulated avifenc crash")

    monkeypatch.setattr(encoders, "encode_highbit_avif", _boom)

    src = tmp_path / "grad.png"
    rgb16 = _write_gradient_png16(src, H=64, W=128)
    loaded = imageio.load(str(src))

    with caplog.at_level(logging.WARNING, logger="facekeep.encoders"):
        data = encoders.encode(loaded.image, codec="avif", quality=90, bit_depth=16)

    assert any("falling back to 8-bit" in r.message for r in caplog.records)
    decoded = encoders.decode(data)  # still a valid 8-bit AVIF
    assert decoded.shape == rgb16.shape


@avifenc_required
def test_highbit_path_only_for_uint16_avif(tmp_path, monkeypatch):
    """The high-bit route fires only for uint16 + avif; 8-bit/JXL stay on Pillow."""
    spy = {"called": False}
    real = encoders.encode_highbit_avif

    def _spy(*a, **k):
        spy["called"] = True
        return real(*a, **k)

    monkeypatch.setattr(encoders, "encode_highbit_avif", _spy)

    # uint8 source: must NOT use the high-bit path even with bit_depth=16.
    img8 = np.full((32, 32, 3), 120, np.uint8)
    encoders.encode(img8, codec="avif", quality=80, bit_depth=16)
    assert spy["called"] is False

    # uint16 + avif + bit_depth>8: high-bit path fires.
    ramp = np.linspace(0, 65535, 64).astype(np.uint16)[None, :].repeat(32, 0)
    img16 = np.stack([ramp, ramp, ramp], axis=-1)
    encoders.encode(img16, codec="avif", quality=80, bit_depth=16)
    assert spy["called"] is True


@avifenc_required
def test_smooth_gradient_no_banding_roundtrip(tmp_path):
    """CORE-GOAL acceptance: a smooth 16-bit gradient must not band on restore.

    The real bar for the high-bit work, now PASSING via the avifenc 10-bit path.
    An 8-bit round-trip caps the smooth ramp at 256 distinct levels; true 10-bit
    keeps far more. We decode the high-bit AVIF with avifdec back to 16-bit and
    count levels — proving >8-bit detail survived (verified ~512 vs 256).
    """
    import subprocess

    src = tmp_path / "grad.png"
    rgb16 = _write_gradient_png16(src, H=64, W=1024)
    loaded = imageio.load(str(src))
    assert loaded.source_bit_depth == 16

    data = encoders.encode(loaded.image, codec="avif", quality=95, bit_depth=16)

    # Decode the encoded AVIF back to a 16-bit PNG via avifdec to read true depth
    # (Pillow decodes AVIF down to 8-bit, which would hide the win). avifdec sits
    # next to avifenc.
    avifenc = encoders._find_avifenc()
    avifdec = str(__import__("pathlib").Path(avifenc).with_name(
        "avifdec.exe" if avifenc.endswith(".exe") else "avifdec"
    ))
    out_avif = tmp_path / "out.avif"
    out_avif.write_bytes(data)
    dec_png = tmp_path / "dec.png"
    proc = subprocess.run(
        [avifdec, "-d", "16", str(out_avif), str(dec_png)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dec_png.exists():
        pytest.skip(f"avifdec unavailable/failed: {proc.stderr[:120]}")

    decoded16 = cv2.imread(str(dec_png), cv2.IMREAD_UNCHANGED)
    assert decoded16.dtype == np.uint16, "expected a true 16-bit decode"

    ref_levels = len(np.unique(rgb16[0, :, 0]))
    out_levels = len(np.unique(decoded16[0, :, 0]))
    # 10-bit gives ~1024 levels max; require comfortably more than the 8-bit cap
    # of 256 to prove banding is gone.
    assert out_levels > 256, f"only {out_levels} levels — still 8-bit banding"
    assert out_levels >= ref_levels * 0.4


# --- The 12-bit output knob (faithful.output_bit_depth) ----------------------
#
# 10-bit is the default; a user can opt into 12-bit via config / --bit-depth for
# maximum precision on a 16-bit source. These pin the config/validation, the
# fingerprint, the threading through encode, and (with avifenc) a real 12-bit
# encode.

def test_validate_accepts_10_and_12():
    for d in (10, 12):
        cfg = FaceKeepConfig()
        cfg.faithful.output_bit_depth = d
        cfg.validate()  # must not raise


@pytest.mark.parametrize("bad", [8, 9, 11, 16, 0])
def test_validate_rejects_other_output_bit_depths(bad):
    cfg = FaceKeepConfig()
    cfg.faithful.output_bit_depth = bad
    with pytest.raises(ConfigError):
        cfg.validate()


def test_default_output_bit_depth_is_10():
    assert FaceKeepConfig().faithful.output_bit_depth == 10


def test_fingerprint_busts_on_output_bit_depth():
    ten = FaceKeepConfig()
    twelve = FaceKeepConfig()
    twelve.faithful.output_bit_depth = 12
    assert settings_fingerprint(ten) != settings_fingerprint(twelve)


def test_encode_threads_output_bit_depth_to_highbit(monkeypatch):
    """``encode`` passes the chosen output depth (not a hardcoded 10) downstream."""
    seen = {}

    def _spy(image, *, bit_depth=10, **k):
        seen["bit_depth"] = bit_depth
        return b"avif-bytes"

    monkeypatch.setattr(encoders, "avifenc_available", lambda: True)
    monkeypatch.setattr(encoders, "encode_highbit_avif", _spy)

    ramp = np.linspace(0, 65535, 32).astype(np.uint16)[None, :].repeat(16, 0)
    img16 = np.stack([ramp, ramp, ramp], axis=-1)

    encoders.encode(img16, codec="avif", quality=80, bit_depth=16, output_bit_depth=12)
    assert seen["bit_depth"] == 12
    encoders.encode(img16, codec="avif", quality=80, bit_depth=16, output_bit_depth=10)
    assert seen["bit_depth"] == 10


@avifenc_required
def test_faithful_compress_12bit_roundtrip(tmp_path):
    """A real 12-bit faithful encode of a 16-bit gradient keeps >8-bit detail."""
    import subprocess

    src = tmp_path / "grad.png"
    rgb16 = _write_gradient_png16(src, H=64, W=1024)

    cfg = FaceKeepConfig()
    cfg.faithful.output_bit_depth = 12
    cfg.faithful.auto_tune = False
    cfg.faithful.quality = 95
    out_path = tmp_path / "out"
    result = faithful.compress(str(src), str(out_path), cfg)
    assert not result.skipped
    assert result.output_path.suffix == ".avif"

    avifenc = encoders._find_avifenc()
    avifdec = str(__import__("pathlib").Path(avifenc).with_name(
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
    assert decoded16.dtype == np.uint16
    out_levels = len(np.unique(decoded16[0, :, 0]))
    ref_levels = len(np.unique(rgb16[0, :, 0]))
    assert out_levels > 256, f"only {out_levels} levels — still 8-bit banding"
    assert out_levels >= ref_levels * 0.4
