"""Error-handling tests (Phase 1: tighten error handling).

These guard the change from broad ``except Exception`` / silent fallbacks to
narrow excepts that either log at warning level or let real bugs propagate. Two
behaviours matter most:

1. **Faithful auto-tune metadata re-embed must not be swallowed.** It re-encodes
   the same image at a quality the search already encoded successfully, so a
   failure is a real bug — and silently falling back to the search bytes would
   drop the ICC/EXIF the previous Phase 1 work added. It must propagate.
2. **Aggressive AI upscale falls back to bicubic only for genuine inference
   failures** (RuntimeError/OOM/cv2.error), not for programming bugs
   (AttributeError, ...), which must surface instead of being masked as "use
   bicubic".
"""

import base64
import struct

import cv2
import numpy as np
import pytest
from PIL import Image

from facekeep import encoders, faithful
from facekeep.aggressive.restorer import Restorer
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import FaceKeepError

# Reuse the same compact, valid Display-P3 profile asset as tests/test_color.py
# (kept independent so neither test file imports the other's internals).
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


@pytest.fixture
def p3_jpeg(tmp_path):
    """A small JPEG carrying a Display-P3 ICC profile (no faces needed)."""
    arr = np.full((48, 48, 3), (200, 140, 110), dtype=np.uint8)
    pil = Image.fromarray(arr, "RGB")
    path = tmp_path / "p3.jpg"
    pil.save(str(path), "JPEG", quality=95, icc_profile=DISPLAY_P3_ICC)
    return path


def _auto_tune_config():
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = True
    # Keep the (now-passing) round-trip verify on; it should not interfere.
    return cfg


# --- 1. Faithful auto-tune metadata re-embed -------------------------------

def test_auto_tune_reembed_failure_propagates(p3_jpeg, tmp_path, monkeypatch):
    """A failure in the auto-tune metadata re-embed must NOT be swallowed.

    The re-embed is the only ``encoders.encode`` call that passes ``icc=``; we
    let the search encodes succeed and make only that final call raise. Before
    the fix this was caught by ``except Exception: pass`` and the stripped bytes
    were written silently; now it propagates as a FaceKeepError.
    """
    real_encode = encoders.encode

    def flaky_encode(*args, **kwargs):
        if kwargs.get("icc"):  # only the metadata re-embed passes icc
            raise encoders.EncodingError("simulated re-embed failure")
        return real_encode(*args, **kwargs)

    monkeypatch.setattr(faithful.encoders, "encode", flaky_encode)

    with pytest.raises(FaceKeepError, match="re-embed"):
        faithful.compress(str(p3_jpeg), str(tmp_path / "out"), _auto_tune_config())


def test_auto_tune_preserves_icc(p3_jpeg, tmp_path):
    """Positive case: auto-tune output still carries the ICC profile.

    Guards against regressing to writing the metadata-stripped search bytes:
    the re-embed must run and its result (not the stripped ``data``) be written.
    """
    result = faithful.compress(
        str(p3_jpeg), str(tmp_path / "out"), _auto_tune_config()
    )
    embedded = Image.open(str(result.output_path)).info.get("icc_profile")
    assert embedded == DISPLAY_P3_ICC


# --- 2. Aggressive AI upscale fallback -------------------------------------

class _RaisingUpsampler:
    """Stand-in Real-ESRGAN upsampler whose enhance() raises a chosen error."""

    def __init__(self, exc):
        self._exc = exc

    def enhance(self, bg, outscale):
        raise self._exc


def _restorer_with_upsampler(upsampler):
    r = Restorer()
    r._tried_init = True  # skip the real Real-ESRGAN import
    r._upsampler = upsampler
    return r


def test_ai_upscale_runtime_error_falls_back_to_bicubic(caplog):
    """A genuine inference failure (RuntimeError, e.g. OOM) -> bicubic fallback.

    The result must still be the correct target size, and a warning logged.
    """
    bg = np.full((20, 30, 3), 120, dtype=np.uint8)
    r = _restorer_with_upsampler(_RaisingUpsampler(RuntimeError("CUDA OOM")))

    with caplog.at_level("WARNING"):
        out, used_ai = r._upscale_background(bg, target_w=60, target_h=40,
                                             bg_scale=0.5)

    assert out.shape[:2] == (40, 60)  # bicubic resize to (H, W)
    assert used_ai is False  # the fallback must report itself honestly
    assert any("AI upscale failed" in rec.message for rec in caplog.records)


def test_ai_upscale_cv2_error_falls_back_to_bicubic():
    """cv2.error is also an inference-side failure -> bicubic, not a crash."""
    bg = np.full((20, 30, 3), 90, dtype=np.uint8)
    r = _restorer_with_upsampler(_RaisingUpsampler(cv2.error("bad buffer")))
    out, used_ai = r._upscale_background(bg, target_w=60, target_h=40, bg_scale=0.5)
    assert out.shape[:2] == (40, 60)
    assert used_ai is False


def test_ai_upscale_programming_bug_propagates():
    """A programming bug (AttributeError) must surface, not become 'use bicubic'.

    This is the whole point of narrowing the except: our own bugs in the call
    path used to be silently masked as a bicubic fallback.
    """
    bg = np.full((20, 30, 3), 90, dtype=np.uint8)
    r = _restorer_with_upsampler(_RaisingUpsampler(AttributeError("typo in call")))
    with pytest.raises(AttributeError):
        r._upscale_background(bg, target_w=60, target_h=40, bg_scale=0.5)


# --- 3. Restore EXIF re-embed warns (doesn't crash, doesn't pass silently) --

def test_restore_metadata_embed_failure_warns(tmp_path, caplog):
    """A bad metadata blob on restore warns but still writes the pixels.

    Restore now writes JPEG via Pillow (carrying EXIF *and* ICC in one save),
    not cv2+piexif. Pillow tolerates short garbage EXIF, but an oversized block
    raises ValueError; the restorer must then fall back to writing pixels
    without metadata rather than losing the restored image.
    """
    out = tmp_path / "restored.jpg"
    result = np.full((16, 16, 3), 100, dtype=np.uint8)
    r = Restorer()
    # >64 KB EXIF overflows the JPEG APP1 segment -> Pillow raises ValueError.
    with caplog.at_level("WARNING"):
        r._write(result, str(out), exif=b"\x00" * 200000)
    assert out.exists()  # pixels were written despite the metadata failure
    assert any("Could not embed metadata" in rec.message for rec in caplog.records)


def test_struct_error_is_catchable():
    """Sanity: a malformed EXIF block raises something our narrow tuple catches.

    Documents why the restorer/imageio excepts include struct.error alongside
    ValueError; if piexif's behaviour changes this test flags it.
    """
    import piexif

    good = piexif.dump({"0th": {piexif.ImageIFD.Orientation: 1}})
    with pytest.raises((ValueError, struct.error)):
        piexif.load(good[: len(good) // 2])  # truncated
