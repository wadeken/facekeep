"""Large-image / memory test — ROADMAP Phase 2.

Encodes a 24MP image in faithful mode and asserts (a) it finishes correctly and
(b) its peak memory stays under a bound. This is the regression guard for the
Phase 3 *bounded-memory* work: it pins today's footprint so a future change that
holds extra full-resolution copies (the thing Phase 3 will trim) fails here.

Measurement is the tricky part, and the design follows from two facts about the
peak-RSS probe (see ``tests/_memprobe.py``):

  1. We need the *codec's* memory too. AVIF's libaom does the heavy allocation in
     C, outside Python, so a Python-only probe (tracemalloc) would miss the bulk.
     We read the OS peak working set / ``ru_maxrss`` instead.
  2. That OS peak is a process-lifetime high-water mark — it never goes down. In
     a shared pytest process, an earlier big encode (e.g. the corpus suite) would
     already have raised the watermark, so measuring a before/after *delta* in
     this process is unreliable (it reads 0 once the watermark is above us), and
     an absolute-peak assertion could false-fail on a peak some *other* test set.

So the encode is run in a **fresh subprocess** (``tests/_memrunner.py``): a clean
process whose peak reflects only this compress, immune to whatever the parent
pytest process did. The child prints its peak; the parent asserts the bound.

The image is *compressible* (smooth gradient + gentle low-frequency texture),
which is what a real photo looks like to the codec — pure random noise is the
worst case for the encoder's working set (~11x raw pixels here vs ~6.5x for the
compressible build) and would inflate the bound into something that no longer
catches real regressions. It is also built frugally (uint8 throughout, no
full-frame float64 temporary) so building it in the parent doesn't itself spike
memory — though that no longer matters for correctness now that measurement is
isolated in the child, it keeps the suite light.

The bound is expressed as a multiple of the raw pixel size (relative, like the
ratio/quality regression lock), not a magic megabyte number, so it scales if the
image size constant changes. Measured peak at 24MP here is ~11.3x raw pixels;
the ceiling is 16x to leave headroom for lossy/codec/platform drift while still
failing if compress starts holding, say, an extra full-resolution copy.

Skips (not fails) when the peak-RSS probe is unavailable on the platform, matching
the project's offline-graceful convention.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import _memprobe
from facekeep import encoders
from facekeep.imageio import load

pytestmark = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)

_RUNNER = Path(__file__).parent / "_memrunner.py"

# Image geometry. 24MP (6000x4000) is large enough to exercise the multi-copy
# footprint and OOM-prone codec paths while still running on a CI box; bump to
# 8000x6000 (48MP) here if a heavier guard is ever wanted.
_W, _H = 6000, 4000
_RAW_PIXELS = _W * _H * 3  # bytes of one full-resolution BGR uint8 copy

# Peak-memory ceiling as a multiple of one raw-pixel copy. Measured ~11.3x;
# 16x gives ~40% headroom for lossy/codec/platform variation but still trips if
# the pipeline starts holding an extra full-resolution buffer.
_PEAK_CEILING_X_RAW = 16
_PEAK_CEILING_BYTES = _PEAK_CEILING_X_RAW * _RAW_PIXELS


def _make_compressible_24mp(path: Path) -> None:
    """Write a 24MP codec-friendly JPEG, built frugally (uint8, no float64 frame).

    Smooth diagonal gradient plus gentle low-frequency texture: realistic for a
    photo (compresses well), unlike random noise which is the encoder's worst
    case and would inflate the memory bound past usefulness.
    """
    yy = np.linspace(20, 230, _H).astype(np.float32)[:, None]
    xx = np.linspace(10, 210, _W).astype(np.float32)[None, :]
    gray = ((yy + xx) * 0.5).astype(np.uint8)  # HxW uint8 directly, no full float frame
    img = np.empty((_H, _W, 3), np.uint8)
    img[..., 0] = gray
    img[..., 1] = (gray * 0.95).astype(np.uint8)
    img[..., 2] = (gray * 0.90).astype(np.uint8)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


@pytest.fixture
def large_image(tmp_path):
    """A 24MP compressible JPEG on disk."""
    path = tmp_path / "large.jpg"
    _make_compressible_24mp(path)
    return path


def _run_compress_in_subprocess(src: Path, out: Path) -> dict:
    """Compress ``src`` in a clean child process; return its JSON result dict.

    Isolating the encode in a fresh process is what makes the peak measurement
    trustworthy: the child's peak reflects only this compress, not any watermark
    raised earlier in the shared pytest process.
    """
    proc = subprocess.run(
        [sys.executable, str(_RUNNER), str(src), str(out), str(Path(__file__).parent)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"compress subprocess failed (rc={proc.returncode}).\n"
        f"STDERR:\n{proc.stderr[-2000:]}"
    )
    # The runner prints a single JSON line last; tolerate leading log noise.
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


def test_large_image_compresses_correctly(large_image, tmp_path):
    """A 24MP image encodes, stays the same shape, and actually gets smaller.

    Runs in-process (no memory probe needed) so it asserts the *functional*
    contract even on platforms where the peak-RSS probe is unavailable.
    """
    from facekeep import faithful
    from facekeep.config import FaceKeepConfig

    # Auto-tune off: this pins the default single-encode path (see _memrunner /
    # IMPROVEMENTS Phase 2). Auto-tune is on by default in production.
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = False
    result = faithful.compress(str(large_image), str(tmp_path / "out"), cfg)
    assert not result.skipped, "24MP compressible image unexpectedly hit skip-if-larger"
    assert result.ratio > 1.0, f"expected compression, got ratio {result.ratio:.3f}"

    original = load(str(large_image)).image
    decoded = encoders.decode(result.output_path.read_bytes())
    assert decoded.shape == original.shape, "decode changed the image dimensions"


def test_large_image_peak_memory_within_bound(large_image, tmp_path):
    """Peak memory for a 24MP encode must stay under the bound (OOM guard).

    Measured in a fresh subprocess so the peak is attributable to this compress
    alone. Skips when the probe can't read a peak on this platform.
    """
    if _memprobe.peak_rss_bytes() is None:
        pytest.skip("peak-RSS probe unavailable on this platform")

    info = _run_compress_in_subprocess(large_image, tmp_path / "out")

    # Sanity: the child must have done a real encode, not bailed via skip-if-larger
    # (which would make the memory number meaningless).
    assert not info["skipped"], "subprocess hit skip-if-larger; memory number is moot"
    assert info["shape_match"], "subprocess decode changed dimensions"

    peak = info["peak"]
    assert peak is not None, "child could not measure peak RSS"
    assert peak <= _PEAK_CEILING_BYTES, (
        f"peak memory {peak / 1048576:.0f} MB exceeds ceiling "
        f"{_PEAK_CEILING_BYTES / 1048576:.0f} MB "
        f"({_PEAK_CEILING_X_RAW}x raw pixels). A regression may be holding extra "
        f"full-resolution copies (see Phase 3 bounded-memory work)."
    )
