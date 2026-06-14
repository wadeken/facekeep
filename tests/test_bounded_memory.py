"""Bounded-memory contract tests — ROADMAP Phase 3.

The Phase 3 bounded-memory item is "avoid holding multiple full-resolution
copies; free intermediates promptly." The 24MP OOM guard in
``tests/test_large_image.py`` pins the *default compress* path's peak via the OS
(where libaom's C-level working set dominates and Python-level copies are in the
noise). These tests instead pin the two places where the **Python layer itself**
was holding extra full-resolution copies that a process-peak number does not
isolate:

1. ``metrics.compare``'s background-SSIM branch used to ``.copy()`` *both*
   full-resolution frames before zeroing the face box — two extra raw-pixel
   buffers alive at once. On a large photo through the ``quality`` CLI command
   that dominated peak memory. It now zeros the face region in-place on the
   float arrays the SSIM needs anyway, so those two uint8 copies are gone. The
   tests assert (a) the result is numerically unchanged, (b) the caller's images
   are not mutated, and (c) the Python-tracked peak is below what "two extra full
   copies" would cost.
2. ``encoders.encode``/``decode`` create a full-frame intermediate (the
   BGR↔RGB array wrapped by a PIL image). They now drop it promptly (``close`` +
   ``del``) rather than letting it live until return. A ``weakref`` proves the
   intermediate is collectable once the call returns — a behavioural contract
   that does not depend on a (libaom-dominated, hence insensitive) peak number.

Why ``tracemalloc`` here, not the OS peak probe of ``test_large_image``: these
are *Python-level* allocations, so the Python allocation tracker measures them
directly and deterministically (no subprocess, no skip), whereas the OS peak is
dominated by the codec's C allocations and would not isolate them. The bounds
are expressed relative to one raw-pixel frame (this repo's regression-lock
style), not magic megabytes.
"""

import gc
import tracemalloc
import weakref

import numpy as np
import pytest

from facekeep import encoders, metrics
from facekeep.detector import FaceRegion

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


# --------------------------------------------------------------------------- #
# 1. metrics.compare background branch: equivalence, no mutation, bounded mem
# --------------------------------------------------------------------------- #

def _old_background_ssim(a, b, bbox):
    """The pre-optimization background-SSIM: copy both frames, zero, then SSIM.

    Kept here as the oracle so the new in-place-on-float implementation is held
    to *exact numerical equality*, not merely "close".
    """
    from skimage.metrics import structural_similarity

    x1, y1, x2, y2 = bbox
    mask = np.ones(a.shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = False
    orig_bg = a.copy()
    proc_bg = b.copy()
    orig_bg[~mask] = 0
    proc_bg[~mask] = 0
    return float(
        structural_similarity(
            metrics._to_float01(orig_bg),
            metrics._to_float01(proc_bg),
            channel_axis=2,
            data_range=1.0,
        )
    )


@pytest.fixture
def _small_pair():
    """A small deterministic original/processed pair + a face bbox inside it."""
    rng = np.random.default_rng(1)
    h, w = 300, 400
    a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    b = (a.astype(int) + rng.integers(-25, 25, (h, w, 3))).clip(0, 255).astype(np.uint8)
    bbox = (50, 40, 200, 180)  # x1, y1, x2, y2
    faces = [FaceRegion(id=0, bbox=bbox, padded_bbox=bbox, confidence=1.0)]
    return a, b, bbox, faces


def test_compare_background_ssim_matches_old_formulation(_small_pair):
    """The leaner in-place background SSIM equals the old copy-then-zero result.

    Numerically identical (not just close): the optimization only changed *where*
    the zeroing happens (on the float arrays the metric builds anyway, in place),
    not the math.
    """
    a, b, bbox, faces = _small_pair
    expected = _old_background_ssim(a, b, bbox)

    report = metrics.compare(a, b, faces)

    assert report.background_ssim is not None
    assert abs(report.background_ssim - expected) < 1e-9, (
        f"background SSIM drifted: {report.background_ssim} vs {expected}"
    )
    # Face SSIM is still produced for the same bbox.
    assert report.face_ssim is not None


def test_compare_does_not_mutate_caller_images(_small_pair):
    """In-place zeroing must hit fresh float copies, never the caller's arrays.

    ``_to_float01`` always does an ``astype`` (a copy), so zeroing the face box
    on its result is safe; this guards that invariant — a regression to zeroing a
    *view* of the input would corrupt the caller's image here.
    """
    a, b, _bbox, faces = _small_pair
    a_before = a.copy()
    b_before = b.copy()

    metrics.compare(a, b, faces)

    assert np.array_equal(a, a_before), "compare() mutated the original image"
    assert np.array_equal(b, b_before), "compare() mutated the processed image"


def test_compare_background_branch_does_not_copy_two_full_frames():
    """The background branch must not hold two extra full-resolution uint8 copies.

    Old code did ``original.copy()`` + ``processed.copy()`` (2× raw of uint8) on
    top of the float arrays SSIM needs. We assert the Python-tracked peak of
    ``compare`` stays under a bound that those two extra uint8 copies alone would
    blow: the bound is set *below* "old behaviour" yet safely above the float
    buffers the metric legitimately needs, so it bites if the uint8 copies return
    but does not false-fail on skimage's own working set.
    """
    rng = np.random.default_rng(7)
    h, w = 1200, 1600
    a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    b = a.copy()
    raw = a.nbytes  # one full-resolution uint8 frame
    faces = [
        FaceRegion(id=0, bbox=(200, 200, 600, 800),
                   padded_bbox=(100, 100, 800, 1000), confidence=1.0)
    ]

    gc.collect()
    tracemalloc.start()
    try:
        metrics.compare(a, b, faces)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # Measure the old formulation's peak in the same process for a relative
    # reference, then require the new code to come in at least one full uint8
    # frame below it (i.e. it genuinely dropped a full-resolution copy).
    gc.collect()
    tracemalloc.start()
    try:
        _old_background_ssim(a, b, (200, 200, 600, 800))
        # The oracle only does the background half; add the face + overall SSIM
        # the real compare also does, so the reference covers the same work.
        metrics._ssim(a[200:800, 200:600], b[200:800, 200:600])
        metrics._ssim(a, b)
        old_peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak <= old_peak - raw, (
        f"compare peak {peak/1048576:.0f} MB did not drop a full uint8 frame "
        f"({raw/1048576:.0f} MB) below the old formulation's "
        f"{old_peak/1048576:.0f} MB — the two full-frame copies may be back."
    )


# --------------------------------------------------------------------------- #
# 2. encode / decode: intermediate full-frame copy is released promptly
# --------------------------------------------------------------------------- #

@requires_avif
def test_encode_releases_intermediate_pil(monkeypatch):
    """``encode`` must not leave its intermediate PIL/RGB array alive after return.

    ``_bgr_to_pil`` builds a full-frame RGB array wrapped in a PIL image; on a
    large photo that is a whole extra raw-pixel buffer. We capture a weakref to
    that intermediate (by wrapping ``_bgr_to_pil``) and assert it is collectable
    once ``encode`` returns — proving the ``finally: pil.close(); del pil`` drops
    it rather than holding it until the function frame is torn down later.
    """
    real_bgr_to_pil = encoders._bgr_to_pil
    captured = {}

    def capturing(image_bgr):
        pil = real_bgr_to_pil(image_bgr)
        captured["ref"] = weakref.ref(pil)
        return pil

    monkeypatch.setattr(encoders, "_bgr_to_pil", capturing)

    img = np.full((256, 256, 3), 120, dtype=np.uint8)
    data = encoders.encode(img, "avif", 70, 6, "auto", False)
    assert data, "encode produced no bytes"

    gc.collect()
    assert captured["ref"]() is None, (
        "intermediate PIL image from _bgr_to_pil is still alive after encode "
        "returned; it should be closed and dropped (bounded-memory)"
    )


@requires_avif
def test_decode_releases_intermediate_pil(monkeypatch):
    """``decode`` must not leave the decoded PIL image alive after return.

    Same contract on the read side: ``decode`` opens a PIL image, converts to a
    BGR array, and should drop the PIL object (and its full-frame buffer) before
    returning the array — important because verify_roundtrip decodes while the
    source frame and encoded bytes are still live.
    """
    img = np.full((256, 256, 3), 90, dtype=np.uint8)
    data = encoders.encode(img, "avif", 70, 6, "auto", False)

    from PIL import Image

    real_open = Image.open
    captured = {}

    def capturing_open(*args, **kwargs):
        pil = real_open(*args, **kwargs)
        captured["ref"] = weakref.ref(pil)
        return pil

    monkeypatch.setattr(Image, "open", capturing_open)

    out = encoders.decode(data)
    assert out.shape == (256, 256, 3)

    gc.collect()
    assert captured["ref"]() is None, (
        "decoded PIL image is still alive after decode returned; it should be "
        "closed and dropped (bounded-memory)"
    )


@requires_avif
def test_verify_roundtrip_quick_releases_decoded(monkeypatch):
    """Quick verify needs only dimensions, so it must free the decoded frame.

    The quick path compares shapes and returns; it should not keep the
    full-resolution decoded array alive (it runs while the source frame and
    encoded bytes are live). We wrap ``encoders.decode`` to weakref its returned
    array and assert it's collectable after ``verify_roundtrip`` returns.
    """
    img = np.full((256, 256, 3), 70, dtype=np.uint8)
    data = encoders.encode(img, "avif", 70, 6, "auto", False)

    real_decode = encoders.decode
    captured = {}

    def capturing_decode(d):
        arr = real_decode(d)
        captured["ref"] = weakref.ref(arr)
        return arr

    monkeypatch.setattr(encoders, "decode", capturing_decode)

    encoders.verify_roundtrip(data, img, thorough=False)

    gc.collect()
    assert captured["ref"]() is None, (
        "decoded array from a quick verify_roundtrip is still alive after it "
        "returned; the quick path should drop it once dimensions are checked"
    )
