"""Image quality metrics for verification and quality auto-tuning.

SSIM and PSNR are provided out of the box. For true "visually lossless"
calibration, perceptual metrics (SSIMULACRA2, butteraugli) are far better
suited; integrating them is tracked in the ROADMAP.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("facekeep.metrics")


@dataclass
class QualityReport:
    """Quality comparison between an original and a processed image."""

    overall_ssim: float
    overall_psnr: float
    face_ssim: Optional[float] = None
    background_ssim: Optional[float] = None
    # Learned perceptual distance (LPIPS). Lower = more perceptually similar.
    # ``None`` unless explicitly requested (``compare(with_lpips=True)``) *and*
    # the optional ``lpips`` package (in the ``[ai]`` extra) is installed —
    # SSIM is the wrong tool for aggressive mode's hallucinated-but-plausible
    # background, but LPIPS pulls torch + downloads weights, so it is opt-in and
    # never on a pipeline default path. See ``lpips_distance``.
    lpips: Optional[float] = None


def _to_float01(x: np.ndarray) -> np.ndarray:
    """Scale an image to float [0, 1] based on its integer dtype range.

    Comparing a high-bit (uint16) original against an 8-bit decoded image — or
    two uint16 images — needs a common scale, otherwise skimage assumes the
    wrong data_range and the SSIM/PSNR numbers are meaningless. Normalizing both
    operands to [0, 1] makes the comparison dtype-agnostic and lets us pass
    data_range=1.0 explicitly.
    """
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    if x.dtype == np.uint16:
        return x.astype(np.float32) / 65535.0
    return x.astype(np.float32)


def _ssim_prepared(af: np.ndarray, bf: np.ndarray) -> float:
    """SSIM of two arrays already normalized to float [0, 1] (data_range=1.0).

    Splitting this out lets callers that have *already* float-normalized their
    inputs (the background-masked comparison in ``compare``) reuse those arrays
    instead of re-converting — avoiding a redundant full-frame copy on top of the
    normalization the metric needs anyway (ROADMAP Phase 3 bounded-memory).
    """
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(af, bf, channel_axis=2, data_range=1.0)
    )


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    return _ssim_prepared(_to_float01(a), _to_float01(b))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import peak_signal_noise_ratio

    return float(
        peak_signal_noise_ratio(_to_float01(a), _to_float01(b), data_range=1.0)
    )


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Public SSIM, dtype-safe (uint8/uint16). Shared by the CLI and auto-tune."""
    return _ssim(a, b)


# --------------------------------------------------------------------------- #
# SSIMULACRA2 — perceptual quality metric (opt-in, [dev] extra)
# --------------------------------------------------------------------------- #
#
# SSIM correlates only loosely with perception and saturates on noisy content, so
# it is a poor auto-tune acceptance target for "the eye can't tell" (see
# IMPROVEMENTS.md). SSIMULACRA2 is a full-reference perceptual metric built for
# exactly this — detecting compression artifacts the way a human would. Score
# interpretation: **higher = better** (~90 ≈ visually lossless, ~70 ≈ high
# quality), the *same direction* as SSIM, so the auto-tune binary search's
# ``score >= target`` comparison is unchanged when this metric is selected.
#
# The ``ssimulacra2`` package is pure Python (numpy/scipy/pillow only — no native
# binary, no model download), so unlike LPIPS it does not need the heavy ``[ai]``
# extra; it lives in ``[dev]``. Like every optional dependency here it is
# lazy-imported and degrades gracefully: missing package -> ``None`` (+ a
# warning), never a crash, so faithful auto-tune falls back to SSIM offline.

_tried_ssimulacra2_init = False
_ssimulacra2_fn = None


def _init_ssimulacra2():
    """Lazily resolve ``compute_ssimulacra2`` once; leave it ``None`` if absent.

    Mirrors ``_init_lpips``: import only here, and an ``ImportError`` logs a
    warning and leaves the function ``None`` so callers no-op (fall back to SSIM)
    instead of crashing.
    """
    global _ssimulacra2_fn, _tried_ssimulacra2_init
    _tried_ssimulacra2_init = True
    try:
        from ssimulacra2 import compute_ssimulacra2

        _ssimulacra2_fn = compute_ssimulacra2
    except ImportError:
        logger.warning(
            "ssimulacra2 not installed; perceptual SSIMULACRA2 scoring "
            "unavailable. Install with: pip install facekeep[dev]"
        )
        _ssimulacra2_fn = None


def ssimulacra2_available() -> bool:
    """True iff SSIMULACRA2 scoring can be used (package installed)."""
    if not _tried_ssimulacra2_init:
        _init_ssimulacra2()
    return _ssimulacra2_fn is not None


def _to_ssimulacra2_buffer(bgr: np.ndarray):
    """BGR uint8/uint16 image -> in-memory PNG file-like the package can open.

    The single BGR->RGB boundary for this metric (like the encoders' PIL
    boundary). ``compute_ssimulacra2`` only accepts paths/file-likes (it does
    ``Image.open(...).convert("RGB")`` internally), so we hand it an in-memory PNG
    rather than writing probe crops to disk. 8-bit PNG is exact for uint8; a
    uint16 source is downscaled to 8-bit (the metric itself loads as 8-bit RGB),
    which is fine — the metric's reference scale is 8-bit sRGB.
    """
    import io

    from PIL import Image

    if bgr.dtype == np.uint16:
        arr8 = (bgr.astype(np.uint32) * 255 // 65535).astype(np.uint8)
    else:
        arr8 = bgr.astype(np.uint8, copy=False)
    rgb = np.ascontiguousarray(arr8[:, :, ::-1])  # BGR -> RGB
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf


def ssimulacra2_score(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Perceptual SSIMULACRA2 score between two BGR images (higher = better).

    Returns ``None`` — never raises — when the optional ``ssimulacra2`` package is
    not installed or a computation failure occurs, so callers degrade gracefully
    (the same contract as the LPIPS / AI restore paths). The two images must share
    spatial dimensions; the metric expects 3-channel images (a single-channel
    image must be made 3-channel by the caller).
    """
    if not _tried_ssimulacra2_init:
        _init_ssimulacra2()
    if _ssimulacra2_fn is None:
        return None

    try:
        buf_a = _to_ssimulacra2_buffer(a)
        buf_b = _to_ssimulacra2_buffer(b)
        return float(_ssimulacra2_fn(buf_a, buf_b))
    # Swallow only genuine computation failures (bad shapes/buffers raise
    # ValueError; numerics raise these). Programming errors stay unmasked.
    except (RuntimeError, ValueError, MemoryError, OSError) as e:
        logger.warning("SSIMULACRA2 scoring failed (%s); skipping.", e)
        return None


# --------------------------------------------------------------------------- #
# LPIPS — learned perceptual distance (opt-in, [ai] extra)
# --------------------------------------------------------------------------- #
#
# SSIM is a structural-similarity metric: it penalizes pixel-level differences
# even when two images look the same. Aggressive mode *reconstructs* (hallucinates)
# the background on restore — the right acceptance question is "does it look
# wrong," not "do the pixels match." LPIPS scores exactly that perceptual
# question, so it is the right tool for evaluating aggressive restores.
#
# It is, however, an *evaluation* tool — never part of a pipeline default path:
#   * it pulls torch (lives in the ``[ai]`` extra, alongside Real-ESRGAN/GFPGAN);
#   * first use downloads small AlexNet linear-layer weights.
# So, like the AI restore paths, it is lazy-imported and degrades gracefully:
# missing package -> ``None`` (+ a warning), never a crash, and the default
# offline compress/restore never touches it.

_lpips_model = None
_tried_lpips_init = False


def _init_lpips():
    """Lazily build the LPIPS model once; leave it ``None`` if unavailable.

    Mirrors the restorer's ``_init_*`` graceful-degradation pattern: ``lpips``
    (and torch) are imported only here, and an ``ImportError`` logs a warning and
    leaves the model ``None`` so callers no-op instead of crashing.
    """
    global _lpips_model, _tried_lpips_init
    _tried_lpips_init = True
    try:
        import lpips as _lpips_pkg

        # net="alex": the LPIPS authors recommend AlexNet when *using LPIPS as a
        # metric* (fast, best human-judgement correlation); VGG is for optimizing
        # through it. We only score, so alex is the right choice.
        _lpips_model = _lpips_pkg.LPIPS(net="alex", verbose=False)
        _lpips_model.eval()
    except ImportError:
        logger.warning(
            "lpips not installed; perceptual LPIPS scoring unavailable. "
            "Install with: pip install facekeep[ai]"
        )
        _lpips_model = None


def lpips_available() -> bool:
    """True iff the LPIPS model can be used (package installed, init succeeded).

    Lets the CLI decide whether to offer/print an LPIPS score without forcing a
    weight download just to find out.
    """
    if not _tried_lpips_init:
        _init_lpips()
    return _lpips_model is not None


def _to_lpips_tensor(bgr: np.ndarray):
    """BGR uint8/uint16 image -> LPIPS input tensor (1, 3, H, W) in RGB, [-1, 1].

    The single BGR->RGB boundary for the metric (like the encoders' PIL
    boundary): LPIPS expects RGB, channels-first, normalized to [-1, 1].
    ``_to_float01`` makes it dtype-agnostic (uint8/uint16).
    """
    import torch

    rgb01 = _to_float01(bgr[:, :, ::-1])  # BGR->RGB, float [0, 1]
    chw = np.ascontiguousarray(rgb01.transpose(2, 0, 1))  # HWC -> CHW
    t = torch.from_numpy(chw).unsqueeze(0)  # -> (1, 3, H, W)
    return t * 2.0 - 1.0  # [0, 1] -> [-1, 1]


def lpips_distance(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Perceptual LPIPS distance between two BGR images (lower = more similar).

    Returns ``None`` — never raises — when the optional ``lpips`` package is not
    installed or an inference-time failure occurs, so callers degrade gracefully
    (the same contract as the AI restore paths). The two images must share
    spatial dimensions.
    """
    if not _tried_lpips_init:
        _init_lpips()
    if _lpips_model is None:
        return None

    try:
        import torch

        with torch.no_grad():
            ta = _to_lpips_tensor(a)
            tb = _to_lpips_tensor(b)
            d = _lpips_model(ta, tb)
        return float(d.item())
    # Only swallow genuine inference-time failures (torch raises RuntimeError on
    # OOM/shape problems; bad buffers raise ValueError). Programming errors stay
    # unmasked.
    except (RuntimeError, ValueError, MemoryError) as e:
        logger.warning("LPIPS scoring failed (%s); skipping.", e)
        return None


def downscaled_ssim(a: np.ndarray, b: np.ndarray, max_side: int = 512) -> float:
    """SSIM computed on downscaled copies — a cheap structural sanity check.

    Used by output round-trip verification: full-resolution SSIM on a large
    photo is slow, but a small downscaled SSIM is plenty to catch a codec that
    produced garbage (wrong/empty/scrambled output). Both images are resized to
    a common size (longest side <= ``max_side``, area-interpolated) before
    comparison, so it also tolerates the decoded image arriving in a different
    dtype than the source (``_ssim`` normalizes by dtype).
    """
    import cv2

    if a.shape[:2] != b.shape[:2]:
        # Match b to a's geometry so SSIM is defined; a true size mismatch is
        # caught separately by verify_roundtrip before this is ever reached.
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    h, w = a.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / longest
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)

    return _ssim(a, b)


def face_union_bbox(
    faces: List, shape: tuple[int, int]
) -> Optional[tuple[int, int, int, int]]:
    """Union bounding box of all face padded_bboxes, clipped to image shape."""
    if not faces:
        return None
    h, w = shape
    x1 = min(f.padded_bbox[0] for f in faces)
    y1 = min(f.padded_bbox[1] for f in faces)
    x2 = max(f.padded_bbox[2] for f in faces)
    y2 = max(f.padded_bbox[3] for f in faces)
    return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))


def compare(
    original_bgr: np.ndarray,
    processed_bgr: np.ndarray,
    faces: Optional[List] = None,
    *,
    with_lpips: bool = False,
) -> QualityReport:
    """Compare two images, optionally reporting face/background SSIM separately.

    Args:
        original_bgr: Ground-truth image (BGR)
        processed_bgr: Compressed/restored image (BGR), same dimensions
        faces: Optional list of FaceRegion to compute regional metrics
        with_lpips: Also compute the perceptual LPIPS distance (opt-in; needs the
            ``[ai]`` extra, downloads weights on first use). Left ``None`` on the
            report when off or unavailable. Off by default so every existing
            caller's behavior is unchanged.

    Returns:
        QualityReport
    """
    if original_bgr.shape != processed_bgr.shape:
        raise ValueError(
            f"Shape mismatch: {original_bgr.shape} vs {processed_bgr.shape}"
        )

    report = QualityReport(
        overall_ssim=_ssim(original_bgr, processed_bgr),
        overall_psnr=_psnr(original_bgr, processed_bgr),
    )

    if with_lpips:
        report.lpips = lpips_distance(original_bgr, processed_bgr)

    bbox = face_union_bbox(faces, original_bgr.shape[:2]) if faces else None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            report.face_ssim = _ssim(
                original_bgr[y1:y2, x1:x2], processed_bgr[y1:y2, x1:x2]
            )
            # Background = everything outside the face union box. Zero the face
            # region in both images, then SSIM the whole frame.
            #
            # Bounded-memory (ROADMAP Phase 3): do the zeroing on the float
            # arrays the SSIM needs anyway (``_to_float01`` builds those
            # regardless), in place, instead of copying both full-resolution
            # uint8 frames first. The old ``original.copy()``/``processed.copy()``
            # held two extra full-resolution copies live at once — on a large
            # photo via the ``quality`` command that dominated peak memory
            # (~2× raw on top of the float buffers). ``_to_float01`` already
            # returns fresh arrays (it always does an ``astype``), so writing
            # zeros into them does not touch the caller's images.
            if y2 - y1 < original_bgr.shape[0] or x2 - x1 < original_bgr.shape[1]:
                orig_bg = _to_float01(original_bgr)
                proc_bg = _to_float01(processed_bgr)
                orig_bg[y1:y2, x1:x2] = 0
                proc_bg[y1:y2, x1:x2] = 0
                report.background_ssim = _ssim_prepared(orig_bg, proc_bg)

    return report
