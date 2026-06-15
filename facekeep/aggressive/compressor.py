"""Aggressive-mode compression: extract face crops + downsample background."""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .. import imageio
from ..config import AggressiveConfig, FaceKeepConfig
from ..detector import (
    DetectionCache,
    FaceRegion,
    _iou,
    create_detector,
    detect_cached,
)
from ..exceptions import SkipFileError
from .blender import create_soft_mask

logger = logging.getLogger("facekeep.aggressive.compressor")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Down-convert to 8-bit for the (analysis-only) edge heuristic.

    The detail heuristic never touches output pixels — it only decides a scale —
    so rounding a uint16 source down to 8-bit here is harmless, and Canny needs
    8-bit input. Mirrors detector._as_uint8 (kept local to avoid importing a
    private name across modules).
    """
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return np.round(image.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def _edge_map(image: np.ndarray) -> Optional[np.ndarray]:
    """Strong-edge map (Canny after a light blur) — the shared text/detail proxy.

    A zero-download heuristic (NOT real text detection). Text, signage, and fine
    regular structure are edge-dense; smooth content (sky, bokeh, plain walls) is
    not. The pre-blur (sigma 1.5) is what makes the proxy honest about *benign*
    fine content: per-pixel camera noise / fine foliage/grass would otherwise
    read as "detailed" (Canny fires on every noisy pixel), but the ROADMAP says
    to keep aggressively compressing that. A mild blur collapses per-pixel noise
    to ~0 while sharp, high-contrast text/structure edges survive — so foliage
    stays benign and text still trips.

    Computed once per photo and shared by the whole-image detail ratio and the
    per-tile text localizer. Returns ``None`` for a degenerate (empty) image.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None
    img8 = _to_uint8(image)
    gray = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY) if img8.ndim == 3 else img8
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return cv2.Canny(gray, 80, 200)


def _background_detail_ratio(image: np.ndarray) -> float:
    """Fraction of the frame on strong edges — a proxy for text/fine structure.

    The whole-image form of the edge heuristic (see ``_edge_map``): a high ratio
    flags a background to compress less. Returns 0.0 for a degenerate image.
    """
    edges = _edge_map(image)
    if edges is None:
        return 0.0
    h, w = image.shape[:2]
    return float(np.count_nonzero(edges)) / float(h * w)


def _has_risky_background_face(
    faces: List[FaceRegion], img_w: int, img_h: int, small_face_ratio: float
) -> bool:
    """True if any detected face is small/distant relative to the frame.

    A small face is a background/distant face: it is the worst case to lose to
    the downsample (an AI reconstruction of a face reads as uncanny), so its
    presence flags a risky background even though the face itself is cropped and
    protected. Measured on the tight ``bbox`` short side as a fraction of the
    frame short side — the same convention as the detector's size filter.
    """
    short = min(img_w, img_h)
    if short == 0:
        return False
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        face_short = min(x2 - x1, y2 - y1)
        if 0 < face_short < small_face_ratio * short:
            return True
    return False


def _risky_regions(
    cfg: AggressiveConfig, faces: List[FaceRegion], img_w: int, img_h: int
) -> List[Tuple[int, int, int, int]]:
    """Localized risky regions to protect at near-original resolution.

    The region-local counterpart to ``_resolve_bg_scale``'s whole-image raise:
    instead of compressing the *whole* background less when risk is found, we
    return the bounding boxes of the risky regions so the caller can store each as
    a sharp patch and composite it back on restore.

    This selector localizes the **small/distant-face** signal — the worst
    failure the AI causes (an uncanny reconstructed background face) and the one
    that already carries a clean bbox. The risky region is the face's *padded*
    bbox: it covers the surrounding background context the AI would otherwise
    hallucinate around the small face (the tight face pixels are separately kept
    as a face crop). Boxes are clamped to the frame. The edge-density/text
    signal is localized by ``_text_regions`` (opt-in) and otherwise stays
    whole-image (handled by ``_resolve_bg_scale``).

    Returns an empty list when content-aware or region-local conservatism is off,
    or when no face is small enough to be risky. Pure (no I/O, no mutation).
    """
    if not (cfg.content_aware and cfg.region_local):
        return []
    short = min(img_w, img_h)
    if short == 0:
        return []
    regions: List[Tuple[int, int, int, int]] = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        face_short = min(x2 - x1, y2 - y1)
        if not (0 < face_short < cfg.small_face_ratio * short):
            continue
        px1, py1, px2, py2 = f.padded_bbox
        cx1, cy1 = max(0, px1), max(0, py1)
        cx2, cy2 = min(img_w, px2), min(img_h, py2)
        if cx2 > cx1 and cy2 > cy1:
            regions.append((cx1, cy1, cx2, cy2))
    return regions


# Text-localization geometry (module constants, like _HAND_ZONE_*; the
# thresholds that need per-user tuning are config fields instead).
_TEXT_GRID = 16          # tiles per side for the coarse edge-density grid
_TEXT_MIN_TILES = 2      # clusters smaller than this many tiles are noise
_TEXT_MAX_CLUSTERS = 8   # more clusters than this -> patching isn't economical
_TEXT_PAD_TILES = 0.5    # pad each cluster bbox outward by this many tile sizes


def _text_regions(
    cfg: AggressiveConfig,
    image: np.ndarray,
    exclude_boxes: List[Tuple[int, int, int, int]],
    edge_map: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int, int]]:
    """Localized text-like clusters to protect as sharp region patches.

    The region-local counterpart of the *edge/text* signal (small faces already
    have ``_risky_regions``). The whole-image detail ratio only fires when the
    entire frame is edge-dense, so a small sign/text block in a big photo gets no
    protection at all — yet text is exactly what the AI upscale mangles. This
    scans a coarse ``_TEXT_GRID``² grid over the shared edge map, marks tiles
    whose edge-pixel fraction exceeds ``cfg.text_region_tile_threshold``, merges
    8-connected risky tiles into clusters, and returns each cluster's padded,
    frame-clamped bbox. Still the zero-download edge *proxy*, NOT OCR — and at
    tile granularity the proxy cannot tell text from benign-but-sharp organic
    content (ferns/ridges measurably trip it on real landscapes), which is why
    ``cfg.protect_text`` is **opt-in** (see the config comment).

    ``exclude_boxes`` (face padded boxes + already-emitted regions) are zeroed in
    the edge map first: those areas are already stored sharp, so a patch there
    would be pure waste (and portraits would otherwise sprout "text" patches on
    eyes/hair). Single-tile clusters are dropped as noise.

    Economy bail-out — returns ``[]`` (caller falls back to the whole-image
    raise) when the clusters cover more than ``cfg.text_region_max_frac`` of the
    frame or there are more than ``_TEXT_MAX_CLUSTERS`` of them: a document-like
    photo is better served by compressing the whole background less than by
    patching most of it. Also returns ``[]`` when disabled
    (``content_aware``/``region_local``/``protect_text``) or nothing is risky.
    Pure (no I/O, no mutation; the edge map is copied before zeroing).
    """
    if not (cfg.content_aware and cfg.region_local and cfg.protect_text):
        return []
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return []
    edges = _edge_map(image) if edge_map is None else edge_map
    if edges is None:
        return []
    edges = edges.copy()
    for bx1, by1, bx2, by2 in exclude_boxes:
        cx1, cy1 = max(0, int(bx1)), max(0, int(by1))
        cx2, cy2 = min(w, int(bx2)), min(h, int(by2))
        if cx2 > cx1 and cy2 > cy1:
            edges[cy1:cy2, cx1:cx2] = 0

    # Per-tile edge density on a coarse grid. linspace boundaries cover the frame
    # exactly even when the size doesn't divide evenly (last tiles a bit larger).
    ys = np.linspace(0, h, _TEXT_GRID + 1, dtype=int)
    xs = np.linspace(0, w, _TEXT_GRID + 1, dtype=int)
    risky = np.zeros((_TEXT_GRID, _TEXT_GRID), np.uint8)
    for i in range(_TEXT_GRID):
        for j in range(_TEXT_GRID):
            ty1, ty2 = ys[i], ys[i + 1]
            tx1, tx2 = xs[j], xs[j + 1]
            area = (ty2 - ty1) * (tx2 - tx1)
            if area <= 0:
                continue
            frac = float(np.count_nonzero(edges[ty1:ty2, tx1:tx2])) / float(area)
            if frac > cfg.text_region_tile_threshold:
                risky[i, j] = 1
    if not risky.any():
        return []

    # Merge 8-connected risky tiles into clusters; each becomes one padded bbox.
    n_labels, labels = cv2.connectedComponents(risky, connectivity=8)
    tile_h = h / _TEXT_GRID
    tile_w = w / _TEXT_GRID
    pad_y = int(round(_TEXT_PAD_TILES * tile_h))
    pad_x = int(round(_TEXT_PAD_TILES * tile_w))
    boxes: List[Tuple[int, int, int, int]] = []
    for label in range(1, n_labels):
        tiles_i, tiles_j = np.nonzero(labels == label)
        if tiles_i.size < _TEXT_MIN_TILES:
            continue  # single-tile blips are noise, not a sign
        ry1 = max(0, int(ys[tiles_i.min()]) - pad_y)
        ry2 = min(h, int(ys[tiles_i.max() + 1]) + pad_y)
        rx1 = max(0, int(xs[tiles_j.min()]) - pad_x)
        rx2 = min(w, int(xs[tiles_j.max() + 1]) + pad_x)
        if rx2 > rx1 and ry2 > ry1:
            boxes.append((rx1, ry1, rx2, ry2))

    if not boxes:
        return []
    if len(boxes) > _TEXT_MAX_CLUSTERS:
        return []  # scattered everywhere -> whole-image conservatism instead
    total = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
    if total / float(h * w) > cfg.text_region_max_frac:
        return []  # document-like -> patching most of the frame isn't economical
    return boxes


# Hand-zone geometry (the C1 offline tier), in units of the *tight* face box.
# Hands in a portrait rest roughly at chest/waist height, to the sides of the
# torso — not in front of the face and usually not dead-centre on the chest. So
# from each face we project two side bands:
#   vertical : from (face_bottom + DOWN_NEAR * face_h) to (+ DOWN_FAR * face_h)
#   horizontal (each side): an outward band from INNER..OUTER * face_w measured
#              from the face centre, leaving the torso centre (where hands usually
#              are not) unprotected — this is the "hand, not whole upper body" knob.
_HAND_ZONE_DOWN_NEAR = 1.0   # band starts ~1 face-height below the face
_HAND_ZONE_DOWN_FAR = 3.0    # band ends ~3 face-heights below the face
_HAND_ZONE_INNER = 0.6       # inner edge of each side band (face-widths from centre)
_HAND_ZONE_OUTER = 2.2       # outer edge of each side band (face-widths from centre)
_HAND_ZONE_MERGE_IOU = 0.2   # union C1 bands overlapping at least this much into one


def _hand_zones_from_faces(
    faces: List[FaceRegion], img_w: int, img_h: int
) -> List[Tuple[int, int, int, int]]:
    """Offline (C1) geometric estimate of where hands likely are, per face.

    OpenCV ships no hand detector, so the zero-download default infers hand-likely
    boxes from human body proportions relative to each detected face: two side
    bands at roughly chest/waist height (see ``_HAND_ZONE_*``). It is a
    *probabilistic guess*, not detection — hands raised overhead, far from the
    body, or in a frame with no nearby face are missed, and occasionally a
    hand-less area is protected. The bands deliberately exclude the torso centre so
    only the *hands* are protected, not the whole upper body. Boxes are clamped to
    the frame; degenerate/empty boxes are dropped. Pure (no I/O, no mutation).
    """
    short = min(img_w, img_h)
    if short == 0:
        return []
    zones: List[Tuple[int, int, int, int]] = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        fw, fh = x2 - x1, y2 - y1
        if fw <= 0 or fh <= 0:
            continue
        cx = (x1 + x2) / 2.0
        top = int(y2 + _HAND_ZONE_DOWN_NEAR * fh)
        bot = int(y2 + _HAND_ZONE_DOWN_FAR * fh)
        ty1, ty2 = max(0, top), min(img_h, bot)
        if ty2 <= ty1:
            continue
        for sign in (-1.0, +1.0):  # left band, right band
            inner = cx + sign * _HAND_ZONE_INNER * fw
            outer = cx + sign * _HAND_ZONE_OUTER * fw
            bx1, bx2 = sorted((inner, outer))
            zx1, zx2 = max(0, int(bx1)), min(img_w, int(bx2))
            if zx2 > zx1:
                zones.append((zx1, ty1, zx2, ty2))
    return zones


def _merge_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int]], iou_thresh: float
) -> List[Tuple[int, int, int, int]]:
    """Union boxes overlapping at >= ``iou_thresh`` into their bounding box.

    A fixed-point merge: any two boxes that overlap enough are replaced by the
    single box bounding both, repeated until nothing more merges. Unlike NMS (which
    *drops* the smaller of an overlapping pair) this keeps the covered area while
    collapsing the redundant, overlapping bands the C1 hand-zone geometry emits for
    adjacent faces — so the same pixels aren't stored in several patches. Pure.
    """
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        out: List[Tuple[int, int, int, int]] = []
        for b in boxes:
            for i, o in enumerate(out):
                if _iou(b, o) >= iou_thresh:
                    out[i] = (min(b[0], o[0]), min(b[1], o[1]),
                              max(b[2], o[2]), max(b[3], o[3]))
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def _c1_hand_zones(
    cfg: AggressiveConfig, faces: List[FaceRegion], img_w: int, img_h: int
) -> List[Tuple[int, int, int, int]]:
    """C1 geometric hand zones, merged and coverage-capped.

    Post-processes the raw per-face bands (``_hand_zones_from_faces``) so the
    offline guess doesn't wreck the ratio on a dense group/family photo:

    1. **Merge** mutually-overlapping bands (adjacent faces produce redundant,
       overlapping bands) so the same pixels aren't stored in several patches.
    2. **Cap + bail** — the bands are a body-proportion *guess*, and when they
       cover more than ``cfg.hand_zone_max_frac`` of the frame (a people-dense
       photo) most of that is torso/lap with no hands; storing it near-original
       destroys the ratio, so drop C1 hand protection entirely (the photo still
       compresses via the face crops + whole-image conservatism, and a user who
       needs real group-hand protection opts into C2). Mirrors the text-region
       ``text_region_max_frac`` guard.

    Only the C1 path is capped — C2 (real detection) boxes are tight and trusted.
    Pure (no I/O, no mutation).
    """
    zones = _hand_zones_from_faces(faces, img_w, img_h)
    if not zones:
        return []
    zones = _merge_overlapping_boxes(zones, _HAND_ZONE_MERGE_IOU)
    frame = float(img_w * img_h)
    if frame > 0:
        # Approximate union (merged boxes are largely disjoint; any residual
        # sub-threshold overlap only over-counts, erring toward the safe bail).
        coverage = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in zones) / frame
        if coverage > cfg.hand_zone_max_frac:
            logger.info(
                "C1 hand zones cover ~%.0f%% of the frame (> %.0f%% cap) — dropping "
                "them (dense group photo; compresses without C1 hand protection)",
                100 * coverage, 100 * cfg.hand_zone_max_frac,
            )
            return []
    return zones


def _hand_regions(
    cfg: AggressiveConfig,
    faces: List[FaceRegion],
    image: np.ndarray,
    hand_detector,
) -> List[Tuple[int, int, int, int]]:
    """Risky regions for hands, per the protect-hands tier (C1 default / C2 opt-in).

    Gated by ``content_aware and region_local and protect_hands`` (hands are a
    region-local protection, so they ride the same switches as small-face regions).
    When a ``hand_detector`` is present (C2, opt-in MediaPipe) its tight per-hand
    boxes are used as-is (trusted real detections, never capped); if it finds none,
    fall back to the offline C1 geometry so an opt-in run with a quiet detector
    still gets the safe guess. With no detector (the default), use C1 geometry. The
    C1 zones are merged and coverage-capped (``_c1_hand_zones``) so the offline
    guess doesn't wreck the ratio on a dense group photo. Returns an empty list
    when disabled. Pure apart from the (best-effort, read-only) hand detector call.
    """
    if not (cfg.content_aware and cfg.region_local and cfg.protect_hands):
        return []
    if hand_detector is not None:
        boxes = hand_detector.detect_hands(image)
        if boxes:
            return boxes
        # Detector ran but found nothing — fall through to the geometric guess.
    h, w = image.shape[:2]
    return _c1_hand_zones(cfg, faces, w, h)


def _dedupe_regions(
    primary: List[Tuple[int, int, int, int]],
    extra: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    contain_frac: float = 0.7,
) -> List[Tuple[int, int, int, int]]:
    """Append ``extra`` boxes not already substantially covered by ``primary``.

    Avoids storing a second sharp patch for a region a small-face patch already
    covers (e.g. a hand zone overlapping a small-face padded box). A box is dropped
    when at least ``contain_frac`` of its area lies inside any primary box. Order is
    preserved: primaries first, then the surviving extras. Pure.
    """
    kept = list(primary)
    for b in extra:
        bx1, by1, bx2, by2 = b
        b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        if b_area == 0:
            continue
        covered = False
        for p in primary:
            px1, py1, px2, py2 = p
            ix1, iy1 = max(bx1, px1), max(by1, py1)
            ix2, iy2 = min(bx2, px2), min(by2, py2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / b_area >= contain_frac:
                covered = True
                break
        if not covered:
            kept.append(b)
    return kept


def _resolve_bg_scale(
    cfg: AggressiveConfig,
    faces: List[FaceRegion],
    image: np.ndarray,
    base_scale: float,
    text_locally_handled: bool = False,
    detail_ratio: Optional[float] = None,
) -> Tuple[float, Optional[str]]:
    """Decide the effective bg_scale given content risk; returns (scale, reason).

    Content-aware conservatism: when ``cfg.content_aware`` is on, a background
    that the AI cannot honestly reconstruct — edge-dense (text/signage/fine
    structure) or carrying a small/distant face — has its scale raised toward
    ``cfg.conservative_bg_scale`` (compress less). This is the same lever the
    no-face fallback pulls, so it composes with it: ``base_scale`` is whatever
    that earlier logic resolved, and we only ever raise it (``max``), never
    lower a scale already made conservative.

    **Region-local interaction:** when ``cfg.region_local`` is on, the
    small/distant-face risk is handled *locally* (a sharp region patch via
    ``_risky_regions``), so this whole-image raise no longer fires on it — only
    the edge-density/text signal does. Likewise ``text_locally_handled=True``
    (the caller emitted text-cluster patches via ``_text_regions``) suppresses
    the edge branch: the risk those patches cover must not *also* raise the
    whole-image scale. When the localizer found nothing or bailed (document-like
    content), the flag stays False and this raise fires exactly as before. With
    ``region_local`` off, both signals raise the whole-image scale as before.

    ``detail_ratio`` may carry a precomputed ``_background_detail_ratio`` value
    so the caller's edge map isn't recomputed; ``None`` computes it here (the
    original behavior, kept for direct callers/tests).

    Pure (no I/O, no mutation) so the decision is unit-testable in isolation.
    Returns ``reason=None`` when the scale is left unchanged.
    """
    if not cfg.content_aware:
        return base_scale, None
    h, w = image.shape[:2]
    if not cfg.region_local and _has_risky_background_face(
        faces, w, h, cfg.small_face_ratio
    ):
        reason = "small/distant background face"
    elif not text_locally_handled and (
        _background_detail_ratio(image) if detail_ratio is None else detail_ratio
    ) > cfg.text_edge_threshold:
        reason = "detailed background (text/fine structure)"
    else:
        return base_scale, None
    new_scale = max(base_scale, cfg.conservative_bg_scale)
    if new_scale <= base_scale:
        # Already at/under the conservative floor (e.g. no-face fallback raised
        # it higher) — nothing to do, don't claim a change we didn't make.
        return base_scale, None
    return new_scale, reason


def _search_bg_scale(
    cfg: AggressiveConfig,
    image: np.ndarray,
    floor_scale: float,
) -> Tuple[float, Optional[str]]:
    """Pick the most aggressive bg_scale whose reconstructed bg meets a target.

    Quality-targeted compression (opt-in via ``cfg.quality_target``): rather than
    using one fixed ``bg_scale`` for every photo, walk ``cfg.quality_scale_candidates``
    and choose the smallest (most aggressive) scale whose *reconstructed*
    background is still perceptually close enough to the original — measured with
    LPIPS (learned perceptual distance; lower = more similar), the right metric
    for a hallucinated-but-plausible background (SSIM is not). A scale "meets the
    target" when its LPIPS distance is ``<= cfg.quality_target``.

    Restore quality is estimated with a fast **bicubic** upscale (downsample →
    bicubic back to original size), not a full Real-ESRGAN restore: it keeps the
    search offline and cheap at *compress* time and is a conservative proxy (real
    AI restore looks at least as good), so the chosen scale errs slightly toward
    quality. The stored scale is the searched value; the real restore may still
    use AI. The whole face region rides along at original quality regardless of
    scale, so comparing the full reconstructed frame is dominated by the
    background — exactly the thing the scale controls.

    Composition with content-aware conservatism: ``floor_scale`` is whatever the
    no-face / content-aware logic already resolved, and the search never returns a
    scale *below* it (``max``) — quality-targeting only ever makes a photo *more*
    conservative than the fixed/heuristic decision, preserving the "only ever
    raise the scale" invariant. Candidates at or below the floor are skipped.

    Graceful degradation (offline-first): returns ``(floor_scale, None)`` unchanged
    when quality-targeting is off, when LPIPS is unavailable (the ``[ai]`` extra
    isn't installed), or when no candidate could be scored — so the fixed
    ``bg_scale`` path is used and nothing crashes. Pure apart from the (read-only)
    LPIPS model; no I/O, no mutation of ``image``.
    """
    if cfg.quality_target is None:
        return floor_scale, None

    from ..metrics import lpips_available, lpips_distance

    if not lpips_available():
        logger.warning(
            "quality_target set but LPIPS is unavailable ([ai] extra not "
            "installed); using fixed bg_scale=%.3f.", floor_scale
        )
        return floor_scale, None

    h, w = image.shape[:2]
    # Ascending so we try the most aggressive (smallest) scale first; the first
    # one that meets the target wins. Skip any candidate at/below the floor (the
    # floor is already the least we may compress).
    candidates = sorted(
        s for s in cfg.quality_scale_candidates if s > floor_scale
    )

    best_scale: Optional[float] = None
    best_dist: Optional[float] = None
    chosen: Optional[float] = None
    for scale in candidates:
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        down = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        recon = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
        dist = lpips_distance(image, recon)
        if dist is None:
            continue
        # Track the closest-to-target candidate as a fallback if none qualify.
        if best_dist is None or dist < best_dist:
            best_dist, best_scale = dist, scale
        if dist <= cfg.quality_target:
            chosen = scale  # meets target — and ascending, so this is the most
            break           # aggressive qualifying scale.

    if chosen is None:
        # No candidate met the target (every reconstruction looked too different):
        # fall back to the gentlest scale that came closest, never below the floor.
        if best_scale is None:
            return floor_scale, None  # nothing scored (LPIPS errored on all)
        chosen = best_scale
        reason = (
            f"quality target LPIPS<= {cfg.quality_target:g} unmet; "
            f"closest bg_scale={chosen:g} (LPIPS {best_dist:.3f})"
        )
    else:
        reason = f"quality target LPIPS<= {cfg.quality_target:g} at bg_scale={chosen:g}"

    chosen = max(chosen, floor_scale)
    return chosen, reason


@dataclass
class CompressedPhoto:
    """In-memory representation of an aggressive-mode compressed photo."""

    original_filename: str
    original_width: int
    original_height: int
    original_size_bytes: int
    original_hash: str
    original_orientation: int
    exif: Optional[bytes]

    background: np.ndarray
    face_crops: List[np.ndarray]
    face_masks: List[np.ndarray]
    faces: List[FaceRegion]
    thumbnail: np.ndarray

    effective_bg_scale: float
    config: AggressiveConfig = field(default_factory=AggressiveConfig)

    # Region-local conservatism: near-original-resolution patches of risky
    # regions (the background around small/distant faces, hand zones, and
    # opt-in text-like clusters) plus their soft
    # masks and frame-coordinate bboxes. Empty unless region_local fired. Each
    # patch is composited onto the upscaled background on restore (before faces),
    # so the risky region stays sharp while the rest of the background keeps the
    # aggressive bg_scale. Defaulted empty so older callers/paths are unaffected.
    region_crops: List[np.ndarray] = field(default_factory=list)
    region_masks: List[np.ndarray] = field(default_factory=list)
    regions: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Original ICC color profile (e.g. Display P3). Stored in the .fkeep as
    # icc.bin and re-embedded on restore so wide-gamut color survives — parity
    # with faithful mode (OpenCV drops ICC, so without this restored P3 photos
    # display duller when shown as sRGB). None when the source had no profile.
    # Defaulted so older construction sites are unaffected.
    icc: Optional[bytes] = None

    # Residual layer (opt-in, aggressive.residual): the full-resolution original
    # pixels (8-bit), attached ONLY when the residual is enabled so the frame's
    # lifetime isn't extended for the default path. format.write_fkeep needs the
    # original to compute `original - bicubic(decoded background)` against the
    # background bytes it just encoded (the bytes restore will actually see) —
    # computing it earlier, pre-encode, would make the residual fight the bg
    # codec's own loss. None => no residual member is written.
    original_image: Optional[np.ndarray] = None


def compress_photo(
    image_path: str,
    config: Optional[FaceKeepConfig] = None,
    detection_cache: Optional[DetectionCache] = None,
    hand_detector=None,
) -> CompressedPhoto:
    """Compress a photo in aggressive mode (crop faces, downsample background).

    ``detection_cache`` (optional): a :class:`~facekeep.detector.DetectionCache`
    that reuses the face-detection result across re-runs (keyed by the input's
    content hash + the *effective* aggressive detector settings). ``None`` (the
    default) detects normally; the cache is a pure speed feature that never
    changes which faces are protected or the output bytes.

    ``hand_detector`` (optional): a constructed opt-in hand detector (C2) for hand
    protection. ``None`` (the default) uses the offline C1 geometric hand zones.
    Like the detection cache it is **parent-process-only** — the MediaPipe
    landmarker isn't picklable, so ``--jobs`` workers always get ``None`` and use
    C1; the CLI passes a real one on the serial path when configured. The caller
    builds it (so each worker doesn't reconstruct/redownload); we do not build it
    here.
    """
    config = config or FaceKeepConfig()
    cfg = config.aggressive

    path = Path(image_path)
    raw = path.read_bytes()
    original_hash = hashlib.sha256(raw).hexdigest()

    loaded = imageio.load(image_path, strip_gps=config.strip_gps)
    image = loaded.image
    h, w = image.shape[:2]

    # Aggressive mode protects every face — including small/distant background
    # ones, whose worst outcome is an uncanny AI reconstruction. resolved_detector
    # applies the aggressive overrides (default: YuNet + relaxed small-face
    # thresholds) on top of the shared detector config, defaulting to higher
    # recall than the faithful path. YuNet falls back to Haar offline.
    det_cfg = cfg.resolved_detector(config.detector)
    detector = create_detector(
        backend=det_cfg.backend,
        confidence=det_cfg.confidence,
        padding=det_cfg.padding,
        nms_iou=det_cfg.nms_iou,
        min_size_ratio=det_cfg.min_size_ratio,
        max_aspect_ratio=det_cfg.max_aspect_ratio,
        roi=det_cfg.roi,
    )
    # The input content hash is already computed above (original_hash); reuse it
    # as the detection-cache key so a re-run reuses the detection result.
    faces = detect_cached(detector, image, original_hash, detection_cache)

    # Zero-face handling. ``effective_bg_scale`` is the working scale (starts at
    # the configured fixed default); ``conservative_floor`` tracks the *raised*
    # protections only (no-face, content-aware) so quality-targeting can search
    # below the fixed default while never dropping under a real protection.
    effective_bg_scale = cfg.bg_scale
    conservative_floor = 0.0
    if not faces:
        if cfg.no_face_strategy == "skip":
            raise SkipFileError(f"No faces detected in {path.name}; skipping.")
        if cfg.no_face_strategy == "conservative":
            effective_bg_scale = cfg.no_face_bg_scale
            conservative_floor = cfg.no_face_bg_scale
            logger.info("No faces in %s; using conservative bg_scale=%.3f",
                        path.name, effective_bg_scale)

    # Region-local conservatism *boxes* are computed first (pure box math — the
    # pixel crops are extracted further down): the text localizer must exclude
    # the face/hand boxes, and _resolve_bg_scale must know whether the text risk
    # was handled locally before deciding the whole-image scale.
    small_face_regions = _risky_regions(cfg, faces, w, h)
    hand_regions = _hand_regions(cfg, faces, image, hand_detector)
    hand_regions = _dedupe_regions(small_face_regions, hand_regions, w, h)[
        len(small_face_regions):
    ]
    text_regions: List[Tuple[int, int, int, int]] = []
    detail_ratio: Optional[float] = None
    if cfg.content_aware:
        edges = _edge_map(image)
        if edges is not None:
            detail_ratio = float(np.count_nonzero(edges)) / float(h * w)
        # Faces are stored as sharp crops and small-face/hand regions as sharp
        # patches already — exclude them so text clusters don't form there.
        exclude = (
            [f.padded_bbox for f in faces] + small_face_regions + hand_regions
        )
        text_regions = _text_regions(cfg, image, exclude, edge_map=edges)
        text_regions = _dedupe_regions(
            small_face_regions + hand_regions, text_regions, w, h
        )[len(small_face_regions) + len(hand_regions):]

    # Content-aware conservatism: a background the AI can't honestly reconstruct
    # (text/fine structure, or a small/distant face) gets its scale raised toward
    # the conservative floor — composes on top of the no-face decision above,
    # only ever raising the scale. A text risk already covered by local patches
    # (text_regions above) is handled, so it doesn't *also* raise the scale; the
    # localizer's bail-out (document-like content) leaves the flag False and the
    # whole-image raise fires as before.
    effective_bg_scale, reason = _resolve_bg_scale(
        cfg, faces, image, effective_bg_scale,
        text_locally_handled=bool(text_regions),
        detail_ratio=detail_ratio,
    )
    if reason is not None:
        conservative_floor = max(conservative_floor, effective_bg_scale)
        logger.info("Content-aware (%s) in %s; using bg_scale=%.3f",
                    reason, path.name, effective_bg_scale)

    # Quality-targeted bg_scale (opt-in): pick the most aggressive candidate scale
    # whose reconstructed background still meets the LPIPS target. It *replaces*
    # the fixed bg_scale baseline (so it may compress harder than cfg.bg_scale),
    # but never drops below ``conservative_floor`` — the raised no-face/content-
    # aware protections (it only ever stays at/above a real protection). No-op
    # (returns the current scale) when quality_target is off or LPIPS is
    # unavailable, so the fixed-scale behavior is byte-for-byte unchanged.
    if cfg.quality_target is not None:
        searched, q_reason = _search_bg_scale(cfg, image, conservative_floor)
        if q_reason is not None:
            effective_bg_scale = searched
            logger.info("Quality-targeted (%s) in %s; using bg_scale=%.3f",
                        q_reason, path.name, effective_bg_scale)

    # Extract face crops at original quality + build blending masks
    face_crops, face_masks = [], []
    for face in faces:
        px1, py1, px2, py2 = face.padded_bbox
        face_crops.append(image[py1:py2, px1:px2].copy())
        margin = max(8, min(32, (px2 - px1) // 8))
        face_masks.append(create_soft_mask((py2 - py1, px2 - px1), margin=margin))

    # Region-local conservatism: instead of raising the *whole-image* bg_scale on
    # a localized risk, keep the risky region sharp locally — store it as a
    # near-original patch + soft mask and composite it back on restore. The
    # benign rest of the frame keeps the aggressive bg_scale. Three sources, all
    # computed above: small/distant-face padded boxes, hand zones (C2 detection
    # or C1 geometry, de-duped against the face regions), and text-like clusters
    # (de-duped against both; a widespread/document-like text risk instead bailed
    # to the whole-image raise in _resolve_bg_scale). Per-patch scale: hand
    # patches use hand_zone_scale, the others region_scale.
    region_scales = (
        [cfg.region_scale] * len(small_face_regions)
        + [cfg.hand_zone_scale] * len(hand_regions)
        + [cfg.region_scale] * len(text_regions)
    )
    regions = small_face_regions + hand_regions + text_regions
    region_crops, region_masks = [], []
    for (rx1, ry1, rx2, ry2), rscale in zip(regions, region_scales):
        patch = image[ry1:ry2, rx1:rx2]
        if rscale < 1.0:
            pw = max(1, int((rx2 - rx1) * rscale))
            ph = max(1, int((ry2 - ry1) * rscale))
            patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            patch = patch.copy()
        region_crops.append(patch)
        margin = max(8, min(32, (rx2 - rx1) // 8))
        region_masks.append(create_soft_mask((ry2 - ry1, rx2 - rx1), margin=margin))
    if regions:
        logger.info(
            "Region-local conservatism: %d risky region(s) in %s kept sharp "
            "(%d small-face, %d hand, %d text)",
            len(regions), path.name, len(small_face_regions), len(hand_regions),
            len(text_regions),
        )

    # Downsample background
    new_w = max(1, int(w * effective_bg_scale))
    new_h = max(1, int(h * effective_bg_scale))
    background = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Thumbnail
    thumb_h = 256
    thumb_w = max(1, int(w * (thumb_h / h)))
    thumbnail = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    return CompressedPhoto(
        original_filename=path.name,
        original_width=w,
        original_height=h,
        original_size_bytes=len(raw),
        original_hash=original_hash,
        original_orientation=loaded.original_orientation,
        exif=loaded.exif,
        icc=loaded.icc,
        background=background,
        face_crops=face_crops,
        face_masks=face_masks,
        faces=faces,
        thumbnail=thumbnail,
        effective_bg_scale=effective_bg_scale,
        config=cfg,
        region_crops=region_crops,
        region_masks=region_masks,
        regions=regions,
        # The .fkeep is an 8-bit container, so the residual is computed against
        # the 8-bit rendering of the source (a uint16 source rides the same
        # warned-elsewhere round-down as the background itself).
        original_image=_to_uint8(image) if cfg.residual else None,
    )
