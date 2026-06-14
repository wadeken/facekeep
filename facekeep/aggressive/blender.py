"""Face-background boundary blending for aggressive-mode restore."""

import cv2
import numpy as np


def create_soft_mask(shape: tuple[int, int], margin: int = 16) -> np.ndarray:
    """Create a feathered alpha mask: 1.0 in the center, fading to 0.0 at edges.

    Each pixel's alpha is its distance to the nearest edge, divided by `margin`
    and clipped to [0, 1]. This produces a concentric feathered border used to
    blend a high-quality face crop into the reconstructed background without a
    hard seam.
    """
    h, w = shape
    if margin <= 0:
        return np.ones((h, w), dtype=np.float32)

    ys = np.arange(h)
    xs = np.arange(w)
    dist_y = np.minimum(ys, (h - 1) - ys)[:, None]
    dist_x = np.minimum(xs, (w - 1) - xs)[None, :]
    edge_dist = np.minimum(dist_y, dist_x)
    return np.clip(edge_dist / float(margin), 0.0, 1.0).astype(np.float32)


def blend_face_onto_background(
    background: np.ndarray,
    face_crop: np.ndarray,
    face_mask: np.ndarray,
    padded_bbox: tuple[int, int, int, int],
    mode: str = "gaussian",
) -> np.ndarray:
    """Composite a face crop onto a full-size background using an alpha mask."""
    x1, y1, x2, y2 = padded_bbox
    region_h, region_w = y2 - y1, x2 - x1

    if face_crop.shape[:2] != (region_h, region_w):
        face_crop = cv2.resize(
            face_crop, (region_w, region_h), interpolation=cv2.INTER_LANCZOS4
        )
    if face_mask.shape[:2] != (region_h, region_w):
        face_mask = cv2.resize(
            face_mask, (region_w, region_h), interpolation=cv2.INTER_LINEAR
        )

    if mode == "poisson":
        return _poisson_blend(background, face_crop, face_mask, padded_bbox)

    if mode == "gaussian":
        ksize = max(3, (min(region_w, region_h) // 16) | 1)
        smooth = cv2.GaussianBlur(face_mask, (ksize, ksize), 0)
    else:  # linear
        smooth = face_mask

    mask3 = np.stack([smooth] * 3, axis=-1)
    bg_region = background[y1:y2, x1:x2].astype(np.float32)
    face_f = face_crop.astype(np.float32)
    blended = face_f * mask3 + bg_region * (1.0 - mask3)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def _poisson_blend(background, face_crop, face_mask, padded_bbox):
    x1, y1, x2, y2 = padded_bbox
    mask_u8 = (face_mask * 255).astype(np.uint8)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    try:
        return cv2.seamlessClone(
            face_crop, background, mask_u8, center, cv2.NORMAL_CLONE
        )
    except cv2.error:
        mask3 = np.stack([face_mask] * 3, axis=-1)
        bg_region = background[y1:y2, x1:x2].astype(np.float32)
        blended = face_crop.astype(np.float32) * mask3 + bg_region * (1.0 - mask3)
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
        return background
