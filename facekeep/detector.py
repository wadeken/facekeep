"""Face detection backends for FaceKeep.

Uses OpenCV's bundled detectors so the baseline works with zero extra downloads:
  - haar  (default): Haar cascade, bundled with opencv-python, always available.
            Fast, frontal-face oriented, occasional false positives.
  - yunet (optional): YuNet DNN detector. Higher accuracy, handles varied poses.
            Requires a ~232 KB ONNX model that is downloaded on first use (via
            models.ensure_weights: shared cache, SHA-256-verified, atomic write)
            and cached; falls back to Haar if the download is unavailable.
  - mediapipe (optional): Google MediaPipe BlazeFace via the **current Tasks
            API** (`mediapipe.tasks.python.vision.FaceDetector`). Needs the
            `mediapipe` package ([detect] extra) plus a ~230 KB `.tflite` model
            downloaded on first use (same shared cache as YuNet). Falls back to
            Haar if the package or model is unavailable.

(MediaPipe's *legacy* `solutions.face_detection` API was dropped in recent
mediapipe builds and is intentionally **not** used — we use the Tasks API above.
MediaPipe stays an optional extra, never a hard dependency: the default path is
Haar, offline and zero-download.)
"""

import hashlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .exceptions import DetectionError, ModelDownloadError
from .models import MODELS_CACHE_DIR, ensure_weights

logger = logging.getLogger("facekeep.detector")

# opencv_zoo stores the ONNX model via Git LFS, so the plain raw.githubusercontent
# path serves a ~131-byte LFS *pointer*, not the model. The media.* endpoint is
# GitHub's official LFS resolver and serves the real file.
YUNET_FILENAME = "yunet_2023mar.onnx"
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# SHA-256 of the real model (matches the LFS pointer's oid); verified on download
# so a changed/redirected URL fails loudly instead of silently caching garbage.
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
# Shared model cache (see facekeep.models) — one source of truth for where
# downloaded models live. The download/verify/atomic-write goes through
# models.ensure_weights (same path, same cache the AI restore weights use).
YUNET_CACHE = MODELS_CACHE_DIR / YUNET_FILENAME

# MediaPipe BlazeFace (short-range) model for the Tasks-API FaceDetector. Google
# serves it from a stable public bucket; routed through models.ensure_weights for
# the same shared cache + SHA-256 verification + atomic write as YuNet.
MEDIAPIPE_FILENAME = "blaze_face_short_range.tflite"
MEDIAPIPE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
# SHA-256 of the real model (verified on download so a changed/redirected URL
# fails loudly instead of silently caching garbage).
MEDIAPIPE_SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"

# MediaPipe Hand Landmarker model (for the opt-in C2 hand-protection backend in
# aggressive mode — see aggressive.compressor). The .task bundle packs a palm
# detector + a hand-landmark model; we use only the bounding region. Google serves
# it from the same stable public bucket as the face model; routed through
# models.ensure_weights for the shared cache + SHA-256 verification + atomic write.
HAND_FILENAME = "hand_landmarker.task"
HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
# SHA-256 of the real model (verified on download so a changed/redirected URL
# fails loudly instead of silently caching garbage).
HAND_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

# MediaPipe can emit two near-identical boxes for one physical hand (observed:
# IoU 0.69 on a real photo). HandDetector.detect_hands runs greedy NMS over its
# output at this IoU so one hand yields one region patch. 0.4 merges the duplicate
# while leaving two genuinely distinct adjacent hands (IoU well below 0.4) intact.
_HAND_NMS_IOU = 0.4


@dataclass
class FaceRegion:
    """A detected face region, in absolute pixel coordinates (origin top-left)."""

    id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) tight detection box
    padded_bbox: tuple[int, int, int, int]  # padded + clipped to image bounds
    confidence: float


def _as_uint8(image: np.ndarray) -> np.ndarray:
    """Down-convert a high-bit image to 8-bit for detection.

    The OpenCV detectors (Haar cascade, YuNet DNN) expect 8-bit input. Detection
    only drives chroma/auto-tune decisions and metadata, never output pixels, so
    rounding a uint16 source down to 8-bit here is harmless to fidelity.
    """
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return np.round(image.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def _pad_and_clip(
    bbox: tuple[int, int, int, int],
    padding: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Expand a bbox by `padding` around its center, clipped to image bounds."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
    return (
        max(0, int(cx - pw / 2)),
        max(0, int(cy - ph / 2)),
        min(img_w, int(cx + pw / 2)),
        min(img_h, int(cy + ph / 2)),
    )


# Subject ROI expansion factors, expressed in units of the *tight* face box.
# Each entry is (down, side): grow the padded box downward by `down * face_h`
# from its current bottom (to cover the torso) and outward by `side * face_w`
# on each side. The downward bias reflects that a body hangs *below* the face.
# `face` keeps the padded box unchanged so the default behaviour is byte-for-byte
# the same as before this feature.
_ROI_FACTORS: dict[str, tuple[float, float]] = {
    "face": (0.0, 0.0),
    "head_shoulders": (1.2, 0.3),
    "person": (4.0, 0.6),
}


def _expand_for_roi(
    padded_bbox: tuple[int, int, int, int],
    tight_bbox: tuple[int, int, int, int],
    roi: str,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Extend a face's padded box to cover the subject's upper body / person.

    Detectors only give a *face* box; the body is extrapolated from the face's
    dimensions. We grow the already-padded box downward (and slightly outward)
    by a multiple of the tight face box, per ``roi``, then clip to the frame.
    Only ``padded_bbox`` is affected — the tight ``bbox`` (which drives the NMS /
    size / aspect filtering) is never touched — and ``roi="face"`` is a no-op, so
    the default path is unchanged. Applied *after* false-positive filtering so it
    never inflates a box that would have been dropped.
    """
    down, side = _ROI_FACTORS.get(roi, (0.0, 0.0))
    if down == 0.0 and side == 0.0:
        return padded_bbox
    px1, py1, px2, py2 = padded_bbox
    tx1, ty1, tx2, ty2 = tight_bbox
    face_w, face_h = tx2 - tx1, ty2 - ty1
    return (
        max(0, int(px1 - side * face_w)),
        max(0, py1),
        min(img_w, int(px2 + side * face_w)),
        min(img_h, int(py2 + down * face_h)),
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_boxes(
    boxes: List[tuple[int, int, int, int]], iou_thresh: float
) -> List[tuple[int, int, int, int]]:
    """Greedy non-maximum suppression over plain (x1,y1,x2,y2) boxes.

    Keeps the larger box of any pair overlapping at ``IoU >= iou_thresh`` (the
    boxes carry no confidence, and the larger one is the better region patch).
    Used to de-duplicate hand detections, where MediaPipe can return two
    near-identical boxes for one physical hand. Pure; preserves no particular order
    beyond "larger first". A no-op when nothing overlaps (the common case).
    """
    def _area(b: tuple[int, int, int, int]) -> int:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    kept: List[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=_area, reverse=True):
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept


def _filter_detections(
    faces: List["FaceRegion"],
    img_w: int,
    img_h: int,
    *,
    nms_iou: float,
    min_size_ratio: float,
    max_aspect_ratio: float,
) -> List["FaceRegion"]:
    """Drop spurious detections using geometric cues, then suppress overlaps.

    Haar gives no confidence score and fires on texture, so we lean on geometry:
    reject boxes that are too small relative to the image or have an implausible
    aspect ratio for a face, then run non-maximum suppression over what remains.
    Shared by both detectors (YuNet benefits too); the only difference is the NMS
    ordering key — YuNet sorts by its real confidence, Haar (all confidence 1.0)
    falls back to box area, so larger boxes win ties.

    Filtering uses the tight `bbox` (not `padded_bbox`): padding inflates every
    box and makes unrelated detections overlap, which would distort the IoU and
    size judgements. `id` is re-assigned to a contiguous 0..n-1 after filtering.
    """
    kept: List["FaceRegion"] = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        # Size: short side must be a meaningful fraction of the image.
        if min(w, h) < min_size_ratio * min(img_w, img_h):
            continue
        # Aspect: a face is roughly 1:1.3; reject very wide/tall boxes.
        if max(w / h, h / w) > max_aspect_ratio:
            continue
        kept.append(f)

    # NMS: prefer higher confidence, then larger area (Haar has uniform
    # confidence, so area is what actually breaks ties for it).
    def _area(f: "FaceRegion") -> int:
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)

    kept.sort(key=lambda f: (f.confidence, _area(f)), reverse=True)

    selected: List["FaceRegion"] = []
    for f in kept:
        if all(_iou(f.bbox, s.bbox) <= nms_iou for s in selected):
            selected.append(f)

    return [
        FaceRegion(
            id=i,
            bbox=f.bbox,
            padded_bbox=f.padded_bbox,
            confidence=f.confidence,
        )
        for i, f in enumerate(selected)
    ]


def _apply_roi(
    faces: List["FaceRegion"], roi: str, img_w: int, img_h: int
) -> List["FaceRegion"]:
    """Expand each surviving face's padded box to its subject ROI (in place-free).

    A no-op for ``roi="face"``. Runs after ``_filter_detections`` so only kept
    boxes are grown. The tight ``bbox`` is preserved; only ``padded_bbox`` grows.
    """
    if _ROI_FACTORS.get(roi, (0.0, 0.0)) == (0.0, 0.0):
        return faces
    return [
        FaceRegion(
            id=f.id,
            bbox=f.bbox,
            padded_bbox=_expand_for_roi(
                f.padded_bbox, f.bbox, roi, img_w, img_h
            ),
            confidence=f.confidence,
        )
        for f in faces
    ]


class FaceDetector(ABC):
    """Abstract face detector interface (Strategy pattern)."""

    # Subclasses set these so a detector can report a settings fingerprint for
    # the detection cache. ``_backend`` names the backend; ``confidence`` is
    # carried by every concrete detector (Haar uses a fixed 1.0).
    _backend: str = "?"
    confidence: float = 1.0
    padding: float = 1.5
    nms_iou: float = 0.3
    min_size_ratio: float = 0.05
    max_aspect_ratio: float = 1.6
    roi: str = "face"

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceRegion]:
        """Detect faces in a BGR image. Returns regions in pixel coordinates."""
        ...

    def fingerprint(self) -> str:
        """Short stable hash of this detector's output-affecting settings.

        Used as the detection-cache key alongside the image content hash. Two
        detectors with the same fingerprint produce the same detections for the
        same image.
        """
        return detector_fingerprint(
            backend=self._backend,
            confidence=self.confidence,
            padding=self.padding,
            nms_iou=self.nms_iou,
            min_size_ratio=self.min_size_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
            roi=self.roi,
        )


class HaarDetector(FaceDetector):
    """Frontal-face detection using OpenCV's bundled Haar cascade."""

    def __init__(
        self,
        padding: float = 1.5,
        nms_iou: float = 0.3,
        min_size_ratio: float = 0.05,
        max_aspect_ratio: float = 1.6,
        roi: str = "face",
    ):
        self._backend = "haar"
        # Haar has no confidence score; pin it so the cache fingerprint is stable
        # (its detect() always reports confidence 1.0).
        self.confidence = 1.0
        self.padding = padding
        self.nms_iou = nms_iou
        self.min_size_ratio = min_size_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.roi = roi
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise DetectionError(f"Failed to load Haar cascade from {cascade_path}")

    def detect(self, image: np.ndarray) -> List[FaceRegion]:
        h, w = image.shape[:2]
        image = _as_uint8(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # minSize scales with image so we don't over-detect tiny noise on big photos
        min_side = max(30, int(min(h, w) * 0.04))
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )

        faces: List[FaceRegion] = []
        # detectMultiScale returns an empty tuple (not array) when nothing found
        for i, (x, y, bw, bh) in enumerate(detections):
            bbox = (int(x), int(y), int(x + bw), int(y + bh))
            faces.append(
                FaceRegion(
                    id=i,
                    bbox=bbox,
                    padded_bbox=_pad_and_clip(bbox, self.padding, w, h),
                    confidence=1.0,  # Haar does not provide a score
                )
            )
        # Haar fires on texture; drop geometric noise and overlaps.
        kept = _filter_detections(
            faces, w, h,
            nms_iou=self.nms_iou,
            min_size_ratio=self.min_size_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
        )
        return _apply_roi(kept, self.roi, w, h)


class YuNetDetector(FaceDetector):
    """DNN-based detection using OpenCV's YuNet model (auto-downloaded)."""

    def __init__(
        self,
        confidence: float = 0.6,
        padding: float = 1.5,
        nms_iou: float = 0.3,
        min_size_ratio: float = 0.05,
        max_aspect_ratio: float = 1.6,
        roi: str = "face",
    ):
        self._backend = "yunet"
        self.confidence = confidence
        self.padding = padding
        self.nms_iou = nms_iou
        self.min_size_ratio = min_size_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.roi = roi
        self._model_path = self._ensure_model()
        self._detector = cv2.FaceDetectorYN.create(
            model=str(self._model_path),
            config="",
            input_size=(320, 320),
            score_threshold=confidence,
        )

    @staticmethod
    def _ensure_model() -> Path:
        """Return a local path to the verified YuNet model, downloading if needed.

        Routes through ``models.ensure_weights`` for the shared cache, SHA-256
        verification, and atomic write. A download/checksum failure surfaces as a
        ``DetectionError`` (translated from ``ModelDownloadError``) so the existing
        offline fallback in ``create_detector`` degrades to Haar unchanged.
        """
        try:
            return ensure_weights(
                YUNET_URL, YUNET_FILENAME, sha256=YUNET_SHA256
            )
        except ModelDownloadError as e:
            raise DetectionError(
                f"Could not download YuNet model ({e}). "
                "Use detector backend 'haar' for offline use."
            ) from e

    def detect(self, image: np.ndarray) -> List[FaceRegion]:
        h, w = image.shape[:2]
        image = _as_uint8(image)
        self._detector.setInputSize((w, h))
        _, results = self._detector.detect(image)

        faces: List[FaceRegion] = []
        if results is None:
            return faces

        for i, det in enumerate(results):
            x, y, bw, bh = det[0], det[1], det[2], det[3]
            score = float(det[14])
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            bbox = (x1, y1, x2, y2)
            faces.append(
                FaceRegion(
                    id=i,
                    bbox=bbox,
                    padded_bbox=_pad_and_clip(bbox, self.padding, w, h),
                    confidence=round(score, 4),
                )
            )
        # Shared geometric/NMS filter (YuNet benefits too; it sorts by score).
        kept = _filter_detections(
            faces, w, h,
            nms_iou=self.nms_iou,
            min_size_ratio=self.min_size_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
        )
        return _apply_roi(kept, self.roi, w, h)


class MediaPipeDetector(FaceDetector):
    """Face detection using MediaPipe BlazeFace via the current Tasks API.

    The legacy ``mediapipe.solutions`` API was removed in recent builds; this
    uses ``mediapipe.tasks.python.vision.FaceDetector`` instead. The package is
    an optional extra ([detect]) and the ~230 KB ``.tflite`` model is downloaded
    on first use (shared cache, SHA-256-verified). A missing package or model
    raises ``DetectionError`` so ``create_detector`` degrades to Haar.
    """

    def __init__(
        self,
        confidence: float = 0.6,
        padding: float = 1.5,
        nms_iou: float = 0.3,
        min_size_ratio: float = 0.05,
        max_aspect_ratio: float = 1.6,
        roi: str = "face",
    ):
        self._backend = "mediapipe"
        self.confidence = confidence
        self.padding = padding
        self.nms_iou = nms_iou
        self.min_size_ratio = min_size_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.roi = roi
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise DetectionError(
                f"MediaPipe is not installed ({e}). Install it with "
                "'pip install facekeep[detect]', or use detector backend 'haar' "
                "for offline use."
            ) from e

        self._mp = mp
        model_path = self._ensure_model()
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    @staticmethod
    def _ensure_model() -> Path:
        """Return a local path to the verified BlazeFace model, downloading if needed.

        Routes through ``models.ensure_weights`` (shared cache, SHA-256 verify,
        atomic write). A download/checksum failure surfaces as ``DetectionError``
        (from ``ModelDownloadError``) so ``create_detector`` falls back to Haar.
        """
        try:
            return ensure_weights(
                MEDIAPIPE_URL, MEDIAPIPE_FILENAME, sha256=MEDIAPIPE_SHA256
            )
        except ModelDownloadError as e:
            raise DetectionError(
                f"Could not download MediaPipe model ({e}). "
                "Use detector backend 'haar' for offline use."
            ) from e

    def detect(self, image: np.ndarray) -> List[FaceRegion]:
        h, w = image.shape[:2]
        image = _as_uint8(image)
        # MediaPipe expects RGB; this is the only BGR->RGB boundary in this path.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._detector.detect(mp_image)

        faces: List[FaceRegion] = []
        for i, det in enumerate(result.detections):
            box = det.bounding_box  # origin_x/origin_y/width/height in pixels
            x1, y1 = max(0, int(box.origin_x)), max(0, int(box.origin_y))
            x2 = min(w, int(box.origin_x + box.width))
            y2 = min(h, int(box.origin_y + box.height))
            # Categories carry the detection score; default to 1.0 if absent.
            score = det.categories[0].score if det.categories else 1.0
            bbox = (x1, y1, x2, y2)
            faces.append(
                FaceRegion(
                    id=i,
                    bbox=bbox,
                    padded_bbox=_pad_and_clip(bbox, self.padding, w, h),
                    confidence=round(float(score), 4),
                )
            )
        # Shared geometric/NMS filter (sorts by the real score here).
        kept = _filter_detections(
            faces, w, h,
            nms_iou=self.nms_iou,
            min_size_ratio=self.min_size_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
        )
        return _apply_roi(kept, self.roi, w, h)


class HandDetector:
    """Opt-in hand detection (MediaPipe Hand Landmarker) for aggressive mode.

    This is the **C2** tier of aggressive-mode hand protection (see
    ``aggressive.compressor``): it returns a tight bounding box per detected hand
    so the compressor can keep that region sharp (a region-local patch), instead
    of letting hands ride the background downsample and get smeared by the AI
    upscaler. It is *not* a ``FaceDetector`` — hands are not faces, so it shares no
    padding/ROI/chroma logic; it just yields boxes.

    Opt-in, never the default: OpenCV ships no hand cascade, so the offline default
    is geometric (C1, in the compressor). This backend needs the ``mediapipe``
    package ([detect] extra) and downloads a ~7.8 MB ``.task`` model on first use
    (shared cache, SHA-256-verified). A missing package/model raises
    ``DetectionError`` so ``create_hand_detector`` degrades to C1.
    """

    def __init__(
        self,
        confidence: float = 0.3,
        num_hands: int = 6,
        detect_long_side: int = 1280,
        padding: float = 1.25,
    ):
        # confidence: the landmarker's detection floor (lower = more hands; the
        #   default leans toward recall — missing a hand is the failure that matters,
        #   a false hand only protects an extra patch).
        # num_hands: cap on hands returned (family photos have several).
        # detect_long_side: downscale the *detection input* to this long side before
        #   running the landmarker — MediaPipe's palm detector is trained for
        #   phone-sized frames, so on a 12 MP photo a hand is too small a fraction
        #   to detect at full res. Landmark coords are normalized, so the boxes map
        #   back to the original frame for free (only the input is scaled). 0 = no
        #   downscale. This is the main recall lever.
        # padding: expansion around the tight landmark box so finger tips/edges
        #   stay inside the kept patch.
        self.confidence = confidence
        self.num_hands = num_hands
        self.detect_long_side = detect_long_side
        self.padding = padding
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise DetectionError(
                f"MediaPipe is not installed ({e}). Install it with "
                "'pip install facekeep[detect]' to enable hand detection, or leave "
                "aggressive.protect_hands_backend=None for offline geometric hand "
                "zones."
            ) from e

        self._mp = mp
        model_path = self._ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=num_hands,
            min_hand_detection_confidence=confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_model() -> Path:
        """Return a local path to the verified Hand Landmarker model.

        Routes through ``models.ensure_weights`` (shared cache, SHA-256 verify,
        atomic write). A download/checksum failure surfaces as ``DetectionError``
        (from ``ModelDownloadError``) so the caller falls back to C1 geometry.
        """
        try:
            return ensure_weights(HAND_URL, HAND_FILENAME, sha256=HAND_SHA256)
        except ModelDownloadError as e:
            raise DetectionError(
                f"Could not download MediaPipe hand model ({e}). "
                "Leave aggressive.protect_hands_backend=None for offline use."
            ) from e

    def detect_hands(
        self, image: np.ndarray
    ) -> List[tuple[int, int, int, int]]:
        """Tight bounding box per detected hand, in frame pixel coordinates.

        Best-effort: returns ``[]`` on no hands or any inference error (swallowed +
        logged) — hand protection is a bonus, never allowed to fail the pipeline.
        The box is the min/max of the 21 hand landmarks (normalized → pixels),
        padded slightly and clamped to the frame.

        The detection input is downscaled to ``detect_long_side`` first (MediaPipe's
        palm detector misses hands that are a tiny fraction of a large frame); since
        landmarks are normalized, the boxes are computed against the *original*
        ``w, h`` and so come out in full-resolution pixels regardless of the
        downscale. Overlapping detections are NMS-de-duplicated (MediaPipe can emit
        two near-identical boxes for one physical hand) so each hand yields one box.

        **Honest limit:** a hand heavily occluded by a held object (e.g. wrapped by
        a snake) can still be missed at any confidence/resolution — that is the palm
        detector's reach, not a tuning gap.
        """
        h, w = image.shape[:2]
        img8 = _as_uint8(image)
        # MediaPipe expects RGB; this is the only BGR->RGB boundary on this path.
        rgb = cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)
        # Downscale the detection input for recall on large photos. Only the input
        # to the landmarker shrinks — landmark coords are normalized, so the boxes
        # below multiply by the ORIGINAL w, h and land in the full-res frame.
        long_side = max(h, w)
        if self.detect_long_side and long_side > self.detect_long_side:
            ds = self.detect_long_side / long_side
            rgb = cv2.resize(
                rgb, (max(1, int(w * ds)), max(1, int(h * ds))),
                interpolation=cv2.INTER_AREA,
            )
        try:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            result = self._landmarker.detect(mp_image)
        except Exception as e:  # noqa: BLE001 - best-effort, never fail the pipeline
            logger.warning("Hand detection failed (%s); skipping hand protection.", e)
            return []

        boxes: List[tuple[int, int, int, int]] = []
        for hand in getattr(result, "hand_landmarks", []) or []:
            xs = [lm.x for lm in hand]
            ys = [lm.y for lm in hand]
            if not xs or not ys:
                continue
            x1, x2 = min(xs) * w, max(xs) * w
            y1, y2 = min(ys) * h, max(ys) * h
            box = _pad_and_clip(
                (int(x1), int(y1), int(x2), int(y2)),
                self.padding, w, h,
            )
            bx1, by1, bx2, by2 = box
            if bx2 > bx1 and by2 > by1:
                boxes.append(box)
        # De-duplicate: MediaPipe can return two near-identical boxes for one
        # physical hand, which would otherwise become two overlapping region
        # patches. NMS keeps one box per hand. (No-op when nothing overlaps.)
        return _nms_boxes(boxes, _HAND_NMS_IOU)


def create_hand_detector(
    backend: Optional[str],
    confidence: float = 0.3,
    num_hands: int = 6,
    detect_long_side: int = 1280,
    padding: float = 1.25,
):
    """Construct an opt-in hand detector, or ``None`` to use C1 geometry.

    ``backend == "mediapipe"`` tries :class:`HandDetector` (threading the C2
    recall/tuning knobs through); if it can't be built (package/model unavailable →
    ``DetectionError``) it logs and returns ``None`` so the caller falls back to the
    offline geometric hand zone (C1). Any other value (including ``None``) returns
    ``None`` — the default offline path. This is the single "construct may fail →
    degrade" seam, mirroring ``create_detector``.
    """
    if backend == "mediapipe":
        try:
            return HandDetector(
                confidence=confidence,
                num_hands=num_hands,
                detect_long_side=detect_long_side,
                padding=padding,
            )
        except DetectionError as e:
            logger.warning(
                "Hand detector unavailable (%s); using offline geometric hand "
                "zones (C1).", e
            )
            return None
    return None


# --------------------------------------------------------------------------- #
# Custom-detector plugin hook (ROADMAP backlog)
#
# A third party can register their own FaceDetector under a new backend name and
# select it via config (`detector.backend: <name>`) exactly like a built-in. The
# registry maps a backend name to a *factory* callable that takes the same
# keyword args `create_detector` resolves (`confidence`, `padding`, `nms_iou`,
# `min_size_ratio`, `max_aspect_ratio`, `roi`) and returns a `FaceDetector`.
#
# Discipline mirroring the built-ins:
#   - The built-in names (haar/yunet/mediapipe) are reserved and cannot be
#     overridden — registering one raises, so a plugin can never silently shadow
#     the offline default.
#   - A registered factory's detector is used *as-is* (no Haar fallback wrapping):
#     a custom backend opts into owning its own degradation. If the factory itself
#     raises, that surfaces (it's the plugin author's bug), matching how an unknown
#     backend already raises rather than guessing.
#   - The detection cache + incremental index key off `detector.fingerprint()` /
#     `index.settings_fingerprint`, which read the detector's `_backend` and the
#     shared filter fields. A custom detector that subclasses `FaceDetector` and
#     sets `_backend` to its registered name (plus the filter fields, e.g. by
#     accepting them in __init__) gets correct cache busting for free; one that
#     does not set `_backend` fingerprints as "?" — still safe (stable), just
#     coarser. This is documented for plugin authors.
# --------------------------------------------------------------------------- #

# Names that ship with FaceKeep and may not be re-registered by a plugin.
_BUILTIN_BACKENDS = ("haar", "yunet", "mediapipe")

# Registered custom backends: name -> factory(**kwargs) -> FaceDetector.
_DETECTOR_REGISTRY: dict = {}


def register_detector(name: str, factory) -> None:
    """Register a custom face-detector backend under ``name``.

    ``factory`` is a callable invoked as ``factory(confidence=, padding=,
    nms_iou=, min_size_ratio=, max_aspect_ratio=, roi=)`` — the same keyword set
    :func:`create_detector` resolves for the built-ins — and must return a
    :class:`FaceDetector`. After registering, set ``detector.backend: <name>`` in
    config (or pass ``backend=<name>`` to :func:`create_detector`) to use it.

    For correct detection-cache / incremental-index behavior, the returned
    detector should subclass :class:`FaceDetector`, set ``_backend`` to ``name``,
    and carry the filter fields it was given (so its ``fingerprint()`` reflects
    its settings); see the registry note above.

    Raises:
        DetectionError: if ``name`` is empty, collides with a built-in backend
            (haar/yunet/mediapipe — reserved so a plugin can't shadow the offline
            default), or is already registered, or if ``factory`` isn't callable.
    """
    if not name or not isinstance(name, str):
        raise DetectionError("Detector backend name must be a non-empty string.")
    if name in _BUILTIN_BACKENDS:
        raise DetectionError(
            f"Cannot register {name!r}: it is a built-in backend and is reserved."
        )
    if name in _DETECTOR_REGISTRY:
        raise DetectionError(
            f"Detector backend {name!r} is already registered "
            "(unregister it first to replace it)."
        )
    if not callable(factory):
        raise DetectionError(f"Detector factory for {name!r} must be callable.")
    _DETECTOR_REGISTRY[name] = factory
    logger.debug("Registered custom detector backend %r", name)


def unregister_detector(name: str) -> None:
    """Remove a previously-registered custom backend (no-op if absent).

    Built-in backends cannot be unregistered.
    """
    if name in _BUILTIN_BACKENDS:
        raise DetectionError(f"Cannot unregister the built-in backend {name!r}.")
    _DETECTOR_REGISTRY.pop(name, None)


def registered_detectors() -> tuple:
    """Return the names of currently-registered custom backends (sorted)."""
    return tuple(sorted(_DETECTOR_REGISTRY))


def known_backends() -> tuple:
    """All selectable backend names: the built-ins plus registered customs."""
    return _BUILTIN_BACKENDS + registered_detectors()


def is_known_backend(name: str) -> bool:
    """True if ``name`` is a built-in or registered custom backend (for validate)."""
    return name in _BUILTIN_BACKENDS or name in _DETECTOR_REGISTRY


def create_detector(
    backend: str = "haar",
    confidence: float = 0.6,
    padding: float = 1.5,
    nms_iou: float = 0.3,
    min_size_ratio: float = 0.05,
    max_aspect_ratio: float = 1.6,
    roi: str = "face",
) -> FaceDetector:
    """Factory for face detectors.

    Args:
        backend: 'haar' (bundled, offline), 'yunet' (DNN, auto-downloads model),
            'mediapipe' (Tasks-API BlazeFace; needs the [detect] extra +
            auto-downloads a model), or any name a plugin registered via
            :func:`register_detector`. The two optional built-in backends fall
            back to Haar if their package/model is unavailable; a custom backend
            owns its own degradation.
        confidence: Minimum detection confidence (yunet / mediapipe only)
        padding: Padding multiplier around the detected face box
        nms_iou: IoU above which overlapping boxes are suppressed
        min_size_ratio: Min face short-side as a fraction of the image short side
        max_aspect_ratio: Max width/height (or height/width) for a plausible face
        roi: high-priority region — 'face' (just the padded face box),
            'head_shoulders', or 'person' (grow it downward to cover the body)

    Returns:
        A FaceDetector. If 'yunet' is requested but its model cannot be obtained,
        falls back to 'haar' with a warning.
    """
    filter_kwargs = dict(
        nms_iou=nms_iou,
        min_size_ratio=min_size_ratio,
        max_aspect_ratio=max_aspect_ratio,
        roi=roi,
    )
    if backend == "haar":
        return HaarDetector(padding=padding, **filter_kwargs)
    if backend == "yunet":
        try:
            return YuNetDetector(confidence=confidence, padding=padding, **filter_kwargs)
        except DetectionError as e:
            logger.warning("YuNet unavailable (%s); falling back to Haar.", e)
            return HaarDetector(padding=padding, **filter_kwargs)
    if backend == "mediapipe":
        try:
            return MediaPipeDetector(
                confidence=confidence, padding=padding, **filter_kwargs
            )
        except DetectionError as e:
            logger.warning("MediaPipe unavailable (%s); falling back to Haar.", e)
            return HaarDetector(padding=padding, **filter_kwargs)

    # Custom plugin backend (ROADMAP backlog). A registered factory is invoked
    # with the same kwargs the built-ins receive and must return a FaceDetector.
    # No Haar-fallback wrapping here: a custom backend owns its degradation, and a
    # factory bug should surface, not be masked as "use Haar".
    factory = _DETECTOR_REGISTRY.get(backend)
    if factory is not None:
        detector = factory(
            confidence=confidence, padding=padding, **filter_kwargs
        )
        if not isinstance(detector, FaceDetector):
            raise DetectionError(
                f"Custom detector backend {backend!r} returned a "
                f"{type(detector).__name__}, not a FaceDetector."
            )
        return detector

    raise DetectionError(f"Unknown detector backend: {backend!r}")


# --------------------------------------------------------------------------- #
# Detection caching (ROADMAP Phase 6)
#
# Detection of a given image under a given set of detector settings is
# deterministic, so its result can be cached by (content hash, detector
# fingerprint) and reused on a re-run. This is a *pure speed* feature, exactly
# like the incremental index (facekeep.index): it never changes which faces a
# pipeline acts on or the output bytes — a cache hit just hands back the same
# FaceRegion list the detector would have produced.
#
# It is complementary to the incremental index, which caches *whole-file*
# outcomes (and so skips detection entirely on a full hit). The detection cache
# helps the cases the file-index misses but detection did NOT change: e.g. a
# re-run with a different --quality (busts the file index, re-encodes) while the
# detector settings and the image are identical — the faces need not be found
# again.
# --------------------------------------------------------------------------- #

# Shared on-disk detection cache, next to the model cache (one cache root). Kept
# separate from the per-output incremental index (facekeep.index), which lives
# beside the outputs it describes; detections are image-intrinsic, so they live
# in the user-global cache and are reusable across output dirs.
DETECTION_CACHE_PATH = MODELS_CACHE_DIR.parent / "detections.sqlite"

# Bump if the row schema changes incompatibly. On a mismatch the table is wiped
# and treated as empty (a stale cache only costs re-detection, never
# correctness) — the same strategy as facekeep.index.
_DETECTION_SCHEMA_VERSION = 1


def detector_fingerprint(
    backend: str,
    confidence: float,
    padding: float,
    nms_iou: float,
    min_size_ratio: float,
    max_aspect_ratio: float,
    roi: str,
) -> str:
    """Short stable hash of every setting that affects detection output.

    Two detectors with the same fingerprint produce the same FaceRegion list for
    the same image, so a cached detection is reusable; any difference here must
    bust the cache. These are exactly the detector fields ``index.py`` already
    treats as output-affecting (kept in sync deliberately).
    """
    blob = json.dumps(
        {
            "backend": backend,
            "confidence": confidence,
            "padding": padding,
            "nms_iou": nms_iou,
            "min_size_ratio": min_size_ratio,
            "max_aspect_ratio": max_aspect_ratio,
            "roi": roi,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _faces_to_json(faces: List[FaceRegion]) -> str:
    """Serialize a FaceRegion list to a compact JSON string for the cache."""
    return json.dumps(
        [
            {
                "id": f.id,
                "bbox": list(f.bbox),
                "padded_bbox": list(f.padded_bbox),
                "confidence": f.confidence,
            }
            for f in faces
        ],
        separators=(",", ":"),
    )


def _faces_from_json(blob: str) -> List[FaceRegion]:
    """Rebuild a FaceRegion list from cached JSON.

    JSON turns the bbox tuples into lists, so they are converted back to tuples
    to match what the detectors return (the rest of the system reads bboxes as
    tuples; a round-tripped FaceRegion must compare equal to a freshly-detected
    one).
    """
    return [
        FaceRegion(
            id=int(d["id"]),
            bbox=tuple(d["bbox"]),
            padded_bbox=tuple(d["padded_bbox"]),
            confidence=float(d["confidence"]),
        )
        for d in json.loads(blob)
    ]


class DetectionCache:
    """A SQLite-backed cache of detections, keyed by (content hash, detector fp).

    Best-effort by design: it is a pure speed optimization, so any error reading
    or writing it is swallowed (logged) and the caller falls back to running the
    detector. Use as a context manager so the connection is always closed::

        with DetectionCache() as cache:
            faces = cache.lookup(content_hash, fp)
            ...
            cache.record(content_hash, fp, faces)

    Concurrency: like ``index.ProcessIndex`` this is intended to be opened in a
    single process. The CLI only attaches it on the serial path (``--jobs<=1``),
    so multiple worker processes never contend on the same DB file.
    """

    def __init__(self, db_path: str | Path = DETECTION_CACHE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def __enter__(self) -> "DetectionCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def _ensure_schema(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version != _DETECTION_SCHEMA_VERSION:
            self._conn.execute("DROP TABLE IF EXISTS detections")
            self._conn.execute(
                """
                CREATE TABLE detections (
                    content_hash         TEXT NOT NULL,
                    detector_fingerprint TEXT NOT NULL,
                    faces_json           TEXT NOT NULL,
                    updated_at           TEXT NOT NULL,
                    PRIMARY KEY (content_hash, detector_fingerprint)
                )
                """
            )
            self._conn.execute(
                f"PRAGMA user_version = {_DETECTION_SCHEMA_VERSION}"
            )
            self._conn.commit()

    def lookup(
        self, content_hash: str, fingerprint: str
    ) -> Optional[List[FaceRegion]]:
        """Return the cached detections, or None on a miss / any error.

        A miss and an error are deliberately indistinguishable to the caller:
        both mean "run the detector", so the cache can never change behaviour.
        """
        try:
            row = self._conn.execute(
                "SELECT faces_json FROM detections "
                "WHERE content_hash = ? AND detector_fingerprint = ?",
                (content_hash, fingerprint),
            ).fetchone()
        except sqlite3.Error as e:
            logger.warning("Detection cache lookup failed (%s); detecting.", e)
            return None
        if row is None:
            return None
        try:
            return _faces_from_json(row["faces_json"])
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Corrupt detection cache row (%s); detecting.", e)
            return None

    def record(
        self, content_hash: str, fingerprint: str, faces: List[FaceRegion]
    ) -> None:
        """Upsert the detections for (content_hash, fingerprint). Best-effort."""
        from datetime import datetime, timezone

        try:
            self._conn.execute(
                """
                INSERT INTO detections
                    (content_hash, detector_fingerprint, faces_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(content_hash, detector_fingerprint) DO UPDATE SET
                    faces_json = excluded.faces_json,
                    updated_at = excluded.updated_at
                """,
                (
                    content_hash,
                    fingerprint,
                    _faces_to_json(faces),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.warning("Detection cache write failed (%s); ignoring.", e)


def detect_cached(
    detector: FaceDetector,
    image: np.ndarray,
    content_hash: str,
    cache: Optional[DetectionCache],
) -> List[FaceRegion]:
    """Detect faces, reusing a cached result when available.

    Looks the result up by ``(content_hash, fingerprint of detector settings)``;
    on a hit returns it without running the detector, on a miss runs the detector
    and records the result. ``cache=None`` (the default in the pipelines, and the
    only behaviour on the ``--jobs`` parallel path) just calls ``detector.detect``
    — so the cache is strictly opt-in and never affects output bytes.

    The detector must expose its settings via a ``fingerprint()`` method (all the
    concrete detectors do); if it does not, caching is skipped.
    """
    if cache is None:
        return detector.detect(image)
    fp = getattr(detector, "fingerprint", None)
    if fp is None:
        return detector.detect(image)
    fingerprint = fp()
    hit = cache.lookup(content_hash, fingerprint)
    if hit is not None:
        logger.debug("Detection cache hit (%s/%s)", content_hash[:8], fingerprint)
        return hit
    faces = detector.detect(image)
    cache.record(content_hash, fingerprint, faces)
    return faces
