"""Image loading with EXIF preservation and orientation correction.

OpenCV's imread/imwrite ignore EXIF entirely, which (a) loses capture date,
GPS, camera info, and (b) leaves rotated phone photos in the wrong orientation
so face detection runs on a sideways image. This module loads the image,
physically applies the EXIF orientation, and carries the EXIF bytes through so
they can be re-embedded on output.
"""

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .exceptions import UnsupportedInputError

logger = logging.getLogger("facekeep.imageio")

SUPPORTED_INPUT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# These additionally require plugins; checked at load time.
PLUGIN_INPUT = {".heic", ".heif", ".avif", ".jxl"}

# EXIF Orientation tag -> OpenCV transform to make pixels upright.
_ORIENTATION_OPS = {
    2: lambda im: cv2.flip(im, 1),
    3: lambda im: cv2.rotate(im, cv2.ROTATE_180),
    4: lambda im: cv2.flip(im, 0),
    5: lambda im: cv2.flip(cv2.transpose(im), 1),
    6: lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
    7: lambda im: cv2.flip(cv2.transpose(im), 0),
    8: lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
}


@dataclass
class LoadedImage:
    """An image loaded with its metadata."""

    image: np.ndarray  # BGR, already upright (orientation applied). uint8 or uint16.
    exif: Optional[bytes]  # Original EXIF bytes (orientation tag normalized to 1)
    original_orientation: int
    width: int
    height: int
    icc: Optional[bytes] = None  # Original ICC color profile (e.g. Display P3)
    source_bit_depth: int = 8  # 8 or 16: the per-channel bit depth of the source.
    # NOTE: the bundled AVIF/JXL encoders have no high-bit path (see encoders.py),
    # so >8-bit sources are currently down-converted to 8-bit at the encode
    # boundary. This field records the source depth for an honest warning and for
    # a future true-high-bit encode path (avifenc CLI — ROADMAP Phase 1 follow-up).


def _strip_gps_from_exif(exif_bytes: Optional[bytes]) -> Optional[bytes]:
    """Return EXIF bytes with the GPS IFD removed (privacy on export).

    Family photos routinely carry the capture location in the GPS IFD. When the
    user opts into ``strip_gps``, we drop *only* that IFD and re-serialize, so
    everything else (capture date, camera, orientation already normalized) is
    preserved. Best-effort: an unparseable block, or one with no GPS, is returned
    unchanged — stripping must never fail the pipeline.
    """
    if not exif_bytes:
        return exif_bytes
    try:
        import piexif

        exif_dict = piexif.load(exif_bytes)
        if not exif_dict.get("GPS"):
            return exif_bytes  # nothing to strip; leave bytes byte-for-byte
        exif_dict["GPS"] = {}
        return piexif.dump(exif_dict)
    # piexif raises InvalidImageDataError (a ValueError) on a non-JPEG/TIFF block
    # and struct.error on a truncated one; GPS-strip is best-effort like the rest.
    except (ValueError, struct.error, UnicodeDecodeError) as e:
        logger.debug("Could not strip GPS from EXIF (%s)", e)
        return exif_bytes


def _normalize_orientation_in_exif(exif_bytes: Optional[bytes]):
    """Parse EXIF bytes, return (normalized_bytes, orientation).

    Reads the orientation tag, then rewrites the tag to 1 in the returned bytes
    (we physically rotate the pixels, so the saved EXIF must not say "rotate"
    again). A missing/unparseable EXIF block yields ``(exif_bytes, 1)``.
    """
    if not exif_bytes:
        return exif_bytes, 1
    try:
        import piexif

        exif_dict = piexif.load(exif_bytes)
        orientation = exif_dict.get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
        if "0th" in exif_dict and piexif.ImageIFD.Orientation in exif_dict["0th"]:
            exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
        return piexif.dump(exif_dict), int(orientation)
    # piexif raises InvalidImageDataError (a ValueError) on a non-JPEG/TIFF block
    # and struct.error on a truncated/malformed one; EXIF here is best-effort.
    except (ValueError, struct.error, UnicodeDecodeError) as e:
        logger.debug("Could not parse EXIF bytes for orientation (%s)", e)
        return exif_bytes, 1


def _read_exif_and_orientation(path: Path):
    """Read EXIF bytes + orientation from a JPEG/TIFF path via piexif.

    Used for the OpenCV-decoded formats (JPEG/PNG/WebP/BMP/TIFF). piexif only
    understands JPEG/TIFF containers, so non-JPEG plugin formats (HEIC/AVIF/JXL)
    must NOT come through here — they read orientation off the opened Pillow
    image instead (see ``_read_exif_orientation_from_pil``); calling
    ``piexif.load`` on those raises "neither JPEG nor TIFF" and the orientation
    would be silently lost.
    """
    try:
        import piexif

        exif_bytes = piexif.dump(piexif.load(str(path)))
    # InvalidImageDataError (a ValueError) for non-JPEG/TIFF, struct.error /
    # FileNotFoundError on malformed or absent data; all mean "no usable EXIF".
    except (ValueError, struct.error, FileNotFoundError, UnicodeDecodeError) as e:
        logger.debug("No EXIF read for %s (%s)", path.name, e)
        return None, 1
    return _normalize_orientation_in_exif(exif_bytes)


def _read_exif_orientation_from_pil(pil) -> tuple[Optional[bytes], int]:
    """Read EXIF bytes + orientation off an already-opened Pillow image.

    This is the single source of truth for the plugin formats (HEIC/AVIF/JXL),
    and it sidesteps the piexif "JPEG/TIFF only" limitation by using the EXIF
    block Pillow already decoded (``info["exif"]``, parseable by piexif).

    It also avoids double-rotation without trusting any per-plugin behaviour:
    we rely solely on the orientation tag Pillow reports on the *decoded* image.
    If a plugin already applied the orientation (pillow-heif rotates the pixels
    and resets the tag to 1), we read 1 and do nothing further. If a plugin did
    not (pillow-avif / pillow-jxl leave pixels unrotated and keep the tag), we
    read the real value and rotate once. Either way the pixels end up upright
    exactly once. (This assumes a plugin that rotates also clears the tag, which
    holds for all three bundled plugins.)
    """
    exif_bytes = pil.info.get("exif")
    if exif_bytes:
        return _normalize_orientation_in_exif(exif_bytes)
    # No raw EXIF block, but Pillow may still expose the orientation tag.
    try:
        orientation = int(pil.getexif().get(0x0112, 1))
    except (AttributeError, KeyError, ValueError, TypeError) as e:
        # No/odd EXIF dict, or a non-int orientation tag; best-effort.
        logger.debug("Could not read orientation from Pillow image (%s)", e)
        orientation = 1
    return None, orientation


def _read_icc(path: Path) -> Optional[bytes]:
    """Read the source ICC color profile via Pillow, if any.

    OpenCV's imread drops ICC entirely, so wide-gamut photos (Display P3 on most
    modern phones) would come out color-shifted. We read the profile here and
    carry it through so it can be re-embedded on output. Best-effort: a missing
    or unreadable profile is normal (most plain sRGB images have none).
    """
    try:
        from PIL import Image

        with Image.open(str(path)) as pil:
            return pil.info.get("icc_profile")
    # UnidentifiedImageError and FileNotFoundError are both OSError subclasses;
    # ICC is best-effort metadata, so an unreadable/odd file just yields None.
    except OSError as e:
        logger.debug("No ICC read for %s (%s)", path.name, e)
        return None


def _normalize_to_bgr(image: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary OpenCV-decoded array to 3-channel BGR.

    `IMREAD_UNCHANGED` may return grayscale (HxW), BGRA (4ch), or palette/odd
    layouts. We always hand the pipeline a 3-channel BGR image, preserving the
    dtype (uint8 or uint16) so a high-bit source stays high-bit.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ch = image.shape[2]
    if ch == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if ch == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image  # already 3-channel BGR


def load(path: str, strip_gps: bool = False) -> LoadedImage:
    """Load an image, applying EXIF orientation and preserving EXIF bytes.

    Args:
        path: Path to the image to load.
        strip_gps: When True, remove the GPS IFD from the carried EXIF bytes
            (capture location) so it is never re-embedded on export. Off by
            default, so the EXIF round-trips byte-for-byte as before. A photo
            with no GPS data is unaffected either way.

    Raises:
        UnsupportedInputError: if the file can't be read or the format needs a
            plugin that isn't installed.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    image = None
    icc: Optional[bytes] = None
    source_bit_depth = 8
    exif_bytes: Optional[bytes] = None
    orientation = 1
    if suffix in PLUGIN_INPUT:
        # HEIC/AVIF/JXL: decode via Pillow (needs plugin)
        try:
            from PIL import Image

            if suffix in {".heic", ".heif"}:
                import pillow_heif

                # The HEIF opener is not auto-registered on import.
                pillow_heif.register_heif_opener()
            elif suffix == ".avif":
                import pillow_avif  # noqa: F401
            elif suffix == ".jxl":
                import pillow_jxl  # noqa: F401
            pil = Image.open(str(p))
            # Grab the ICC profile before convert() (convert may drop it).
            icc = pil.info.get("icc_profile")
            # Read orientation off the decoded Pillow image: piexif.load(path)
            # cannot parse HEIC/AVIF/JXL containers, so the path-based reader
            # would silently lose orientation for these formats.
            exif_bytes, orientation = _read_exif_orientation_from_pil(pil)
            # Pillow's high-bit modes (I;16, I) signal a >8-bit source even
            # though convert("RGB") below renders 8-bit (no high-bit decode here).
            if pil.mode in {"I", "I;16", "I;16B", "I;16L", "I;16N"}:
                source_bit_depth = 16
            image = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        except ImportError as e:
            raise UnsupportedInputError(
                f"Reading {suffix} requires an extra plugin: {e}"
            ) from e
        except Exception as e:  # noqa: BLE001
            raise UnsupportedInputError(f"Cannot read {path}: {e}") from e
    else:
        # IMREAD_UNCHANGED preserves 16-bit depth (and alpha); we normalize to
        # 3-channel BGR below. IMREAD_COLOR would silently downconvert to 8-bit.
        image = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if image is not None:
            if image.dtype == np.uint16:
                source_bit_depth = 16
            image = _normalize_to_bgr(image)
        # OpenCV ignores ICC; read it separately so P3 photos aren't shifted.
        icc = _read_icc(p)
        # piexif handles JPEG/TIFF; safe for the OpenCV-decoded formats.
        exif_bytes, orientation = _read_exif_and_orientation(p)

    if image is None:
        raise UnsupportedInputError(f"Cannot read image: {path}")

    if orientation in _ORIENTATION_OPS:
        image = _ORIENTATION_OPS[orientation](image)

    if strip_gps:
        exif_bytes = _strip_gps_from_exif(exif_bytes)

    h, w = image.shape[:2]
    return LoadedImage(
        image=image,
        exif=exif_bytes,
        original_orientation=orientation,
        width=w,
        height=h,
        icc=icc,
        source_bit_depth=source_bit_depth,
    )
