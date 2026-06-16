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
    # A 16-bit source — a 16-bit PNG/TIFF (IMREAD_UNCHANGED) or a 10/12-bit HDR
    # HEIC decoded high-bit via pillow_heif.open_heif (see _decode_heif) — carries
    # uint16 pixels with source_bit_depth=16, which encoders.encode routes to the
    # true 10/12-bit avifenc AVIF output path. Without the avifenc binary (or for
    # JXL/WebP) it rounds down to 8-bit at the encode boundary with a loud warning
    # (never a silent truncation).


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


def _decode_heif(path: Path) -> tuple[np.ndarray, int, Optional[bytes], Optional[bytes]]:
    """Decode a HEIC/HEIF fully via ``pillow_heif.open_heif`` — never PIL ``Image.open``.

    Returns ``(bgr, source_bit_depth, exif_bytes, icc)``.

    **Why open_heif and never Image.open for HEIC.** Opening the *same* HEIC file
    with both the PIL HEIF plugin (``Image.open``) and ``open_heif`` in one
    process segfaults inside libheif — two libheif contexts on one file collide,
    and a ``close()`` does not help (verified). ``open_heif`` is stable on its
    own and is the only API here that yields the genuine >8-bit samples, so HEIC
    reads pixels *and* EXIF *and* ICC from ``open_heif`` exclusively (this is why
    HEIC no longer goes through the shared ``Image.open`` plugin path that AVIF /
    JXL still use — those have no high-bit decode here and no dual-API hazard).

    **High bit depth (the point of this path).** ``convert_hdr_to_8bit=False``
    decodes a 10/12-bit HDR source to a ``uint16`` array (pillow-heif mode
    ``"RGB;16"``, the 10-bit values scaled into the 16-bit range), so
    ``source_bit_depth=16`` carries real high-bit data into the ``avifenc``
    10/12-bit AVIF output path (``encoders.encode`` routes uint16+avif there). An
    8-bit HEIC decodes to ``uint8`` (``"RGB"``) — the same result the old
    ``Image.open`` path produced. Without ``avifenc`` the uint16 array simply
    rounds down to 8-bit at the encode boundary with the standard warning, the
    same honest fallback as a 16-bit PNG (offline-first / graceful degradation).

    **Orientation.** ``open_heif`` applies the EXIF/irot orientation to the
    pixels itself (they come out upright) but does *not* clear the EXIF
    orientation tag. So the caller must apply NO further rotation, and we
    normalize the carried EXIF's tag to 1 here (matching what the PIL plugin
    exposes on the 8-bit HEIC path) so a re-embed never double-rotates.
    """
    import pillow_heif

    # register_heif_opener() only patches PIL's Image.open; open_heif is the
    # low-level reader and does not require it, but calling it is harmless and
    # keeps the plugin initialized. It opens nothing, so it adds no second
    # libheif context (no dual-API crash).
    pillow_heif.register_heif_opener()
    heif_file = pillow_heif.open_heif(str(path), convert_hdr_to_8bit=False)
    himg = heif_file[0]
    arr = np.asarray(himg)  # uint8 (8-bit) or uint16 (HDR); RGB / RGBA / grayscale
    if arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    source_bit_depth = 16 if arr.dtype == np.uint16 else 8
    icc = himg.info.get("icc_profile")
    # open_heif leaves the orientation tag set even though it rotated the pixels,
    # so normalize the carried tag to 1 (the raw value is discarded — the pixels
    # are already upright, so the caller applies orientation 1 = a no-op).
    exif_bytes, _orig_orientation = _normalize_orientation_in_exif(himg.info.get("exif"))
    return bgr, source_bit_depth, exif_bytes, icc


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
    if suffix in {".heic", ".heif"}:
        # HEIC/HEIF: decode via pillow_heif.open_heif ONLY (never PIL Image.open)
        # so a 10/12-bit HDR source keeps its high-bit pixels — see _decode_heif
        # for the full rationale (and why mixing the two APIs on one HEIC crashes).
        try:
            image, source_bit_depth, exif_bytes, icc = _decode_heif(p)
            # open_heif already applied the orientation (pixels upright) and
            # _decode_heif normalized the carried tag to 1, so the uniform
            # orientation step below must NOT rotate again.
            orientation = 1
        except ImportError as e:
            raise UnsupportedInputError(
                f"Reading {suffix} requires an extra plugin: {e}"
            ) from e
        except Exception as e:  # noqa: BLE001
            raise UnsupportedInputError(f"Cannot read {path}: {e}") from e
    elif suffix in {".avif", ".jxl"}:
        # AVIF/JXL: decode via Pillow (needs plugin). The bundled plugins render
        # 8-bit (no high-bit decode here), and there is no dual-API hazard since
        # we never call open_heif on these — so this keeps the original path.
        try:
            from PIL import Image

            if suffix == ".avif":
                import pillow_avif  # noqa: F401
            elif suffix == ".jxl":
                import pillow_jxl  # noqa: F401
            pil = Image.open(str(p))
            # Grab the ICC profile before convert() (convert may drop it).
            icc = pil.info.get("icc_profile")
            # Read orientation off the decoded Pillow image: piexif.load(path)
            # cannot parse AVIF/JXL containers, so the path-based reader would
            # silently lose orientation for these formats.
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
