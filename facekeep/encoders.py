"""Modern image codec wrappers for faithful mode (AVIF / JPEG XL).

These encode a whole image with a modern codec. The codec's own adaptive
quantization allocates bits perceptually (more on faces/edges, fewer on flat
regions), so we get region-aware quality "for free" without manual region
manipulation, and the output is a single standard image file.
"""

import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from . import metrics
from .exceptions import EncodingError

logger = logging.getLogger("facekeep.encoders")

# Register codec plugins on import (best-effort; absence handled at call time)
try:
    import pillow_avif  # noqa: F401
    _AVIF_OK = True
except ImportError:
    _AVIF_OK = False

try:
    import pillow_jxl  # noqa: F401
    _JXL_OK = True
except ImportError:
    _JXL_OK = False


CODEC_EXTENSION = {"avif": ".avif", "jxl": ".jxl", "webp": ".webp"}

# WebP's bitstream caps each dimension at 16383 px (libwebp limit). A larger
# frame cannot be encoded; callers surface a clear EncodingError rather than a
# cryptic Pillow failure. (AVIF/JXL have no such practical cap.)
_WEBP_MAX_DIM = 16383


def codec_available(codec: str) -> bool:
    """Return True if the given codec's encoder is installed.

    ``webp`` is built into Pillow (no plugin), so it is always available — it is
    the maximum-compatibility fallback for old viewers that can't open AVIF/JXL.
    """
    return {"avif": _AVIF_OK, "jxl": _JXL_OK, "webp": True}.get(codec, False)


def _check_webp_dims(image_bgr: np.ndarray) -> None:
    """Raise EncodingError if the image exceeds WebP's per-dimension cap.

    libwebp caps each side at 16383 px; a larger frame fails to encode. Surface a
    clear message (with the offending size and the AVIF/JXL alternative) rather
    than letting Pillow raise a cryptic libwebp error deep in ``save``.
    """
    h, w = image_bgr.shape[:2]
    if w > _WEBP_MAX_DIM or h > _WEBP_MAX_DIM:
        raise EncodingError(
            f"Image {w}x{h} exceeds WebP's {_WEBP_MAX_DIM}px per-dimension limit; "
            "use the avif or jxl codec for images this large."
        )


def _to_uint8_for_encode(image_bgr: np.ndarray) -> np.ndarray:
    """Down-convert a high-bit image to 8-bit, loudly, for the 8-bit-only codecs.

    The bundled Pillow codec plugins have NO high-bit encode path: pillow-jxl
    raises on non-8-bit modes, and pillow-avif silently ``convert()``s anything
    that isn't RGB/RGBA down to 8-bit. To avoid that *silent* truncation (a
    [CORE-GOAL] fidelity leak — banding on smooth gradients), we down-convert
    explicitly here, by rounding (not truncating), and warn so the loss is
    visible. True 10/12-bit output is a follow-up that needs the avifenc CLI
    (ROADMAP Phase 1 follow-up / Phase 5 delta-Q share that path).
    """
    if image_bgr.dtype == np.uint8:
        return image_bgr
    if image_bgr.dtype == np.uint16:
        logger.warning(
            "Input is 16-bit but the bundled AVIF/JXL encoders have no high-bit "
            "path; rounding down to 8-bit (smooth gradients may band). True "
            "10/12-bit output is pending the avifenc CLI."
        )
        # Round to nearest 8-bit level rather than truncating low bits.
        return np.round(image_bgr.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    # Any other dtype (e.g. float): coerce conservatively to 8-bit.
    logger.warning(
        "Input dtype %s is not 8-bit; coercing to 8-bit for encoding.",
        image_bgr.dtype,
    )
    return np.clip(image_bgr, 0, 255).astype(np.uint8)


def _find_avifenc() -> Optional[str]:
    """Locate the external ``avifenc`` binary, or ``None`` if unavailable.

    True 10/12-bit AVIF *output* (ROADMAP Phase 1) needs the libavif ``avifenc``
    CLI: the bundled Pillow codecs are 8-bit-only, so high-bit sources otherwise
    round down to 8-bit (banding on smooth gradients). ``avifenc`` is an
    **opt-in, machine-local external binary** — it is never a Python dependency
    and the default 8-bit path never touches it. Resolution order:

    1. ``$FACEKEEP_AVIFENC`` (explicit override — an exact path to the binary),
    2. ``avifenc`` on the system ``PATH`` (``shutil.which``),
    3. ``None`` → callers fall back to the warned 8-bit round-down.

    Offline-first / graceful degradation: a missing binary is *not* an error —
    high-bit output is a best-effort upgrade, never required.
    """
    env = os.environ.get("FACEKEEP_AVIFENC")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        logger.warning(
            "FACEKEEP_AVIFENC=%s is not a file; falling back to PATH/8-bit.", env
        )
    return shutil.which("avifenc")


def avifenc_available() -> bool:
    """Return True if the external ``avifenc`` binary can be located."""
    return _find_avifenc() is not None


def _find_avifdec() -> Optional[str]:
    """Locate the external ``avifdec`` binary, or ``None`` if unavailable.

    Reading a true high-bit (10/12-bit) AVIF back at full depth needs the libavif
    ``avifdec`` CLI: the bundled Pillow plugin decodes AVIF down to 8-bit, which
    would flatten a high-bit crop on aggressive restore. ``avifdec`` ships beside
    ``avifenc`` in the libavif release, so resolution order is:

    1. ``$FACEKEEP_AVIFDEC`` (explicit override — an exact path to the binary),
    2. ``avifdec`` as a sibling of the located ``avifenc`` (so a single
       ``$FACEKEEP_AVIFENC`` enables *both* the high-bit encode and decode paths),
    3. ``avifdec`` on the system ``PATH`` (``shutil.which``),
    4. ``None`` → callers fall back to the 8-bit Pillow decode.

    Offline-first / graceful degradation: a missing binary is not an error.
    """
    env = os.environ.get("FACEKEEP_AVIFDEC")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        logger.warning(
            "FACEKEEP_AVIFDEC=%s is not a file; falling back to sibling/PATH.", env
        )
    enc = _find_avifenc()
    if enc:
        sibling = Path(enc).with_name(
            "avifdec.exe" if enc.lower().endswith(".exe") else "avifdec"
        )
        if sibling.is_file():
            return str(sibling)
    return shutil.which("avifdec")


def avifdec_available() -> bool:
    """Return True if the external ``avifdec`` binary can be located."""
    return _find_avifdec() is not None


def encode_highbit_avif(
    image_bgr: np.ndarray,
    *,
    bit_depth: int = 10,
    quality: int = 70,
    chroma: str = "auto",
    has_faces: bool = False,
    exif: Optional[bytes] = None,
    icc: Optional[bytes] = None,
) -> bytes:
    """Encode a 16-bit BGR image to true 10/12-bit AVIF via the ``avifenc`` CLI.

    This is the high-bit-output path the bundled Pillow codec cannot do. The
    source ``uint16`` pixels are written to a temporary 16-bit PNG (OpenCV writes
    real 16-bit PNG), then ``avifenc -d {bit_depth}`` encodes it; EXIF/ICC are
    passed through ``avifenc``'s own ``--exif``/``--icc`` file flags so wide-gamut
    color and orientation survive. The encoded bytes are read back and returned.

    Args:
        image_bgr: A ``uint16`` BGR image (high-bit source). Lower dtypes are
            accepted and promoted, but the point is to preserve >8-bit detail.
        bit_depth: Output bit depth — 10 or 12.
        quality: 0-100 (same scale as :func:`encode`; higher = better/larger).
        chroma: '444', '420', or 'auto' (444 when ``has_faces`` else 420).
        has_faces: Hint for 'auto' chroma selection.
        exif: Optional EXIF bytes to embed.
        icc: Optional ICC profile to embed.

    Returns:
        Encoded 10/12-bit AVIF bytes.

    Raises:
        EncodingError: if ``avifenc`` is unavailable, or the subprocess fails.
            Callers (faithful mode) catch this and fall back to the 8-bit path,
            so high-bit output stays a best-effort upgrade.
    """
    if bit_depth not in (10, 12):
        raise EncodingError(f"Unsupported high-bit depth {bit_depth} (expected 10 or 12).")

    binary = _find_avifenc()
    if binary is None:
        raise EncodingError(
            "avifenc not found (set FACEKEEP_AVIFENC or put avifenc on PATH); "
            "cannot write true high-bit AVIF."
        )

    # Promote to uint16 if needed (e.g. a uint8 array routed here): scale up so
    # the PNG carries 16-bit samples avifenc reads at full depth.
    if image_bgr.dtype == np.uint8:
        png_img = (image_bgr.astype(np.uint16) * 257)
    elif image_bgr.dtype == np.uint16:
        png_img = image_bgr
    else:
        png_img = np.clip(image_bgr, 0, 65535).astype(np.uint16)

    # cv2.imwrite already converts a BGR array to a correct RGB PNG (which avifenc
    # then reads as RGB), so write the BGR ``png_img`` straight through. An
    # explicit BGR->RGB cvtColor here would *double*-swap — the PNG, and thus the
    # AVIF, would come back with R and B exchanged (pinned by a known-color
    # round-trip in tests/test_bit_depth.py).
    subsampling = "444" if (chroma == "444" or (chroma == "auto" and has_faces)) else "420"

    tmpdir = tempfile.mkdtemp(prefix="facekeep_avif_")
    try:
        in_png = os.path.join(tmpdir, "in.png")
        out_avif = os.path.join(tmpdir, "out.avif")
        if not cv2.imwrite(in_png, png_img):
            raise EncodingError("Failed to write temporary 16-bit PNG for avifenc.")

        cmd = [
            binary,
            "-d", str(bit_depth),
            "-y", subsampling,
            "-q", str(quality),
            "--ignore-exif", "--ignore-xmp", "--ignore-icc",  # control metadata ourselves
        ]
        if exif:
            exif_path = os.path.join(tmpdir, "meta.exif")
            Path(exif_path).write_bytes(exif)
            cmd += ["--exif", exif_path]
        if icc:
            icc_path = os.path.join(tmpdir, "meta.icc")
            Path(icc_path).write_bytes(icc)
            cmd += ["--icc", icc_path]
        cmd += [in_png, out_avif]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as e:
            raise EncodingError(f"avifenc invocation failed: {e}") from e
        if proc.returncode != 0:
            raise EncodingError(
                f"avifenc exited {proc.returncode}: {proc.stderr.strip()[:300]}"
            )
        if not os.path.isfile(out_avif):
            raise EncodingError("avifenc produced no output file.")
        return Path(out_avif).read_bytes()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def decode_highbit_avif(data: bytes) -> np.ndarray:
    """Decode AVIF bytes to a **uint16** BGR image via the ``avifdec`` CLI.

    The bundled Pillow AVIF plugin decodes everything down to 8-bit RGB, which
    would silently flatten a 10/12-bit (HDR) AVIF crop on aggressive restore.
    ``avifdec -d 16`` writes a true 16-bit PNG, which OpenCV reads back as uint16
    BGR (``cv2.imread`` returns BGR for a standard RGB PNG — the exact inverse of
    :func:`encode_highbit_avif`, which writes its BGR source straight to a PNG).

    Returns:
        A uint16 BGR array. (An 8-bit AVIF routed here is promoted to 16-bit by
        ``avifdec -d 16``; callers only use this for crops the manifest declares
        high-bit.)

    Raises:
        EncodingError: if ``avifdec`` is unavailable or the subprocess fails.
            Callers that want graceful degradation (read a high-bit crop on a box
            without ``avifdec``) catch this and fall back to the 8-bit Pillow
            :func:`decode`.
    """
    binary = _find_avifdec()
    if binary is None:
        raise EncodingError(
            "avifdec not found (set FACEKEEP_AVIFDEC or FACEKEEP_AVIFENC, or put "
            "avifdec on PATH); cannot decode high-bit AVIF at full depth."
        )
    tmpdir = tempfile.mkdtemp(prefix="facekeep_avifdec_")
    try:
        in_avif = os.path.join(tmpdir, "in.avif")
        out_png = os.path.join(tmpdir, "out.png")
        Path(in_avif).write_bytes(data)
        cmd = [binary, "-d", "16", in_avif, out_png]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as e:
            raise EncodingError(f"avifdec invocation failed: {e}") from e
        if proc.returncode != 0:
            raise EncodingError(
                f"avifdec exited {proc.returncode}: {proc.stderr.strip()[:300]}"
            )
        img = cv2.imread(out_png, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise EncodingError("avifdec produced no decodable PNG.")
        return img
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _find_avifgainmaputil() -> Optional[str]:
    """Locate the external ``avifgainmaputil`` binary, or ``None`` if unavailable.

    Re-attaching an iPhone HDR gain map on aggressive restore (Phase 9) needs
    libavif's ``avifgainmaputil`` CLI — its ``combine`` subcommand is the only
    tool here that can write an AVIF with an *embedded gain map* (no bundled
    codec can). It ships beside ``avifenc`` in the libavif release, so the
    resolution order mirrors ``_find_avifdec``:

    1. ``$FACEKEEP_AVIFGAINMAPUTIL`` (explicit override — exact path),
    2. ``avifgainmaputil`` as a sibling of the located ``avifenc`` (a single
       ``$FACEKEEP_AVIFENC`` enables the whole avif tool family),
    3. ``avifgainmaputil`` on the system ``PATH``,
    4. ``None`` → callers fall back to the SDR write with a warning.

    Offline-first / graceful degradation: a missing binary is not an error —
    HDR re-attach is a best-effort upgrade, never required.
    """
    env = os.environ.get("FACEKEEP_AVIFGAINMAPUTIL")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        logger.warning(
            "FACEKEEP_AVIFGAINMAPUTIL=%s is not a file; falling back to "
            "sibling/PATH.", env,
        )
    enc = _find_avifenc()
    if enc:
        sibling = Path(enc).with_name(
            "avifgainmaputil.exe" if enc.lower().endswith(".exe")
            else "avifgainmaputil"
        )
        if sibling.is_file():
            return str(sibling)
    return shutil.which("avifgainmaputil")


def avifgainmaputil_available() -> bool:
    """Return True if the external ``avifgainmaputil`` binary can be located."""
    return _find_avifgainmaputil() is not None


# SDR diffuse white in nits for the PQ-encoded HDR alternate (BT.2408's 203).
_GAINMAP_SDR_WHITE_NITS = 203.0


def _srgb_eotf(v: np.ndarray) -> np.ndarray:
    """sRGB electro-optical transfer: encoded [0,1] -> linear light [0,1].

    Display P3 uses the same transfer curve as sRGB, so this covers both the
    sRGB and the (iPhone-default) Display P3 base images.
    """
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _pq_oetf(nits: np.ndarray) -> np.ndarray:
    """SMPTE ST 2084 (PQ) inverse EOTF: linear nits -> PQ-encoded [0,1]."""
    L = np.clip(nits / 10000.0, 0.0, 1.0)
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    lm1 = L**m1
    return ((c1 + c2 * lm1) / (1 + c3 * lm1)) ** m2


def encode_gainmap_avif(
    image_bgr: np.ndarray,
    gain_map: np.ndarray,
    *,
    headroom: float = 3.0,
    quality: int = 70,
    gain_map_quality: int = 75,
    exif: Optional[bytes] = None,
    icc: Optional[bytes] = None,
) -> bytes:
    """Encode a BGR image + HDR gain map to a gain-map AVIF via ``avifgainmaputil``.

    Produces a **backward-compatible HDR AVIF**: SDR viewers show the base
    image; HDR viewers multiply it by the gain map (scaled to the display's
    headroom) to extend highlights — the same mechanism as an iPhone HDR photo.
    ``combine`` wants the base plus the *fully-applied* HDR alternate, so this
    rebuilds the alternate the way the OS would: linearize the base (sRGB/P3
    transfer), boost by ``2^(headroom * gain)`` per pixel (validated against a
    libavif-converted reference of a real iPhone photo — the raw Apple gain map
    is carried value-for-value at gamma 1), then PQ-encode at an SDR white of
    203 nits into a 16-bit PNG.

    Metadata honesty (both verified by probing the tool):

    * **EXIF rides through** — it is embedded in the temp base PNG (eXIf chunk)
      and ``combine`` copies it into the output.
    * **ICC cannot** — ``combine`` refuses inputs with ICC profiles ("not
      supported"), so color is declared via CICP instead: a Display P3 profile
      (sniffed by name) maps to primaries 12, anything else to sRGB primaries 1
      (transfer 13 base / 16 (PQ) alternate). For P3/sRGB sources — every
      iPhone — CICP is color-equivalent to the profile; an exotic profile would
      be approximated, which is the documented limit of this path.

    Args:
        image_bgr: The restored base image, uint8 BGR (a uint16 input is
            rounded down — the gain-map HDR mechanism is an 8-bit base by
            construction).
        gain_map: The stored gain map (2-D uint8, any resolution; a 3-channel
            BGR map is applied per channel). Resized to the base size.
        headroom: HDR headroom in stops (``aggressive.gain_map_headroom``).
        quality: 0-100 quality for the base/alternate color.
        gain_map_quality: 0-100 quality for the embedded gain map.
        exif: Optional EXIF bytes to carry into the output.
        icc: Optional source ICC bytes — used only to *choose* the CICP
            primaries (see above), never embedded.

    Returns:
        Encoded gain-map AVIF bytes.

    Raises:
        EncodingError: if ``avifgainmaputil`` is unavailable or the subprocess
            fails. Callers (aggressive restore) catch this and fall back to the
            plain SDR AVIF write, warned.
    """
    binary = _find_avifgainmaputil()
    if binary is None:
        raise EncodingError(
            "avifgainmaputil not found (set FACEKEEP_AVIFENC or put the libavif "
            "tools on PATH); cannot write a gain-map (HDR) AVIF."
        )

    if image_bgr.dtype == np.uint16:
        image_bgr = np.round(image_bgr.astype(np.float32) / 257.0).astype(np.uint8)
    elif image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    h, w = image_bgr.shape[:2]
    gm = cv2.resize(gain_map, (w, h), interpolation=cv2.INTER_CUBIC)
    gain = gm.astype(np.float32) / 255.0
    if gain.ndim == 2:
        gain = gain[..., None]

    # Rebuild the fully-applied HDR alternate: linear boost, PQ-encoded 16-bit.
    base_lin = _srgb_eotf(image_bgr.astype(np.float32) / 255.0)
    alt_lin = base_lin * (2.0 ** (float(headroom) * gain))
    alt_pq = _pq_oetf(alt_lin * _GAINMAP_SDR_WHITE_NITS)
    alt16 = np.round(alt_pq * 65535.0).astype(np.uint16)

    # CICP by profile: Display P3 (the iPhone default) -> primaries 12, else
    # sRGB primaries 1. Transfer: 13 (sRGB) base, 16 (PQ) alternate; matrix 6.
    primaries = 12 if (icc and b"Display P3" in icc) else 1
    cicp_base = f"{primaries}/13/6"
    cicp_alt = f"{primaries}/16/6"

    tmpdir = tempfile.mkdtemp(prefix="facekeep_gainmap_")
    try:
        base_png = os.path.join(tmpdir, "base.png")
        alt_png = os.path.join(tmpdir, "alt.png")
        out_avif = os.path.join(tmpdir, "out.avif")

        # Base via PIL so the EXIF bytes ride along as a PNG eXIf chunk (cv2
        # can't write one); BGR->RGB at the PIL boundary per the repo rule. No
        # ICC on purpose — combine rejects profiled inputs (see docstring).
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        save_kwargs = {"exif": exif} if exif else {}
        pil.save(base_png, **save_kwargs)
        pil.close()
        # Alternate via cv2: writes the BGR uint16 array to a correct RGB PNG
        # (straight through — a cvtColor here would double-swap R/B, the same
        # rule as encode_highbit_avif).
        if not cv2.imwrite(alt_png, alt16):
            raise EncodingError("Failed to write the temporary HDR alternate PNG.")

        cmd = [
            binary, "combine", base_png, alt_png, out_avif,
            "--cicp-base", cicp_base,
            "--cicp-alternate", cicp_alt,
            "-q", str(quality),
            "--qgain-map", str(gain_map_quality),
            # Store the gain map at half resolution — Apple's own native scale.
            "--downscaling", "2",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as e:
            raise EncodingError(f"avifgainmaputil invocation failed: {e}") from e
        if proc.returncode != 0:
            raise EncodingError(
                f"avifgainmaputil exited {proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()[:300]}"
            )
        if not os.path.isfile(out_avif):
            raise EncodingError("avifgainmaputil produced no output file.")
        return Path(out_avif).read_bytes()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def encode_lossless_avif(
    image_bgr: np.ndarray,
    *,
    exif: Optional[bytes] = None,
    icc: Optional[bytes] = None,
) -> bytes:
    """Encode a BGR image to **mathematically lossless** AVIF via ``avifenc -l``.

    The bundled pillow-avif has no truly-lossless path (even q100 4:4:4 is not
    bit-exact), so real lossless AVIF needs the external ``avifenc`` CLI — the
    same opt-in, machine-local binary as the high-bit path. ``avifenc -l`` forces
    lossless (identity color transform + 4:4:4 + lossless quantizer). 8-bit and
    16-bit sources are both written through a real PNG so avifenc reads exact
    samples; EXIF/ICC are passed through avifenc's own file flags.

    Raises:
        EncodingError: if ``avifenc`` is unavailable or the subprocess fails.
            The faithful caller catches this and falls back to lossless JXL, so
            lossless output stays guaranteed even without the binary.
    """
    binary = _find_avifenc()
    if binary is None:
        raise EncodingError(
            "avifenc not found (set FACEKEEP_AVIFENC or put avifenc on PATH); "
            "cannot write lossless AVIF."
        )

    # PNG carries exact samples at the source depth (8- or 16-bit). cv2.imwrite
    # converts the BGR array to a correct RGB PNG (which avifenc reads as RGB), so
    # write ``png_img`` straight through — an explicit cvtColor here would double-
    # swap R/B (see encode_highbit_avif).
    if image_bgr.dtype not in (np.uint8, np.uint16):
        png_img = np.clip(image_bgr, 0, 255).astype(np.uint8)
    else:
        png_img = image_bgr

    tmpdir = tempfile.mkdtemp(prefix="facekeep_avif_ll_")
    try:
        in_png = os.path.join(tmpdir, "in.png")
        out_avif = os.path.join(tmpdir, "out.avif")
        if not cv2.imwrite(in_png, png_img):
            raise EncodingError("Failed to write temporary PNG for avifenc.")

        # -l forces lossless (identity transform, 4:4:4, lossless quantizer).
        cmd = [
            binary,
            "-l",
            "--ignore-exif", "--ignore-xmp", "--ignore-icc",  # control metadata ourselves
        ]
        if exif:
            exif_path = os.path.join(tmpdir, "meta.exif")
            Path(exif_path).write_bytes(exif)
            cmd += ["--exif", exif_path]
        if icc:
            icc_path = os.path.join(tmpdir, "meta.icc")
            Path(icc_path).write_bytes(icc)
            cmd += ["--icc", icc_path]
        cmd += [in_png, out_avif]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as e:
            raise EncodingError(f"avifenc invocation failed: {e}") from e
        if proc.returncode != 0:
            raise EncodingError(
                f"avifenc exited {proc.returncode}: {proc.stderr.strip()[:300]}"
            )
        if not os.path.isfile(out_avif):
            raise EncodingError("avifenc produced no output file.")
        return Path(out_avif).read_bytes()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(_to_uint8_for_encode(image_bgr), cv2.COLOR_BGR2RGB))


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def encode(
    image_bgr: np.ndarray,
    codec: str = "avif",
    quality: int = 70,
    speed: int = 6,
    chroma: str = "auto",
    has_faces: bool = False,
    exif: Optional[bytes] = None,
    icc: Optional[bytes] = None,
    bit_depth: int = 8,
    output_bit_depth: int = 10,
    lossless: bool = False,
) -> bytes:
    """Encode a BGR image to modern-codec bytes.

    Args:
        image_bgr: Image in BGR (OpenCV) order
        codec: 'avif' or 'jxl'
        quality: 0-100 (higher = better quality, larger file)
        speed: Encoder effort 0-10 (avif only; lower = slower/smaller)
        chroma: '444', '420', or 'auto' (444 when has_faces else 420)
        has_faces: Hint for 'auto' chroma selection
        exif: Optional EXIF bytes to embed
        icc: Optional ICC color profile to embed (preserves wide-gamut color)
        lossless: Encode mathematically lossless (bit-exact), ignoring ``quality``
            / ``chroma`` / auto-tune. JXL does this natively (``lossless=True``).
            AVIF has no lossless path in the bundled Pillow plugin, so it routes
            through the ``avifenc -l`` CLI and raises ``EncodingError`` if that
            binary is absent — the faithful caller catches it and falls back to
            lossless JXL, so lossless output is always genuine.
        bit_depth: Source per-channel bit depth (8 or 16). When >8 *and* the
            source is genuinely uint16 *and* codec is avif *and* the ``avifenc``
            CLI is available, route through the true high-bit AVIF path
            (:func:`encode_highbit_avif`) instead of the 8-bit Pillow codec,
            preserving >8-bit detail (no banding). Any of those not met → the
            existing 8-bit path (warned round-down) — so the default stays
            byte-for-byte unchanged and offline-first.
        output_bit_depth: Output depth (10 or 12) for the high-bit AVIF path.
            Only consulted when that path is taken (see ``bit_depth``); 10 is the
            widely-supported default that clears banding, 12 keeps maximum
            precision for a 16-bit source. Ignored by the 8-bit path.

    Returns:
        Encoded image bytes.
    """
    if not codec_available(codec):
        raise EncodingError(
            f"Codec {codec!r} is not available. Install the plugin "
            f"(pip install pillow-{'avif-plugin' if codec == 'avif' else 'jxl-plugin'})."
        )

    # Mathematically-lossless output (ROADMAP backlog: archival of irreplaceable
    # originals). Bypasses quality/chroma entirely. JXL is bit-exact natively;
    # AVIF needs the avifenc CLI (the bundled plugin has no lossless path) and
    # raises EncodingError if it's absent — the faithful caller catches that and
    # falls back to lossless JXL, so lossless output is always genuine.
    if lossless:
        if codec == "avif":
            return encode_lossless_avif(image_bgr, exif=exif, icc=icc)
        # JXL and WebP both encode lossless natively via Pillow ``lossless=True``
        # (WebP lossless is bit-exact 8-bit; a >8-bit source rounds down with the
        # standard loud warning, since WebP has no high-bit path — the honest
        # 8-bit limit shared with the lossy WebP path and the bundled JXL).
        pil_format = "WEBP" if codec == "webp" else "JXL"
        if codec == "webp":
            _check_webp_dims(image_bgr)
        pil = _bgr_to_pil(image_bgr)
        buf = io.BytesIO()
        save_kwargs = {"lossless": True}
        if exif:
            save_kwargs["exif"] = exif
        if icc:
            save_kwargs["icc_profile"] = icc
        try:
            pil.save(buf, pil_format, **save_kwargs)
        except Exception as e:  # noqa: BLE001
            raise EncodingError(f"Failed to encode lossless {codec}: {e}") from e
        finally:
            pil.close()
            del pil
        return buf.getvalue()

    # True high-bit AVIF output (ROADMAP Phase 1). Only when every precondition
    # holds — high-bit source, genuinely uint16, AVIF, and the avifenc binary
    # present. JXL has no high-bit path here, so it always uses the 8-bit route.
    # On any avifenc failure we fall back to the 8-bit Pillow path (graceful
    # degradation), so high-bit output is a best-effort upgrade, never required.
    if (
        bit_depth > 8
        and image_bgr.dtype == np.uint16
        and codec == "avif"
        and avifenc_available()
    ):
        try:
            return encode_highbit_avif(
                image_bgr,
                bit_depth=output_bit_depth,  # 10 (default) or 12, per config
                quality=quality,
                chroma=chroma,
                has_faces=has_faces,
                exif=exif,
                icc=icc,
            )
        except EncodingError as e:
            logger.warning(
                "High-bit AVIF encode failed (%s); falling back to 8-bit.", e
            )

    pil = _bgr_to_pil(image_bgr)
    buf = io.BytesIO()
    save_kwargs: dict = {"quality": quality}
    if exif:
        save_kwargs["exif"] = exif
    if icc:
        save_kwargs["icc_profile"] = icc

    subsampling = "4:4:4" if (chroma == "444" or (chroma == "auto" and has_faces)) else "4:2:0"

    try:
        if codec == "avif":
            save_kwargs["speed"] = speed
            # pillow-avif accepts subsampling as e.g. "4:4:4"
            save_kwargs["subsampling"] = subsampling
            pil.save(buf, "AVIF", **save_kwargs)
        elif codec == "webp":
            # WebP is the maximum-compatibility fallback (built into Pillow, opens
            # in any browser/old viewer). It is 8-bit-only with no chroma/effort
            # knobs we expose — Pillow uses 4:2:0 below q100, 4:4:4 at q100 — so
            # ``speed``/``chroma``/``has_faces`` don't apply; only ``quality``,
            # plus the dimension cap checked above.
            _check_webp_dims(image_bgr)
            pil.save(buf, "WEBP", **save_kwargs)
        else:  # jxl
            pil.save(buf, "JXL", **save_kwargs)
    except Exception as e:  # noqa: BLE001
        raise EncodingError(f"Failed to encode {codec}: {e}") from e
    finally:
        # Free the intermediate full-resolution copy promptly (ROADMAP Phase 3
        # bounded-memory). ``pil`` wraps a full-frame RGB array created by
        # ``_bgr_to_pil``; on a large photo that is a whole extra raw-pixel
        # buffer. Dropping it here (rather than letting it live until return)
        # keeps the peak from carrying the source BGR *and* this RGB copy *and*
        # the codec's working set all at once.
        pil.close()
        del pil

    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    """Decode modern-codec bytes back to a BGR image."""
    pil = None
    try:
        pil = Image.open(io.BytesIO(data))
        return _pil_to_bgr(pil)
    except Exception as e:  # noqa: BLE001
        raise EncodingError(f"Failed to decode image: {e}") from e
    finally:
        # Release the decoded PIL image (and its full-frame buffer) as soon as
        # we've converted to BGR (ROADMAP Phase 3 bounded-memory). ``_pil_to_bgr``
        # has already produced the independent BGR array we return, so on a large
        # photo this stops the decoded RGB copy from lingering past the call —
        # which matters for verify_roundtrip, run while the source BGR + encoded
        # bytes are still live.
        if pil is not None:
            pil.close()
            del pil


def verify_roundtrip(
    data: bytes,
    original_bgr: np.ndarray,
    *,
    thorough: bool = False,
    ssim_floor: float = 0.90,
) -> Optional[float]:
    """Confirm encoded output decodes and matches the source — fail loudly if not.

    Faithful mode otherwise writes the file without ever checking that it can be
    read back, so a corrupt/garbage encode would pass silently. This is the
    safety net (ROADMAP Phase 1):

    - Always (quick check): the bytes must decode, and the decoded image must
      have the same height/width as the source.
    - ``thorough=True`` (the ``--verify`` path): additionally compute a cheap
      downscaled SSIM against the source and require it to stay above
      ``ssim_floor``. The floor is deliberately low — it detects an encoder that
      blew up, not lossy-compression quality, so it must not false-fail on
      legitimately low-quality encodes.

    Returns:
        The downscaled SSIM score when ``thorough=True`` (so callers — e.g. the
        ``--report`` ledger — can surface a number that was *actually measured*),
        else ``None``. The quick path measures no quality score, so it returns
        ``None``; do not synthesize a fidelity number that wasn't computed.

    Raises:
        EncodingError: if the output cannot be decoded, its dimensions differ
            from the source, or (thorough) the downscaled SSIM is below the floor.
    """
    decoded = decode(data)  # raises EncodingError on undecodable bytes

    decoded_shape = decoded.shape[:2]
    if decoded_shape != original_bgr.shape[:2]:
        raise EncodingError(
            "Round-trip verification failed: decoded dimensions "
            f"{decoded_shape[1]}x{decoded_shape[0]} != source "
            f"{original_bgr.shape[1]}x{original_bgr.shape[0]}."
        )

    if not thorough:
        # Quick path needs only the dimensions; release the full-resolution
        # decoded copy now rather than holding it until return (ROADMAP Phase 3
        # bounded-memory — verify runs while the source BGR + encoded bytes are
        # still live, so this is one of the spots most prone to stacking copies).
        del decoded
        return None

    score = metrics.downscaled_ssim(original_bgr, decoded)
    # Done with the decoded full-resolution copy; downscaled_ssim already made
    # its own small resized copies for the comparison.
    del decoded
    if score < ssim_floor:
        raise EncodingError(
            "Round-trip verification failed: decoded output is structurally "
            f"unlike the source (downscaled SSIM {score:.3f} < {ssim_floor:.2f}); "
            "the encode is likely corrupt."
        )
    return score


_KNOWN_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
    ".heic", ".heif", ".avif", ".jxl", ".fkeep",
}


def _with_extension(path: Path, ext: str) -> Path:
    """Return `path` ending with `ext`, without mangling dotted filenames.

    Strips a *known* image/container extension if present, then appends `ext`.
    'photo' -> 'photo.avif'; 'photo.jpg' -> 'photo.avif';
    '2024.05.20_trip' -> '2024.05.20_trip.avif' (dot kept, not a known ext).
    """
    if path.suffix.lower() == ext:
        return path
    if path.suffix.lower() in _KNOWN_IMAGE_EXTS:
        return path.with_suffix(ext)
    return path.parent / (path.name + ext)


def write_encoded(data: bytes, output_path: str, codec: str) -> Path:
    """Write encoded bytes to a file, ensuring the correct extension."""
    ext = CODEC_EXTENSION.get(codec, Path(output_path).suffix)
    path = _with_extension(Path(output_path), ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def copy_original(src: str, output_path: str) -> Path:
    """Copy the source file to the output location, keeping its real extension.

    Used by faithful mode's skip-if-larger guard: when the encoded output would
    be no smaller than the input (already-optimized files), we keep the original
    bytes rather than write a larger file. The output keeps the *source* suffix
    (it is the original, not an AVIF/JXL), routed through `_with_extension` so
    dotted filenames aren't mangled. If the resolved destination is the source
    itself, nothing is copied (copying a file onto itself would truncate it).
    """
    src_path = Path(src)
    dest = _with_extension(Path(output_path), src_path.suffix.lower())

    # Don't copy a file onto itself (would zero it out). samefile needs both to
    # exist; fall back to a normalized-path compare when dest doesn't exist yet.
    same = False
    if dest.exists():
        try:
            same = os.path.samefile(src_path, dest)
        except OSError:
            same = False
    else:
        same = os.path.normcase(os.path.abspath(src_path)) == os.path.normcase(
            os.path.abspath(dest)
        )
    if same:
        return src_path

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dest)
    return dest
