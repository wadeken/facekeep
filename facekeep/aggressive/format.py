""".fkeep container format for aggressive mode.

A .fkeep file is a ZIP archive (so anyone can inspect it with `unzip`):
  manifest.json     - metadata: original size, faces, settings, EXIF/ICC flags
  exif.bin          - original EXIF bytes (if any)
  icc.bin           - original ICC color profile bytes (if any, e.g. Display P3)
  background.(jpg|avif|jxl) - downsampled background, encoded with
                      AggressiveConfig.bg_codec (avif/jxl stored 4:2:0; jpg is
                      the default). Readers locate it by extension in that order.
  face_NNN.(png|avif|jxl|jpg) - face crops. PNG when face_quality >= 100
                      (lossless); otherwise the crop codec from
                      AggressiveConfig.face_codec (avif/jxl stored 4:4:4, else
                      JPEG). Readers locate crops by extension in that order.
  face_mask_NNN.png - 8-bit alpha masks for blending
  thumbnail.jpg     - small preview (always JPEG, compatibility)
  residual.(jxl|jpg) - optional (manifest 1.6.0+, aggressive.residual): the real
                      high-frequency delta the downsample lost, offset-encoded
                      as uint8 (value/2 + 128) at residual_scale resolution.
                      Restore adds it back to a bicubic upscale, making the
                      background real (lossy) data instead of a hallucination.
  gainmap.jpg       - optional (manifest 1.10.0+, aggressive.preserve_gain_map):
                      the source's iPhone HDR gain map (single-channel JPEG,
                      typically half resolution, upright). Restore re-attaches
                      it into an HDR AVIF (`restore -f avif`) via
                      avifgainmaputil; manifest flag: gain_map_preserved.
"""

import hashlib
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .. import __version__
from ..exceptions import EncodingError, FormatError
from .compressor import CompressedPhoto, _to_uint8

logger = logging.getLogger("facekeep.aggressive.format")

FKEEP_EXTENSION = ".fkeep"


def _jpg(image: np.ndarray, q: int = 85) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def _png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


def _mask_png(mask: np.ndarray) -> bytes:
    return cv2.imencode(".png", (mask * 255).astype(np.uint8))[1].tobytes()


def _decode(data: bytes, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def _decode_mask(data: bytes) -> np.ndarray:
    m = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return m.astype(np.float32) / 255.0


# Face-crop extensions, in reader search order. PNG is the lossless escape hatch
# (face_quality >= 100); avif/jxl are the compact 4:4:4 codecs; jpg is the
# universal default. A crop is stored as exactly ONE of these; the reader picks
# the first present, so this order is the load-bearing contract (see
# docs/fkeep-format.md). The bundled OpenCV build cannot decode the AVIF/JXL that
# pillow writes, so those two go through the faithful-mode encoder's decode
# (Pillow) — only png/jpg use cv2 here.
_CROP_EXTS = ("png", "avif", "jxl", "jpg")

# Background extensions, in reader search order (manifest 1.5.0+ may store the
# background as AVIF/JXL via AggressiveConfig.bg_codec; older files always have
# background.jpg, found first). Like _CROP_EXTS this order is the load-bearing
# contract (see docs/fkeep-format.md), and the same decode split applies: the
# bundled OpenCV build cannot decode plugin-written AVIF/JXL, so those two go
# through encoders.decode (Pillow) — only jpg uses cv2 here.
_BG_EXTS = ("jpg", "avif", "jxl")


def _encode_bg(bg: np.ndarray, cfg) -> Tuple[str, bytes]:
    """Encode the downsampled background, returning (extension, bytes).

    ``cfg.bg_codec`` picks the codec: avif/jxl go through the faithful-mode
    encoder at ``bg_quality`` with 4:2:0 chroma (it's background — no face
    needing 4:4:4), giving the restore upscaler a cleaner input than JPEG's
    block artifacts; jpg (the default) stays the byte-identical cv2 JPEG path.
    """
    if cfg.bg_codec in ("avif", "jxl"):
        from .. import encoders

        data = encoders.encode(
            bg, codec=cfg.bg_codec, quality=cfg.bg_quality, chroma="420",
        )
        return cfg.bg_codec, data
    return "jpg", _jpg(bg, cfg.bg_quality)


def _read_bg(zf: zipfile.ZipFile, names: set) -> Optional[np.ndarray]:
    """Decode the background member from an open archive, or None if absent.

    Tries ``background.(jpg|avif|jxl)`` in ``_BG_EXTS`` order and decodes via
    the right path: avif/jxl through the faithful-mode codec (Pillow -> BGR),
    jpg through cv2 — mirroring :func:`_read_crop`. Returns a BGR array or
    None when no background member exists (the caller decides whether absence
    is an error).
    """
    for ext in _BG_EXTS:
        member = f"background.{ext}"
        if member not in names:
            continue
        raw = zf.read(member)
        if ext in ("avif", "jxl"):
            from .. import encoders

            return encoders.decode(raw)
        return _decode(raw)
    return None


# Residual-member extensions, in reader search order (manifest 1.6.0+ may store
# the residual layer when aggressive.residual is on). A high-bit (HDR) residual is
# stored as a true 10/12-bit ``avif`` (manifest 1.9.0+, only when high-bit storage
# is engaged); an 8-bit residual is ``jxl`` (preferred — it wins on the residual's
# noise-like content) or ``jpg`` (the warned fallback when the JXL plugin is
# unavailable). A file has exactly ONE residual member, so the ``avif`` extension
# self-describes a high-bit residual (unlike a crop's ambiguous ``.avif``).
# Load-bearing order like _CROP_EXTS/_BG_EXTS (see docs/fkeep-format.md), with the
# decode split: avif high-bit through avifdec (uint16), jxl through encoders.decode
# (Pillow), jpg through cv2.
_RESIDUAL_EXTS = ("avif", "jxl", "jpg")


def _offset_encode_residual(residual: np.ndarray, *,
                            high_bit: bool = False) -> np.ndarray:
    """Offset-encode a signed float residual into uint8/uint16: ``clip(r/2 + mid)``.

    Halving costs ~1 bit of precision — fine for a correction layer — and maps the
    signed range into the unsigned container. 8-bit (default): ``r/2 + 128`` into
    ``uint8`` (the +-255 range). ``high_bit`` (HDR residual): ``r/2 + 32768`` into
    ``uint16`` (the +-65535 range), stored as a true 10/12-bit AVIF so a uint16
    source's background delta keeps its depth. The inverse is
    :func:`_offset_decode_residual` (which infers the depth from dtype); both
    transforms are documented in docs/fkeep-format.md so manual recovery works
    without FaceKeep.
    """
    if high_bit:
        return np.clip(np.rint(residual / 2.0 + 32768.0), 0, 65535).astype(np.uint16)
    return np.clip(np.rint(residual / 2.0 + 128.0), 0, 255).astype(np.uint8)


def _offset_decode_residual(encoded: np.ndarray) -> np.ndarray:
    """Invert :func:`_offset_encode_residual`: uint8/uint16 -> signed float32 delta.

    The depth is inferred from dtype: a ``uint16`` member is a high-bit (HDR)
    residual (``r = e*2 - 65536``); a ``uint8`` member is the 8-bit residual
    (``r = e*2 - 256``). So a high-bit residual decoded at full depth via avifdec
    (uint16) reconstructs correctly without any extra flag.
    """
    if encoded.dtype == np.uint16:
        return encoded.astype(np.float32) * 2.0 - 65536.0
    return encoded.astype(np.float32) * 2.0 - 256.0


def _encode_residual(original: np.ndarray, bg_bytes: bytes, bg_ext: str,
                     cfg) -> Tuple[str, bytes, int]:
    """Encode the residual layer, returning (extension, bytes, stored_bit_depth).

    ``residual = original - bicubic(decoded background)`` — computed against the
    background bytes *as just encoded* (decode them back), NOT the pre-encode
    array, so the residual corrects the bg codec's loss instead of fighting it;
    and with the same interpolation restore uses (INTER_CUBIC both sides — a
    pinned contract, see restorer._apply_residual). The signed residual is
    downscaled (INTER_AREA) to ``residual_scale``, offset-encoded, and stored.

    **8-bit (default):** offset-encoded to uint8 and stored as JXL at
    ``residual_quality`` (jpg fallback, warned, when the JXL plugin is
    unavailable); ``stored_bit_depth`` is 8.

    **High-bit (HDR):** a ``uint16`` original (high-bit storage engaged, so the
    compressor hands the full-depth original here) is reconstructed toward its
    real uint16 values — the stored background is 8-bit, so its bicubic upscale is
    promoted to the 16-bit scale (``x257``) before differencing
    (``restorer._apply_residual`` mirrors this). The delta is offset-encoded to
    uint16 and stored as a true 10/12-bit AVIF via ``encode_highbit_avif`` (4:4:4 —
    a correction layer must not lose chroma); ``stored_bit_depth`` is that depth.
    If ``avifenc`` is unavailable (or the high-bit encode fails) the residual
    cleanly degrades to the 8-bit path above (the uint16 original down-converted
    with ``_to_uint8`` + warned) — offline-first holds, the 8-bit residual is
    unchanged.
    """
    from .. import encoders

    if bg_ext in ("avif", "jxl"):
        decoded_bg = encoders.decode(bg_bytes)
    else:
        decoded_bg = _decode(bg_bytes)
    h, w = original.shape[:2]
    ref = cv2.resize(decoded_bg, (w, h), interpolation=cv2.INTER_CUBIC)  # 8-bit BGR

    rw = max(1, int(round(w * cfg.residual_scale)))
    rh = max(1, int(round(h * cfg.residual_scale)))

    def _downscale(res: np.ndarray) -> np.ndarray:
        if (rw, rh) != (w, h):
            return cv2.resize(res, (rw, rh), interpolation=cv2.INTER_AREA)
        return res

    # High-bit (HDR) residual: a uint16 original + a locatable avifenc. The stored
    # background is 8-bit, so promote its upscale to the uint16 scale before
    # differencing. On any avifenc failure, fall through to the 8-bit path.
    if original.dtype == np.uint16 and encoders.avifenc_available():
        residual = original.astype(np.float32) - ref.astype(np.float32) * 257.0
        encoded = _offset_encode_residual(_downscale(residual), high_bit=True)
        try:
            data = encoders.encode_highbit_avif(
                encoded, bit_depth=cfg.output_bit_depth,
                quality=cfg.residual_quality, chroma="444", has_faces=True,
            )
            return "avif", data, cfg.output_bit_depth
        except encoders.EncodingError as e:
            logger.warning("High-bit residual encode failed (%s); storing it 8-bit.", e)

    # 8-bit residual (default, or the high-bit fallback). Down-convert a uint16
    # original (high-bit requested but avifenc unavailable) to match the 8-bit
    # background it corrects — the clean /257 round-down, not a CV_8U cast.
    if original.dtype == np.uint16:
        if not encoders.avifenc_available():
            logger.warning(
                "output_bit_depth=%d residual requested but avifenc is "
                "unavailable; storing the residual 8-bit (install avifenc / set "
                "FACEKEEP_AVIFENC to keep HDR).",
                cfg.output_bit_depth,
            )
        original = _to_uint8(original)
    residual = original.astype(np.float32) - ref.astype(np.float32)
    encoded = _offset_encode_residual(_downscale(residual))
    if encoders.codec_available("jxl"):
        return "jxl", encoders.encode(
            encoded, codec="jxl", quality=cfg.residual_quality,
        ), 8
    logger.warning(
        "JXL plugin unavailable; storing the residual layer as JPEG instead "
        "(larger / lossier on this noise-like content)."
    )
    return "jpg", _jpg(encoded, cfg.residual_quality), 8


def _read_residual(zf: zipfile.ZipFile, names: set, *,
                   avifdec_strict: bool = True) -> Optional[np.ndarray]:
    """Decode the residual member from an open archive, or None if absent.

    Tries ``residual.(avif|jxl|jpg)`` in ``_RESIDUAL_EXTS`` order. Returns the
    offset-encoded array as stored (uint16 for a high-bit ``avif`` residual, uint8
    for ``jxl``/``jpg``); decoding the offset back to a signed delta is the
    restorer's job (:func:`_offset_decode_residual`, which keys off the dtype).

    The decode split: ``jxl`` via ``encoders.decode`` (Pillow), ``jpg`` via cv2.
    A high-bit ``avif`` residual depends on ``avifdec_strict``:

    * ``True`` (restore/preview, via :func:`read_fkeep`): decode at full depth via
      ``avifdec`` (uint16). If ``avifdec`` is unavailable, returns ``None`` + warns
      — restore then falls back to the AI/bicubic path. A Pillow 8-bit decode is
      *not* used here: the offset was 16-bit (``r/2 + 32768``), so an 8-bit decode
      would mis-scale into garbage; skipping the residual is the only safe option.
    * ``False`` (``verify_fkeep``): decode via Pillow (8-bit) purely to confirm the
      member is a structurally valid image — no avifdec dependency, mirroring how
      verify checks high-bit ``avif`` *crops*.
    """
    for ext in _RESIDUAL_EXTS:
        member = f"residual.{ext}"
        if member not in names:
            continue
        raw = zf.read(member)
        if ext == "avif":
            from .. import encoders

            if avifdec_strict:
                try:
                    return encoders.decode_highbit_avif(raw)
                except EncodingError as e:
                    logger.warning(
                        "High-bit residual needs avifdec (%s); restoring without "
                        "the residual layer (install avifdec / set FACEKEEP_AVIFENC "
                        "or FACEKEEP_AVIFDEC for the faithful background).", e
                    )
                    return None
            return encoders.decode(raw)
        if ext == "jxl":
            from .. import encoders

            return encoders.decode(raw)
        return _decode(raw)
    return None


def _encode_crop(crop: np.ndarray, cfg) -> Tuple[str, bytes, int]:
    """Encode one face/region crop, returning (extension, bytes, stored_bit_depth).

    A **uint16** crop reaches here only when high-bit storage was requested
    (``output_bit_depth`` 10/12 + ``face_codec == 'avif'`` + ``face_quality < 100``;
    the compressor keeps such crops uint16). It is stored as a true high-bit AVIF
    (``encode_highbit_avif``, 4:4:4) so HDR survives, and ``stored_bit_depth`` is
    that depth. If avifenc is unavailable (or the encode fails) the crop is cleanly
    down-converted to 8-bit and stored as usual with ``stored_bit_depth`` 8 —
    graceful degradation, so high-bit is best-effort and offline-first holds.

    An 8-bit crop takes the existing path unchanged: ``face_quality >= 100`` wins as
    lossless PNG; otherwise ``cfg.face_codec`` (avif/jxl stored 4:4:4, else JPEG).
    ``stored_bit_depth`` is 8 in every 8-bit case.
    """
    from .. import encoders

    if crop.dtype == np.uint16:
        if encoders.avifenc_available():
            try:
                data = encoders.encode_highbit_avif(
                    crop, bit_depth=cfg.output_bit_depth,
                    quality=cfg.face_quality, chroma="444", has_faces=True,
                )
                return "avif", data, cfg.output_bit_depth
            except encoders.EncodingError as e:
                logger.warning("High-bit crop encode failed (%s); storing it 8-bit.", e)
        else:
            logger.warning(
                "output_bit_depth=%d requested but avifenc is unavailable; storing "
                "crops 8-bit (install avifenc / set FACEKEEP_AVIFENC to keep HDR).",
                cfg.output_bit_depth,
            )
        crop = _to_uint8(crop)  # clean /257 down-convert for the 8-bit path below

    if cfg.face_quality >= 100:
        return "png", _png(crop), 8
    if cfg.face_codec in ("avif", "jxl"):
        data = encoders.encode(
            crop, codec=cfg.face_codec, quality=cfg.face_quality,
            chroma="444", has_faces=True,
        )
        return cfg.face_codec, data, 8
    return "jpg", _jpg(crop, cfg.face_quality), 8


def _read_crop(
    zf: zipfile.ZipFile, names: set, i: int, prefix: str = "face",
    high_bit: bool = False,
) -> Optional[np.ndarray]:
    """Decode crop ``i`` (``<prefix>_NNN.*``) from an open archive, or None.

    Used for both face crops (``prefix="face"``) and region-local conservatism
    patches (``prefix="region"``); both store one of ``_CROP_EXTS`` per index in
    the same load-bearing order. Tries the crop extensions in ``_CROP_EXTS`` order
    and decodes via the right path: avif/jxl through the faithful-mode codec
    (Pillow -> BGR, since cv2 can't decode them here), png/jpg through cv2.
    Returns a BGR array (matching the other crops/background) or None when the
    crop is absent — the caller decides whether absence is an error.

    ``high_bit`` (set by ``read_fkeep`` from the manifest's ``settings.bit_depth``)
    decodes an AVIF crop at true 10/12-bit depth via the ``avifdec`` CLI
    (``encoders.decode_highbit_avif`` -> uint16 BGR), so HDR survives restore. If
    ``avifdec`` is unavailable it falls back to the 8-bit Pillow decode (warned) —
    graceful degradation, offline-first.
    """
    for ext in _CROP_EXTS:
        member = f"{prefix}_{i:03d}.{ext}"
        if member not in names:
            continue
        raw = zf.read(member)
        if ext in ("avif", "jxl"):
            from .. import encoders

            if high_bit and ext == "avif":
                try:
                    return encoders.decode_highbit_avif(raw)
                except EncodingError as e:
                    logger.warning(
                        "High-bit crop decode failed (%s); falling back to 8-bit "
                        "(install avifdec / set FACEKEEP_AVIFENC for full HDR).", e
                    )
            return encoders.decode(raw)
        return _decode(raw)
    return None


def _fkeep_path(output_path: str) -> Path:
    """Resolve the .fkeep path for an output target without mangling dotted names."""
    path = Path(output_path)
    if path.suffix.lower() != FKEEP_EXTENSION:
        # Append rather than replace, so dotted filenames (e.g. 2024.05.20_trip)
        # are not mangled by suffix replacement.
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic",
                                    ".heif", ".avif", ".jxl", ".tif", ".tiff", ".bmp"}:
            path = path.with_suffix(FKEEP_EXTENSION)
        else:
            path = path.parent / (path.name + FKEEP_EXTENSION)
    return path


def _write_archive(zf: zipfile.ZipFile, photo: CompressedPhoto, manifest: dict,
                   bg_ext: str, bg_bytes: bytes, thumb_bytes: bytes,
                   face_payloads: list, mask_payloads: list,
                   region_payloads: list, region_mask_payloads: list,
                   residual_payload: Optional[Tuple[str, bytes]] = None,
                   gain_map_payload: Optional[bytes] = None) -> None:
    """Write all .fkeep entries into an open ZipFile (file- or memory-backed).

    Shared by the real write and the dry-run estimate so both pack byte-for-byte
    identical archives — the dry-run packs into an in-memory buffer and never
    touches disk, but its size is exactly what a real write would produce.
    """
    zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    if photo.exif:
        zf.writestr("exif.bin", photo.exif)
    # Original ICC color profile (e.g. Display P3), re-embedded on restore so
    # wide-gamut color survives. Same optional-member pattern as exif.bin: no
    # member when the source had no profile, so such a photo packs identically
    # to the pre-1.4.0 layout.
    if photo.icc:
        zf.writestr("icc.bin", photo.icc)
    zf.writestr(f"background.{bg_ext}", bg_bytes)
    zf.writestr("thumbnail.jpg", thumb_bytes)
    for i, (ext, payload) in enumerate(face_payloads):
        zf.writestr(f"face_{i:03d}.{ext}", payload)
    for i, m in enumerate(mask_payloads):
        zf.writestr(f"face_mask_{i:03d}.png", m)
    # Region-local conservatism patches (one crop + one mask per risky region),
    # mirroring the face members. Empty list -> no region members written, so a
    # photo without risky regions packs byte-identically to the pre-1.3.0 layout.
    for i, (ext, payload) in enumerate(region_payloads):
        zf.writestr(f"region_{i:03d}.{ext}", payload)
    for i, m in enumerate(region_mask_payloads):
        zf.writestr(f"region_mask_{i:03d}.png", m)
    # Residual layer (manifest 1.6.0+, aggressive.residual): the real delta the
    # downsample lost. None -> no member, so a residual-less photo packs
    # byte-identically to the pre-1.6.0 layout (manifest aside).
    if residual_payload is not None:
        zf.writestr(f"residual.{residual_payload[0]}", residual_payload[1])
    # iPhone HDR gain map (manifest 1.10.0+, aggressive.preserve_gain_map):
    # stored so restore can re-attach it into an HDR AVIF. None -> no member.
    if gain_map_payload is not None:
        zf.writestr("gainmap.jpg", gain_map_payload)


def write_fkeep(photo: CompressedPhoto, output_path: str,
                dry_run: bool = False) -> int:
    """Write a CompressedPhoto to a .fkeep file. Returns the size in bytes.

    With ``dry_run=True`` the archive is packed into an in-memory buffer instead
    of being written: no directory is created and no file is produced, but the
    returned size is exactly the real file size (same packing path), so the CLI
    can report an accurate projected ratio without writing anything.
    """
    path = _fkeep_path(output_path)

    cfg = photo.config

    # Pre-encode all payloads once. The background codec is configurable
    # (bg_codec: jpg default | avif | jxl); the thumbnail stays JPEG (it is a
    # compatibility preview, not a fidelity surface).
    bg_ext, bg_bytes = _encode_bg(photo.background, cfg)
    thumb_bytes = _jpg(photo.thumbnail, 80)
    # Faces use high-quality JPEG by default (visually lossless, ~10x smaller
    # than PNG for photographic content). Optionally AVIF/JXL 4:4:4 (~2x smaller
    # again, same perceptual quality), lossless PNG when face_quality >= 100, or
    # true high-bit (10/12-bit) AVIF when output_bit_depth + face_codec allow it.
    # _encode_crop returns (ext, bytes, stored_bit_depth); regions are just
    # non-face crops, encoded identically. Strip the depth for the writer and take
    # the max across all crops/regions as the container's stored real-pixel depth.
    face_enc = [_encode_crop(crop, cfg) for crop in photo.face_crops]
    region_enc = [_encode_crop(crop, cfg) for crop in photo.region_crops]
    face_payloads = [(ext, data) for ext, data, _ in face_enc]
    region_payloads = [(ext, data) for ext, data, _ in region_enc]
    mask_payloads = [_mask_png(m) for m in photo.face_masks]
    region_mask_payloads = [_mask_png(m) for m in photo.region_masks]
    # Residual layer (opt-in): needs both the flag and the attached original
    # (compress_photo gates the latter on the former). Encoded here, in the
    # shared pre-encode block, so dry-run and the real write pack identically.
    # _encode_residual returns its stored depth too — 10/12 for a high-bit (HDR)
    # residual stored as residual.avif (a uint16 original + avifenc), else 8.
    residual_payload = None
    residual_bit_depth = 8
    if cfg.residual and photo.original_image is not None:
        r_ext, r_bytes, residual_bit_depth = _encode_residual(
            photo.original_image, bg_bytes, bg_ext, cfg
        )
        residual_payload = (r_ext, r_bytes)
    # Container's stored real-data depth = max across face/region crops AND the
    # residual. Folding the residual in is what lets a *faceless* high-bit residual
    # still record settings.bit_depth (so restore writes the right depth). All
    # 8-bit -> 8 -> no bit_depth key, byte-identical manifest.
    stored_bit_depth = max(
        [d for _, _, d in face_enc + region_enc] + [residual_bit_depth]
    )

    # iPhone HDR gain map (manifest 1.10.0+): stored as a single-channel JPEG
    # (Apple itself stores it lossy; q90 on a half-res gray map costs little).
    # compress_photo attaches it only when the source carried one AND
    # aggressive.preserve_gain_map is on, so None here means "no member".
    gain_map_payload = None
    if photo.gain_map is not None:
        gm = photo.gain_map
        if gm.dtype != np.uint8:
            gm = _to_uint8(gm)
        ok, buf = cv2.imencode(".jpg", gm, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise EncodingError("Failed to encode the HDR gain map member.")
        gain_map_payload = buf.tobytes()

    payload_size = (
        len(bg_bytes) + len(thumb_bytes)
        + sum(len(p) for _, p in face_payloads)
        + sum(len(m) for m in mask_payloads)
        + sum(len(p) for _, p in region_payloads)
        + sum(len(m) for m in region_mask_payloads)
        + (len(residual_payload[1]) if residual_payload else 0)
        + (len(gain_map_payload) if gain_map_payload else 0)
    )

    manifest = {
        # 1.10.0 added the optional gainmap.jpg member + the gain_map_preserved
        # flag (iPhone HDR gain-map preservation, Phase 9); 1.9.0 made the
        # residual layer high-bit too (residual.avif; manifest bit_depth then
        # covers it); 1.8.0 added settings.bit_depth + high-bit crops; 1.7.0 the
        # optional preset key; 1.6.0 the residual layer. Readers are tolerant by
        # structure, so older readers restore 1.10.0 files unchanged (one that
        # ignores gainmap.jpg just restores SDR, exactly as before Phase 9).
        "version": "1.10.0",
        "mode": "aggressive",
        "original": {
            "filename": photo.original_filename,
            "width": photo.original_width,
            "height": photo.original_height,
            "size_bytes": photo.original_size_bytes,
            "hash_sha256": photo.original_hash,
            "orientation": photo.original_orientation,
        },
        "exif_preserved": photo.exif is not None,
        # True iff an icc.bin member is present (the source had an ICC profile,
        # e.g. Display P3). Added in manifest 1.4.0; absent on older files.
        "icc_preserved": photo.icc is not None,
        # True iff a gainmap.jpg member is present (the source carried an iPhone
        # HDR gain map and preserve_gain_map was on). Added in manifest 1.10.0;
        # absent on older files. Same flag family as exif/icc_preserved.
        "gain_map_preserved": gain_map_payload is not None,
        "settings": {
            "bg_scale": photo.effective_bg_scale,
            "bg_quality": cfg.bg_quality,
            # The background codec actually used (jpg|avif|jxl). Informational:
            # readers locate the background by member extension, not this field.
            # Added in manifest 1.5.0; absent on older files (always jpg there).
            "bg_codec": cfg.bg_codec,
            "face_quality": cfg.face_quality,
            # The crop codec actually used (jpg|avif|jxl). Informational: readers
            # locate crops by member extension, not this field. PNG crops (the
            # lossless face_quality>=100 case) still report the configured codec.
            "face_codec": cfg.face_codec,
            "blend_mode": cfg.blend_mode,
            "model": cfg.model,
            # Residual layer (manifest 1.6.0+). `residual` is the presence flag
            # for the residual.(avif|jxl|jpg) member (False when the layer is off;
            # avif = the high-bit HDR residual, 1.9.0+); scale/quality describe how
            # it was stored. Readers locate the member by extension; verify uses
            # the flag to require it.
            "residual": residual_payload is not None,
            "residual_scale": cfg.residual_scale,
            "residual_quality": cfg.residual_quality,
        },
        "faces": [
            {
                "id": f.id,
                "bbox": list(f.bbox),
                "padded_bbox": list(f.padded_bbox),
                "confidence": f.confidence,
            }
            for f in photo.faces
        ],
        # Region-local conservatism (manifest 1.3.0+): risky regions kept sharp as
        # patches. ``bbox`` is the frame-coordinate box the patch covers and where
        # restore composites it; ``scale`` is the resolution the stored patch was
        # downscaled to (1.0 = original). Empty/absent => no region patches (an
        # older 1.2.0 reader sees no key and behaves exactly as before).
        "regions": [
            {
                "id": i,
                "bbox": list(bbox),
                "scale": cfg.region_scale,
            }
            for i, bbox in enumerate(photo.regions)
        ],
        "estimated_payload_bytes": payload_size,
        # Second precision (fixed-width ISO-8601, e.g. 2026-06-01T09:30:00+00:00).
        # Microsecond precision is noise for a creation stamp and, being
        # variable-width (Python omits/zero-trims the microseconds field), it made
        # two packs of the same photo differ by a byte — breaking the dry-run
        # "estimate == real size" guarantee. Seconds keeps packing deterministic
        # within a second. See tests/test_dry_run.py.
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "facekeep_version": __version__,
    }

    # Preset (manifest 1.7.0+): the aggressive-mode preset this file was
    # compressed with, when one was used — informational plus a restore hint
    # (restore auto-applies the preset's restore-side knobs unless they are
    # explicitly overridden). Omitted entirely on presetless runs, so those
    # manifests carry no new key; readers are tolerant by structure either way.
    if cfg.preset is not None:
        manifest["settings"]["preset"] = cfg.preset

    # Bit depth of the stored high-bit real-data members (manifest 1.8.0+): the max
    # depth across face/region crops AND (1.9.0+) the residual layer. Present only
    # when something went high-bit (10/12-bit AVIF), so an 8-bit container carries
    # no new key and packs byte-identically to a 1.7.0 file. read_fkeep uses it to
    # decode high-bit AVIF crops at full depth via avifdec; the residual instead
    # self-describes its depth by its .avif extension.
    if stored_bit_depth > 8:
        manifest["settings"]["bit_depth"] = stored_bit_depth

    if dry_run:
        # Pack into memory to measure the real archive size; write nothing.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            _write_archive(zf, photo, manifest, bg_ext, bg_bytes, thumb_bytes,
                           face_payloads, mask_payloads,
                           region_payloads, region_mask_payloads,
                           residual_payload, gain_map_payload)
        return buf.tell()

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
        _write_archive(zf, photo, manifest, bg_ext, bg_bytes, thumb_bytes,
                       face_payloads, mask_payloads,
                       region_payloads, region_mask_payloads,
                       residual_payload, gain_map_payload)

    return path.stat().st_size


def read_fkeep(fkeep_path: str) -> dict:
    """Read a .fkeep file. Returns dict with manifest + decoded arrays + exif."""
    try:
        with zipfile.ZipFile(fkeep_path, "r") as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read("manifest.json"))
            background = _read_bg(zf, names)
            if background is None:
                raise KeyError("background.(jpg|avif|jxl)")
            exif = zf.read("exif.bin") if "exif.bin" in names else None
            icc = zf.read("icc.bin") if "icc.bin" in names else None

            # High-bit crops (manifest 1.8.0+, settings.bit_depth 10/12) decode at
            # full depth via avifdec; older/8-bit files have no key -> 8 -> False,
            # so they read exactly as before.
            settings = manifest.get("settings", {}) or {}
            high_bit = int(settings.get("bit_depth", 8) or 8) > 8

            face_crops, face_masks = [], []
            for i in range(len(manifest["faces"])):
                crop = _read_crop(zf, names, i, high_bit=high_bit)
                if crop is None:
                    raise KeyError(f"face_{i:03d}.(png|avif|jxl|jpg)")
                face_crops.append(crop)
                face_masks.append(_decode_mask(zf.read(f"face_mask_{i:03d}.png")))

            # Region-local conservatism patches (manifest 1.3.0+). Absent on older
            # files -> manifest.get("regions") is empty -> no region members read.
            region_crops, region_masks = [], []
            for i in range(len(manifest.get("regions", []) or [])):
                crop = _read_crop(zf, names, i, prefix="region", high_bit=high_bit)
                if crop is None:
                    raise KeyError(f"region_{i:03d}.(png|avif|jxl|jpg)")
                region_crops.append(crop)
                region_masks.append(_decode_mask(zf.read(f"region_mask_{i:03d}.png")))

            thumbnail = _decode(zf.read("thumbnail.jpg"))

            # Residual layer (manifest 1.6.0+). Absent on older files / when the
            # layer is off -> None, and restore takes the normal AI/bicubic path.
            residual = _read_residual(zf, names)

            # iPhone HDR gain map (manifest 1.10.0+). Absent on older files /
            # gain-map-less photos -> None, and restore writes SDR as before.
            # IMREAD_UNCHANGED keeps a grayscale member 2-D (the normal case).
            gain_map = (
                _decode(zf.read("gainmap.jpg"), cv2.IMREAD_UNCHANGED)
                if "gainmap.jpg" in names else None
            )
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        raise FormatError(f"Malformed .fkeep file {fkeep_path}: {e}") from e

    return {
        "manifest": manifest,
        "background": background,
        "face_crops": face_crops,
        "face_masks": face_masks,
        "region_crops": region_crops,
        "region_masks": region_masks,
        "thumbnail": thumbnail,
        "exif": exif,
        "icc": icc,
        "residual": residual,
        "gain_map": gain_map,
    }


def read_fkeep_info(fkeep_path: str) -> dict:
    """Read only the manifest (fast, no image decoding)."""
    try:
        with zipfile.ZipFile(fkeep_path, "r") as zf:
            return json.loads(zf.read("manifest.json"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        raise FormatError(f"Malformed .fkeep file {fkeep_path}: {e}") from e


@dataclass
class VerifyReport:
    """Result of ``verify_fkeep`` — a structural-integrity check of a .fkeep.

    ``ok`` is True only when the container is internally consistent: it opened,
    the manifest parsed, every entry the manifest promises is present *and*
    decodable, the face/crop/mask counts line up, and the dimensions are sane.

    Honesty note: the manifest stores the SHA-256 of the *original input file*,
    but the original pixels are gone (only the downsampled background + face
    crops survive), so the hash cannot be recomputed from the .fkeep alone.
    ``stored_hash`` is surfaced as metadata; ``hash_match`` is only populated
    (True/False) when the caller supplies the original file to match against,
    and stays ``None`` otherwise — never a fabricated pass.
    """

    path: str
    ok: bool
    problems: List[str] = field(default_factory=list)
    faces_declared: int = 0
    crops_found: int = 0
    masks_found: int = 0
    regions_declared: int = 0
    region_crops_found: int = 0
    region_masks_found: int = 0
    residual_declared: bool = False  # manifest settings.residual flag (1.6.0+)
    residual_ok: bool = False  # the declared residual member exists and decodes
    gain_map_declared: bool = False  # manifest gain_map_preserved flag (1.10.0+)
    gain_map_ok: bool = False  # the declared gainmap.jpg exists and decodes
    background_size: Optional[Tuple[int, int]] = None  # (width, height)
    original_size: Optional[Tuple[int, int]] = None     # (width, height)
    thumbnail_ok: bool = False
    stored_hash: Optional[str] = None
    hash_match: Optional[bool] = None  # None = not checked (no original given)


def _bbox_well_formed(bbox: object) -> bool:
    """A bbox is a length-4 sequence [x1, y1, x2, y2] with x2>x1 and y2>y1.

    Accepts any sized sequence of numbers (list/tuple from JSON, but also a numpy
    array or numpy scalars), since face coords originate from the detector as
    numpy types; gating on ``list``/``tuple`` only would wrongly reject those.
    """
    try:
        if len(bbox) != 4:  # type: ignore[arg-type]
            return False
        x1, y1, x2, y2 = (int(v) for v in bbox)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def verify_fkeep(fkeep_path: str, original_path: Optional[str] = None) -> VerifyReport:
    """Structurally verify a .fkeep container; return a :class:`VerifyReport`.

    Checks that the archive is self-consistent and complete:

    1. it is a valid ZIP and ``manifest.json`` parses (else ``FormatError``);
    2. the background (``background.(jpg|avif|jxl)``, located in that order)
       and ``thumbnail.jpg`` are present and decode;
    3. for each of the *N* faces the manifest declares, both a crop
       (``face_NNN.(png|avif|jxl|jpg)``, located in that order) and
       ``face_mask_NNN.png`` are present and decode — this is exactly the
       contract :func:`read_fkeep` relies on at restore time;
    4. the declared face count equals the number of decodable crops and masks;
    5. the background is non-empty and no larger than the manifest's original
       size (a downsampled background cannot exceed the original);
    6. every face's ``bbox``/``padded_bbox`` is well-formed;
    7. (manifest 1.3.0+) the same crop+mask+count+bbox checks for each declared
       region-local conservatism patch (``region_NNN.*`` / ``region_mask_NNN.png``).
       Absent ``regions`` (older files) means zero regions — verified trivially;
    8. (manifest 1.6.0+) when ``settings.residual`` declares a residual layer,
       the ``residual.(jxl|jpg)`` member (located in that order) is present and
       decodes. A residual-less file is unchanged.
    9. (manifest 1.10.0+) when ``gain_map_preserved`` declares an HDR gain map,
       the ``gainmap.jpg`` member is present and decodes. A gain-map-less file
       is unchanged.

    Any failure of 2-7 leaves the file *readable but inconsistent*: the report's
    ``ok`` is False with the specifics in ``problems`` (no exception). Only a
    truly unopenable file (bad ZIP / missing or corrupt manifest) raises
    :class:`FormatError` — the same contract as the other readers here.

    If ``original_path`` is given, its bytes are SHA-256'd and compared to the
    manifest's stored original hash; the result is reported in ``hash_match``.
    Without it, ``hash_match`` stays ``None`` (we do not invent a pass — the
    original cannot be reconstructed from the .fkeep to self-verify the hash).
    """
    # Manifest first (raises FormatError on a non-zip / unparseable manifest).
    manifest = read_fkeep_info(fkeep_path)

    problems: List[str] = []

    original = manifest.get("original", {}) or {}
    stored_hash = original.get("hash_sha256")
    orig_w, orig_h = original.get("width"), original.get("height")
    original_size = (
        (int(orig_w), int(orig_h))
        if isinstance(orig_w, int) and isinstance(orig_h, int)
        else None
    )

    faces = manifest.get("faces", []) or []
    faces_declared = len(faces)
    for i, f in enumerate(faces):
        if not _bbox_well_formed(f.get("bbox")):
            problems.append(f"face {i}: malformed bbox {f.get('bbox')!r}")
        if not _bbox_well_formed(f.get("padded_bbox")):
            problems.append(f"face {i}: malformed padded_bbox {f.get('padded_bbox')!r}")

    regions = manifest.get("regions", []) or []
    regions_declared = len(regions)
    for i, r in enumerate(regions):
        if not _bbox_well_formed(r.get("bbox")):
            problems.append(f"region {i}: malformed bbox {r.get('bbox')!r}")

    settings = manifest.get("settings", {}) or {}
    residual_declared = bool(settings.get("residual"))
    gain_map_declared = bool(manifest.get("gain_map_preserved"))

    crops_found = 0
    masks_found = 0
    region_crops_found = 0
    region_masks_found = 0
    residual_ok = False
    gain_map_ok = False
    background_size: Optional[Tuple[int, int]] = None
    thumbnail_ok = False

    try:
        with zipfile.ZipFile(fkeep_path, "r") as zf:
            names = set(zf.namelist())

            # Background: present + decodes + sane size. Located by extension
            # in _BG_EXTS order (bg_codec may store it as avif/jxl on 1.5.0+
            # files). A present-but-undecodable background is a *problem*, not
            # a crash — the cv2 (jpg) path signals that with None, the Pillow
            # (avif/jxl) path with an EncodingError, same as the crop checks.
            bg_member = next(
                (f"background.{e}" for e in _BG_EXTS
                 if f"background.{e}" in names),
                None,
            )
            if bg_member is None:
                problems.append("missing background.(jpg|avif|jxl)")
            else:
                try:
                    bg = _read_bg(zf, names)
                except EncodingError:
                    bg = None
                if bg is None:
                    problems.append(f"{bg_member} does not decode")
                else:
                    bh, bw = bg.shape[:2]
                    background_size = (bw, bh)
                    if bw <= 0 or bh <= 0:
                        problems.append(f"background has empty size {background_size}")
                    elif original_size and (bw > original_size[0] or bh > original_size[1]):
                        problems.append(
                            f"background {background_size} is larger than the "
                            f"declared original {original_size}"
                        )

            # Thumbnail: present + decodes.
            if "thumbnail.jpg" not in names:
                problems.append("missing thumbnail.jpg")
            else:
                thumb = _decode(zf.read("thumbnail.jpg"))
                if thumb is None:
                    problems.append("thumbnail.jpg does not decode")
                else:
                    thumbnail_ok = True

            # One crop + one mask per declared item (face or region), each
            # decodable. A crop is stored as exactly one of _CROP_EXTS; locate it
            # in that order. A present-but-undecodable crop is a *problem*, not a
            # crash — the cv2 path signals that with None, the Pillow (avif/jxl)
            # path with an EncodingError, so catch it and report consistently.
            def _count_crops_masks(prefix: str, declared: int) -> Tuple[int, int]:
                crops = masks = 0
                for i in range(declared):
                    present = next(
                        (e for e in _CROP_EXTS if f"{prefix}_{i:03d}.{e}" in names),
                        None,
                    )
                    if present is None:
                        problems.append(
                            f"missing {prefix} crop {i:03d} "
                            f"({prefix}_{i:03d}.png/.avif/.jxl/.jpg)"
                        )
                    else:
                        try:
                            crop = _read_crop(zf, names, i, prefix=prefix)
                        except EncodingError:
                            crop = None
                        if crop is None:
                            problems.append(
                                f"{prefix} crop {i:03d} (.{present}) does not decode"
                            )
                        else:
                            crops += 1

                    mask_name = f"{prefix}_mask_{i:03d}.png"
                    if mask_name not in names:
                        problems.append(f"missing {prefix} mask {mask_name}")
                    else:
                        mask = cv2.imdecode(
                            np.frombuffer(zf.read(mask_name), np.uint8),
                            cv2.IMREAD_GRAYSCALE,
                        )
                        if mask is None:
                            problems.append(f"{mask_name} does not decode")
                        else:
                            masks += 1
                return crops, masks

            crops_found, masks_found = _count_crops_masks("face", faces_declared)
            region_crops_found, region_masks_found = _count_crops_masks(
                "region", regions_declared
            )

            # Residual layer (manifest 1.6.0+): a declared residual must have a
            # decodable member — restore relies on it for the faithful-background
            # path. Same problem-not-crash contract as the crops/background.
            if residual_declared:
                res_member = next(
                    (f"residual.{e}" for e in _RESIDUAL_EXTS
                     if f"residual.{e}" in names),
                    None,
                )
                if res_member is None:
                    problems.append(
                        "manifest declares a residual layer but "
                        "residual.(jxl|jpg) is missing"
                    )
                else:
                    try:
                        # Structural check only: a high-bit residual.avif decodes
                        # via Pillow (8-bit) here so verify needs no avifdec —
                        # mirroring how high-bit avif *crops* are verified.
                        res = _read_residual(zf, names, avifdec_strict=False)
                    except EncodingError:
                        res = None
                    if res is None:
                        problems.append(f"{res_member} does not decode")
                    else:
                        residual_ok = True

            # HDR gain map (manifest 1.10.0+): a declared gain map must have a
            # decodable gainmap.jpg — restore relies on it for the HDR AVIF
            # path. Same problem-not-crash contract as the residual.
            if gain_map_declared:
                if "gainmap.jpg" not in names:
                    problems.append(
                        "manifest declares an HDR gain map but gainmap.jpg "
                        "is missing"
                    )
                else:
                    gm = _decode(zf.read("gainmap.jpg"), cv2.IMREAD_UNCHANGED)
                    if gm is None:
                        problems.append("gainmap.jpg does not decode")
                    else:
                        gain_map_ok = True
    except (zipfile.BadZipFile, KeyError) as e:
        # The manifest read above succeeded, so a failure here means a member is
        # named in the directory but unreadable — a corrupt archive.
        raise FormatError(f"Malformed .fkeep file {fkeep_path}: {e}") from e

    if crops_found != faces_declared:
        problems.append(
            f"face-crop count {crops_found} != declared faces {faces_declared}"
        )
    if masks_found != faces_declared:
        problems.append(
            f"face-mask count {masks_found} != declared faces {faces_declared}"
        )
    if region_crops_found != regions_declared:
        problems.append(
            f"region-crop count {region_crops_found} != declared regions "
            f"{regions_declared}"
        )
    if region_masks_found != regions_declared:
        problems.append(
            f"region-mask count {region_masks_found} != declared regions "
            f"{regions_declared}"
        )

    hash_match: Optional[bool] = None
    if original_path is not None:
        actual = hashlib.sha256(Path(original_path).read_bytes()).hexdigest()
        hash_match = (stored_hash is not None) and (actual == stored_hash)
        if not hash_match:
            problems.append(
                "original hash mismatch: "
                f"{('no stored hash' if stored_hash is None else stored_hash[:12] + '…')} "
                f"!= {actual[:12]}…"
            )

    return VerifyReport(
        path=str(fkeep_path),
        ok=not problems,
        problems=problems,
        faces_declared=faces_declared,
        crops_found=crops_found,
        masks_found=masks_found,
        regions_declared=regions_declared,
        region_crops_found=region_crops_found,
        region_masks_found=region_masks_found,
        residual_declared=residual_declared,
        residual_ok=residual_ok,
        gain_map_declared=gain_map_declared,
        gain_map_ok=gain_map_ok,
        background_size=background_size,
        original_size=original_size,
        thumbnail_ok=thumbnail_ok,
        stored_hash=stored_hash,
        hash_match=hash_match,
    )
