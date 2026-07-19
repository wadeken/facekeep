"""Faithful mode: whole-image modern-codec compression.

This is the default mode. It encodes the entire image with a modern codec
(AVIF or JPEG XL), relying on the codec's adaptive quantization to spend more
bits on faces/edges and fewer on flat background. Real pixels everywhere, no
reconstruction, no seams. Output is a single standard image file that opens in
any modern viewer; "restoring" is just opening the file.

Faces are still detected, for two reasons:
  1. Chroma decision: use 4:4:4 (no chroma subsampling) when faces are present
     to keep skin-tone and lip color crisp.
  2. Quality auto-tuning (on by default): search for the lowest quality whose
     face region meets a perceptual-quality target (faces are the acceptance
     criterion). The default target metric is SSIMULACRA2 (~90 = visually
     lossless), with an SSIM fallback when that optional package is absent.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import encoders, imageio, metrics
from .config import FaceKeepConfig
from .detector import DetectionCache, FaceRegion, create_detector, detect_cached
from .exceptions import CompressionError

logger = logging.getLogger("facekeep.faithful")

# Default acceptance thresholds per auto-tune metric. SSIM is 0-1 (config default
# 0.985); SSIMULACRA2 is ~0-100 with ~90 ≈ visually lossless. When the configured
# metric is unavailable we fall back to SSIM — and if the configured target_value
# was an SSIMULACRA2-scale number, it would be a nonsensical SSIM threshold, so
# the fallback re-bases to this SSIM default and warns.
_SSIM_DEFAULT_TARGET = 0.985


@dataclass
class FaithfulResult:
    """Result of a faithful-mode compression."""

    output_path: Path
    original_size: int
    compressed_size: int
    faces_detected: int
    quality_used: int
    codec: str
    skipped: bool = False  # True if the original was kept (encode wasn't smaller)
    # Quality score, only when one was *actually measured*. Faithful measures a
    # downscaled-SSIM round-trip only under --verify-thorough; otherwise this
    # stays None (the report leaves the cell blank rather than fabricate a
    # fidelity number we never computed). ``quality_metric`` names the method.
    quality_score: Optional[float] = None
    quality_metric: Optional[str] = None
    # True when the output is a gain-map (HDR) AVIF — the source's HDR gain map
    # was really carried into the written file (ROADMAP 9.6). False for SDR
    # output, map-less sources, and every degraded/fallback path.
    gain_map_carried: bool = False

    @property
    def ratio(self) -> float:
        return self.original_size / self.compressed_size if self.compressed_size else 0.0


def _resolve_tune_metric(cfg):
    """Pick the auto-tune acceptance scorer + threshold from ``cfg.target_metric``.

    Returns ``(scorer, target_value)`` where ``scorer(region, decoded) -> float``
    is higher-is-better (so the binary search's ``score >= target`` holds for both
    metrics). SSIMULACRA2 is the perceptual option; if it is selected but the
    optional package is unavailable we **fall back to SSIM** (graceful
    degradation, offline-first) and re-base the threshold to the SSIM default —
    because a target_value chosen on the SSIMULACRA2 scale (~90) would be a
    meaningless SSIM threshold. SSIM is always available (skimage).
    """
    if cfg.target_metric == "ssimulacra2":
        if metrics.ssimulacra2_available():
            def _score(region, decoded):
                s = metrics.ssimulacra2_score(region, decoded)
                # A per-probe computation failure -> treat as "below target" so
                # the search raises quality rather than accepting blindly.
                return s if s is not None else float("-inf")

            return _score, cfg.target_value
        logger.warning(
            "target_metric='ssimulacra2' but the ssimulacra2 package is "
            "unavailable; falling back to SSIM (target %.3f). Install with "
            "pip install facekeep[dev].",
            _SSIM_DEFAULT_TARGET,
        )
        return metrics.ssim, _SSIM_DEFAULT_TARGET

    # Default / explicit SSIM.
    return metrics.ssim, cfg.target_value


def _auto_tune_quality(
    image: np.ndarray,
    faces: List[FaceRegion],
    cfg,
    has_faces: bool,
    codec: Optional[str] = None,
    exif: Optional[bytes] = None,
    icc: Optional[bytes] = None,
    bit_depth: int = 8,
) -> tuple[bytes, int]:
    """Binary-search the lowest quality whose face region meets the target.

    Returns (encoded_bytes, quality_used). Falls back to the configured quality
    if no face region is available or the target cannot be evaluated.

    ``codec`` is the concrete codec to encode with; ``None`` (the default) uses
    ``cfg.codec`` — so the per-image ``"both"`` path passes an explicit codec
    while direct callers can omit it.

    Efficiency (ROADMAP Phase 3). Two changes cut the cost of the old search,
    which did ~6 *full-image* probe encodes plus a separate full-image re-encode
    just to re-attach metadata:

    1. **Search on the face region, not the whole image.** The acceptance
       criterion is the face region's SSIM, and ``face_union_bbox`` already
       returns the union of the detector-*padded* face boxes (clipped to the
       image) — i.e. the faces plus realistic surrounding context. The binary
       search encodes only that region (typically a small fraction of the
       frame); the *returned* encode is still the full image at the chosen
       quality. This is also exactly the region the old search scored, so the
       chosen quality is unchanged except for the codec allocating bits slightly
       differently on a region encoded in isolation vs in the full frame.
    2. **No separate metadata re-attach.** Probes encode without metadata (EXIF/
       ICC don't affect SSIM); only the final full-image encode carries
       ``exif``/``icc``, so the returned bytes are already metadata-bearing.

    The win scales with how clustered the faces are: a single/centred face
    shrinks each probe to a small tile; faces spread across the frame (or a
    distant false positive) make the padded union large and the saving smaller.
    Worst case it degrades to roughly the old cost, never worse.

    ``bit_depth`` (high-bit output, ROADMAP Phase 1) is applied to the *final*
    full-image encode only — the probe encodes deliberately stay 8-bit (the
    search just picks a quality number; an 8-bit proxy is enough, and it keeps
    the ~6 probes off the slower avifenc CLI path). So a 16-bit source is searched
    in 8-bit and emitted at the chosen quality in true 10-bit.
    """
    if codec is None:
        codec = cfg.codec
    bbox = metrics.face_union_bbox(faces, image.shape[:2])
    if bbox is None:
        # No face region to tune against: encode once at the configured quality,
        # with metadata, and return it.
        data = encoders.encode(
            image, codec, cfg.quality, cfg.speed, cfg.chroma, has_faces,
            exif=exif, icc=icc, bit_depth=bit_depth,
            output_bit_depth=cfg.output_bit_depth,
        )
        return data, cfg.quality

    # Tune on the padded face-union region rather than the whole image. This is
    # the same region the search's acceptance metric covers, so scoring it in
    # isolation barely shifts the chosen quality while encoding far fewer pixels.
    x1, y1, x2, y2 = bbox
    region = image[y1:y2, x1:x2]

    # Probes deliberately run in 8-bit (an 8-bit proxy is enough to pick a
    # quality number, and it keeps the ~6 probes off the slower avifenc CLI path).
    # For a high-bit (uint16) source, down-convert the *probe* region once here so
    # the probe encode doesn't repeatedly emit the encoder's "rounding down to
    # 8-bit / pending avifenc" warning — which would be misleading since the
    # *final* encode below uses the true high-bit path. The score then compares
    # the 8-bit region against its 8-bit decode (consistent), and the final
    # full-image encode still carries bit_depth for true 10-bit output.
    if region.dtype == np.uint16:
        probe_region = np.round(region.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    elif region.dtype != np.uint8:
        probe_region = np.clip(region, 0, 255).astype(np.uint8)
    else:
        probe_region = region

    # Resolve the acceptance metric + threshold once (not per probe), so a
    # fallback warning is emitted at most once and availability is checked once.
    scorer, target_value = _resolve_tune_metric(cfg)

    lo, hi = 40, 95
    best_q: Optional[int] = None

    for _ in range(6):  # ~log2(55) iterations
        q = (lo + hi) // 2
        # Probe encodes the face region only, without metadata (cheaper;
        # metadata bytes don't affect the score the search reads).
        data = encoders.encode(probe_region, codec, q, cfg.speed, cfg.chroma, has_faces)
        decoded = encoders.decode(data)
        # Both scorers are dtype-safe / higher-is-better; probe_region is 8-bit.
        score = scorer(probe_region, decoded)
        logger.debug("auto-tune q=%d %s=%.4f", q, cfg.target_metric, score)
        if score >= target_value:
            best_q = q
            hi = q - 1
        else:
            lo = q + 1

    # Chosen quality: the lowest that met the target, else the highest we tried.
    quality_used = best_q if best_q is not None else hi
    # Single metadata-bearing encode of the FULL image at the chosen quality.
    # This is the only encode whose bytes we return, so EXIF/ICC are embedded
    # exactly once with no extra re-encode. A failure here is a real bug (the
    # search already encoded this quality successfully on the region) and
    # propagates as EncodingError — never swallowed (would write stripped bytes).
    data = encoders.encode(
        image, codec, quality_used, cfg.speed, cfg.chroma, has_faces,
        exif=exif, icc=icc, bit_depth=bit_depth,
        output_bit_depth=cfg.output_bit_depth,
    )
    return data, quality_used


def _encode_one(
    image: np.ndarray,
    faces: List[FaceRegion],
    cfg,
    has_faces: bool,
    codec: str,
    exif: Optional[bytes],
    icc: Optional[bytes],
    bit_depth: int = 8,
) -> tuple[bytes, int, str]:
    """Encode the image with a single concrete codec, honoring auto-tune/lossless.

    Returns ``(encoded_bytes, quality_used, codec_used)``. This is the per-codec
    unit of work shared by the single-codec path and the ``both`` (per-image
    choice) path; it never sees ``cfg.codec`` so the caller decides which codec to
    use. ``codec_used`` usually equals ``codec`` but may differ in lossless mode:
    a lossless AVIF request without the ``avifenc`` CLI honestly falls back to
    lossless JXL (so the user still gets a genuinely lossless file), and the
    returned ``codec_used`` reflects that so the extension/verify/index agree.

    ``bit_depth`` routes a high-bit source through the true 10-bit AVIF path on
    the final encode (see :func:`_auto_tune_quality` / :func:`encoders.encode`);
    it is a no-op for JXL and for 8-bit sources.
    """
    if cfg.lossless:
        # Lossless bypasses auto-tune/quality entirely (it's bit-exact). A
        # lossless AVIF needs avifenc; without it, fall back to lossless JXL so
        # the "lossless" promise is never quietly broken.
        if codec == "avif" and not encoders.avifenc_available():
            if encoders.codec_available("jxl"):
                logger.warning(
                    "Lossless AVIF needs the avifenc CLI (not found); writing "
                    "lossless JXL instead (set FACEKEEP_AVIFENC or put avifenc on "
                    "PATH for lossless AVIF)."
                )
                codec = "jxl"
            # else: no JXL either — fall through and let encode() raise the
            # standard EncodingError (with the install hint) for AVIF lossless.
        data = encoders.encode(
            image, codec, exif=exif, icc=icc, lossless=True,
        )
        # quality_used is reported as 100 for a lossless encode (informational).
        return data, 100, codec
    if cfg.auto_tune:
        data, quality_used = _auto_tune_quality(
            image, faces, cfg, has_faces, codec, exif=exif, icc=icc,
            bit_depth=bit_depth,
        )
        return data, quality_used, codec
    data = encoders.encode(
        image, codec, cfg.quality, cfg.speed, cfg.chroma, has_faces,
        exif=exif, icc=icc, bit_depth=bit_depth,
        output_bit_depth=cfg.output_bit_depth,
    )
    return data, cfg.quality, codec


def _encode_best_codec(
    image: np.ndarray,
    faces: List[FaceRegion],
    cfg,
    has_faces: bool,
    exif: Optional[bytes],
    icc: Optional[bytes],
    bit_depth: int = 8,
) -> tuple[bytes, int, str]:
    """Resolve the encode for ``cfg.codec``, including per-image ``"both"`` choice.

    Returns ``(encoded_bytes, quality_used, codec_used)`` — ``codec_used`` is the
    concrete codec the returned bytes are in (``avif`` or ``jxl``), which for
    ``"both"`` is whichever produced the *smaller* output.

    ``"both"`` (per-image codec choice, ROADMAP Phase 5): trial-encode with both
    codecs — each at its own auto-tuned/configured quality, so the comparison is
    "at equal perceptual target, which codec is smaller" rather than equal raw
    quality number — and keep the smaller. It costs an extra encode (or auto-tune
    search) per image, so it is opt-in via ``codec: both``.

    Graceful degradation (offline-first): if a codec's plugin is missing, it is
    dropped from the trial; if only one of the two is available, ``both`` falls
    back to that one with a warning. ``encoders.encode`` raises ``EncodingError``
    if *none* is available, matching the single-codec path.
    """
    if cfg.codec != "both":
        # _encode_one reports the *actual* codec used (a lossless-AVIF-without-
        # avifenc request falls back to lossless JXL), so trust its codec_used.
        return _encode_one(
            image, faces, cfg, has_faces, cfg.codec, exif, icc, bit_depth,
        )

    candidates = [c for c in ("avif", "jxl") if encoders.codec_available(c)]
    if len(candidates) < 2:
        if len(candidates) == 1:
            logger.warning(
                "codec='both' but only the %s codec is available; using it.",
                candidates[0],
            )
            chosen = candidates[0]
        else:
            # Neither plugin present: defer to encode() so it raises the standard
            # EncodingError with the install hint (same as the single-codec path).
            chosen = "avif"
        return _encode_one(
            image, faces, cfg, has_faces, chosen, exif, icc, bit_depth,
        )

    best: Optional[tuple[bytes, int, str]] = None
    for codec in candidates:
        data, quality_used, codec_used = _encode_one(
            image, faces, cfg, has_faces, codec, exif, icc, bit_depth,
        )
        logger.debug("codec=both probe %s -> %d bytes", codec_used, len(data))
        if best is None or len(data) < len(best[0]):
            best = (data, quality_used, codec_used)
    assert best is not None  # candidates is non-empty
    logger.debug("codec=both chose %s (%d bytes)", best[2], len(best[0]))
    return best


def _attach_gain_map(
    data: bytes,
    codec_used: str,
    quality_used: int,
    loaded: "imageio.LoadedImage",
    cfg,
) -> tuple[bytes, bool]:
    """Carry the source's HDR gain map into the output (ROADMAP 9.6).

    When the source carried a gain map (``loaded.gain_map``, Phase 9.1) and the
    concrete output codec is AVIF, re-encode the output as a backward-compatible
    **gain-map (HDR) AVIF** via ``encoders.encode_gainmap_avif`` (the 9.2
    restore recipe: rebuild the fully-applied HDR alternate — honoring the
    source's declared hdrgm math when it rode in — and ``avifgainmaputil
    combine``). Unlike the aggressive path there is no hallucinated-background
    caveat: the base here is the real full-resolution image at the tuned
    quality, so the carry is as honest as faithful mode itself.

    Returns ``(encoded_bytes, carried)`` — on any non-AVIF/degraded path the
    input ``data`` is returned unchanged with ``carried=False`` (graceful
    degradation, offline-first: the default zero-download install emits
    byte-identical SDR output plus a warning).

    Documented trades on the HDR path (both inherited from ``combine``): color
    is declared via CICP (P3-by-name / sRGB) rather than the embedded ICC
    profile, and the base is encoded by libavif's own encoder at
    ``quality_used`` (the tuned quality number carries over; the plugin's
    chroma/speed knobs do not apply to this encoder).
    """
    if loaded.gain_map is None or not cfg.preserve_gain_map:
        return data, False
    if cfg.lossless:
        # Lossless promises a bit-exact base; combine re-encodes it lossily.
        logger.warning(
            "Source carries an HDR gain map, but lossless mode keeps its "
            "bit-exact promise and writes SDR (the gain-map AVIF path "
            "re-encodes the base). Disable faithful.lossless to carry HDR."
        )
        return data, False
    if loaded.source_bit_depth > 8:
        # Gain-map HDR is an 8-bit base by construction; a genuine uint16
        # source already keeps its HDR through the deep-color 10/12-bit path.
        logger.warning(
            "Source is high-bit AND carries a gain map; keeping the 10/12-bit "
            "deep-color output and dropping the gain map (the gain-map AVIF "
            "base is 8-bit by construction)."
        )
        return data, False
    if codec_used != "avif":
        logger.warning(
            "Source carries an HDR gain map, but only an AVIF output can "
            "carry it; writing SDR %s. Set faithful.codec: avif to preserve "
            "HDR.", codec_used,
        )
        return data, False
    if not encoders.avifgainmaputil_available():
        logger.warning(
            "Source carries an HDR gain map, but the avifgainmaputil binary "
            "was not found (set FACEKEEP_AVIFENC or put the libavif tools on "
            "PATH); writing SDR AVIF."
        )
        return data, False
    try:
        hdr = encoders.encode_gainmap_avif(
            loaded.image, loaded.gain_map,
            headroom=cfg.gain_map_headroom,
            gain_map_params=(loaded.gain_map_meta or {}).get("hdrgm"),
            quality=quality_used,
            exif=loaded.exif, icc=loaded.icc,
        )
        return hdr, True
    except Exception as e:  # noqa: BLE001 - HDR carry must never fail the encode
        logger.warning(
            "HDR gain-map carry failed (%s); writing SDR AVIF instead.", e,
        )
        return data, False


def compress(
    image_path: str,
    output_path: Optional[str] = None,
    config: Optional[FaceKeepConfig] = None,
    dry_run: bool = False,
    detection_cache: Optional[DetectionCache] = None,
) -> FaithfulResult:
    """Compress an image in faithful mode.

    Args:
        image_path: Path to the input image
        output_path: Output path (extension is forced to match the codec).
            If None, uses the input path's stem with the codec extension.
        config: Configuration (uses defaults if None)
        dry_run: If True, run the full pipeline (load, detect, encode, verify,
            skip-if-larger) to compute the real projected size/ratio, but do
            **not** write any file. ``output_path`` on the result is the path
            that *would* be written (so the CLI can report it); nothing is
            created on disk. The faithful pipeline encodes the same bytes
            either way, so a dry-run's numbers match the real run exactly.
        detection_cache: Optional :class:`~facekeep.detector.DetectionCache`. When
            given, the face-detection result is reused across re-runs (keyed by
            the input's content hash + the detector settings). ``None`` (the
            default) detects normally — the cache is a pure speed feature and
            never changes the output bytes.

    Returns:
        FaithfulResult with paths and statistics.
    """
    config = config or FaceKeepConfig()
    cfg = config.faithful

    src = Path(image_path)
    original_size = src.stat().st_size

    loaded = imageio.load(image_path, strip_gps=config.strip_gps)
    image = loaded.image

    # Detect faces (for chroma + optional auto-tune + metadata). When a detection
    # cache is supplied, the result is looked up by (file content hash, detector
    # fingerprint) and reused on a hit — a pure speed optimization that does not
    # change which faces are used or the encoded bytes.
    try:
        detector = create_detector(
            backend=config.detector.backend,
            confidence=config.detector.confidence,
            padding=config.detector.padding,
            nms_iou=config.detector.nms_iou,
            min_size_ratio=config.detector.min_size_ratio,
            max_aspect_ratio=config.detector.max_aspect_ratio,
            roi=config.detector.roi,
        )
        if detection_cache is not None:
            content_hash = hashlib.sha256(src.read_bytes()).hexdigest()
            faces = detect_cached(detector, image, content_hash, detection_cache)
        else:
            faces = detector.detect(image)
    except Exception as e:  # noqa: BLE001 - detection failure shouldn't block faithful encode
        logger.warning("Face detection failed (%s); encoding without face info.", e)
        faces = []

    has_faces = len(faces) > 0

    # Encode
    #
    # Both paths embed EXIF/ICC in their returned bytes (auto-tune does it in its
    # final chosen-quality encode — see _auto_tune_quality), so there is no
    # separate metadata re-attach step. A metadata-embed failure surfaces as
    # EncodingError rather than being swallowed (which would write bytes with the
    # ICC/EXIF stripped, dropping color/orientation we worked to preserve).
    # ``codec_used`` is the concrete codec the returned bytes are in. It equals
    # cfg.codec for a single codec; for cfg.codec == "both" it is whichever of
    # avif/jxl produced the smaller output (per-image codec choice). Everything
    # downstream (verify, skip-if-larger, the output extension, the reported
    # codec) uses codec_used, not cfg.codec.
    try:
        data, quality_used, codec_used = _encode_best_codec(
            image, faces, cfg, has_faces, exif=loaded.exif, icc=loaded.icc,
            bit_depth=loaded.source_bit_depth,
        )
    except Exception as e:  # noqa: BLE001
        raise CompressionError(f"Faithful encode failed for {image_path}: {e}") from e

    # HDR gain-map carry (ROADMAP 9.6): a gain-map-bearing source with an AVIF
    # output becomes a backward-compatible gain-map (HDR) AVIF; every other
    # combination keeps `data` unchanged (warned where HDR is really lost).
    # This runs BEFORE verify and skip-if-larger on purpose: verify must check
    # the bytes actually written, and the size decision must see the final
    # (slightly larger) HDR file — in a dry run too, so its numbers stay real.
    data, gain_map_carried = _attach_gain_map(
        data, codec_used, quality_used, loaded, cfg
    )

    # Verify the encoded output decodes and matches the source before writing,
    # so a corrupt encode fails loudly instead of silently producing a bad file.
    # Under --verify-thorough this also returns the downscaled-SSIM it measured,
    # which we surface in the result (for --report) — a real number, not one we
    # invented. The quick check measures nothing, so the score stays None.
    quality_score: Optional[float] = None
    quality_metric: Optional[str] = None
    if cfg.verify:
        quality_score = encoders.verify_roundtrip(
            data, image, thorough=cfg.verify_thorough
        )
        if quality_score is not None:
            quality_metric = "ssim_downscaled"

    if output_path is None:
        output_path = str(src.with_suffix(""))

    # Skip-if-larger: never make a file worse. If the encode is no smaller than
    # the input (already-optimized files), keep the original bytes instead.
    #
    # Lossless mode opts out: a bit-exact encode is *expected* to be larger than a
    # lossy original (that's the archival trade-off the user asked for), so the
    # skip would otherwise always keep the original and never write the lossless
    # file. The user explicitly wants the lossless file regardless of size.
    #
    # dry_run reflects this decision faithfully (the report must say "would keep
    # original") but writes nothing: it resolves the path the write *would*
    # produce — the same `_with_extension` logic the real writers use — without
    # touching disk.
    if cfg.skip_if_larger and not cfg.lossless and len(data) >= original_size:
        logger.info(
            "Encoded %s is not smaller (%d >= %d bytes); keeping the original.",
            src.name, len(data), original_size,
        )
        if dry_run:
            out = encoders._with_extension(Path(output_path), src.suffix.lower())
        else:
            out = encoders.copy_original(str(src), output_path)
        return FaithfulResult(
            output_path=out,
            original_size=original_size,
            compressed_size=original_size,  # we kept the original; ratio = 1.0
            faces_detected=len(faces),
            quality_used=quality_used,
            codec=codec_used,
            skipped=True,
        )

    if dry_run:
        out = encoders._with_extension(
            Path(output_path), encoders.CODEC_EXTENSION.get(codec_used, Path(output_path).suffix)
        )
    else:
        out = encoders.write_encoded(data, output_path, codec_used)

    return FaithfulResult(
        output_path=out,
        original_size=original_size,
        compressed_size=len(data),
        faces_detected=len(faces),
        quality_used=quality_used,
        codec=codec_used,
        quality_score=quality_score,
        quality_metric=quality_metric,
        gain_map_carried=gain_map_carried,
    )
