"""Configuration management for FaceKeep."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .exceptions import ConfigError
# Single source of truth for the video defaults (video.py documents the
# measured rationale); the dataclass below mirrors them so the YAML/CLI layer
# and the library API can never drift apart. video.py imports only stdlib +
# exceptions, so this adds no import cycle and no heavy dependency.
from .video import (
    DEFAULT_CRF,
    DEFAULT_FACE_VMAF_TARGET,
    DEFAULT_PRESET,
    DEFAULT_VMAF_TARGET,
)


@dataclass
class DetectorConfig:
    """Face/ROI detection settings (shared by both modes)."""

    backend: str = "haar"  # haar | yunet | mediapipe
    confidence: float = 0.6  # Min detection confidence (yunet / mediapipe only)
    padding: float = 1.5  # Padding multiplier around detected face bbox

    # High-priority region per detected subject. Detectors give only a *face*
    # box; this optionally grows the padded box downward (and slightly outward)
    # to cover the upper body, so family subjects keep detail beyond the face.
    #   face          : just the padded face box (default — unchanged behaviour)
    #   head_shoulders: extend down ~1.2x face height to cover head + shoulders
    #   person        : extend down ~4x face height to cover a standing body
    # It only enlarges padded_bbox (the high-priority region used for aggressive
    # crops and the faithful auto-tune acceptance region); the tight detection
    # box, NMS, and chroma decision are unchanged.
    roi: str = "face"  # face | head_shoulders | person

    # False-positive filtering (Haar has no confidence score, so use geometry).
    nms_iou: float = 0.3  # Suppress overlapping boxes above this IoU
    min_size_ratio: float = 0.05  # Min face short-side as fraction of image short side
    max_aspect_ratio: float = 1.6  # Reject boxes wider/taller than this (w/h or h/w)


@dataclass
class FaithfulConfig:
    """Faithful mode: whole-image modern-codec encoding.

    The whole image is encoded with a modern codec at `quality`. The codec's
    own adaptive quantization handles perceptual bit allocation (more bits on
    faces/edges, fewer on flat background). Output is a single standard image
    file. Real pixels everywhere; no reconstruction, no seams.
    """

    # avif | jxl | webp | both. "both" trial-encodes the image with avif *and*
    # jxl (each at its own auto-tuned/configured quality) and keeps whichever
    # output is smaller — per-image codec choice. It costs an extra encode (or
    # auto-tune search) per image, so it is opt-in; the default stays a single
    # codec. "webp" is the maximum-compatibility fallback: built into Pillow (no
    # plugin) and opens in any browser / older viewer that can't yet read AVIF or
    # JXL, at the cost of a larger file. WebP is 8-bit-only (a 16-bit source
    # rounds down, warned) and caps each side at 16383 px; it is deliberately not
    # part of "both" (it never beats avif/jxl on size — it's a compatibility
    # choice, not a size contender).
    codec: str = "avif"  # avif | jxl | webp | both
    quality: int = 70  # 0-100. ~70 is visually lossless for most photos.

    # Mathematically-lossless output for archival of irreplaceable originals.
    # When on, the whole image is encoded bit-exact and quality/auto-tune are
    # ignored (a lossless file is much larger — this is "keep this original
    # exactly", not "small"). JXL is lossless natively; AVIF needs the external
    # avifenc CLI (the bundled plugin has no lossless path), and without it the
    # encode honestly falls back to lossless JXL with a warning (so the user
    # always gets genuine lossless). Off by default. Output-affecting → in
    # index.settings_fingerprint.
    lossless: bool = False
    speed: int = 6  # Encoder effort 0-10 (avif). Lower = slower but smaller.

    # Auto-tuning: search for the lowest quality whose face region meets a
    # perceptual quality target, so users get "visually lossless" output without
    # picking a quality number. **On by default** now that a perceptual metric
    # (SSIMULACRA2) has landed — the search accepts on perception, not a guessed
    # `quality`. The CLI turns it off when an explicit `-q` is given (an explicit
    # quality is a deliberate override) and exposes `--auto-tune/--no-auto-tune`.
    auto_tune: bool = True
    # Acceptance metric for the auto-tune search (higher = better for both).
    #   ssimulacra2: perceptual quality (~0-100; ~90 visually lossless, ~70 high
    #                quality) — the "the eye can't tell" target, far better
    #                perceptual correlation than SSIM. This is the default. It
    #                needs the ssimulacra2 package ([dev], pure-Python — no native
    #                binary, no model download); a plain install that lacks it
    #                falls back to SSIM automatically (the threshold is re-based to
    #                the SSIM default — see faithful._resolve_tune_metric), so the
    #                default is "perceptual when available, SSIM otherwise".
    #   ssim:        structural similarity (0-1; target ~0.985). Always available,
    #                but correlates only loosely with perception.
    # (butteraugli is a separate, lower-is-better metric tracked in the ROADMAP;
    # not wired here — it is deferred behind the same external-binary wall as
    # 10-bit AVIF / delta-Q ROI: no pure-Python butteraugli package exists, so
    # the real distance needs an external libjxl/butteraugli binary.)
    target_metric: str = "ssimulacra2"  # ssimulacra2 | ssim
    target_value: float = 90.0  # acceptance threshold (SSIMULACRA2 scale, ~90 = visually lossless)

    # Chroma subsampling. 4:4:4 preserves skin-tone/lip color fidelity on
    # faces; 4:2:0 is smaller. "auto" uses 4:4:4 when faces are present.
    chroma: str = "auto"  # auto | 444 | 420

    # Output bit depth for the true high-bit AVIF path (10 | 12). Only takes
    # effect when the *source* is genuinely high-bit (uint16) AND codec is avif
    # AND the external `avifenc` CLI is available (see encoders.encode_highbit_avif);
    # an 8-bit source, JXL, or a missing avifenc binary ignores it and uses the
    # 8-bit path regardless. 10-bit clears banding on smooth gradients and is the
    # widely-supported default; 12 keeps maximum precision for 16-bit sources at
    # the cost of compatibility with older decoders. Never widens an 8-bit source.
    output_bit_depth: int = 10  # 10 | 12 (high-bit AVIF output only)

    # Output round-trip verification. `verify` (on by default) runs a cheap
    # sanity check after encoding — decode the output and confirm dimensions —
    # so a corrupt encode fails loudly instead of silently. `verify_thorough`
    # additionally requires a downscaled-SSIM floor against the source.
    verify: bool = True
    verify_thorough: bool = False

    # If the encoded output is no smaller than the input (already-optimized
    # files), keep the original instead of writing a larger file. On by default
    # so we never make a file worse.
    skip_if_larger: bool = True


@dataclass
class AggressiveConfig:
    """Aggressive mode: crop faces + downsample background + AI restore.

    Extreme compression at the cost of background fidelity. The background is
    discarded and reconstructed (hallucinated) on restore, so it will look
    plausible but differ from the original. Faces are kept at original quality.
    """

    # Background downsampling
    bg_scale: float = 0.25  # 0.125 = 1/8, 0.25 = 1/4
    bg_quality: int = 85  # quality for the stored background (0-100, any codec)
    # Codec for the stored background. avif/jxl give the restore upscaler a
    # cleaner input (JPEG block artifacts are exactly what SR amplifies into
    # false texture) and a smaller .fkeep — but AVIF can lose to JPEG on noisy
    # content, so jpg stays the default (default output is byte-identical).
    bg_codec: str = "jpg"  # jpg | avif | jxl (avif/jxl stored 4:2:0)
    face_quality: int = 95  # Quality for face crops (>=100 -> lossless PNG)
    # Codec for face crops. avif/jxl match jpg q95's perceptual quality at ~2x
    # smaller (always stored 4:4:4 so skin/lips stay crisp). face_quality>=100
    # still forces lossless PNG regardless of this. jpg is the universal default.
    face_codec: str = "jpg"  # jpg | avif | jxl

    # Output bit depth for the real-pixel members (face crops + region patches).
    # 8 (default) = the 8-bit container: a high-bit source (e.g. a 10/12-bit HDR
    # HEIC, decoded as uint16) is rounded down to 8-bit like every other member.
    # 10 | 12 = store crops/regions at true high bit depth via the external
    # `avifenc` CLI, so iPhone-style HDR survives the round-trip. The background,
    # thumbnail, and residual stay 8-bit (the background is hallucinated on
    # restore, so high-bit there buys nothing — the win is the real-pixel crops).
    # High-bit storage ENGAGES only when output_bit_depth in (10, 12) AND
    # face_codec == "avif" AND avifenc/avifdec are locatable; otherwise it degrades
    # gracefully to the warned 8-bit round-down (offline-first holds, and the
    # default 8 is byte-identical). It makes the .fkeep larger (HDR crops exceed
    # 8-bit) — the explicit fidelity/ratio trade — so it is off by default.
    # Output-affecting -> in index.settings_fingerprint. Restoring a high-bit
    # .fkeep to true HDR needs an avif/jxl output (`restore -f avif`); a JPEG
    # output rounds down (warned).
    output_bit_depth: int = 8  # 8 (off) | 10 | 12 — high-bit crops via avifenc

    # Detection override (aggressive mode only). The worst failure here is a
    # small/distant background face that detection *missed*: it gets downsampled
    # and the AI reconstructs it into something uncanny — emotionally worse than
    # a soft background. So aggressive mode protects every face more aggressively
    # than faithful mode, which only uses faces for a chroma hint. Each field is
    # None = "inherit the shared DetectorConfig value"; set it to override only
    # for this mode (faithful keeps DetectorConfig untouched).
    #   detector_backend: None (inherit) keeps the offline, zero-download default
    #     (Haar) so the default aggressive run never needs the network — the
    #     CLAUDE.md offline-first convention. For the best small/profile/background
    #     face recall, set this to "yunet" (DNN): it auto-downloads a ~232 KB model
    #     on first use and falls back to Haar offline. It is an opt-in upgrade, not
    #     the default, precisely so the default path stays offline.
    #   detector_confidence: lower than faithful so faint background faces survive.
    #   detector_min_size_ratio: much smaller so a distant face (a small fraction
    #     of a big frame) is not discarded by the false-positive size filter —
    #     missing a real face is far worse here than keeping an extra crop. This is
    #     a pure parameter (no download), so it helps even on the default Haar.
    detector_backend: Optional[str] = None  # None -> inherit (haar); "yunet" | "mediapipe"
    detector_confidence: Optional[float] = 0.5  # None -> inherit DetectorConfig
    detector_min_size_ratio: Optional[float] = 0.02  # None -> inherit

    # Zero-face handling
    no_face_strategy: str = "conservative"  # conservative | skip | normal
    no_face_bg_scale: float = 0.5

    # Content-aware conservatism. The aggressive downsample is safe on benign
    # content (foliage, sky, bokeh, plain surfaces) but mangles content the AI
    # cannot honestly reconstruct: text/signage, fine regular structure, and
    # small/distant background faces (the worst failure — an uncanny face). When
    # `content_aware` is on (default), a risky photo has its *global* bg_scale
    # raised toward `conservative_bg_scale` (compress the whole background less),
    # the same lever the no-face fallback uses — never lowered. (This is the
    # whole-image first step; per-region scale maps are a future item.)
    content_aware: bool = True  # raise bg_scale on risky content
    conservative_bg_scale: float = 0.5  # floor bg_scale is raised to on a risk
    # Region-local conservatism. When on (default), a risky region is protected
    # *locally* — stored as a near-original-resolution patch composited back on
    # restore — instead of raising the *whole-image* bg_scale toward
    # conservative_bg_scale. So the benign majority of the frame keeps aggressive
    # compression while only the risky region is kept sharp. Gated by content_aware
    # (region_local only matters when content_aware is on); when region_local is
    # off, the whole-image conservatism above is used as before. Small/distant
    # faces and hands are localized by default; the edge-density/text signal is
    # localized only via the opt-in protect_text below (otherwise whole-image).
    region_local: bool = True  # protect risky regions locally, not whole-image
    region_scale: float = 1.0  # resolution to store region patches at (1.0 = orig)
    # Hand protection (region-local, gated by content_aware + region_local). Hands
    # aren't faces, so the detector never finds them: they ride the aggressive
    # bg_scale downsample and the AI upscaler smears their thin finger structure on
    # restore. When on (default), hand regions are kept sharp *locally* — the same
    # near-original patch mechanism as small-face regions (region_NNN.* + mask +
    # bbox), so it changes no .fkeep format/manifest. It is tiered, mirroring the
    # offline-Haar / opt-in-YuNet split, because OpenCV ships no hand cascade:
    #   protect_hands_backend = None (default): C1 — infer a hand-likely zone from
    #     each detected face's geometry (a band below/beside the face, where hands
    #     rest in a portrait). Zero-download, offline; a probabilistic guess
    #     (overhead/off-body/face-less hands are missed).
    #   protect_hands_backend = "mediapipe": C2 — real MediaPipe Hand Landmarker
    #     detection (tight per-hand boxes, catches off-body hands). Opt-in upgrade:
    #     needs the [detect] extra + a model download; missing/offline gracefully
    #     falls back to C1, never crashes.
    protect_hands: bool = True  # keep hands sharp via region patches (default C1)
    protect_hands_backend: Optional[str] = None  # None -> C1 geometry; "mediapipe" -> C2
    hand_zone_scale: float = 1.0  # resolution to store hand patches at (1.0 = orig)
    # C1 over-coverage guard (no effect on C2 real detection). The C1 geometric
    # hand zones are a body-proportion *guess*; on a dense group/family photo each
    # face's bands stack up to cover a large slice of the frame (mostly torsos/laps
    # with no hands), which encodes to most of the .fkeep and destroys the ratio.
    # So the merged C1 zones are dropped entirely when their union covers more than
    # this fraction of the frame — the photo still compresses via the face crops +
    # whole-image conservatism, and a user who needs real group-hand protection
    # opts into C2. Mirrors aggressive.text_region_max_frac. Tuned on the corpus:
    # a dense 5-face group covers ~43% (its .fkeep was *larger* than the source) →
    # bails; a 3-face photo covers ~22% → kept (C1 still useful there).
    hand_zone_max_frac: float = 0.30  # drop C1 hand zones above this frame coverage
    # C2 (MediaPipe) detection tuning — recall knobs (no effect on C1). MediaPipe's
    # palm detector is trained for phone-camera-sized frames, so on a big photo
    # (e.g. 12 MP) a hand is a tiny fraction of the frame and is missed. So C2
    # *downscales the detection input* to hand_detect_long_side before running the
    # landmarker (landmark coords are normalized, so the resulting boxes map back to
    # the full-resolution frame for free) — this roughly doubles recall on large
    # photos. The defaults lean toward recall (catch the man's occluded hand too):
    # missing a hand is the failure we care about; a false hand only protects an
    # extra background patch (slightly larger file), never corrupts output.
    hand_detect_confidence: float = 0.3  # C2 min detection confidence (lower = more hands)
    hand_detect_max_hands: int = 6  # C2 num_hands cap (family photos have several)
    hand_detect_long_side: int = 1280  # downscale detection input to this long side (0 = full-res)
    hand_detect_padding: float = 1.25  # padding around the tight C2 landmark box
    # Edge-density heuristic for "detailed" (text/signage/fine structure). It is
    # a zero-download proxy, NOT true OCR/text detection: the fraction of the
    # frame on strong edges; above this, treat the background as risky.
    text_edge_threshold: float = 0.05  # edge-pixel fraction -> "detailed"
    # A detected face whose short side is below this fraction of the frame is a
    # small/distant "background face" — its presence flags a risky background.
    small_face_ratio: float = 0.04  # short-side fraction below this = risky face
    # Text protection (region-local, gated by content_aware + region_local). The
    # whole-image edge heuristic above only fires when the *frame* is edge-dense,
    # so a small sign/text block in a big photo gets no protection at all — yet
    # text is exactly what the AI upscale mangles (garbled glyphs). When on, a
    # coarse per-tile edge scan finds localized text-like clusters and stores
    # each as a sharp region patch (the same region_NNN.* mechanism as small
    # faces/hands, at region_scale), leaving bg_scale aggressive for the benign
    # rest. When the risky clusters cover too much of the frame
    # (text_region_max_frac — e.g. a document), patches aren't economical and
    # the whole-image raise above handles it as before.
    #   OPT-IN (default off), unlike the other region protections: this is still
    # the zero-download edge *proxy*, NOT OCR, and at tile granularity it cannot
    # tell text from benign-but-sharp organic content — measured on the real
    # corpus, default-on would patch fern/ridge clusters on landscapes (up to
    # ~24% of a frame) that AI reconstruction handles fine. A false patch is
    # only size waste (never corrupts output), but silently growing benign
    # photos' .fkeep breaks the defaults-must-not-change-benign-output rule, so
    # users with signage-heavy libraries opt in instead (the yunet/mediapipe
    # precedent: protective upgrades needing judgment are opt-in).
    protect_text: bool = False  # opt-in: patch text-like clusters (edge proxy)
    # Per-tile edge-pixel fraction above which a tile reads as text-like.
    # Calibrated on synthetic sign/text fixtures + the real corpus: sign tiles
    # measure ~0.06-0.23 (glyph gaps dilute a tile), benign-sharp organic
    # clusters mostly bail via text_region_max_frac at this level. Raising it
    # trades sign recall for fewer organic false patches.
    text_region_tile_threshold: float = 0.06
    # When the merged text clusters exceed this fraction of the frame area,
    # patching is not economical — fall back to the whole-image raise.
    text_region_max_frac: float = 0.30

    # Residual layer (opt-in "middle mode"). Stores what the downsample lost —
    # the signed delta between the original and the *decoded* stored background,
    # bicubic-upscaled back to size — as one extra member (residual.jxl,
    # offset-encoded uint8 at residual_scale resolution). On restore the delta is
    # added back to a plain bicubic upscale, so the background is real (lossy)
    # data instead of an AI hallucination; the AI upscale and GFPGAN are skipped
    # on that path (never replace real pixels with a hallucination). The honest
    # framing: it does NOT recover full fidelity (the residual is half-res +
    # lossy) and the .fkeep gets larger — the explicit trade. Off by default.
    residual: bool = False  # store the real high-frequency delta (CLI --residual)
    residual_scale: float = 0.5  # resolution the residual is stored at (1.0 = full)
    residual_quality: int = 60  # encode quality for the residual member (1-100)

    # iPhone HDR gain-map preservation (Phase 9). A modern iPhone HDR photo is
    # an 8-bit base plus an HDR *gain map* (imageio.load extracts it). When on
    # (the default — "just preserve it") and the source carried one, the gain
    # map is stored in the .fkeep (gainmap.jpg + manifest flag, 1.10.0+) so
    # `restore -f avif` can re-attach it and emit a real HDR AVIF. Sources
    # without a gain map are byte-identical either way. Compress-side and
    # output-affecting -> in index.settings_fingerprint.
    preserve_gain_map: bool = True
    # Restore-side: HDR headroom in stops used to rebuild the HDR alternate
    # when re-attaching the gain map (linear boost = 2^(headroom * gain)).
    # Apple's exact per-photo headroom lives in maker notes there is no offline
    # parser for; 3.0 matches the real-photo reference libavif derived from
    # Apple's own metadata. Restore-only -> NOT fingerprinted.
    gain_map_headroom: float = 3.0

    # Quality-targeted bg_scale (opt-in). When `quality_target` is set, instead
    # of compressing every photo at the fixed `bg_scale`, search the candidate
    # scales for the *most aggressive* one whose reconstructed background still
    # meets a target perceptual quality (LPIPS distance <= quality_target; lower
    # LPIPS = more similar). So each photo is compressed as hard as it can be
    # without looking wrong, rather than a one-size-fits-all scale. It is opt-in
    # (None = off, the fixed-scale behavior) and needs the [ai] extra for LPIPS —
    # if LPIPS is unavailable the search is skipped and the fixed `bg_scale` is
    # used (graceful degradation, offline-first preserved). The search estimates
    # restore quality with a fast bicubic upscale (no Real-ESRGAN needed at
    # *compress* time); the chosen scale is stored and the real restore may still
    # use AI. Composes with content-aware conservatism (it never picks a scale
    # *more* aggressive than the conservative floor).
    quality_target: Optional[float] = None  # target LPIPS; None = off (fixed bg_scale)
    # Candidate bg_scales the search walks, ascending (most aggressive first).
    quality_scale_candidates: list = field(
        default_factory=lambda: [0.125, 0.1667, 0.25, 0.3333, 0.5]
    )

    # Restore (AI super-resolution)
    model: str = "realesrgan-x4plus"
    # Real-ESRGAN tiles the upscale so a 24MP+ background never materializes a
    # full-resolution intermediate at once — `tile` is the tile side in pixels
    # (smaller = lower peak memory, slightly slower; 0 disables tiling) and
    # `tile_pad` is the overlap that hides tile seams. Restore-only knobs, so they
    # do not affect compress output and are *not* in index.settings_fingerprint
    # (like `model`/`face_enhance`). The bicubic fallback path is a single C-level
    # cv2.resize (one output buffer, no multi-copy), so tiling does not apply there.
    tile: int = 512  # Real-ESRGAN tile side in px (0 = no tiling)
    tile_pad: int = 10  # tile overlap padding to hide seams
    # Restore-only: run GFPGAN face restoration on the reconstructed background so
    # a face the *detector missed* (downsampled with the bg, then upscaled) doesn't
    # melt into something uncanny. Only the faces GFPGAN finds in the upscaled bg
    # are touched (soft-mask blended); detected faces are real crops composited on
    # top afterward, so they are never replaced. Needs the [ai] extra (gfpgan);
    # missing -> skipped gracefully. Does not affect compress output, so it is not
    # in index.settings_fingerprint.
    face_enhance: bool = True  # GFPGAN on reconstructed background faces (restore)
    # Restore-only: which face-restoration model `face_enhance` uses. "gfpgan"
    # (default, [ai] extra) is the established safety net; "codeformer" (opt-in
    # [codeformer] extra, used together with [ai]) is more robust on heavily
    # degraded faces and exposes a fidelity dial — but note its code/weights are
    # S-Lab License 1.0 (non-commercial), which the user accepts by installing
    # the extra. Missing package/weights -> enhancement is skipped with a warning
    # (never a silent switch back to gfpgan). Not in index.settings_fingerprint.
    face_enhance_backend: str = "gfpgan"  # gfpgan | codeformer
    # Restore-only: CodeFormer's fidelity weight w in [0,1] — higher stays closer
    # to the input face, lower lets the model "beautify" more. The
    # closer-to-original <-> prettier dial; ignored by the gfpgan backend.
    face_enhance_fidelity: float = 0.7  # CodeFormer w (codeformer backend only)
    # Restore-only: blend each restored face with the un-enhanced pixels at this
    # alpha (1.0 = full enhancement, today's behavior; ~0.6-0.8 softens the
    # "too-perfect face on a real photo" look; 0 = no visible enhancement).
    # Applies to BOTH backends at paste time. Not in index.settings_fingerprint.
    face_enhance_strength: float = 1.0  # restored-face blend alpha (0..1)
    # Restore-only: anchor the AI-upscaled background's low frequencies to the
    # stored background.jpg. Real-ESRGAN drifts in color/brightness/low-frequency
    # structure (it optimizes perceptual realism, not fidelity), but every spatial
    # frequency below the stored background's Nyquist is real measured data — so
    # the low band of the AI output is replaced with the reference's (pure
    # NumPy/OpenCV, zero new deps). Applied only when the AI upsampler actually
    # ran: the bicubic fallback is consistent with the stored background by
    # construction, and skipping keeps that path byte-identical. Does not affect
    # compress output, so it is not in index.settings_fingerprint (like
    # `model`/`face_enhance`/`tile`).
    restore_anchor: bool = True  # anchor AI-restored bg low frequencies to real data
    # Restore-only, experimental (default off): after anchoring, run N gentle
    # iterative back-projection steps pinning the *mid* band to the stored
    # background too (x <- x + lambda*up(bg - down(x))). Off by default because the
    # stored background carries JPEG q85 artifacts that strict consistency would
    # pull back into the restore. Same AI-path-only gating and fingerprint
    # exemption as restore_anchor.
    restore_backproject_iters: int = 0  # back-projection iterations (0 = off)
    # Restore-only: synthesize matched grain on the reconstructed background
    # before compositing. The real face crops / region patches carry natural
    # sensor noise + JPEG texture while the GAN/bicubic upscale is too smooth
    # ("plastic"), so even a perfectly feathered paste is findable by texture
    # discontinuity — the biggest visible tell. The grain level is estimated
    # from the real crops (robust MAD statistic) and applied as seeded,
    # deterministic mono/luma noise to BOTH the AI and bicubic paths (a bicubic
    # upscale is just as smooth). Skipped when the file has no crops (nothing
    # to mismatch). Does not affect compress output, so it is not in
    # index.settings_fingerprint (like `restore_anchor`).
    restore_grain: bool = True  # matched grain on the reconstructed bg (restore)
    blend_mode: str = "gaussian"  # gaussian | linear | poisson
    # The aggressive-mode preset this config was expanded from (None = no
    # preset; set by apply_preset, never by hand). Recorded in the .fkeep
    # manifest (settings.preset, 1.7.0+) so restore can auto-apply the preset's
    # restore-side knobs. Deliberately NOT in index.settings_fingerprint: the
    # expanded fields already are, so a preset and the same values set by hand
    # fingerprint identically.
    preset: Optional[str] = None

    def resolved_detector(self, detector: "DetectorConfig") -> "DetectorConfig":
        """The DetectorConfig aggressive mode should use.

        Starts from the shared ``detector`` config and applies any aggressive
        overrides that are set (non-None). Returns a *new* DetectorConfig so the
        caller's shared config is never mutated. This is the single place that
        resolves "inherit vs override", so the fingerprint (index.py) and the
        compressor agree on exactly which detector settings are in effect.
        """
        return DetectorConfig(
            backend=self.detector_backend
            if self.detector_backend is not None else detector.backend,
            confidence=self.detector_confidence
            if self.detector_confidence is not None else detector.confidence,
            padding=detector.padding,
            nms_iou=detector.nms_iou,
            min_size_ratio=self.detector_min_size_ratio
            if self.detector_min_size_ratio is not None else detector.min_size_ratio,
            max_aspect_ratio=detector.max_aspect_ratio,
            roi=detector.roi,
        )


@dataclass
class VideoConfig:
    """Faithful video re-encode settings (ROADMAP Phase 10).

    Videos in a compress run (a video file, or videos found in a folder) are
    faithfully re-encoded with SVT-AV1 into a standard ``.mp4`` — the same
    bargain as faithful photos: real pixels, the codec's own psychovisual bit
    allocation, opening the file *is* the restore. Aggressive mode deliberately
    does NOT apply to video (per-frame AI SR is computationally absurd and
    temporally unstable). Needs the external ``ffmpeg`` binary
    (``$FACEKEEP_FFMPEG`` -> PATH — the avifenc pattern, never a Python
    dependency); without it videos are skipped with an install hint and photos
    are unaffected. Videos always encode serially (SVT-AV1 already saturates
    the cores, so ``--jobs`` fan-out would only slow everything down).
    """

    # Include videos in compress runs. Off = photos only (the pre-Phase-10
    # behavior). The escape hatch exists because video encoding is *slow*
    # (~0.25x realtime for 4K on a desktop CPU — an overnight-batch feature),
    # so a quick photo pass must be one flag away: CLI `--no-videos`.
    enabled: bool = True
    # SVT-AV1 CRF (0-63, lower = better quality / larger). The default is the
    # conservative end of the measured visually-lossless band on real phone
    # clips; the VMAF gate below catches content it was never measured on.
    crf: int = DEFAULT_CRF
    # SVT-AV1 speed preset (0-13, lower = slower/smaller). The default is the
    # measured speed/quality point (~0.25x realtime for 4K).
    preset: int = DEFAULT_PRESET
    # Post-encode VMAF quality gate: score the encode against the source and
    # re-encode at a lower CRF when the per-frame 1%-low (p1) misses this
    # target ("the worst moments still look good"). None disables the gate —
    # scoring costs about as much as the encode itself on 4K. Needs libvmaf in
    # the ffmpeg build; a build without it skips the gate with a warning.
    vmaf_target: Optional[float] = DEFAULT_VMAF_TARGET
    # Opt-in sampled CRF auto-tune: probe a few short spans to find the highest
    # CRF that still meets vmaf_target, instead of the fixed `crf`. Costs ~6
    # short probe encodes per file before the real encode (the reason it is
    # opt-in); the gate then verifies the full file.
    auto_tune: bool = False
    # Skip sources that are already efficiently encoded (AV1, or a low
    # bits/pixel/frame) instead of burning hours adding a lossy generation for
    # nothing. Also what keeps our own outputs from being re-eaten on a re-run.
    skip_efficient: bool = True
    # Carry a Dolby Vision source's per-frame RPU (tone-mapping refinement)
    # into the AV1 output (DV profile 8.x -> 10.x) so a DV display renders the
    # same picture as the original — phone HDR clips are routinely DV 8.4, and
    # without the RPU they degrade to the plain HLG base (measured on a real
    # phone: visibly less saturated). Needs an ffmpeg whose libsvtav1 has
    # Dolby Vision support; an older build keeps the HLG base with a warning.
    preserve_dolby_vision: bool = True
    # Face-aware quality (the photo chroma/auto-tune analog): run the shared
    # face detector on a few sampled frames and, when faces are present, raise
    # the VMAF p1 target to face_vmaf_target — a clip's worst moments are held
    # to a higher bar exactly when people are in it. A missed face keeps the
    # base target (never worse than face-less); a false positive only costs
    # bytes. Needs the gate/auto-tune (and libvmaf) to have any effect.
    face_aware: bool = True
    # The raised p1 target for face-bearing clips (never lowers vmaf_target).
    face_vmaf_target: float = DEFAULT_FACE_VMAF_TARGET
    # Live-Photo pair policy (ROADMAP 11.1, measured): keep a Live Photo's
    # paired ~3 s .mov VERBATIM (copied, never re-encoded) when a same-stem
    # photo sibling sits beside it AND the .mov really carries Apple's pairing
    # key (com.apple.quicktime.content.identifier). The pairing key itself
    # survives our re-encode, but the still-image-time marker lives in a mebx
    # timed-metadata TRACK that the encode's -map drops structurally — a
    # re-encoded motion side is no longer a Live Photo to anything. The clip
    # is tiny, so keeping it costs little and preserves everything. False =
    # re-encode it like any video (the pairing key is carried; the loss is
    # documented).
    preserve_live_photos: bool = True


# ---------------------------------------------------------------------------
# Aggressive-mode presets: one-word intent -> a tuned knob bundle.
#
# A preset expands to ordinary config fields (dotted keys on FaceKeepConfig),
# applied as a layer between the dataclass defaults and anything explicitly
# written: defaults < preset < explicit YAML keys < explicit CLI flags. A
# hand-written field always beats a preset, regardless of where the preset was
# named (a CLI --preset only *chooses* the preset; it does not outrank an
# explicit YAML key). The preset *name* is not output-affecting on its own (it
# is not fingerprinted — the expanded fields already are), but it is recorded
# in the .fkeep manifest (settings.preset, 1.7.0+) so restore can auto-apply
# the preset's restore-side knobs.
#
# Builders are functions, not constant dicts, because `pretty` resolves "best
# available face enhancer" on the machine doing the work (compress records the
# name; the *restoring* machine re-resolves). The values are opinions tuned
# against `facekeep bench`; the names are the contract.


def _best_face_enhance_backend() -> str:
    """codeformer if importable, else gfpgan (the `pretty` preset's semantic).

    Unlike an explicit ``face_enhance_backend: codeformer`` (which must never
    silently fall back — the user named a model), a preset asks for "the best
    available enhancer", so probing is the documented behavior. Installing the
    [codeformer] extra is the S-Lab non-commercial license acceptance; a
    preset never installs anything.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("codeformer") is not None:
            return "codeformer"
    except (ImportError, ValueError):  # broken package metadata -> gfpgan
        pass
    return "gfpgan"


def _preset_ratio() -> dict:
    """Smallest .fkeep: hardest downsample + modern codecs for every member."""
    return {
        "aggressive.bg_scale": 0.125,
        "aggressive.bg_codec": "jxl",
        "aggressive.face_codec": "avif",
        "aggressive.face_quality": 90,
        "aggressive.detector_backend": "yunet",
    }


def _preset_pretty() -> dict:
    """Best-looking restore: clean SR input + the strongest face enhancement."""
    return {
        "aggressive.bg_scale": 0.25,
        "aggressive.bg_codec": "avif",
        "aggressive.detector_backend": "yunet",
        "aggressive.face_enhance_backend": _best_face_enhance_backend(),
        "aggressive.face_enhance_fidelity": 0.5,
        "aggressive.face_enhance_strength": 1.0,
    }


def _preset_fidelity() -> dict:
    """Closest to the original: store the real residual, keep people sharp."""
    return {
        "aggressive.residual": True,
        "aggressive.residual_quality": 75,
        "aggressive.bg_scale": 0.25,
        "aggressive.bg_codec": "jxl",
        "aggressive.face_quality": 98,
        "aggressive.detector_backend": "yunet",
        "detector.roi": "head_shoulders",
    }


def _preset_family() -> dict:
    """Never a melted face/hand: every recall/protection upgrade at once."""
    return {
        "aggressive.detector_backend": "yunet",
        "aggressive.protect_hands_backend": "mediapipe",
        "aggressive.small_face_ratio": 0.06,
        "detector.roi": "head_shoulders",
    }


def _preset_share() -> dict:
    """Compact + private for sending out: the ratio bundle + GPS stripping."""
    return {**_preset_ratio(), "strip_gps": True}


PRESETS = {
    "ratio": _preset_ratio,
    "pretty": _preset_pretty,
    "fidelity": _preset_fidelity,
    "family": _preset_family,
    "share": _preset_share,
}
PRESET_NAMES = tuple(PRESETS)

# The restore-side subset of preset keys (knobs that do not affect compress
# output). Restore auto-applies these from a manifest's settings.preset unless
# the user explicitly set them; compress-side keys are never re-applied at
# restore time (the .fkeep already embodies them).
RESTORE_SIDE_PRESET_KEYS = frozenset({
    "aggressive.face_enhance_backend",
    "aggressive.face_enhance_fidelity",
    "aggressive.face_enhance_strength",
})


def _set_dotted(config: "FaceKeepConfig", dotted: str, value) -> None:
    """Set a dotted config field (e.g. ``aggressive.bg_scale``), loudly."""
    obj = config
    *parents, attr = dotted.split(".")
    for p in parents:
        obj = getattr(obj, p)
    if not hasattr(obj, attr):
        raise ConfigError(f"Preset references unknown config field: {dotted}")
    setattr(obj, attr, value)


def apply_preset(config: "FaceKeepConfig", name: str,
                 explicit_keys: frozenset = frozenset()) -> None:
    """Apply preset ``name``'s expansion onto ``config`` (the preset layer).

    Dotted fields listed in ``explicit_keys`` are skipped — an explicitly
    written field always beats a preset (precedence: defaults < preset <
    explicit YAML < explicit CLI). Implies ``mode: aggressive``; combining a
    preset with an explicitly non-aggressive mode is a loud error, never a
    silent flip.
    """
    if name not in PRESETS:
        raise ConfigError(
            f"Unknown preset: {name!r} (expected one of: {', '.join(PRESET_NAMES)})"
        )
    if "mode" in explicit_keys and config.mode != "aggressive":
        raise ConfigError(
            f"Preset {name!r} implies aggressive mode; it cannot be combined "
            f"with an explicit mode={config.mode!r}"
        )
    for dotted, value in PRESETS[name]().items():
        if dotted in explicit_keys:
            continue
        _set_dotted(config, dotted, value)
    config.mode = "aggressive"
    config.aggressive.preset = name


def preset_restore_overrides(name: Optional[str],
                             explicit_keys: frozenset = frozenset()) -> dict:
    """Restore-side knob overrides for a manifest-recorded preset name.

    Returns ``{dotted_key: value}`` limited to ``RESTORE_SIDE_PRESET_KEYS``,
    minus anything the user explicitly configured (an explicit setting beats
    the manifest hint). Unknown or absent names return ``{}`` — restore stays
    tolerant by structure, so a .fkeep written by a future preset still
    restores on this reader.
    """
    builder = PRESETS.get(name) if name else None
    if builder is None:
        return {}
    return {
        k: v for k, v in builder().items()
        if k in RESTORE_SIDE_PRESET_KEYS and k not in explicit_keys
    }


def default_config_yaml() -> str:
    """A commented ``facekeep.yaml`` template at the shipped defaults.

    Used by ``facekeep init``. This is hand-authored (so it can carry the
    explanatory comments ``yaml.dump`` cannot) but the values mirror the
    dataclass defaults above — keep them in sync. It is intentionally a curated
    subset of the most-tuned knobs; every field is optional and falls back to its
    dataclass default, so omitting a line is the same as the default.
    """
    c = FaceKeepConfig()  # source the defaults so a drift is at least visible here
    f, a, d, v = c.faithful, c.aggressive, c.detector, c.video
    return f"""\
# FaceKeep configuration. Every key is optional; an omitted key uses its
# default (shown here). Delete what you don't need. See docs/architecture.md.
#
# Run:  facekeep compress photo.jpg            # faithful (default) -> photo.avif
#       facekeep compress photo.jpg -m aggressive   # -> photo.fkeep (needs restore)

# faithful (default): whole-image modern-codec encode, real pixels everywhere.
# aggressive: crop faces + downsample background + AI restore (a .fkeep).
# Left commented so `preset:` below (which implies aggressive) can be enabled
# without editing two lines — an explicit `mode:` in the same file as a
# `preset:` is a contradiction and errors loudly.
# mode: {c.mode}

# Optional aggressive-mode preset: one word that tunes the aggressive knobs
# toward a goal (implies `mode: aggressive`). Any key you write explicitly in
# this file still wins over the preset. CLI equivalent: `compress --preset`
# (the CLI flag, like `-m`, also overrides a `mode:` written here).
#   ratio:    smallest .fkeep (1/8 background, modern codecs everywhere)
#   pretty:   best-looking restore (best available face enhancer, beauty-leaning)
#   fidelity: closest to the original (stores the real residual; larger .fkeep)
#   family:   max face/hand protection (YuNet + MediaPipe hands + upper-body ROI)
#   share:    ratio + strip GPS (small and private, for sending out)
# preset: family

# Strip the GPS (location) EXIF from the output for privacy (both modes).
# Keeps date/camera/orientation. Off by default (EXIF round-trips unchanged).
strip_gps: {str(c.strip_gps).lower()}

# Face detection (shared by both modes).
detector:
  # haar: bundled, offline, zero-download (default). yunet: higher-accuracy DNN,
  # auto-downloads a small model on first use (opt-in). mediapipe: [detect] extra.
  backend: {d.backend}
  confidence: {d.confidence}    # min detection confidence (yunet/mediapipe)
  # face | head_shoulders | person: grow the high-priority region beyond the face.
  roi: {d.roi}

# Faithful mode (the default): one standard .avif/.jxl, no restore step.
faithful:
  # avif | jxl | webp | both. both keeps the smaller of avif/jxl per image; webp
  # is the maximum-compatibility fallback (opens anywhere, larger file).
  codec: {f.codec}
  # auto_tune (on by default) searches for a visually-lossless quality, so you
  # normally do NOT set `quality`. Setting `quality` (0-100) turns auto_tune off.
  auto_tune: {str(f.auto_tune).lower()}
  # quality: {f.quality}        # uncomment to pin a fixed quality (disables auto_tune)
  target_metric: {f.target_metric}   # ssimulacra2 (perceptual) | ssim
  chroma: {f.chroma}          # auto (4:4:4 when faces present) | 444 | 420
  # lossless: bit-exact archival (ignores quality/auto-tune, much larger file).
  # JXL is lossless natively; lossless AVIF needs the avifenc CLI, else falls
  # back to lossless JXL. Off by default.
  lossless: {str(f.lossless).lower()}
  # High-bit AVIF output depth (10 | 12); only used for a 16-bit source via the
  # external avifenc CLI, otherwise ignored. Never widens an 8-bit source.
  output_bit_depth: {f.output_bit_depth}

# Aggressive mode (only when `mode: aggressive`): extreme compression; the
# background is hallucinated on restore. Faces/hands/risky regions kept sharp.
aggressive:
  bg_scale: {a.bg_scale}        # background downsample (0.25 = 1/4)
  bg_codec: {a.bg_codec}        # jpg (default) | avif | jxl (4:2:0; cleaner SR input)
  face_quality: {a.face_quality}       # face-crop quality (>=100 -> lossless PNG)
  face_codec: {a.face_codec}        # jpg (default) | avif | jxl (4:4:4)
  # Store face/region crops at true high bit depth (10 | 12) so HDR sources
  # (e.g. iPhone HDR HEIC) survive — needs face_codec: avif + the avifenc CLI.
  # 8 = 8-bit container (default; background/residual stay 8-bit). Larger .fkeep.
  output_bit_depth: {a.output_bit_depth}
  content_aware: {str(a.content_aware).lower()}   # keep text/fine-detail/small-face regions sharp
  protect_hands: {str(a.protect_hands).lower()}   # keep hands sharp (region patches)
  # Opt-in: also patch localized text-like clusters (signage) so the AI never
  # repaints them. An edge proxy, not OCR — may also patch sharp organic
  # content (the size cost of a false patch), which is why it is off by default.
  protect_text: {str(a.protect_text).lower()}
  # Residual layer ("middle mode"): also store the real detail the downsample
  # lost, so restore adds it back instead of hallucinating (larger .fkeep,
  # background faithful-but-lossy; AI upscale + GFPGAN skipped on restore).
  residual: {str(a.residual).lower()}
  residual_scale: {a.residual_scale}   # resolution the residual is stored at
  residual_quality: {a.residual_quality}   # encode quality for the residual member
  # iPhone HDR: store the source's HDR gain map in the .fkeep when it has one;
  # `restore -f avif` re-attaches it -> a backward-compatible HDR AVIF (needs
  # the avifgainmaputil binary at restore; otherwise SDR + a warning).
  preserve_gain_map: {str(a.preserve_gain_map).lower()}
  gain_map_headroom: {a.gain_map_headroom}   # HDR headroom (stops) used at re-attach
  model: {a.model}   # restore super-resolution model ([ai] extra)
  # Face restoration of detector-missed background faces on restore.
  # gfpgan ([ai] extra) | codeformer ([codeformer] extra, with [ai]; S-Lab
  # non-commercial license; has a fidelity dial).
  face_enhance_backend: {a.face_enhance_backend}
  # CodeFormer fidelity w (0..1): higher = closer to the input face (gfpgan ignores).
  face_enhance_fidelity: {a.face_enhance_fidelity}
  # Blend restored faces with the un-enhanced pixels (1.0 = full enhancement;
  # ~0.6-0.8 softens the "too-perfect face" look). Both backends.
  face_enhance_strength: {a.face_enhance_strength}

# Faithful video re-encode (videos in a folder run, or a video file given
# directly). SVT-AV1 into a standard .mp4 - real pixels, plays anywhere modern,
# no restore step; HDR (10-bit/HLG) and VFR timestamps survive. Needs the
# external ffmpeg binary (FACEKEEP_FFMPEG or PATH; a build with libsvtav1);
# without it videos are skipped with a hint and photos are unaffected.
# Videos always encode serially (--jobs applies to photos only).
video:
  enabled: {str(v.enabled).lower()}      # include videos in compress runs (CLI --no-videos skips once)
  crf: {v.crf}             # SVT-AV1 CRF 0-63 (lower = better quality, larger file)
  preset: {v.preset}            # SVT-AV1 speed preset 0-13 (lower = slower, smaller)
  # Post-encode VMAF quality gate: re-encode at a lower CRF when the worst-1%
  # frame score (p1) misses this target. Scoring costs about as much as the
  # encode itself on 4K; set to null to skip verification. Needs libvmaf in
  # the ffmpeg build (skipped with a warning otherwise).
  vmaf_target: {v.vmaf_target}
  # Opt-in: probe short samples to find the highest CRF meeting vmaf_target
  # per clip (instead of the fixed crf above). Slower at compress time.
  auto_tune: {str(v.auto_tune).lower()}
  # Carry a Dolby Vision source's per-frame tone-mapping metadata (RPU) into
  # the AV1 output, so a DV display renders it like the original (phone HDR
  # clips are routinely DV). Needs a recent ffmpeg; older builds keep the
  # plain HDR base with a warning.
  preserve_dolby_vision: {str(v.preserve_dolby_vision).lower()}
  # Face-aware quality: detect faces on a few sampled frames and hold
  # face-bearing clips to the higher face_vmaf_target (people are the
  # subject). Face-less footage keeps vmaf_target.
  face_aware: {str(v.face_aware).lower()}
  face_vmaf_target: {v.face_vmaf_target}
  # Live Photos: keep a pair's ~3 s .mov verbatim (same-stem photo sibling +
  # Apple's pairing key) instead of re-encoding it — a re-encode drops the
  # still-image-time track, after which nothing treats the clip as a Live
  # Photo. false = re-encode it like any video.
  preserve_live_photos: {str(v.preserve_live_photos).lower()}
"""


@dataclass
class FaceKeepConfig:
    """Top-level configuration."""

    mode: str = "faithful"  # faithful | aggressive
    # Privacy: strip the GPS (location) EXIF IFD from the output. A shared,
    # both-mode export concern (faithful re-embeds EXIF in the encoded file;
    # aggressive stores it as exif.bin and re-embeds on restore), applied once at
    # load time so both inherit it. Off by default → EXIF round-trips byte-for-byte
    # as before; only the GPS IFD is dropped when on (date/camera/orientation kept).
    strip_gps: bool = False
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    faithful: FaithfulConfig = field(default_factory=FaithfulConfig)
    aggressive: AggressiveConfig = field(default_factory=AggressiveConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    def validate(self) -> None:
        """Validate configuration values, raising ConfigError on problems."""
        if self.mode not in ("faithful", "aggressive"):
            raise ConfigError(f"Unknown mode: {self.mode!r}")
        # Accept the built-in backends (haar/yunet/mediapipe) *and* any backend a
        # plugin registered via detector.register_detector (the custom-detector
        # hook). Lazy import so config has no hard dependency on detector and a
        # plugin registered at runtime is honoured here.
        from .detector import is_known_backend
        if not is_known_backend(self.detector.backend):
            raise ConfigError(f"Unknown detector backend: {self.detector.backend!r}")
        if not 0 <= self.detector.nms_iou <= 1:
            raise ConfigError("detector.nms_iou must be between 0 and 1")
        if self.detector.min_size_ratio < 0:
            raise ConfigError("detector.min_size_ratio must be >= 0")
        if self.detector.max_aspect_ratio < 1:
            raise ConfigError("detector.max_aspect_ratio must be >= 1")
        if self.detector.roi not in ("face", "head_shoulders", "person"):
            raise ConfigError(
                f"Unknown detector.roi: {self.detector.roi!r} "
                "(expected 'face', 'head_shoulders', or 'person')"
            )
        if self.faithful.codec not in ("avif", "jxl", "webp", "both"):
            raise ConfigError(f"Unknown codec: {self.faithful.codec!r}")
        if not 0 <= self.faithful.quality <= 100:
            raise ConfigError("faithful.quality must be 0-100")
        if self.faithful.output_bit_depth not in (10, 12):
            raise ConfigError(
                "faithful.output_bit_depth must be 10 or 12 (high-bit AVIF "
                "output only; an 8-bit source is never widened)"
            )
        if self.faithful.target_metric not in ("ssim", "ssimulacra2"):
            raise ConfigError(
                f"Unknown faithful.target_metric: {self.faithful.target_metric!r} "
                "(expected 'ssim' or 'ssimulacra2')"
            )
        if not 0.05 <= self.aggressive.bg_scale <= 1.0:
            raise ConfigError("aggressive.bg_scale must be between 0.05 and 1.0")
        if self.aggressive.face_codec not in ("jpg", "avif", "jxl"):
            raise ConfigError(
                f"Unknown aggressive.face_codec: {self.aggressive.face_codec!r}"
            )
        if self.aggressive.bg_codec not in ("jpg", "avif", "jxl"):
            raise ConfigError(
                f"Unknown aggressive.bg_codec: {self.aggressive.bg_codec!r}"
            )
        if self.aggressive.output_bit_depth not in (8, 10, 12):
            raise ConfigError(
                "aggressive.output_bit_depth must be 8 (8-bit container), 10, or 12 "
                "(high-bit crops via the avifenc CLI; needs face_codec='avif')"
            )
        # None (inherit) or any built-in/registered custom backend (same hook as
        # the shared detector.backend above), so a plugin can also be the
        # aggressive-mode override.
        if self.aggressive.detector_backend is not None and not is_known_backend(
            self.aggressive.detector_backend
        ):
            raise ConfigError(
                "aggressive.detector_backend must be None, a built-in "
                "('haar'/'yunet'/'mediapipe'), or a registered custom backend"
            )
        if (
            self.aggressive.detector_min_size_ratio is not None
            and self.aggressive.detector_min_size_ratio < 0
        ):
            raise ConfigError("aggressive.detector_min_size_ratio must be >= 0")
        if not 0.05 <= self.aggressive.conservative_bg_scale <= 1.0:
            raise ConfigError(
                "aggressive.conservative_bg_scale must be between 0.05 and 1.0"
            )
        if not 0 <= self.aggressive.text_edge_threshold <= 1:
            raise ConfigError("aggressive.text_edge_threshold must be between 0 and 1")
        if not 0 <= self.aggressive.small_face_ratio <= 1:
            raise ConfigError("aggressive.small_face_ratio must be between 0 and 1")
        if not 0 < self.aggressive.text_region_tile_threshold <= 1:
            raise ConfigError(
                "aggressive.text_region_tile_threshold must be between "
                "0 (exclusive) and 1"
            )
        if not 0 < self.aggressive.text_region_max_frac <= 1:
            raise ConfigError(
                "aggressive.text_region_max_frac must be between 0 (exclusive) and 1"
            )
        if not 0 < self.aggressive.region_scale <= 1.0:
            raise ConfigError(
                "aggressive.region_scale must be between 0 (exclusive) and 1.0"
            )
        if not 0 < self.aggressive.residual_scale <= 1.0:
            raise ConfigError(
                "aggressive.residual_scale must be between 0 (exclusive) and 1.0"
            )
        if not 1 <= self.aggressive.residual_quality <= 100:
            raise ConfigError(
                "aggressive.residual_quality must be between 1 and 100"
            )
        if not 0 < self.aggressive.gain_map_headroom <= 6:
            raise ConfigError(
                "aggressive.gain_map_headroom must be between 0 (exclusive) "
                "and 6 stops"
            )
        if self.aggressive.protect_hands_backend not in (None, "mediapipe"):
            raise ConfigError(
                "aggressive.protect_hands_backend must be None (offline geometry) "
                "or 'mediapipe' (no hand Haar cascade exists)"
            )
        if not 0 < self.aggressive.hand_zone_scale <= 1.0:
            raise ConfigError(
                "aggressive.hand_zone_scale must be between 0 (exclusive) and 1.0"
            )
        if not 0 < self.aggressive.hand_zone_max_frac <= 1.0:
            raise ConfigError(
                "aggressive.hand_zone_max_frac must be between 0 (exclusive) and 1.0"
            )
        if not 0 <= self.aggressive.hand_detect_confidence <= 1:
            raise ConfigError(
                "aggressive.hand_detect_confidence must be between 0 and 1"
            )
        if self.aggressive.hand_detect_max_hands < 1:
            raise ConfigError("aggressive.hand_detect_max_hands must be >= 1")
        if self.aggressive.hand_detect_long_side < 0:
            raise ConfigError("aggressive.hand_detect_long_side must be >= 0")
        if self.aggressive.hand_detect_padding < 1.0:
            raise ConfigError("aggressive.hand_detect_padding must be >= 1.0")
        if (
            self.aggressive.quality_target is not None
            and self.aggressive.quality_target <= 0
        ):
            raise ConfigError("aggressive.quality_target must be > 0 (or None to disable)")
        for s in self.aggressive.quality_scale_candidates:
            if not 0.05 <= s <= 1.0:
                raise ConfigError(
                    "aggressive.quality_scale_candidates entries must be "
                    "between 0.05 and 1.0"
                )
        if self.aggressive.tile < 0:
            raise ConfigError("aggressive.tile must be >= 0")
        if self.aggressive.tile_pad < 0:
            raise ConfigError("aggressive.tile_pad must be >= 0")
        if self.aggressive.restore_backproject_iters < 0:
            raise ConfigError("aggressive.restore_backproject_iters must be >= 0")
        if self.aggressive.face_enhance_backend not in ("gfpgan", "codeformer"):
            raise ConfigError(
                "aggressive.face_enhance_backend must be 'gfpgan' or 'codeformer', "
                f"got {self.aggressive.face_enhance_backend!r}"
            )
        if not 0 <= self.aggressive.face_enhance_fidelity <= 1:
            raise ConfigError(
                "aggressive.face_enhance_fidelity must be between 0 and 1"
            )
        if not 0 <= self.aggressive.face_enhance_strength <= 1:
            raise ConfigError(
                "aggressive.face_enhance_strength must be between 0 and 1"
            )
        if not 0 <= self.video.crf <= 63:
            raise ConfigError("video.crf must be 0-63 (SVT-AV1 CRF range)")
        if not 0 <= self.video.preset <= 13:
            raise ConfigError("video.preset must be 0-13 (SVT-AV1 preset range)")
        if not 0 < self.video.face_vmaf_target <= 100:
            raise ConfigError(
                "video.face_vmaf_target must be between 0 (exclusive) and 100 "
                "(VMAF score range)"
            )
        if self.video.vmaf_target is not None and not (
            0 < self.video.vmaf_target <= 100
        ):
            raise ConfigError(
                "video.vmaf_target must be between 0 (exclusive) and 100 "
                "(or null to disable the quality gate)"
            )
        if self.aggressive.preset is not None:
            if self.aggressive.preset not in PRESETS:
                raise ConfigError(
                    f"Unknown preset: {self.aggressive.preset!r} "
                    f"(expected one of: {', '.join(PRESET_NAMES)})"
                )
            # A preset is an aggressive-mode bundle; silently flipping the mode
            # under an explicit faithful request would be worse than erroring.
            if self.mode != "aggressive":
                raise ConfigError(
                    f"Preset {self.aggressive.preset!r} implies aggressive "
                    f"mode; it cannot be combined with mode={self.mode!r}"
                )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FaceKeepConfig":
        """Load config from YAML. Falls back to defaults if not found."""
        if path is None:
            for candidate in (
                Path("facekeep.yaml"),
                Path("facekeep.yml"),
                Path.home() / ".config" / "facekeep" / "config.yaml",
            ):
                if candidate.exists():
                    path = candidate
                    break

        config = cls()
        if path is None or not path.exists():
            config.explicit_keys = frozenset()
            config.validate()
            return config

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Record which dotted keys the YAML explicitly sets. An explicit key
        # outranks a preset expansion (precedence: defaults < preset < explicit
        # YAML < explicit CLI flags); the CLI's --preset layer and the restore
        # auto-apply read this set off the returned config for the same reason.
        explicit = set()
        for top in ("mode", "strip_gps"):
            if top in data:
                explicit.add(top)
        for section in ("detector", "faithful", "aggressive", "video"):
            if isinstance(data.get(section), dict):
                for key in data[section]:
                    explicit.add(f"{section}.{key}")

        # Preset layer FIRST, the explicit YAML keys after (so they win). The
        # name may be the top-level `preset:` key or `aggressive.preset:` (the
        # form a saved config round-trips through — a saved file also carries
        # every expanded field explicitly, so re-expanding is a no-op there).
        preset_name = data.get("preset")
        if not preset_name and isinstance(data.get("aggressive"), dict):
            preset_name = data["aggressive"].get("preset")
        if preset_name:
            apply_preset(config, preset_name)

        if "mode" in data:
            config.mode = data["mode"]
        if "strip_gps" in data:
            config.strip_gps = data["strip_gps"]

        for section, obj in (
            ("detector", config.detector),
            ("faithful", config.faithful),
            ("aggressive", config.aggressive),
            ("video", config.video),
        ):
            if section in data and isinstance(data[section], dict):
                for key, value in data[section].items():
                    if hasattr(obj, key):
                        setattr(obj, key, value)

        config.explicit_keys = frozenset(explicit)
        config.validate()
        return config

    def save(self, path: Path) -> None:
        """Save config to YAML."""
        data = {
            "mode": self.mode,
            "strip_gps": self.strip_gps,
            "detector": vars(self.detector),
            "faithful": vars(self.faithful),
            "aggressive": vars(self.aggressive),
            "video": vars(self.video),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
