# Architecture

This document explains how FaceKeep is designed and, more importantly, *why* —
especially why the default mode keeps real pixels instead of downsampling and
reconstructing with AI.

## The two goals are in tension

FaceKeep aims for two things at once:

1. **Strong compression** — smaller files.
2. **Faithful restoration** — the result is imperceptibly different from the
   original.

These pull against each other, and how you resolve the tension determines the
whole architecture. FaceKeep offers two modes for two different answers.

## Why "downsample + AI restore" is the wrong default

An intuitive design is: keep faces sharp, shrink the background, and use AI
super-resolution to "restore" the background on the way back. It compresses
aggressively. But it fails goal (2) for a fundamental reason:

> **Information that has been discarded cannot be recovered.** A super-resolution
> model does not reconstruct the original background; it *generates* a new,
> statistically plausible one. The output looks sharp, but it is a believable
> fiction, not the original.

For smooth, regular content (sky, bokeh, foliage) the invented detail is often
imperceptible. For structured content (text, signage, tile patterns, distant
faces, architectural lines) it visibly diverges on close inspection. There is
also a second problem: cropping a high-quality face and compositing it over a
reconstructed background creates a **seam** at the boundary — itself a source of
artifacts.

Benchmarks during design confirmed both the promise and the cost: downsampling
to 1/4 and upscaling can reach very high ratios, but fidelity to the *original*
drops (SSIM falls well below the visually-lossless threshold on detailed
content), and the gap is structural, not a tuning issue.

## Why a modern codec is the right default

The better answer to "small *and* faithful" is to **keep the real background and
just compress it more efficiently**, using a modern codec:

- **AVIF** (AV1 intra) and **JPEG XL** are ~2–3× more efficient than JPEG at the
  same perceptual quality.
- Their encoders already perform **adaptive quantization**: they spend more bits
  on visually important, high-detail regions (faces, edges) and fewer on flat
  areas. This is region-aware quality *for free*, in the frequency domain, with
  **no seam** because it is one continuous image.
- The output is a **standard file** (`.avif` / `.jxl`) that opens in any modern
  viewer. "Restoring" is just opening it — there is no custom container and no
  reconstruction step that could drift from the original.

A design experiment tried to "help" the codec by manually reducing background
detail (edge-preserving blur outside faces) before encoding. It **hurt** overall
fidelity for only modest size gains — because any manual modification of the
background is itself a loss, and the codec's internal perceptual bit allocation
is smarter than a crude spatial blur. **Lesson: in faithful mode, don't
manipulate regions manually — let the codec do it.**

## Faithful mode (default)

```
input image
   │
   ▼
imageio.load()         apply EXIF orientation, preserve EXIF + ICC bytes  (BGR)
   │
   ▼
detector.detect()      faces -> chroma decision (4:4:4 on faces) +
   │                            optional quality auto-tune target
   ▼
encoders.encode()      whole image -> AVIF (or JXL), adaptive quantization
   │
   ▼
encoders.write_encoded()   single standard .avif / .jxl file
```

With `codec: both` the encode step trial-encodes the image as *both* AVIF and JXL
(each at its own auto-tuned quality) and keeps the smaller — per-image codec
choice — while still writing one standard file in the winning codec.

Faces influence two things only:

1. **Chroma subsampling.** With faces present, use 4:4:4 (no chroma
   subsampling) so skin tone and lip color stay crisp; otherwise 4:2:0 is
   smaller.
2. **Auto-tuning (on by default).** Binary-search the lowest quality whose
   *face region* meets a perceptual target (faces are the acceptance criterion;
   the background rides along at the same codec quality, where adaptive
   quantization keeps it efficient). The default acceptance metric is the
   perceptual **SSIMULACRA2** (~90 = visually lossless); a plain install lacking
   that optional package falls back to SSIM (threshold re-based). An explicit
   `--quality` turns auto-tune off (`--no-auto-tune` too), so a chosen quality is
   honored directly.

Detection failure never blocks encoding — faithful mode degrades gracefully to a
plain whole-image encode.

## Aggressive mode (optional)

For users who explicitly want extreme compression and accept a reconstructed
background:

```
input → detect faces → crop faces (original quality; JPEG q95 by default, or
                        AVIF/JXL 4:4:4 via aggressive.face_codec)
                      → keep risky regions sharp (region-local conservatism:
                        a patch + mask per small/distant-face region AND per hand —
                        C1 face-geometry default / opt-in C2 MediaPipe hand detect)
                      → downsample background (INTER_AREA)
                      → optionally store the residual the downsample lost
                        (aggressive.residual, off by default: original minus the
                        bicubic upscale of the *decoded* background, half-res,
                        offset-encoded as residual.jxl — the "middle mode")
                      → pack into .fkeep (a ZIP: manifest + bg + faces + masks +
                        region patches [+ residual] [+ HDR gain map] + EXIF +
                        ICC profile)

restore: .fkeep → AI super-resolution upscale background (Real-ESRGAN;
                  bicubic fallback) → re-anchor the upscale's low frequencies
                  to the stored background (AI path only — real measured tones
                  over invented detail; aggressive.restore_anchor, default on)
                  → GFPGAN-restore any *missed* faces in the
                  upscaled background (opt-in via [ai]; soft-mask blended, skipped
                  if absent) → add matched grain to the reconstructed background
                  (estimated from the real crops, so the smooth upscale stops
                  being findable by texture; aggressive.restore_grain, default
                  on; both AI and bicubic paths)
                  → composite region patches, then real face crops, with
                  feathered soft masks → full-resolution standard image (.jpg
                  default via Pillow, or .avif/.jxl; EXIF *and* ICC color profile
                  re-embedded) — so a .fkeep is never a dead end
                  → (1.10.0+, .avif output only) re-attach the stored iPhone HDR
                  gain map via avifgainmaputil combine → a backward-compatible
                  HDR AVIF (SDR viewers see the base; missing binary/other
                  formats → SDR + warn)

         (residual files, 1.6.0+: the background is instead reconstructed from
          REAL data — bicubic upscale + the stored residual delta — and the AI
          upscale, anchor, and GFPGAN are all skipped: they exist to make
          hallucination plausible, and real pixels are never replaced by a
          hallucination. Grain and the composites apply unchanged.)
```

**Presets: one word instead of twenty knobs.** Aggressive mode's knobs (above
and below) are individually documented but nobody should have to read them all;
`--preset <name>` (or YAML `preset:`) names the *goal* and expands to a tuned
bundle, applied as a layer with fixed precedence — defaults < preset < explicit
YAML keys < explicit CLI flags — so any hand-written field still wins. Five
presets: **`ratio`** (smallest `.fkeep`: 1/8 background, JXL background + AVIF
crops, YuNet recall), **`pretty`** (best-looking restore: clean AVIF SR input;
restore uses the *best available* face enhancer — CodeFormer iff its opt-in
extra is installed, else GFPGAN — at a beauty-leaning fidelity), **`fidelity`**
(closest to the original: the residual layer + higher crop quality +
upper-body ROI), **`family`** (never a melted face/hand: YuNet + MediaPipe
hand detection + upper-body ROI + a wider small-face net), and **`share`**
(`ratio` + GPS stripping, for sending out). A preset implies aggressive mode —
a CLI `--preset` overrides a config file's `mode: faithful` exactly like
`-m aggressive` would, while a *same-level* contradiction (`--preset` with
`-m faithful`, or `preset:` beside `mode: faithful` in one YAML) errors
loudly. It selects opt-in upgrades without
weakening any graceful-degradation chain (offline still falls back Haar/C1),
and is recorded in the manifest (`settings.preset`, 1.7.0+) so `restore`
auto-applies its restore-side knobs unless explicitly overridden. The preset
name is not fingerprinted — the expanded fields already are.

**Protecting every face matters more here than in faithful mode.** A face the
detector *misses* is downsampled with the background and then reconstructed by
the AI on restore — which can come back uncanny, the worst failure for a family
tool (emotionally worse than a soft background). So aggressive mode biases
toward recall via a per-mode detector override on `AggressiveConfig`
(`detector_backend` / `detector_confidence` / `detector_min_size_ratio`, each
`None` = inherit the shared `DetectorConfig`):

- **Relaxed small-face thresholds by default** (`min_size_ratio` 0.05 → 0.02,
  `confidence` 0.6 → 0.5) so a distant face is kept and cropped at original
  quality rather than discarded by the false-positive size filter. These need no
  model download, so they help even on the default Haar — missing a real face is
  far worse here than keeping one extra crop.
- **YuNet as an opt-in upgrade** (`detector_backend: yunet`) for the best
  small/profile/background recall. It is *opt-in, not the default*: the default
  `detector_backend` is `None` → inherit Haar, so the default aggressive run
  stays offline and zero-download (principle 4 below). YuNet auto-downloads its
  model and falls back to Haar offline.

`AggressiveConfig.resolved_detector()` is the single place that merges the
override onto the shared config, so the compressor and the incremental index's
`settings_fingerprint` always agree on the *effective* detector. (Faithful mode
does not use this override — it reads the shared `DetectorConfig` directly,
where faces only pick chroma and the auto-tune acceptance region.)

**Content-aware conservatism** is the second guardrail (on by default,
`aggressive.content_aware`). The aggressive downsample is safe on *benign*
content (sky, bokeh, foliage, plain walls) but mangles content the AI cannot
honestly reconstruct: text/signage, fine regular structure, and small/distant
background faces. Risk is judged by two zero-download signals: a small/distant
detected face (short side below `small_face_ratio` of the frame), and an
**edge-density heuristic** (the fraction of the frame on strong edges after a
light blur, above `text_edge_threshold`). The edge heuristic is an honest
*proxy*, **not** real OCR/text detection — the blur is what keeps benign fine
content (camera noise, foliage) from reading as risky while sharp text/structure
still trips it.

The guardrail acts in one of two ways, depending on `aggressive.region_local`
(on by default):

- **Region-local** (the default, for the small/distant-face signal): the benign
  majority of the frame keeps the aggressive `bg_scale`, and only the *risky
  region* — the background around a small/distant face — is kept sharp, stored as
  a near-original-resolution patch (`region_NNN.*` + mask + bbox) and composited
  back on restore. This reuses the exact face-crop mechanism (a region patch is a
  non-face crop), so the global background and its restore math are unchanged and
  the new `.fkeep` members are purely additive (manifest `1.3.0`). `region_scale`
  tunes the patch resolution.
- **Whole-image** (the fallback, and still how the *edge-density/text* signal
  behaves): the single `bg_scale` is **raised toward `conservative_bg_scale`** —
  the whole background is compressed less — using the *same lever* the no-face
  fallback pulls, and only ever raising the scale (it composes with, never undoes,
  the no-face decision). With `region_local` off, the small-face signal also takes
  this path.

**Text-region localization is available but OPT-IN** (`aggressive.protect_text`,
default off). The whole-image edge signal only fires on a frame-wide ratio, so a
localized sign in a big photo gets no protection at all — yet garbled signage is
one of the mode's worst visible failures. When opted in, a coarse per-tile scan
over the same edge map marks text-like tiles (`text_region_tile_threshold`),
merges adjacent ones into clusters, and stores each cluster as an ordinary
region patch; document-like content (clusters over `text_region_max_frac` of the
frame) bails back to the whole-image raise, and emitted patches suppress that
raise so a risk is never charged twice. It is opt-in — unlike the small-face and
hand protections — for an honest, *measured* reason: the zero-download edge
proxy cannot tell text from benign-but-sharp organic content at tile granularity
(on the real corpus it patches fern/ridge clusters on landscapes, pure size
waste on content the AI reconstructs convincingly), and a default must not
silently grow benign photos' files. Users with signage-heavy libraries opt in;
a false patch never corrupts output, it only costs bytes.

**Remaining scope:** a true per-tile *scale map* of the background stays a
tracked follow-up. All of these switches/thresholds (`content_aware`,
`conservative_bg_scale`, `text_edge_threshold`, `small_face_ratio`,
`region_local`, `region_scale`, `protect_text`, `text_region_tile_threshold`,
`text_region_max_frac`) are output-affecting, so they feed
`settings_fingerprint`.

**Hands are protected by the same region-local mechanism** (`aggressive.protect_hands`,
on by default). Hands aren't faces, so the detector never finds them; left alone
they ride the `bg_scale` downsample and the AI upscaler melts their thin finger
structure on restore (the user-visible defect this addresses). So `compress_photo`
emits hand regions as ordinary region patches — same `region_NNN.*` + mask + bbox,
de-duped against the small-face regions, **no `.fkeep` format change** (a hand
region is just another `regions[]` entry; restore is generic over them). Because
OpenCV ships no hand cascade, hand detection is **tiered**, exactly mirroring the
Haar-default / YuNet-opt-in detector split:

- **C1 (default, offline, zero-download):** infer a hand-likely band below/beside
  each detected face from body proportions (`_hand_zones_from_faces`; the torso
  centre is deliberately excluded so only the *hands* are protected, not the whole
  upper body — far less compression cost than `roi: person`). A probabilistic
  guess: overhead/off-body/face-less hands are missed. Because it is a *guess*, a
  dense group/family photo stacks one guess per face until they cover a large
  slice of the frame (mostly torsos/laps with no hands) — a 5-face group once made
  a `.fkeep` *larger than the source*. So C1 zones are post-processed
  (`_c1_hand_zones`): overlapping bands are **merged** (so adjacent faces don't
  store the same pixels twice), and the whole set is **dropped** once its coverage
  exceeds `aggressive.hand_zone_max_frac` (default 0.30, corpus-tuned: a ~43%
  5-face group bails to no C1 protection — it still compresses via the face crops
  + whole-image conservatism — while a ~22% 3-face photo is kept). This guard is
  C1-only; C2's tight real detections are never capped.
- **C2 (opt-in upgrade):** real MediaPipe Hand Landmarker detection
  (`detector.HandDetector`, `[detect]` extra, model via `models.ensure_weights`)
  gives tight per-hand boxes and catches off-body hands. Because MediaPipe's palm
  detector is trained for phone-sized frames, `detect_hands` **downscales the
  detection input** to `hand_detect_long_side` (default 1280) before running the
  landmarker — on a 12 MP photo a hand is otherwise too small a fraction to detect
  (this ~doubled recall on the real repro photo); landmark coords are normalized, so
  the boxes still come out in the original frame. Its knobs
  (`hand_detect_confidence`/`_max_hands`/`_long_side`/`_padding`) lean toward
  recall — a missed hand is the failure that matters; a false hand only protects an
  extra patch (larger file), never corrupts output. `detect_hands` also
  **NMS-de-duplicates** its boxes (MediaPipe can return two near-identical boxes for
  one physical hand), so each hand yields one patch. Graceful degradation
  (principle 4): missing package/model/offline → fall back to C1; a quiet detector
  → C1's geometric guess; inference error → no hand regions, never a crash.
  **Honest limit:** a hand *heavily occluded by a held object* (in the repro photo,
  a hand cradling a snake) is missed at any confidence/resolution — the palm
  detector's reach, not a tuning gap; we accept the miss rather than drop confidence
  far enough to invent false hands elsewhere.

Hand detection is parent-process-only (the landmarker isn't picklable), so
`--jobs` workers use C1 — the same discipline as the detection cache. The new
fields (`protect_hands`, `protect_hands_backend`, `hand_zone_scale`,
`hand_zone_max_frac`, and the C2 tuning knobs `hand_detect_*`) feed
`settings_fingerprint`; the C1 geometry/merge factors are module constants. **Honest limitation:** the C1 default is a body-proportion
*guess*, not detection — for precise off-body hands a user opts into C2.

**Quality-targeted `bg_scale`** is the opt-in alternative to a single fixed
scale (`aggressive.quality_target`, off by default). Rather than compressing
every photo at the configured `bg_scale`, `compressor._search_bg_scale` walks
`aggressive.quality_scale_candidates` (ascending, most aggressive first) and
chooses the *most aggressive* scale whose **reconstructed** background still
meets a target perceptual quality — LPIPS distance `<= quality_target` (lower =
more similar), the right metric for a hallucinated-but-plausible background
(SSIM is not; see the metrics component). So each photo is compressed as hard as
it can be without looking *wrong*, instead of one-size-fits-all. The search
estimates restore quality with a fast **bicubic** upscale (downsample → bicubic
back to size), not a full Real-ESRGAN restore: that keeps the search **offline
and cheap at compress time** and is a conservative proxy (real AI restore looks
at least as good), so the chosen scale errs toward quality; the stored scale is
the searched value and the real restore may still use AI. It **replaces the
fixed `bg_scale` baseline** (so it may compress harder than `bg_scale`) but
**never drops below the conservative floor** the no-face / content-aware logic
raised — `compress_photo` tracks that floor separately and passes it in, so the
"only ever stay at/above a real protection" invariant holds. It changes **no**
`.fkeep` member (it only picks the existing single `effective_bg_scale`), and
degrades gracefully: `quality_target` set but LPIPS unavailable (the `[ai]`
extra absent) → the search is skipped and the fixed `bg_scale` is used (offline-
first preserved). `quality_target` and `quality_scale_candidates` are
output-affecting, so they also feed `settings_fingerprint`.

**Low-frequency anchoring** (`aggressive.restore_anchor`, on by default) is the
restore-side *fidelity* lever. Real-ESRGAN optimizes perceptual realism, not
pixel accuracy, so its output drifts in color/brightness/low-frequency structure
vs the real photo — but the stored `background.jpg` *is* a real measurement:
every spatial frequency below its Nyquist is data, not guesswork. So after the
AI upscale, the result's low band is replaced with the reference's
(`restorer._anchor_low_frequencies`: `out = sr − blur(sr, σ) + blur(bicubic(bg),
σ)`, with σ derived from the upscale factor so only certainly-measured
frequencies are transplanted — erring toward "swap less", since a wider band
would pull the stored JPEG's artifacts back in). The AI's invented high-frequency
detail is kept; its tonal drift is corrected with real data. **It runs only when
the AI upsampler actually ran** (`_upscale_background` reports `(out, used_ai)`):
the bicubic fallback is consistent with the stored background by construction —
anchoring it is an identity up to rounding — and gating keeps that path
byte-identical, so the bicubic-based aggressive regression lock is untouched. An
optional second half, `aggressive.restore_backproject_iters` (default 0 = off),
adds gentle iterative back-projection (`x ← x + λ·up(bg − down(x))`) pinning the
*mid* band too; it is off by default because the stored background carries JPEG
q85 artifacts that strict consistency would pull back in. Both knobs are
**restore-only**: no `.fkeep`/manifest change, every existing `.fkeep` benefits
immediately, and neither is in `settings_fingerprint` (like
`model`/`face_enhance`/`tile`). Pipeline order: upscale → **anchor** (→
back-project) → GFPGAN background faces → composite; `preview()` is untouched
(pure bicubic — nothing to anchor).

**Grain matching** (`aggressive.restore_grain`, on by default) is the
restore-side *seamlessness* lever — it fixes the biggest visible tell. The
composite mixes *real* pixels (face crops / region patches, carrying natural
sensor noise and JPEG texture) with a GAN/bicubic background that is too smooth
("plastic"), so even a perfectly feathered paste is findable by texture
discontinuity — and GFPGAN-restored background faces are smooth the same way.
So restore estimates the grain level from the real crops
(`restorer._estimate_grain_sigma`: the luma high-frequency residual measured
with **MAD × 1.4826, not a std** — real edges are sparse outliers that would
inflate a std far above the noise floor; medianed across crops, preferring face
crops and falling back to region patches) and adds matched grain to the
reconstructed background before compositing (`restorer._apply_grain`: one
**seeded** Gaussian field, lightly blurred so it reads as grain rather than
salt-and-pepper and renormalized back to the estimated strength, added
identically to all three BGR channels — mono/luma grain, since chroma noise
looks wrong). The fixed seed keeps restore **deterministic**: the same `.fkeep`
restores to the same bytes every run. Unlike the anchor it applies to **both**
the AI and bicubic paths (a bicubic upscale is just as smooth) and is skipped
only when the file has no crops at all (no real pixels → no mismatch to hide).
Pipeline order: upscale → anchor → GFPGAN background faces → **grain** →
composite; `preview()` skips it (speed — the GFPGAN precedent), which is also
why the bench/corpus-lock bicubic *proxy* (preview-based) cannot see it —
measuring 8.2 takes a real `restore()` (on the corpus film photo it moved
restore LPIPS 0.42 → 0.16, toward the grainy original). Restore-only: no
`.fkeep`/manifest change, every existing `.fkeep` benefits immediately, and the
knob is not in `settings_fingerprint`.

**The residual layer** (`aggressive.residual`, CLI `--residual`, off by
default) is the opt-in "middle mode" between aggressive and faithful — the
biggest *fidelity* lever, and a compress-side one. Aggressive mode's ceiling is
information-theoretic: detail the downsample discarded can only be invented
back. The residual stores that detail: at compress time the just-encoded
background member is **decoded back** and the signed delta
`original − bicubic(decoded bg)` is computed — against the decoded bytes
restore will actually have (not the pre-encode array, which would make the
residual fight the bg codec's own loss) and with the same INTER_CUBIC restore
uses (a pinned contract) — then downscaled to `residual_scale` (0.5 default),
offset-encoded to uint8 (`value/2 + 128`), and stored as `residual.jxl`
(manifest 1.6.0; JPEG fallback, warned, if the JXL plugin is absent). When
high-bit storage is engaged the residual is instead a true 10/12-bit
`residual.avif` carrying a uint16 offset (`value/2 + 32768`), so a uint16
source's background delta keeps its depth too (manifest 1.9.0; the 8-bit
background's upscale is promoted `× 257` before differencing, and restore
mirrors that). On
restore the delta is added back to a plain bicubic upscale, so the background
is **real (lossy) data** — and the AI upscale **and GFPGAN are skipped** on
that path: both exist to make hallucination plausible, and repainting real
data would violate "never replace real pixels with a hallucination"
(soft-but-real beats fake-but-sharp). 8.1's low-frequency anchor is moot there
(the low band already is the stored background's); 8.2's grain still applies
(the residual is half-res + lossy, so the background remains smoother than the
real crops). `preview()` applies the residual too — it is stored *data*, not
an enhancement, and the bench bicubic proxy is preview-based, so skipping it
would hide the fidelity win. **Honest framing:** this converts "hallucinated"
into "lossy but real"; it does **not** recover full fidelity (half-res + lossy
residual), and the `.fkeep` gets larger — the explicit trade. All three knobs
(`residual`/`residual_scale`/`residual_quality`) are output-affecting and feed
`settings_fingerprint`.

**High-bit (HDR) crops** (`aggressive.output_bit_depth`, default `8` = off |
`10` | `12`) let an iPhone-style HDR source survive aggressive mode. The `.fkeep`
is an 8-bit container by default — every member is rounded down to 8-bit (a
10/12-bit HDR HEIC, decoded as uint16, included), so only faithful mode preserved
HDR. When enabled, the **real-pixel** members — face crops *and* region patches —
are stored at true high bit depth via the `avifenc` 10/12-bit AVIF path, so the
detail the user actually cares about (faces, hands, sharp regions) keeps its
depth. The **background, thumbnail, and residual stay 8-bit**: the background is
hallucinated on restore, so high-bit there buys nothing — the win concentrates on
the real pixels, which is the honest scope. It is **gated** on
`face_codec: avif` plus the locatable `avifenc` (encode) and `avifdec` (decode)
binaries, and degrades gracefully to the warned 8-bit round-down otherwise
(offline-first holds; the default 8-bit container is byte-identical). Restore
decodes the high-bit crops at full depth (`avifdec`), promotes the 8-bit
hallucinated background to 16-bit, composites the real uint16 crops, and writes
true HDR **only to an `.avif` output** (`restore -f avif`); a JPEG/PNG/JXL output
rounds down with a warning, since those are 8-bit here. It is an explicit
fidelity↔ratio trade — high-bit crops are larger, so the `.fkeep` grows — hence
off by default; `output_bit_depth` is output-affecting and feeds
`settings_fingerprint`, and the stored depth is recorded as the optional
`settings.bit_depth` manifest key (1.8.0+, absent on an 8-bit container). The
residual layer is **also high-bit** when it is on (manifest 1.9.0+):
`residual.avif` stores the background delta at the same 10/12-bit depth via a
uint16 offset (`value/2 + 32768`), so the residual + high-bit-crops combination —
aggressive mode's most-faithful path — keeps HDR end to end. It degrades to the
8-bit residual without `avifenc`, and (offline-first) is skipped at restore on a
box without `avifdec` (the background then hallucinates, as without a residual).
*(Aside: implementing this surfaced and fixed a latent R/B-swap bug in
the shared `avifenc` encode path — `encode_highbit_avif` / `encode_lossless_avif`
double-applied a BGR→RGB conversion, so faithful mode's 10/12-bit and lossless
AVIF output had red/blue exchanged; now pinned by a color round-trip test.)*

**iPhone HDR gain-map preservation** (`aggressive.preserve_gain_map`, on by
default; manifest 1.10.0+) is the *real*-iPhone-HDR path — Phase 9's insight is
that a modern iPhone HDR still is **not** a 10/12-bit deep-color image but an
8-bit Display-P3 base plus an Apple HDR **gain map** (an auxiliary image the OS
multiplies onto the base, scaled to the display's headroom). `imageio.load`
extracts that gain map (HEIC aux image / JPEG MPF second frame,
XMP-discriminated); aggressive compress stores it as one small `gainmap.jpg`
member (grayscale, typically half-resolution — Apple's own native scale) plus a
`gain_map_preserved` manifest flag, automatically whenever the source carries
one (the "just preserve it" decision; the flag opts out). On `restore -f avif`
the gain map is **re-attached**: the restored base is linearized, boosted per
pixel by `2^(headroom × gain)` (`aggressive.gain_map_headroom`, default 3 stops
— validated value-for-value against libavif's own conversion of a real iPhone
photo), PQ-encoded into an HDR alternate, and `avifgainmaputil combine` writes
a **backward-compatible HDR AVIF** (SDR viewers show the base; HDR displays
extend the highlights — the same mechanism as the original photo). Graceful
degradation everywhere (principle 4): a non-`.avif` output, a machine without
the `avifgainmaputil` binary (the same opt-in `.tools`/PATH family as
`avifenc`), or any re-attach failure falls back to the normal SDR write with a
warning — never a hard fail, and `preview()` never re-attaches (an external
re-encode is too slow for interactive use, and preview *pixels* are identical
either way). Two honest limits, documented in the format spec: the tool
rejects ICC-profiled inputs, so output color is declared via CICP (equivalent
for the P3/sRGB profiles every iPhone uses); and the background under the gain
map is reconstructed, so its HDR is approximate — the faces/patches are real
pixels with real HDR boost, the same "plausible, not faithful" bargain the mode
already makes. `preserve_gain_map` is compress-side → fingerprinted;
`gain_map_headroom` is restore-only → not.

**Face enhancement of reconstructed background faces** is the restore-side
safety net for the recall guardrails above. Detection biases toward recall, but
a face it still misses is downsampled with the background and then *upscaled* by
Real-ESRGAN/bicubic on restore — which tends to melt it into something uncanny,
again the worst family-tool failure. So restore (gated by
`aggressive.face_enhance`, on by default) runs a face-restoration model over the
upscaled background to re-synthesize plausible face detail. This is
**restore-only** — it changes nothing about compress, the `.fkeep` format, or
the cache fingerprint — and it is careful in two ways: it blends back **only the
face regions the enhancer itself detected** (feathered soft mask;
`bg_upsampler=None` so it never repaints non-face background), and it runs
**before** the real face crops are composited, so a missed face is improved while
a *detected* face is still overlaid with its original-quality crop and never
replaced by a hallucination. Like Real-ESRGAN it degrades gracefully: the
enhancer lives in optional extras, and if it (or its weights) are unavailable
the step is skipped and restore proceeds with the plain upscale (`preview` skips
it unconditionally for speed).

Two **backends** are available (`aggressive.face_enhance_backend`), plus a
shared de-uncanny dial:

- **GFPGAN v1.4** (the default; `[ai]` extra). The established safety net; its
  known weakness is the "too-perfect face on a real photo" look.
- **CodeFormer** (opt-in; `[codeformer]` extra = the `codeformer-pip` package,
  used together with `[ai]`, which provides facexlib/torch). More robust on
  heavily degraded inputs, and its fidelity weight
  `aggressive.face_enhance_fidelity` (CodeFormer's `w ∈ [0,1]`, default 0.7,
  higher = closer to the input face) is exactly the closer-to-original ↔
  prettier dial. **License honesty:** CodeFormer's code and weights are S-Lab
  License 1.0 (**non-commercial**); the arch comes from `codeformer-pip` (never
  vendored into this repo), weights are fetched at runtime through
  `models.ensure_weights` (SHA-256-verified, local cache), and the extra is
  deliberately excluded from the `all` convenience bundle so accepting that
  license is always an explicit opt-in. A selected-but-unavailable CodeFormer
  **warns and skips enhancement — it never silently substitutes GFPGAN** (the
  user asked for a specific model; silently swapping would misreport what
  restored their photo). Internally `_CodeFormerEnhancer` duck-types `GFPGANer`
  (`enhance`/`face_helper`/`upscale`), so the bounded one-face-at-a-time
  self-paste (the GFPGAN-OOM lesson) is a single shared code path.
- **`aggressive.face_enhance_strength`** (default 1.0, both backends) lerps each
  restored face toward the un-enhanced pixels at paste time: ~0.6–0.8 directly
  softens the "too-perfect" look, 1.0 is byte-identical to full enhancement, and
  0 skips inference entirely.

The `.fkeep` container is a plain ZIP so it can be inspected with `unzip`. Face
crops are stored as high-quality JPEG by default (optionally AVIF/JXL 4:4:4 via
`aggressive.face_codec`, which match JPEG's perceptual quality at a smaller size
on photographic content; PNG only when lossless is explicitly requested), because
PNG on photographic crops is large enough to defeat the purpose. The
**downsampled background** is likewise JPEG by default, and `aggressive.bg_codec`
may store it as AVIF/JXL (4:2:0) instead — a double win when it fits the content:
a smaller `.fkeep`, and a *cleaner input for the restore upscaler* (JPEG block
artifacts in the background are exactly what SR amplifies into false texture).
The default stays JPEG because AVIF can lose to JPEG on noisy content (the same
content-dependence as the face-crop codecs); readers locate the background by
extension (`background.jpg → .avif → .jxl`), so old files are untouched. The
restore step
is where the background is reconstructed — this is the
mode's defining trade-off. The full on-disk spec (archive layout, the complete
`manifest.json` schema, and how to recover pixels by hand with just `unzip`) is
documented in [fkeep-format.md](fkeep-format.md).

## Components

- **config.py** — `FaceKeepConfig` (dataclasses) with `validate()`, YAML
  `load()`/`save()`. Sub-configs: `DetectorConfig`, `FaithfulConfig`,
  `AggressiveConfig`. Also home to the **aggressive-mode presets** (`PRESETS`,
  `apply_preset`, `preset_restore_overrides`): named knob bundles applied as a
  precedence layer (defaults < preset < explicit YAML < explicit CLI), with
  `load()` recording the YAML-explicit keys so the layer order holds wherever
  the preset is named.
- **detector.py** — `FaceDetector` strategy interface; `HaarDetector` (bundled,
  offline default), `YuNetDetector` (DNN; its ~232 KB ONNX model is downloaded
  via `models.ensure_weights` — shared cache, SHA-256-verified, atomic — from
  GitHub's **LFS resolver** URL, falling back to Haar offline), and
  `MediaPipeDetector` (optional `[detect]` extra; Google BlazeFace via the current
  **Tasks API**, with a ~230 KB `.tflite` model fetched through the same
  `models.ensure_weights` cache/verify path — a missing package or model falls
  back to Haar). Both optional backends are opt-in upgrades, never the default,
  so the default path stays offline and zero-download. `create_detector()`
  factory. **Custom-detector plugin hook:** a third party registers their own
  `FaceDetector` under a new backend name (`register_detector(name, factory)`)
  and selects it via `detector.backend: <name>` exactly like a built-in — the
  factory gets the same resolved kwargs and the detector is used as-is (no Haar
  wrapping; a custom backend owns its degradation). The built-in names are
  reserved so a plugin can't shadow the offline default; `config.validate()`
  (`is_known_backend`) and the index/detection-cache fingerprint (which key off
  the backend string) accept and bust on a custom name. A separate
  **`HandDetector`** (also optional `[detect]`; MediaPipe Hand
  Landmarker via the Tasks API, model fetched through `models.ensure_weights`)
  powers aggressive-mode **hand protection** (C2): it returns tight per-hand boxes
  so hands can be kept as region patches — it is *not* a `FaceDetector` (hands
  share no padding/ROI/chroma logic). `create_hand_detector()` returns `None` when
  the package/model is unavailable, so the compressor falls back to the offline C1
  geometric hand zones; the default (`protect_hands_backend=None`) is C1. `DetectorConfig.roi` (`face` default | `head_shoulders` | `person`)
  optionally grows each subject's high-priority region beyond the face to the
  upper body — implemented purely as a downward/outward expansion of `padded_bbox`
  (the region aggressive crops and faithful auto-tune already use), applied after
  the false-positive filter and leaving the tight detection box (and the chroma
  decision) unchanged; `face` is a no-op. A **`DetectionCache`** (stdlib
  `sqlite3`, user-global at `~/.cache/facekeep/detections.sqlite`) caches a
  detection result by `(content hash, detector_fingerprint())` so a re-run reuses
  it (`detect_cached()`) — the per-image counterpart to `index.py`'s whole-file
  skip, helping when the file index misses but detection didn't change (e.g. a
  different `--quality`). It is a pure speed feature and best-effort (any cache
  error → just detect), attached only on the serial path (`--jobs` workers detect
  normally), and stores only coordinates so it stays offline/zero-download.
- **models.py** — shared model/weights cache under `~/.cache/facekeep/models`
  (`MODELS_CACHE_DIR`, the one cache-dir source of truth — `detector.YUNET_CACHE`
  points here too). `ensure_weights(url, filename, sha256=)` downloads a weights
  file once, **SHA-256-verifies** it, writes it atomically (temp + replace, so an
  interrupted download never leaves a half-written cache entry), and re-downloads
  a checksum-failing cache. On a download/checksum failure it raises
  `ModelDownloadError` with a message pointing at the `[ai]` extra. The aggressive
  AI restore (Real-ESRGAN, GFPGAN) routes its weights through this and hands the
  underlying package the resulting *local verified path* (not the URL), so weights
  are validated and live in the documented cache rather than buried in
  `site-packages`. The opt-in **YuNet** detector download also routes through it
  (`detector._ensure_model`), so its model gets the same checksum/atomic-write
  guarantees. Only these *opt-in* paths touch it; the default faithful pipeline
  (and the default Haar detector) never do.
- **encoders.py** — faithful-mode codec wrappers; `encode`/`decode`/
  `write_encoded`; `codec_available`. Robust extension handling for dotted
  filenames. **Per-image codec choice (`codec: both`)** lives in `faithful.py`
  (`_encode_best_codec`): it trial-encodes each image with *both* AVIF and JXL,
  each at its own auto-tuned/configured quality, and keeps the smaller output —
  the concrete winner (never `"both"`) is what verify, skip-if-larger, the output
  extension, `FaithfulResult.codec`, and the index/report all see. Falls back to
  the one available plugin (warned) if only one is installed.
- **metrics.py** — SSIM/PSNR and regional (face/background) comparison; shared
  by the CLI and auto-tune. Also an opt-in **LPIPS** (learned perceptual
  distance) evaluator (`[ai]` extra, lazy-imported, graceful `None` when absent)
  for scoring aggressive-mode restores — SSIM is the wrong tool for a
  hallucinated-but-plausible background. LPIPS is an *evaluation* metric only,
  never on a pipeline default path (it pulls torch and downloads weights), exposed
  via `facekeep quality --lpips`. Also an opt-in **SSIMULACRA2** perceptual
  metric (`[dev]` extra — pure Python, no native binary or model download; lazy-
  imported, graceful `None` when absent) that is the **default** acceptance target
  of the faithful **auto-tune** search (which is itself on by default):
  `faithful.target_metric: ssimulacra2`, higher = better, ~90 visually lossless —
  far better perceptual correlation than SSIM. When selected but unavailable (a
  plain install without the `[dev]` extra) the search falls back to SSIM
  (re-basing the threshold, since the scales differ), so the default is
  "perceptual when available, SSIM otherwise." Also exposed for inspection via
  `facekeep quality --ssimulacra2`.
- **imageio.py** — `load()` with EXIF orientation correction and EXIF + ICC
  color-profile preservation; the single entry point for reading images. A
  **10/12-bit HDR HEIC** is decoded high-bit (`_decode_heif` →
  `pillow_heif.open_heif(convert_hdr_to_8bit=False)` → `uint16` BGR,
  `source_bit_depth=16`), so it feeds the `avifenc` 10/12-bit AVIF output path
  instead of being flattened to 8-bit; HEIC uses `open_heif` *exclusively* (never
  PIL `Image.open`) because opening one HEIC with both APIs in a process
  segfaults libheif. It also carries the **iPhone HDR gain map** (Phase 9.1): a
  HEIC's `…aux:hdrgainmap` auxiliary image (exposed via
  `pillow_heif.options.AUX_IMAGES`) or a JPEG's MPF second frame (accepted only
  when its XMP names a gain map, so a stereo MPO is never misread) rides
  `LoadedImage.gain_map` / `gain_map_meta`, kept upright and aligned with the
  base pixels. Extraction is best-effort (`None` on any failure — a gain map
  never fails a load); nothing consumes it yet — storing and re-attaching it is
  Phase 9.2.
- **faithful.py** — the default pipeline.
- **index.py** — incremental-processing cache (stdlib `sqlite3`): records each
  file's content hash + settings fingerprint + output path so a re-run skips
  unchanged photos whose output still exists. Opened only in the parent process
  (the CLI), so it never contends with `--jobs` workers. (The per-image detection
  cache that complements it lives in `detector.py` — see above; both are
  parent-process-only and share the detector-field set that busts them.)
- **aggressive/** — `compressor`, `blender`, `format`, `restorer`.
- **cli.py** — Click CLI: `compress`, `restore`, `info`, `quality`, `compare`,
  `verify`, `bench`, `init`, `gui`.
- **compare.py** — `facekeep compare`: a read-only before/after viewer. Loads an
  original and a compressed artifact, reconstructs the "after" (faithful decode /
  `.fkeep` restore or `--preview` bicubic / a standard image), and writes a single
  **self-contained** HTML report (before/after slider + difference heatmap +
  SSIM/PSNR, optional LPIPS/SSIMULACRA2), images inlined as base64. It changes no
  output pixels — only visualizes existing outputs — so it adds no fidelity
  surface; its pure helpers (`diff_map`/`_embed`/`build_html`) are unit-tested
  browser-free, mirroring `gui.py`. The `.fkeep` **real-restore** "after" is
  **preset-aware**: `load_after` reads the file's recorded `settings.preset`
  (manifest 1.7.0+) and auto-applies its restore-side knobs onto a copy of the
  aggressive config (`_restore_agg_config`, mirroring `cli.restore._restorer_for`;
  explicit keys win; unreadable/absent preset → base config), so the comparison
  matches what `facekeep restore` actually produces *per file* — both here and in
  the GUI Compare tab, which share `load_after`. The `--preview` path is untouched
  (it never enhances faces).
- **gui.py** — optional local web GUI (`facekeep gui`; the `[gui]` extra,
  Gradio). A deliberately thin wrapper over the library API: drag a photo in,
  pick a mode, see a before/after with the real stats, download the real output.
  It runs the *same* faithful/aggressive pipeline as `compress` (byte-identical
  output — no new fidelity surface). Gradio is imported **lazily** inside
  `build_demo`/`launch`, so the module and its pure handlers import without it
  (the handlers are unit-tested browser-free); a missing `[gui]` makes
  `facekeep gui` print an install hint, not crash (graceful degradation,
  principle 4). `launch()` is **local-only with sharing and telemetry off** — a
  photo tool must not phone home or open a public tunnel by default. Aggressive
  mode's before/after "after" is the fast **bicubic `preview()`** (a full AI
  restore is far too slow for an interactive UI); the download is the real
  `.fkeep`. A second **Compare** tab (`compare_images`, the same pure-handler
  shape) reuses the `compare.py` helpers (`load_after`/`align`/`diff_map` +
  metrics) to show a live `gr.ImageSlider` before/after wipe, a difference
  heatmap, and SSIM/PSNR for an original vs *any* compressed output — the
  interactive sibling of the `facekeep compare` HTML export. Its `.fkeep` "after"
  defaults to the fast bicubic preview, with an **opt-in real AI restore**
  (`use_ai`): the toggle is honest about availability (`realesrgan_available()`
  in `restorer.py` — it falls back to the preview and says so when the `[ai]`
  extra is absent, rather than running the same bicubic slowly and mislabeling
  it) and warns up front that it is slow (a `gr.Info`/`gr.Warning` toast + a
  `show_progress="full"` spinner). The underlying `restore()` is unchanged — the
  tab only exposes the existing path — so no new fidelity surface.

## Design principles

1. **Faithful by default; aggressive only on request.** The honest, universal
   result is the default.
2. **Let the codec do perceptual work.** No manual region manipulation in
   faithful mode.
3. **Standard outputs.** Faithful mode emits files that need no special software.
   Both modes preserve color fidelity end-to-end: the source ICC profile (e.g.
   Display P3) is carried through and embedded in the output — faithful via
   `encoders.encode`, aggressive via the `.fkeep` `icc.bin` member re-embedded on
   restore (the restore JPEG is written through Pillow, which can carry a profile;
   OpenCV cannot).
4. **Graceful degradation.** Missing AI extras → bicubic background upscale and
   GFPGAN background-face restore skipped; offline (or a weights checksum failure)
   → `ModelDownloadError` caught → same bicubic/skip fallback; offline detector →
   Haar; detection failure → plain encode.
5. **Inspectable formats.** `.fkeep` is a ZIP; `manifest.json` is human-readable.
