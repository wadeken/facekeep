# The `.fkeep` container format

This is the on-disk specification for FaceKeep's **aggressive-mode** output. It
is written so that your photos are never trapped: a `.fkeep` is a plain ZIP of
standard image files plus one human-readable JSON manifest, and this document
tells you exactly how to get your pixels back **without FaceKeep installed at
all** (see [Manual recovery](#manual-recovery-without-facekeep)).

> **Faithful mode does not use this format.** The default (faithful) mode writes
> a single standard `.avif` / `.jxl` file with no container and no restore step —
> opening it *is* the restore. `.fkeep` exists only for aggressive mode, where
> the background is downsampled and reconstructed on restore. See
> [architecture.md](architecture.md) for why faithful is the default.

- **Extension:** `.fkeep`
- **Container:** a ZIP archive (`zipfile.ZIP_DEFLATED`). Anything that opens a
  `.zip` opens a `.fkeep` — just rename it, or point `unzip` straight at it.
- **Current manifest version:** `1.10.0` (see [Versioning](#versioning--compatibility)).

---

## Design promise: your photos aren't trapped

A photo-backup tool that locks files in a proprietary box is an adoption risk —
if the tool disappears, so do the photos. FaceKeep deliberately avoids that:

1. **The container is a standard ZIP.** No custom binary framing. `unzip x.fkeep`
   (or any archive manager) extracts every member.
2. **Every payload is a standard image.** Background, thumbnail, and face crops
   are JPEG or PNG by default (optionally AVIF/JXL); masks are 8-bit grayscale
   PNG. They open in any image viewer (AVIF/JXL in any modern one).
3. **The manifest is human-readable JSON.** Sizes, the face list with bounding
   boxes, and the settings used are all plain text you can inspect.

What you **cannot** get back is the original full-resolution *background* pixels:
aggressive mode discards them on purpose and stores only a downsampled copy, so
restore reconstructs (hallucinates) plausible background detail. That trade-off
is the whole point of the mode — but it is the *only* thing you lose, and the
faces are kept at original quality. If you need the original background pixels,
use faithful mode instead, which never discards them.

> **Middle ground (manifest 1.6.0+):** the opt-in **residual layer**
> (`aggressive.residual`, CLI `--residual`) additionally stores the *real*
> high-frequency delta the downsample lost, so restore adds it back instead of
> hallucinating — the background becomes faithful-but-lossy (the residual is
> half-resolution and lossy-encoded, so this is **not** full fidelity), at the
> cost of a larger `.fkeep`. See the `residual.(avif|jxl|jpg)` member below.

---

## Archive layout

A `.fkeep` contains the following members. `NNN` is a zero-padded three-digit
index (`000`, `001`, …) matching the face's `id` in the manifest (and, for the
`region_*` members, the region's `id`).

| Member | When present | Format | Contents |
| --- | --- | --- | --- |
| `manifest.json` | always | UTF-8 JSON | Metadata: original dimensions, faces, regions, settings, flags. The source of truth. |
| `background.jpg` | always (default codec) | JPEG (q = `settings.bg_quality`) | The **downsampled** background, scaled by `settings.bg_scale`. |
| `background.avif` / `background.jxl` | when `settings.bg_codec` is `avif`/`jxl` (manifest 1.5.0+) | AVIF / JPEG XL, **4:2:0** (q = `settings.bg_quality`) | Same downsampled background, stored with a modern codec — fewer block artifacts for the SR upscaler to amplify, smaller. Replaces the `.jpg`. |
| `thumbnail.jpg` | always | JPEG (q80) | A small preview, fixed 256-px height. Not used by restore. |
| `face_NNN.jpg` | one per face (default codec) | JPEG (q = `settings.face_quality`) | A face crop at **original** resolution, covering `padded_bbox`. |
| `face_NNN.avif` / `face_NNN.jxl` | one per face (when `settings.face_codec` is `avif`/`jxl`) | AVIF / JPEG XL, **4:4:4** (q = `settings.face_quality`) | Same crop, stored with a modern codec — same perceptual quality, smaller. Replaces the `.jpg` for that index. |
| `face_NNN.png` | one per face (only when `face_quality >= 100`) | PNG (lossless) | Same crop, lossless. Replaces the `.jpg`/`.avif`/`.jxl` for that index. |
| `face_mask_NNN.png` | one per face | 8-bit grayscale PNG | Soft alpha mask for feathered compositing of the crop. |
| `region_NNN.(jpg\|avif\|jxl\|png)` | one per region (manifest 1.3.0+; only when region-local conservatism fired) | same codec/precedence as face crops | A **region-local conservatism** patch: a near-original-resolution crop of a risky region — the background around a small/distant face, a **hand**, or a **text-like cluster** (signage; opt-in `protect_text`) — kept sharp instead of downsampled. Readers need no new key: a region is a region. Covers the region's `bbox`. |
| `region_mask_NNN.png` | one per region | 8-bit grayscale PNG | Soft alpha mask for feathered compositing of the region patch. |
| `residual.avif` / `residual.jxl` / `residual.jpg` | only when `settings.residual` is true (manifest 1.6.0+) | high-bit (10/12-bit) AVIF when `settings.bit_depth` > 8 (manifest 1.9.0+); else JPEG XL (q = `settings.residual_quality`), or JPEG as a warned fallback when the JXL plugin is unavailable | The **residual layer**: the real high-frequency delta the background downsample lost, at `settings.residual_scale` resolution, offset-encoded (`value/2 + 128` into uint8; high-bit: `value/2 + 32768` into uint16 — see [Image members](#image-members-in-detail)). Restore adds it back to a bicubic upscale, so the background is real (lossy) data, not a hallucination. |
| `gainmap.jpg` | only when `gain_map_preserved` is true (manifest 1.10.0+) | grayscale JPEG q90 (typically half the original resolution) | The source's **iPhone HDR gain map** (the auxiliary image that makes an iPhone photo HDR — the base image is plain 8-bit SDR). Stored upright (aligned with the original frame). `facekeep restore -f avif` re-attaches it via `avifgainmaputil combine` into a backward-compatible **HDR AVIF**; any other output (or a machine without the binary) restores SDR with a warning. |
| `exif.bin` | only if the source had EXIF | raw bytes | The original EXIF block, re-embedded into the restored image. |
| `icc.bin` | only if the source had an ICC profile (manifest 1.4.0+) | raw bytes | The original ICC color profile (e.g. Display P3), re-embedded into the restored image so wide-gamut color survives. |

Notes:

- **Exactly one crop per face.** It is *one* of `face_NNN.png`,
  `face_NNN.avif`, `face_NNN.jxl`, or `face_NNN.jpg` — never more than one for a
  given index. **Readers locate it by trying those extensions in that exact
  order** (`png` → `avif` → `jxl` → `jpg`); this order is the load-bearing
  contract. The default codec is JPEG q95; `settings.face_codec` may select
  `avif`/`jxl` (stored 4:4:4 to keep skin/lip color crisp), which match JPEG's
  perceptual quality at a smaller size on photographic content. **PNG always
  wins:** `face_quality >= 100` stores the crop lossless as PNG regardless of
  `face_codec` (PNG is the explicit lossless request).
- **Decoding the modern-codec crops.** `face_NNN.avif`/`.jxl` are decoded with
  the same Pillow plugins faithful mode uses, **not** OpenCV — the bundled
  OpenCV build cannot decode the AVIF/JXL that those plugins write. `png`/`jpg`
  crops decode with OpenCV as before. All crops decode to standard BGR pixels.
- **Size is content-dependent.** AVIF/JXL beat JPEG on clean, smooth
  photographic crops (the intended case). On a crop that already carries JPEG
  artifacts or per-pixel noise — AV1-intra's worst case — an AVIF crop can be
  *larger* than the JPEG equivalent (JXL still tends to win). The codec is opt-in
  for that reason; the default stays JPEG.
- **One mask per face**, always PNG, always named `face_mask_NNN.png`.
- **Exactly one background member.** It is *one* of `background.jpg`,
  `background.avif`, or `background.jxl`. **Readers locate it by trying those
  extensions in that exact order** (`jpg` → `avif` → `jxl`) — the load-bearing
  contract, like the crop order (older files always have `background.jpg`, found
  first). The `avif`/`jxl` variants decode with the faithful-mode Pillow plugins
  (OpenCV cannot decode them here); `jpg` decodes with OpenCV. The same
  content-dependent size caveat as for crops applies (AVIF can lose to JPEG on
  noisy content), which is why the default stays JPEG.
- **Region patches mirror face crops exactly.** A `region_NNN.*` crop is located
  by the *same* `png → avif → jxl → jpg` order and decoded the same way (avif/jxl
  via Pillow, png/jpg via OpenCV), and each has exactly one `region_mask_NNN.png`.
  They are present only on manifest 1.3.0+ files where region-local conservatism
  fired; an older file (or a photo with no risky region) simply has no `region_*`
  members and an empty/absent `regions` array.
- **At most one gain-map member** *(1.10.0+)*. When the top-level
  `gain_map_preserved` flag is true there is exactly one `gainmap.jpg` (fixed
  name — no extension search). It is a plain grayscale JPEG any viewer opens;
  values encode the per-pixel HDR boost (`linear boost = 2^(headroom ×
  value/255)`, headroom ≈ 3 stops). Files with the flag false/absent have no
  member, and a reader that ignores it restores SDR exactly as before Phase 9.
- **At most one residual member** *(1.6.0+)*. When `settings.residual` is true
  there is exactly one of `residual.avif` / `residual.jxl` / `residual.jpg`,
  **located in that order**. A high-bit (HDR) residual is a true 10/12-bit
  `avif` (manifest 1.9.0+, only when high-bit storage is engaged) — its `.avif`
  extension self-describes the depth. An 8-bit residual is `jxl` (the intended
  codec — the residual is noise-like content where JXL wins) or `jpg` (the warned
  fallback when the JXL plugin was unavailable at compress time). `residual.avif`
  decodes high-bit via `avifdec`, `residual.jxl` with the faithful-mode Pillow
  plugin, `residual.jpg` with OpenCV. Files with `settings.residual` false/absent
  have no residual member.
- **High-bit (HDR) crops + residual** *(1.8.0+ crops; 1.9.0+ residual)*. When
  `settings.bit_depth` is `10` or `12`, the **real-data** members are stored at
  true high bit depth via the `avifenc` CLI: the **real-pixel** crops (face crops
  + region patches, `face_NNN.avif` / `region_NNN.avif`, since 1.8.0) and — when
  the residual layer is on — the **residual** (`residual.avif`, since 1.9.0), so a
  10/12-bit HDR source keeps its depth on every real-data member. The
  **background and thumbnail are always 8-bit** (the background is hallucinated on
  restore, so high-bit there buys nothing). Restore decodes these members at full
  depth with `avifdec` and writes true HDR only to an `.avif`/`.jxl` output (a
  JPEG output rounds down, warned). Absent `bit_depth` (the default 8-bit
  container and all pre-1.8.0 files) means every member is 8-bit, exactly as
  before.
- `thumbnail.jpg` is a convenience preview only; restore ignores it.
- A well-formed `.fkeep` therefore has `2 + 2·N + 2·R` image members plus the
  manifest (and `+1` each for `exif.bin` / `icc.bin` if the source carried EXIF /
  an ICC profile, `+1` for the residual member on a 1.6.0+ file with
  `settings.residual` true, and `+1` for `gainmap.jpg` on a 1.10.0+ file with
  `gain_map_preserved` true), where `N` is the number of faces and `R` the number
  of region patches (`R = 0` on pre-1.3.0 files; `icc.bin` only on 1.4.0+ files
  whose source had a profile). `facekeep verify` checks the crop/mask
  consistency — see [Integrity & verification](#integrity--verification).
  (`exif.bin`/`icc.bin` are optional metadata and not required by `verify`; a
  *declared* residual member **is** required by it.)

---

## `manifest.json` schema

The manifest is written with `indent=2` and `ensure_ascii=False`. Top-level
keys (all present on a v1.1.0 file written by the current code):

| Key | Type | Meaning |
| --- | --- | --- |
| `version` | string | **Manifest schema version** (`"1.9.0"`). Independent of the tool version. |
| `mode` | string | Always `"aggressive"` (only aggressive mode writes `.fkeep`). |
| `original` | object | Facts about the original input file — see below. |
| `exif_preserved` | bool | `true` iff an `exif.bin` member is present. |
| `icc_preserved` | bool | `true` iff an `icc.bin` member is present (the source had an ICC color profile). *(Added in manifest `1.4.0`; absent on older files.)* |
| `gain_map_preserved` | bool | `true` iff a `gainmap.jpg` member is present (the source carried an iPhone HDR gain map and `aggressive.preserve_gain_map` was on — the default). *(Added in manifest `1.10.0`; absent on older files.)* |
| `settings` | object | The aggressive-mode parameters used to produce this file — see below. |
| `faces` | array | One entry per detected face — see below. May be empty (no faces found). |
| `regions` | array | One entry per region-local conservatism patch — see below. *(Added in manifest `1.3.0`; absent/empty on older files and on photos with no risky region.)* |
| `estimated_payload_bytes` | int | Sum of the encoded image members' sizes (pre-ZIP), for diagnostics. |
| `created_at` | string | ISO-8601 UTC timestamp of when the file was written. |
| `facekeep_version` | string | The FaceKeep package version that wrote the file. |

### `original`

| Key | Type | Meaning |
| --- | --- | --- |
| `filename` | string | The original file's base name (e.g. `2024.05.20_trip.jpg`). |
| `width` | int | Original width in pixels (after EXIF-orientation correction). |
| `height` | int | Original height in pixels. |
| `size_bytes` | int | Size of the original file on disk, in bytes. |
| `hash_sha256` | string | **SHA-256 of the original input file's bytes.** See the honesty note below. |
| `orientation` | int | The original EXIF orientation tag (1–8) that was applied on load. |

> **Honesty note on `hash_sha256`.** This is the hash of the *original input
> file*, recorded at compress time. The original pixels are **not** stored in the
> `.fkeep` (only a downsampled background survives), so this hash **cannot be
> recomputed from the `.fkeep` alone**. It is provenance metadata: to verify it,
> you must still have the original file to hash and compare. `facekeep verify`
> reflects this — it never fabricates a "hash OK"; it only reports a match when
> you pass the original with `--original`.

### `settings`

| Key | Type | Meaning |
| --- | --- | --- |
| `bg_scale` | float | The background downsample factor actually used (e.g. `0.25` = ¼). Restore upscales by `1 / bg_scale`. May differ from the configured default when the no-face fallback **or** whole-image content-aware conservatism (a text/fine-detail background) raised it, **or** when **quality-targeting** (`aggressive.quality_target`) searched a per-photo scale to hit a perceptual-quality (LPIPS) target — which may be *more or less* aggressive than the configured `bg_scale`, but never below the conservative floor the above protections set. *(Region-local conservatism does **not** raise this — a small/distant-face risk is instead protected by a `regions[]` patch, leaving `bg_scale` aggressive for the benign rest of the frame.)* Whatever value lands here, restore reads it verbatim — the field is self-describing, so a `.fkeep` restores correctly regardless of how the scale was chosen. |
| `bg_quality` | int | Quality (0-100) used for the background member, whatever its codec. |
| `bg_codec` | string | Codec for the background: `jpg` (default) \| `avif` \| `jxl`. `avif`/`jxl` are stored 4:2:0. *(Added in manifest `1.5.0`; absent on older files, where it is `jpg`.)* Informational — readers locate the background by member extension, not this field. |
| `face_quality` | int | Quality used for face crops; `>= 100` means crops are lossless PNG (overrides `face_codec`). |
| `face_codec` | string | Codec for face crops: `jpg` (default) \| `avif` \| `jxl`. `avif`/`jxl` are stored 4:4:4. *(Added in manifest `1.2.0`; absent on older files, where it is `jpg`.)* Informational — readers locate crops by member extension, not this field. |
| `bit_depth` | int | *(1.8.0+, optional)* Max bit depth of the stored high-bit **real-data** members: the face crops + region patches (1.8.0+) and — when the residual layer is on — the residual (1.9.0+). `10` or `12` when stored as true high-bit AVIF (via the `avifenc` CLI, for a 10/12-bit HDR source). **Absent** on the default 8-bit container and all older files — readers treat absent as `8`. The background/thumbnail are always 8-bit. Restore decodes high-bit AVIF crops at full depth (`avifdec`) and writes true HDR only to an avif/jxl output (a JPEG output rounds down, warned). |
| `blend_mode` | string | Soft-mask compositing mode (`gaussian` \| `linear` \| `poisson`). |
| `model` | string | The super-resolution model name requested for restore (e.g. `realesrgan-x4plus`). |
| `residual` | bool | *(1.6.0+)* Presence flag for the residual layer: `true` iff a `residual.(avif\|jxl\|jpg)` member is present (`avif` = the high-bit HDR residual, 1.9.0+). Restore then reconstructs the background from real data (bicubic + residual) and skips the AI upscale. |
| `residual_scale` | float | *(1.6.0+)* The resolution the residual was downscaled to before encoding (`0.5` default = half the original per side). Informational — restore resizes the decoded residual to full resolution regardless. |
| `residual_quality` | int | *(1.6.0+)* Quality (1-100) used for the residual member. |
| `preset` | string | *(1.7.0+, optional)* The aggressive-mode **preset** (`ratio` \| `pretty` \| `fidelity` \| `family` \| `share`) the file was compressed with. **Absent entirely** on presetless runs and on older files. Informational plus a restore hint: `facekeep restore` auto-applies the preset's *restore-side* knobs (face-enhance backend/fidelity/strength) unless the user explicitly overrode them; an unknown or absent name is simply ignored (tolerant by structure), so a file written by a future preset still restores. |

### `faces[]`

Each entry describes one face. Coordinates are pixel offsets in the
**original-resolution** image, as `[x1, y1, x2, y2]` (left, top, right, bottom).

| Key | Type | Meaning |
| --- | --- | --- |
| `id` | int | Zero-based index; matches the `NNN` in `face_NNN.*` / `face_mask_NNN.png`. |
| `bbox` | `[int,int,int,int]` | The tight detected face box. |
| `padded_bbox` | `[int,int,int,int]` | The padded box (per `detector.padding`) that the **crop and mask cover** and that restore composites at. |
| `confidence` | float | Detector confidence (YuNet gives real scores; Haar reports a uniform value). |

### `regions[]`

*(Manifest 1.3.0+.)* Each entry describes one **region-local conservatism**
patch — a risky region kept sharp instead of downsampled. A region may be the
background context around a **small/distant face**, *(since hand protection)* a
**hand** (which the detector never finds, so it would otherwise be smeared by the
AI upscale), or *(opt-in, `aggressive.protect_text`)* a **text-like cluster**
(signage — what the AI upscale garbles worst). All are stored identically — the
manifest does not distinguish them
(a region is a region), so none of these needed **a schema/version change**.
Coordinates
are pixel offsets in the **original-resolution** image, as `[x1, y1, x2, y2]`. May
be empty.

| Key | Type | Meaning |
| --- | --- | --- |
| `id` | int | Zero-based index; matches the `NNN` in `region_NNN.*` / `region_mask_NNN.png`. |
| `bbox` | `[int,int,int,int]` | The frame-coordinate box the patch covers and that restore composites it at (clamped to the frame). |
| `scale` | float | The resolution the stored patch was downscaled to before encoding (`1.0` = original; per `aggressive.region_scale` for small-face and text regions, `aggressive.hand_zone_scale` for hand regions). The patch is resized up to `bbox` on restore regardless, so this is a size/quality knob, not a placement field. |

### Example

A real two-face manifest (values illustrative):

```json
{
  "version": "1.10.0",
  "mode": "aggressive",
  "original": {
    "filename": "2024.05.20_trip.jpg",
    "width": 4032,
    "height": 3024,
    "size_bytes": 5183920,
    "hash_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "orientation": 1
  },
  "exif_preserved": true,
  "icc_preserved": true,
  "gain_map_preserved": false,
  "settings": {
    "bg_scale": 0.25,
    "bg_quality": 85,
    "bg_codec": "jpg",
    "face_quality": 95,
    "face_codec": "jpg",
    "blend_mode": "gaussian",
    "model": "realesrgan-x4plus",
    "residual": false,
    "residual_scale": 0.5,
    "residual_quality": 60
  },
  "faces": [
    {
      "id": 0,
      "bbox": [1180, 760, 1520, 1180],
      "padded_bbox": [1010, 590, 1690, 1350],
      "confidence": 0.94
    },
    {
      "id": 1,
      "bbox": [2240, 820, 2560, 1200],
      "padded_bbox": [2080, 660, 2720, 1360],
      "confidence": 0.91
    },
    {
      "id": 2,
      "bbox": [3650, 980, 3740, 1090],
      "padded_bbox": [3560, 890, 3830, 1180],
      "confidence": 0.62
    }
  ],
  "regions": [
    {
      "id": 0,
      "bbox": [3560, 890, 3830, 1180],
      "scale": 1.0
    }
  ],
  "estimated_payload_bytes": 612345,
  "created_at": "2026-06-01T09:30:00+00:00",
  "facekeep_version": "0.2.0"
}
```

In this example the third face (`id: 2`) is small/distant (90×110 px in a
4032-wide frame ≈ 3 % of the short side), so region-local conservatism kept the
background around it sharp: a `regions[0]` patch covering that face's padded box
(`region_000.jpg` + `region_mask_000.png`). The two large faces produced no
region. `bg_scale` stays at the aggressive `0.25` for the rest of the frame.

---

## Image members in detail

- **`background.(jpg|avif|jxl)`** — the original image resized by
  `settings.bg_scale` with area-averaging (`INTER_AREA`), then encoded at
  `bg_quality` with `settings.bg_codec` (JPEG by default; AVIF/JXL 4:2:0 when
  selected, located in the `jpg → avif → jxl` order above). Its pixel
  dimensions are therefore approximately `original.width·bg_scale` ×
  `original.height·bg_scale`, and it must never be larger than the original.
- **`thumbnail.jpg`** — the original resized to a fixed 256-px height
  (width scaled to keep aspect), JPEG q80. Preview only.
- **`face_NNN.(png|avif|jxl|jpg)`** — the original-resolution pixels inside that
  face's `padded_bbox`, copied **before** any downsampling, so faces keep full
  detail. JPEG at `face_quality` by default; AVIF/JXL 4:4:4 at `face_quality`
  when `settings.face_codec` selects them; lossless PNG only when
  `face_quality >= 100` (PNG wins over the codec). Exactly one extension is
  present per index, located in the `png → avif → jxl → jpg` order above. The
  `avif`/`jxl` variants decode with the faithful-mode Pillow plugins (OpenCV
  cannot decode them here); `png`/`jpg` decode with OpenCV.
- **`face_mask_NNN.png`** — an 8-bit grayscale PNG the same size as the crop. It
  is a feathered soft mask: stored as `0–255`, interpreted on restore as alpha in
  `[0,1]` (value ÷ 255). It blends the crop's edges into the background so the
  composited face has no hard seam.
- **`region_NNN.(png|avif|jxl|jpg)`** *(1.3.0+)* — a region-local conservatism
  patch: the pixels inside that region's `bbox`, copied **before** downsampling
  (optionally pre-downscaled to `regions[].scale`), encoded with the *same* codec
  precedence as face crops. It carries real, sharp detail for a risky region so
  restore doesn't have to hallucinate it.
- **`region_mask_NNN.png`** *(1.3.0+)* — the region patch's feathered soft mask,
  identical in form and meaning to `face_mask_NNN.png`.
- **`residual.(avif|jxl|jpg)`** *(1.6.0+, only when `settings.residual` is true)* —
  the residual layer. At compress time FaceKeep decodes the background member it
  just encoded, bicubic-upscales it back to the original size (`INTER_CUBIC` —
  the same interpolation restore uses, a pinned contract), and computes the
  signed delta `original − upscale`. That delta is downscaled (`INTER_AREA`) to
  `settings.residual_scale` and **offset-encoded**, then stored:
  - **8-bit (default):** `clip(round(value/2 + 128), 0, 255)` into uint8 (halving
    costs ~1 bit of precision — fine for a correction layer), encoded as JXL at
    `residual_quality` (JPEG only as a warned fallback when the JXL plugin is
    unavailable; located `jxl → jpg`). Inverse: `value × 2 − 256`.
  - **High-bit (HDR), 1.9.0+:** when high-bit storage is engaged the original is
    uint16, so the 8-bit background's bicubic upscale is promoted to the 16-bit
    scale (`× 257`) before differencing, and the delta is offset-encoded as
    `clip(round(value/2 + 32768), 0, 65535)` into uint16, stored as a true
    10/12-bit `residual.avif` (via `avifenc`, 4:4:4). Inverse: `value × 2 −
    65536`. Restore promotes the upscale to 16-bit and adds the delta, so the
    background comes back at full depth.
- **`gainmap.jpg`** *(1.10.0+, only when `gain_map_preserved` is true)* — the
  source's iPhone HDR gain map, carried through from load (HEIC aux image /
  JPEG MPF second frame): a single-channel image, typically half the original
  resolution, stored upright (aligned with the original frame — the same
  orientation the restored image has). Grayscale JPEG q90 (Apple itself stores
  the map lossy). Semantics: `linear boost = 2^(headroom × value/255)` per
  pixel, with a headroom of ≈3 stops (validated value-for-value against
  libavif's own conversion of a real iPhone photo). Restore uses it to rebuild
  the fully-applied HDR alternate and embeds a gain map in the output AVIF via
  `avifgainmaputil combine`.

---

## How restore reconstructs the image

`facekeep restore x.fkeep` performs:

1. Read the manifest; take `original.width/height` and `settings.bg_scale`.
2. *(1.6.0+, only when a residual member is present)* **Reconstruct the
   background from real data instead**: bicubic-upscale the background member to
   the original dimensions, decode `residual.(avif|jxl|jpg)`, turn it back into
   the signed delta (`value × 2 − 256`; a high-bit `residual.avif` decodes uint16
   via `avifdec` with the delta `value × 2 − 65536`, and the upscale is promoted
   to the 16-bit scale `× 257` first so the result is uint16 HDR), resize it to
   full resolution (`INTER_CUBIC`) and add it on. The background is now **real
   (lossy) data**,
   so the AI upscale **and** the GFPGAN background-face step below are skipped —
   both exist to make hallucination plausible, and FaceKeep never replaces real
   pixels with a hallucination. Matched grain (see step 3) still applies; the
   low-frequency anchor is moot (the low band already is the stored
   background's). Steps 4-6 proceed unchanged.
3. **Otherwise, upscale the background member** (`background.(jpg|avif|jxl)`)
   back to the original dimensions. With the `[ai]`
   extra this uses Real-ESRGAN super-resolution; otherwise it falls back to
   bicubic. **This step invents detail** — the reconstructed background is
   plausible, *not* the original pixels. When the AI upscaler ran, its
   **low-frequency band is then re-anchored** to the real stored-background data
   (`aggressive.restore_anchor`, on by default): every spatial frequency below
   the stored background's Nyquist is measured data, so the AI's invented
   detail is kept but its color/brightness/large-structure drift is replaced
   with the real signal. Restore-only — it changes nothing in this format, and
   the bicubic fallback skips it (it is consistent with the stored background
   by construction). Finally, **matched grain is added** to the reconstructed
   background (`aggressive.restore_grain`, on by default): the grain level is
   estimated from the real face/region crops and synthesized as seeded,
   deterministic mono noise, so the too-smooth upscale stops being findable by
   texture against the grainy real crops composited next. Both AI and bicubic
   paths; skipped when the file has no crops. Restore-only — it changes
   nothing in this format.
4. **Composite each region patch** *(1.3.0+)*. For every entry in `regions[]`,
   paste `region_NNN.*` onto the upscaled background at its `bbox`, feathered by
   `region_mask_NNN.png`. These restore **real** sharp detail over the
   hallucinated upscale, so they go *under* the faces (next step). No-op on files
   with no `regions`.
5. **Composite each face.** For every entry in `faces[]`, paste `face_NNN.*` onto
   the background at `padded_bbox`, feathered by `face_mask_NNN.png` using
   `settings.blend_mode`.
6. **Write a standard image** at the format the output extension implies. The
   default is a universal `.jpg` (written via **Pillow**, not OpenCV, so it can
   carry a color profile); `--format avif`/`--format jxl` (or a `-o *.avif`/`*.jxl`
   path) instead writes a real AVIF/JXL through the faithful-mode codec. If
   `exif.bin` and/or `icc.bin` are present they are **re-embedded** — into JPEG in
   a single Pillow save carrying both EXIF and the ICC profile, and into AVIF/JXL
   by the codec. So a restored wide-gamut (e.g. Display P3) photo keeps its color;
   without the profile a viewer would fall back to sRGB and the image would look
   duller. (A `.fkeep` is never a dead end: every photo comes back as a file that
   opens anywhere. Point `restore` at a folder to un-fkeep a whole library at once.)
   *Note: AVIF carries the ICC bytes verbatim; JXL re-serializes the profile from
   its internal color model (still a valid embedded profile, just not byte-identical).*
7. *(1.10.0+, only when a `gainmap.jpg` member is present and the output is
   `.avif`)* **Re-attach the HDR gain map.** The restored base is linearized,
   boosted per pixel by `2^(headroom × gain)` (headroom =
   `aggressive.gain_map_headroom`, default 3 stops), PQ-encoded into an HDR
   alternate, and `avifgainmaputil combine` writes a **backward-compatible HDR
   AVIF**: SDR viewers show the base, HDR displays extend the highlights — the
   same mechanism as the original iPhone photo. EXIF rides along; color is
   declared via CICP (Display P3 → primaries 12) because the tool does not
   accept ICC-profiled inputs — equivalent for P3/sRGB sources, which is every
   iPhone. A non-`.avif` output, a machine without the `avifgainmaputil`
   binary, or any re-attach failure falls back to the normal SDR write above
   with a warning (offline-first; never a hard fail). The aggressive caveat
   applies honestly: the background under the gain map is reconstructed, so its
   HDR is approximate — the faces/patches are real pixels with real HDR boost.

The result is full-resolution: **real face pixels** over a **reconstructed
background**. The seam between them is hidden by the soft mask.

---

## Manual recovery without FaceKeep

Because a `.fkeep` is just a ZIP of standard files, you can recover your photos
even if FaceKeep is gone. This is the format's safety guarantee.

```bash
# 1. Inspect / extract — it's a ZIP.
unzip -l photo.fkeep            # list members
unzip photo.fkeep -d photo/     # extract everything

# 2. Read the manifest (plain JSON).
cat photo/manifest.json
```

What you can recover by hand, and how good it is:

- **The faces — at original quality.** `face_000.*`, `face_001.*`, … (`.jpg` by
  default, or `.avif`/`.jxl`/`.png` depending on `settings.face_codec`) are the
  real, full-resolution face crops. Open them directly in any image viewer; an
  `.avif`/`.jxl` crop needs a viewer that supports that format (any modern one).
- **The whole scene — at reduced quality.** `background.jpg` (or
  `background.avif`/`.jxl` when `settings.bg_codec` selected a modern codec —
  any modern viewer opens those) is the real image,
  just downscaled (by `settings.bg_scale`). Open it as-is for a smaller but
  faithful view of the entire photo, or upscale it yourself (any image editor's
  resize) and paste the face crops back at their `padded_bbox` coordinates for a
  full-resolution composite. Without an AI upscaler the background will be soft,
  but it is genuinely your photo, not a fabrication.
- **Risky regions — at original quality** *(1.3.0+)*. If the manifest has a
  `regions[]` array, `region_000.*`, … are real, near-original-resolution crops of
  the risky regions (e.g. the background around a small/distant face). Paste each
  back at its `regions[].bbox` over the upscaled background for a sharp result
  there — exactly what restore does, only by hand.
- **The lost background detail** *(1.6.0+, only if `settings.residual` is
  true)*. `residual.jxl` (or `.jpg`, or a high-bit `residual.avif` since 1.9.0)
  is the real high-frequency delta the downsample lost, offset-encoded. By hand:
  upscale the background member to the original size (bicubic), decode the
  residual, compute `value × 2 − 256` per pixel (it is stored as `value/2 + 128`;
  for a high-bit `residual.avif` it is uint16 stored as `value/2 + 32768`, so
  `value × 2 − 65536`, and promote the upscale `× 257` first), resize that delta
  to the full size, and add it to the upscaled background — the result is a
  faithful (if lossy) background, no AI needed. This is exactly what `facekeep
  restore` does on a residual-bearing file.
- **The HDR gain map** *(1.10.0+, only if `gain_map_preserved` is true)*.
  `gainmap.jpg` is the real per-pixel HDR boost map from the original iPhone
  photo — a plain grayscale JPEG any viewer opens. By hand: brighten the
  recovered image per pixel by `2^(3 × value/255)` in linear light for the
  full HDR rendition, or hand the base + map to any gain-map-aware tool
  (e.g. libavif's `avifgainmaputil combine`) — exactly what `facekeep restore
  -f avif` does.
- **A quick preview.** `thumbnail.jpg` is an immediate small preview of the
  whole image.
- **The color profile** *(1.4.0+)*. If the source was wide-gamut, `icc.bin` is the
  original ICC profile (e.g. Display P3). Any editor can attach/assign it to the
  recovered images so their color matches the original; FaceKeep re-embeds it
  automatically on restore.

What you **cannot** recover, by design: the original full-resolution background
*pixels*. They were discarded at compress time; only the downsampled
background member exists — plus, on a 1.6.0+ file with the opt-in residual
layer, the lossy half-res delta above, which narrows (but does not close) the
gap. (Faithful mode is the option that keeps the real pixels.)

---

## Integrity & verification

Use `facekeep verify` to check a `.fkeep` is complete and self-consistent
without restoring it:

```bash
facekeep verify photo.fkeep
facekeep verify photo.fkeep --original original.jpg   # also match the stored hash
```

It confirms the archive opens and the manifest parses; that the background
(`background.jpg`/`.avif`/`.jxl`, located in that order) and
`thumbnail.jpg` are present and decodable; that **for each declared face** both a
decodable crop (`face_NNN.png`/`.avif`/`.jxl`/`.jpg`) **and** a decodable
`face_mask_NNN.png` exist; that **for each declared region** (1.3.0+) both a
decodable `region_NNN.*` **and** `region_mask_NNN.png` exist; that the crop/mask
counts equal the declared face *and* region counts; that the background is
non-empty and no larger than the declared original; that every bounding box
(face and region) is well-formed; and (1.6.0+) that **when `settings.residual`
declares a residual layer** the `residual.(avif|jxl|jpg)` member is present and
decodes — a high-bit `residual.avif` is decoded with the Pillow plugin here
(8-bit) purely as a structural check, so `verify` needs no `avifdec` (a
residual-less file is unchanged).

A file that *opens but is inconsistent* (a missing crop, a count mismatch) is
reported as a structured list of problems — it is not treated as a crash. Only a
truly unreadable archive (corrupt ZIP, missing/invalid manifest) is a hard error.

Per the honesty note above, the stored `hash_sha256` cannot be self-verified from
the `.fkeep` alone, so `verify` reports a hash match **only** when you supply the
original file via `--original`; otherwise it states the hash was not checked
rather than implying it passed.

---

## Versioning & compatibility

- **`version`** is the manifest schema version (currently `1.9.0`) and is
  separate from `facekeep_version` (the tool version that wrote the file). Schema
  history: `1.2.0` added `settings.face_codec` (AVIF/JXL face crops); `1.3.0`
  added the `regions[]` array and the `region_NNN.*` / `region_mask_NNN.png`
  members (region-local conservatism); `1.4.0` added the optional `icc.bin` member
  and the `icc_preserved` flag (ICC color-profile preservation, e.g. Display P3);
  `1.5.0` added `settings.bg_codec` and the `background.avif`/`background.jxl`
  member variants (AVIF/JXL background — older files always have
  `background.jpg`, which readers try first); `1.6.0` added the opt-in residual
  layer — the `residual.(jxl|jpg)` member and the `settings.residual` /
  `residual_scale` / `residual_quality` keys (older files have neither, and a
  reader that ignores them still restores correctly via the normal upscale path,
  just without the residual's fidelity); `1.7.0` added the optional
  `settings.preset` key (the aggressive-mode preset the file was compressed
  with — a restore hint; presetless files carry no key at all and a reader that
  ignores it restores identically); `1.8.0` added the optional
  `settings.bit_depth` key and high-bit (10/12-bit) AVIF face/region crops (an
  HDR source via the `avifenc` CLI — the background stays 8-bit; absent on the
  default 8-bit container and older files, where every member is 8-bit, and a
  reader that ignores it still restores via the 8-bit Pillow decode); `1.9.0`
  extended high-bit storage to the **residual layer** (`residual.avif`, via
  `avifenc`, when the residual is on and high-bit storage is engaged) and widened
  `settings.bit_depth` to cover it — older files have no `residual.avif` and a
  reader that ignores it just restores via the normal upscale path (hallucinating
  the background), exactly as a residual-less file does; `1.10.0` added the
  optional `gainmap.jpg` member and the top-level `gain_map_preserved` flag
  (iPhone HDR gain-map preservation — `aggressive.preserve_gain_map`, on by
  default, stores the map only when the source carried one; `restore -f avif`
  re-attaches it into a backward-compatible HDR AVIF via `avifgainmaputil`) —
  older files have neither, and a reader that ignores them restores SDR exactly
  as before Phase 9.
- Readers are **tolerant by structure, not by strict schema validation**: they
  locate members by name (`background.(jpg|avif|jxl)`, `face_NNN.*`,
  `face_mask_NNN.png`,
  `region_NNN.*`, `exif.bin`, `icc.bin`) and read the fields they need from the
  manifest, rather than rejecting a file for unknown extra keys. New optional keys
  can therefore be added without breaking older readers — an older reader that
  ignores `regions` still restores correctly from the global background + faces
  (just without the region patches' sharpness), and one that ignores `icc.bin`
  restores correctly without the color profile.
- The container framing (ZIP) and member naming (`%03d` zero-padded indices,
  fixed member names) are the stable contract this document specifies. Treat the
  member names and the `faces[]`↔`face_NNN.*` (and `regions[]`↔`region_NNN.*`)
  index correspondence as load-bearing.
