# FaceKeep

<p align="center">
  <img src="https://raw.githubusercontent.com/wadeken/facekeep/main/assets/hero.png"
       alt="FaceKeep — face-aware photo compression: shrink a photo library 8–12× while keeping every face pixel-perfect"
       width="100%">
</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Face-aware photo compression that never ruins a face — shrink a family photo
library ~8–12×, with every face kept pixel-perfect.**

FaceKeep compresses a photo library *hard* by spending bytes only where your eye
goes. Its headline **aggressive mode** keeps every **face — plus hands, fine
detail, and (opt-in) signage — at original quality**, downsamples only the benign
background (sky, foliage, plain walls), and reconstructs that background on
restore. You get roughly **8–12× smaller** than JPEG — **up to ~40×** on photos
that are mostly scenery — with the people still sharp.

Naive "AI shrink" tools downsample the *whole* photo and hallucinate it back —
faces included, which come back uncanny. FaceKeep won't: faces are detected and
protected, never invented. And because a reconstructed background is *plausible,
not the original*, FaceKeep is honest about the trade — aggressive output is a
`.fkeep` you bring back with one `restore` command, and when you want **every
pixel real in a file that opens anywhere**, the **faithful default mode** does
exactly that.

And FaceKeep has grown past still photos: **videos** re-encode faithfully
(SVT-AV1, measured **2–14× smaller** on real phone clips, quality-gated by
VMAF), **phone HDR survives** in both modes (iPhone/Android gain maps are
carried, not flattened), and **`facekeep watch`** keeps a synced folder
compressed automatically.

## See it work

<p align="center">
  <img src="https://raw.githubusercontent.com/wadeken/facekeep/main/assets/proof.png"
       alt="FaceKeep before/after: a 10.5 MB family photo compressed to 260 KB (41× smaller) with every face kept pixel-perfect"
       width="100%">
</p>

A real 17-megapixel family photo in aggressive mode: **10.5 MB → 260 KB (41×
smaller)**. The three faces are stored as original-quality crops — **byte-for-byte
identical, not reconstructed** (see the zoom strip) — while only the benign
background (sky, mountains, lake, foliage) is downsampled and rebuilt on restore.
Restored above with real Real-ESRGAN: smooth content is indistinguishable and
fine detail holds up under a 100% zoom.

> **41× is a scenic, few-face best case** — people-heavy photos land nearer the
> typical 8–12×. And the background is *plausible, not the original*: that's the
> aggressive-mode trade. Need every pixel real? The faithful default keeps them.

## Two modes

| | **aggressive** (the hero) | **faithful** (default) |
|---|---|---|
| For | Shrink a whole library, dramatically | A faithful, universal backup |
| How | Keep faces/hands/detail real, rebuild the background | Whole image → AVIF/JXL, all real pixels |
| Output | `.fkeep` (a documented ZIP) | `.avif` / `.jxl` (standard) |
| Restore step | `facekeep restore` | None — just open it |
| Background | Reconstructed (plausible) | Real pixels, efficiently coded |
| Fidelity | The people are real; background is plausible | Visually lossless (SSIM > 0.98) |
| Typical ratio | ~8–12× vs JPEG (up to ~40×) | ~2.5–3× vs JPEG |
| Opens anywhere | Needs `facekeep restore` | ✅ |
| Videos | — (photos only) | ✅ AV1 re-encode (`.mp4`) |
| Needs AI / torch | Yes, for the best restore (bicubic without) | No |

> **Faithful stays the default** for a bare `facekeep compress`: a backup tool
> shouldn't silently hand you a reconstructed background. Reach for aggressive
> mode (or a preset) when shrinking the library matters more than a pixel-exact
> background.

## Quick start

```bash
pip install "facekeep[ai]"            # aggressive mode with AI restore (Real-ESRGAN; pulls torch)

# Shrink a whole folder with a one-word goal (see Presets)
facekeep compress ./photos -o ./small --preset family

# Bring a photo back to a standard image
facekeep restore ./small/photo.fkeep -o photo.jpg
```

Faithful-only, no heavy deps:

```bash
pip install facekeep                  # faithful mode
facekeep compress photo.jpg           # -> photo.avif, opens in any modern viewer
```

Requires Python 3.10+. Faithful mode needs no GPU and no model downloads.
(Aggressive mode also runs without `[ai]` — it just restores with a bicubic
upscale instead of Real-ESRGAN.)

## Presets — one word instead of twenty knobs

Aggressive mode has many knobs; a preset names the *goal* and expands to a tuned
bundle. Every underlying field stays individually overridable.

| Preset | Goal |
|---|---|
| `family` | Never ruin a face or hand — the strongest people protection |
| `ratio` | The smallest possible `.fkeep` |
| `pretty` | The best-looking restore |
| `fidelity` | Closest to the original (stores a real-detail residual layer) |
| `share` | Small **and** private — strips GPS, for sending out |

```bash
facekeep compress ./photos --preset family     # implies aggressive mode
```

## Videos too

`facekeep compress` takes the **videos** in your camera roll along with the
photos. Phones record with a real-time hardware encoder that buys quality with
bitrate; FaceKeep re-encodes offline with **SVT-AV1**, spending the time your
phone couldn't — measured **2–14× smaller** on real phone clips at
visually-lossless quality. Every encode is checked by a **VMAF quality gate**
(scored against the source, re-encoded at higher quality on a miss — and if
AV1 can't beat the original, the original is kept). The output is a standard
`.mp4` that plays anywhere modern: opening it *is* the restore, exactly like
faithful photos.

Measured on the real phone clips that drove development (all defaults):

| Real phone clip | Original | FaceKeep | VMAF p1 |
|---|---|---|---|
| Android 4K30 HLG (31 Mbps HEVC) | 45.7 MB | **5.2 MB — 8.8× smaller** | 95.4 |
| iPhone 4K30 Dolby Vision (25 Mbps HEVC) | 44.3 MB | **23.0 MB — 1.9× smaller** | 96.4 |

The iPhone clip has a person in it, so **face-aware quality** raised the VMAF
target and the gate re-encoded once at higher quality — bytes spent exactly
where the design says they should be. Opt-in per-clip auto-tune
(`video.auto_tune`) pushed the Android clip to **13.4×**, still gate-confirmed
at p1 93. (VMAF p1 = the worst-1%-frame score; ~95 is visually lossless.)

What survives the re-encode (verified on real phone footage):

- **HDR** — 10-bit and HLG carry through, and so does the per-frame
  **Dolby Vision** metadata phones record (frame-for-frame).
- **Variable frame rate** — timestamps pass through verbatim, so audio never
  drifts out of sync.
- **Faces raise the bar** — a clip with people in it gets a higher VMAF
  target: the video analog of the photo modes' face-aware quality.
- **Live Photos** — a paired `.mov` is kept verbatim, never re-encoded: a
  re-encode would silently break the Live-Photo pairing to save ~1–3 MB.

```bash
facekeep compress clip.mov                  # -> clip.mp4 (AV1)
facekeep compress camera_roll/              # photos + videos in one run
facekeep compress camera_roll/ --no-videos  # photos only this run
```

> **Honest costs:** video needs an external `ffmpeg` (built with SVT-AV1;
> libvmaf for the quality gate) on your machine — an opt-in binary, never a
> Python dependency. Without it, videos are skipped with an install hint and
> photos are unaffected. Encoding is *slow* by design (~4 minutes per minute
> of 4K on a desktop CPU) — an overnight-batch feature. And video is
> **faithful-only**: aggressive mode never applies to a video.

## Your HDR stays HDR

A modern phone "HDR photo" isn't deep-color pixels — it's an SDR base **plus a
gain map** the display multiplies on. Most tools silently drop the map, and
the photo comes back flat. FaceKeep carries it, in both modes:

- **Faithful mode** re-encodes an iPhone/Android HDR photo as a
  backward-compatible **HDR AVIF**: SDR viewers see the normal image, HDR
  displays get the real highlights — the same mechanism as the original file.
- **Aggressive mode** stores the gain map inside the `.fkeep` and re-attaches
  it on restore: the default `.jpg` output becomes an **Ultra HDR JPEG**
  (built in — pure Python, no external tool), and `restore -f avif` an HDR
  AVIF. An Android Ultra HDR source keeps its own declared HDR math (verified
  against a real Android photo, not just iPhone).

Measured on real phone photos: an iPhone HDR photo went **7.5 MB → 4.7 MB**
as a faithful HDR AVIF, and a real Android Ultra HDR photo **2.8 MB → 0.5 MB
(5.3×)** faithful — or a **443 KB `.fkeep` (6.5×)** in aggressive mode, with
the gain map riding byte-for-byte back into the restored Ultra HDR JPEG.

**The pair below is the proof, not a screenshot of it.** Both are the same
FaceKeep aggressive-mode restore of a real Android phone photo. The left one
had its gain map stripped — what a map-dropping tool leaves you. The right one
is FaceKeep's actual output: an Ultra HDR JPEG rebuilt from a `.fkeep`, the
gain map carried byte-for-byte. View this page in Chrome or Edge **on an HDR
display**: only the right lamp genuinely glows brighter than the page's white
— something no screenshot can fake. (On an SDR screen the two look identical;
that's the backward-compatible design doing its job.)

<table align="center">
  <tr>
    <th align="center">gain map dropped (most tools)</th>
    <th align="center">FaceKeep restore — Ultra HDR kept</th>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/wadeken/facekeep/main/assets/hdr_proof_sdr.jpg"
             alt="SDR reference: the same restored photo with the HDR gain map stripped — the lamp stays flat even on an HDR display"
             width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/wadeken/facekeep/main/assets/hdr_proof.jpg"
             alt="FaceKeep HDR proof: an actual Ultra HDR JPEG restored from a .fkeep — on an HDR display in Chrome the ceiling lamp glows beyond SDR white"
             width="100%"></td>
  </tr>
</table>

> The faithful-mode HDR-AVIF carry uses libavif's `avifgainmaputil` binary
> (opt-in, machine-local); without it you get today's SDR output with a
> warning. The aggressive-restore Ultra HDR JPEG needs nothing extra.

## Set-and-forget: `facekeep watch`

Point FaceKeep at the folder your phone already syncs into, and everything
that lands there gets compressed into your archive automatically:

```bash
facekeep watch inbox/ -o archive/           # scan -> compress new files -> sleep -> repeat
facekeep watch inbox/ -o archive/ --once    # one pass, for Task Scheduler / cron
```

It rides any phone→computer transport — iCloud for Windows, OneDrive / Google
Drive camera upload, Syncthing, a manual import folder. Idle cycles are
near-free (metadata-only checks, no re-hashing), a file still mid-sync is left
alone until it stops changing, and **your source files are never deleted or
modified** — cleanup stays yours.

## Why FaceKeep

The failure that matters in a family photo is a **melted or uncanny face** —
emotionally worse than a slightly soft background. Every other aggressive
compressor risks it, because they downsample and reconstruct *everything*.
FaceKeep's whole design is built around *not* doing that:

- **Protect every face** with robust detection (Haar offline by default, YuNet
  opt-in), including small and distant background faces.
- **Protect more than faces** — hands and fine structure are kept as
  original-quality patches (signage/text too, opt-in), because the AI upscaler
  smears exactly those.
- **Reconstruct only what's safe** — sky, foliage, bokeh, plain walls, where
  invented detail is invisible.
- **Be honest about the rest** — a reconstructed background is *plausible, not
  faithful*; when you need the real pixels, the faithful default keeps them in a
  standard file.

See [docs/architecture.md](docs/architecture.md) for the full design and the
information-theory reasoning behind the two modes.

## Install

```bash
pip install facekeep                 # faithful mode (no heavy deps)
pip install "facekeep[ai]"           # + aggressive-mode AI restore
pip install "facekeep[heic]"         # + HEIC/HEIF input
pip install "facekeep[gui]"          # + local drag-and-drop web GUI
```

> **iPhone HEIC/HEIF?** Add the `[heic]` extra above. Without it, `.heic`/`.heif`
> inputs are skipped with a one-line install hint (your other photos still
> process). HEIC is already an efficient format, so it shrinks most in
> **aggressive** mode (`-m aggressive`) — a faithful re-encode saves only a little.

> **Optional external binaries** (never Python dependencies): video needs
> `ffmpeg` (with SVT-AV1 + libvmaf), and the HDR-AVIF paths use libavif's
> `avifenc` / `avifgainmaputil`. A missing binary always degrades gracefully —
> a clear warning and a sensible fallback, never a crash.

### Docker

For reproducible batch runs without installing anything on the host:

```bash
docker build -t facekeep .                                  # faithful image (lean, offline)
docker run --rm -v "$PWD:/work" facekeep compress photos/ -o compressed/

docker build --target ai -t facekeep:ai .                   # + aggressive-mode AI restore
```

See [docs/docker.md](docs/docker.md) for the full guide (volume mounts, the AI
image, and a persistent model-cache volume).

**On a NAS** (Synology / QNAP / Unraid)? The repo ships a
[`docker-compose.yml`](docker-compose.yml) for scheduled library backups — see
[docs/nas.md](docs/nas.md) (Container Manager setup + DSM Task Scheduler).

### Web GUI

Prefer clicking to typing? Launch a local drag-and-drop interface:

```bash
pip install "facekeep[gui]"
facekeep gui                          # opens http://127.0.0.1:7860
```

Drop a photo in, pick a mode, and see a before/after with the size/ratio and a
download of the result. A second **Compare** tab pairs an original against any
compressed output (a faithful `.avif`/`.jxl`/`.webp`, or a `.fkeep` reconstructed
on the fly — the instant bicubic preview, or an opt-in real AI restore that warns
it's slow and shows a spinner) with a live before/after wipe slider, a difference
heatmap, and SSIM/PSNR — the interactive sibling of `facekeep compare`. It runs
**locally only** — no public link and no telemetry. (It's a thin wrapper over the
same engine as the CLI, so the output is identical.)

## Usage

```bash
# Aggressive (the hero): dramatic shrink, faces kept perfect -> .fkeep
facekeep compress photo.jpg -m aggressive
facekeep compress ./photos --preset family       # one-word goal (implies aggressive)
facekeep compress photo.jpg -m aggressive --bg-scale 0.2

# Restore an aggressive-mode file to a standard image
facekeep restore photo.fkeep -o restored.jpg
facekeep restore photo.fkeep --preview           # fast, no AI
facekeep info photo.fkeep                         # inspect a .fkeep
facekeep verify photo.fkeep                       # structural integrity

# Faithful (default): photo.jpg -> photo.avif, opens in any modern viewer
facekeep compress photo.jpg
facekeep compress photo.jpg --codec jxl -q 85
facekeep compress ./photos -o ./compressed

# Video (faithful-only; needs ffmpeg with SVT-AV1)
facekeep compress clip.mov                        # -> clip.mp4 (AV1, VMAF-gated)
facekeep compress camera_roll/ --no-videos        # skip videos this run

# Keep a synced folder compressed automatically
facekeep watch inbox/ -o archive/                 # or --once for Task Scheduler / cron

# Compare / report on any pair
facekeep quality original.jpg compressed.avif
facekeep compare original.jpg photo.fkeep         # before/after HTML report (restores on the fly)
```

## How it works

**Aggressive mode** — detect and crop every face (plus hands and other risky
regions) at original quality, downsample the benign background, and pack it into
a `.fkeep` (a documented ZIP). On `restore`, the background is upscaled
(Real-ESRGAN, or bicubic without `[ai]`), grain-matched so it isn't plastic-
smooth, and the real face/region patches are composited back with feathered
masks. **Your photos aren't trapped:** a `.fkeep` is a plain ZIP of standard
images plus a JSON manifest, and `restore` always produces a standard file — see
[docs/fkeep-format.md](docs/fkeep-format.md).

**Faithful mode** — load (EXIF orientation applied, EXIF + ICC profile
preserved), detect faces to pick chroma subsampling (4:4:4 on faces keeps
skin-tone/lips crisp) and auto-tune quality so the face region hits a
visually-lossless target, then encode the whole image with AVIF (or JPEG XL) and
write one standard file. No container, no restore — opening the file *is* the
restore.

## Project status

Alpha. Both photo modes, the faithful video re-encode, HDR gain-map carry, and
`facekeep watch` are implemented and tested; quality tuning and distribution
(PyPI) are ongoing.

## License

MIT — see [LICENSE](LICENSE).
