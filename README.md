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

Alpha. Both modes are implemented and tested; quality tuning and distribution
(PyPI) are ongoing.

## License

MIT — see [LICENSE](LICENSE).
