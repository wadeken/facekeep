# Running FaceKeep in Docker

A container is the simplest way to get **reproducible batch runs** without
installing Python, the codecs, or (for aggressive mode) the heavy AI stack on
the host. The [Dockerfile](../Dockerfile) ships **two build targets**:

| Target | Contents | Size | Use it for |
| --- | --- | --- | --- |
| **`slim`** (default) | Faithful mode only — AVIF/JXL encode. No torch. | Lean | Shrinking a library to standard `.avif`/`.jxl`. Fully offline. |
| **`ai`** (opt-in) | `slim` **+** aggressive-mode AI restore (Real-ESRGAN / GFPGAN / LPIPS) on **CPU-only** torch. | Multi-GB | Compressing to `.fkeep` *and* restoring with AI super-resolution. |

The split mirrors the project itself: **faithful is the default and stays
offline/zero-download**, AI is an explicit opt-in. A plain `docker build .`
gives you the lean image.

> **Running on a NAS?** The repo also ships a
> [`docker-compose.yml`](../docker-compose.yml) that wraps these targets for
> scheduled library backups (Synology Container Manager / QNAP / Unraid). See
> [nas.md](nas.md).

## Build

```bash
# Faithful image (default target) — small, offline
docker build -t facekeep .

# AI image (aggressive-mode restore) — adds CPU torch + Real-ESRGAN/GFPGAN/LPIPS
docker build --target ai -t facekeep:ai .
```

> The AVIF/JXL codecs come from the bundled pip wheels (`pillow-avif-plugin` /
> `pillow-jxl-plugin`), so the faithful image needs **no** system libavif/libjxl
> and builds the same everywhere. `mediapipe` is deliberately left out of both
> images (its wheel ships its own OpenCV and would clobber the pinned build).

## Run — volume-mounted batch

The image's entry point is `facekeep`, and its working directory is `/work`.
Mount your photo folder at `/work` and pass any normal `facekeep` command:

```bash
# Faithful: compress a folder to ./compressed (standard .avif files)
docker run --rm -v "$PWD:/work" facekeep compress photos/ -o compressed/

# Try both codecs per image and keep the smaller, across 4 worker processes
docker run --rm -v "$PWD:/work" facekeep compress photos/ --codec both --jobs 4

# No arguments → prints help
docker run --rm facekeep
```

### File ownership on Linux

The container runs as a non-root user (`uid 10001`). Files it writes to the
mounted folder are owned by that uid on the host. To have outputs owned by *you*,
override the uid/gid:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" facekeep compress photos/ -o out/
```

(On Windows/macOS Docker Desktop handles ownership for you; the override is a
Linux concern.)

## Aggressive mode (the `ai` image)

Compress to `.fkeep` and restore with AI super-resolution. The AI models
(Real-ESRGAN / GFPGAN weights) download on first use into `~/.cache/facekeep`
**inside** the container — which is ephemeral. Mount a **named volume** there so
the weights are fetched once and reused (and so a run is reproducible/offline
afterward):

```bash
# Compress to .fkeep (aggressive)
docker run --rm -v "$PWD:/work" facekeep:ai \
    compress photos/ -m aggressive -o fkeeps/

# Restore with AI — persist the model cache across runs in a named volume.
# --tile bounds peak memory on large frames (recommended on CPU).
docker run --rm \
    -v "$PWD:/work" \
    -v facekeep-models:/home/app/.cache/facekeep \
    facekeep:ai \
    restore fkeeps/photo.fkeep -o restored/photo.jpg --tile 512
```

Without the `[ai]` extra a restore still works — it falls back to a bicubic
upscale — so the **default `slim` image can also restore** (just without AI
super-resolution):

```bash
docker run --rm -v "$PWD:/work" facekeep restore fkeeps/photo.fkeep -o restored/
```

> CPU AI restore is slow (minutes per image) and memory-hungry — always pass
> `--tile` on large photos. There is no GPU support baked in; the image installs
> **CPU-only** torch so it runs anywhere. GPU users should build their own image
> with a CUDA torch wheel.

## Notes

- **Config file.** `facekeep` auto-discovers a `facekeep.yaml` in the working
  directory, so a mounted `/work/facekeep.yaml` is picked up automatically. Run
  `docker run --rm -v "$PWD:/work" facekeep init` to write a commented template.
- **High-bit (10/12-bit) AVIF** is not enabled by default (high-bit sources
  round down to 8-bit with a warning, same as a plain `pip install`). To enable
  it in-container, add `RUN apt-get update && apt-get install -y --no-install-recommends libavif-bin`
  to the Dockerfile — `avifenc` then provides the high-bit path.
- **Pinning for strict reproducibility.** The base is `python:3.11-slim`; pin it
  to a digest (`python:3.11-slim@sha256:…`) if you need bit-for-bit repeatable
  builds.
