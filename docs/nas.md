# Running FaceKeep on a NAS (Synology / QNAP / Unraid)

FaceKeep is a natural fit for a NAS: point it at your photo share, and it
shrinks the library on a schedule. The simplest way to run it there is the
[`docker-compose.yml`](../docker-compose.yml) at the repo root — a thin wrapper
over the same two-target image documented in [docker.md](docker.md). It works in
**Synology Container Manager**, **QNAP Container Station**, **Unraid**, or any
host with `docker compose`.

> Why compose and not a native Synology `.spk` package? Compose runs on every
> NAS that has Docker, builds directly on the existing image, and needs no
> Synology-specific toolchain — one artifact for all NAS brands. (A native DSM
> package would be Synology-only.)

## What the compose file gives you

Two services:

| Service | Image target | What it does |
| --- | --- | --- |
| **`facekeep`** (default) | `slim` (faithful, no torch) | One-shot **batch backup**: compress everything under the photo folder into the output folder, then exit. |
| **`ai`** (profile `ai`) | `ai` (+ CPU torch / Real-ESRGAN / GFPGAN) | Aggressive-mode compress to `.fkeep` and AI restore. Pulled only when you ask for the `ai` profile, so a normal run never downloads the multi-GB image. |

Both mount your **source photos read-only** (a backup tool must never modify
originals) and write to a separate output folder.

## Quick start

From an SSH session on the NAS (or any Docker host), in the repo directory:

```bash
# Build the lean faithful image once
docker compose build facekeep

# Back up a share: compress /volume1/photos -> /volume1/compressed
FACEKEEP_PHOTOS=/volume1/photos FACEKEEP_OUT=/volume1/compressed \
  docker compose run --rm facekeep
```

`docker compose run --rm facekeep` runs the default backup command
(`compress /work/photos -o /work/compressed`) once and removes the container.
Override it with any `facekeep` command:

```bash
# Try both codecs and keep the smaller, 4 workers
FACEKEEP_PHOTOS=/volume1/photos FACEKEEP_OUT=/volume1/compressed \
  docker compose run --rm facekeep compress /work/photos -o /work/compressed --codec both --jobs 4

# Inspect a result
docker compose run --rm facekeep info /work/compressed/trip.fkeep
```

### Configuration knobs

These environment variables parameterize the compose file:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FACEKEEP_PHOTOS` | `./photos` | Host path to the source photo folder (mounted read-only at `/work/photos`). |
| `FACEKEEP_OUT` | `./compressed` | Host path for compressed output (and the incremental index). |
| `PUID` / `PGID` | `10001` | uid/gid the `facekeep` service runs as, so outputs are owned by your NAS user. Find yours with `id` over SSH. |

Put them in a `.env` file next to `docker-compose.yml` so you don't repeat them:

```dotenv
FACEKEEP_PHOTOS=/volume1/photos
FACEKEEP_OUT=/volume1/compressed
PUID=1026
PGID=100
```

## Synology Container Manager

DSM 7.2+ ships **Container Manager** (older DSM: "Docker"), which can run a
compose file as a *Project*:

1. Copy this repo (or at least `Dockerfile`, `docker-compose.yml`,
   `pyproject.toml`, `README.md`, `LICENSE`, and the `facekeep/` folder) to a
   shared folder, e.g. `/volume1/docker/facekeep`.
2. Container Manager → **Project** → **Create** → set the path to that folder;
   it picks up `docker-compose.yml`.
3. Edit the `FACEKEEP_PHOTOS` / `FACEKEEP_OUT` paths (or add a `.env`) to point
   at your photo share, then build and run.

The container's working directory is `/work` and its entry point is `facekeep`,
so the mounted paths in the commands (`/work/photos`, `/work/compressed`) are
what the service sees.

## Scheduling periodic backups

A backup tool earns its keep by running automatically. Use **DSM Control Panel →
Task Scheduler → Create → Scheduled Task → User-defined script** (run as a user
in the `docker` group), on whatever cadence you like (e.g. nightly):

```bash
cd /volume1/docker/facekeep && /usr/local/bin/docker compose run --rm facekeep
```

(QNAP: use *crontab*; Unraid: the *User Scripts* plugin — same `docker compose
run --rm facekeep` line.)

Re-runs are cheap and safe: FaceKeep keeps an **incremental index** in the
output folder, so a scheduled job **skips every photo that is byte-identical to
the last run** with the same settings — only new or changed photos are
re-encoded. The first run does the bulk of the work; later runs are quick.

## File ownership

The container runs as `PUID:PGID` (default `10001:10001`). Set them to your NAS
user so the compressed files are owned by you rather than by an anonymous uid:

```bash
PUID=$(id -u) PGID=$(id -g) docker compose run --rm facekeep
```

## Aggressive mode + AI restore (the `ai` profile)

Aggressive mode produces smaller `.fkeep` files whose background is
reconstructed with AI on restore. It needs the heavier `ai` image, gated behind
the `ai` profile so it is never pulled by accident:

```bash
# Build the AI image once (multi-GB; CPU-only torch)
docker compose --profile ai build ai

# Compress to .fkeep (aggressive)
FACEKEEP_PHOTOS=/volume1/photos FACEKEEP_OUT=/volume1/fkeeps \
  docker compose --profile ai run --rm ai

# Restore a .fkeep with AI super-resolution (--tile bounds memory on CPU)
FACEKEEP_OUT=/volume1/fkeeps docker compose --profile ai run --rm ai \
  restore /work/compressed/trip.fkeep -o /work/compressed/ --tile 512
```

The AI weights download on first use into a **named volume**
(`facekeep-models`), so they are fetched once and reused across runs. The `ai`
service runs as the image's app user (uid 10001), which owns that volume — keep
it that way (don't set `PUID` on `ai`) so weight downloads stay writable.

> CPU AI restore is **slow** (minutes per image) and memory-hungry — always pass
> `--tile` on large photos, and prefer a NAS with plenty of RAM. There is no GPU
> support baked in. Faithful mode (the default `facekeep` service) needs none of
> this and is the right choice for most backups.

## Notes

- **Config file.** `facekeep` auto-discovers a `facekeep.yaml` in `/work`, so a
  `facekeep.yaml` at the root of your mounted output/work folder is picked up.
  Generate one with `docker compose run --rm facekeep init /work/facekeep.yaml`.
- **Originals are safe.** The photo folder is mounted read-only; FaceKeep only
  writes to the output folder.
- **High-bit (10/12-bit) AVIF** is not enabled in the image (high-bit sources
  round down to 8-bit with a warning) — see [docker.md](docker.md) for how to add
  the optional `avifenc` CLI if you need it.
