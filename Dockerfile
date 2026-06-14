# syntax=docker/dockerfile:1
#
# FaceKeep container image — reproducible batch runs.
#
# Two build targets:
#   slim (DEFAULT) — faithful mode only. Lean, fully offline, no torch.
#                    `docker build -t facekeep .`
#   ai             — + aggressive-mode AI restore (Real-ESRGAN / GFPGAN / LPIPS)
#                    on CPU-only torch. Multi-GB.
#                    `docker build --target ai -t facekeep:ai .`
#
# Run a volume-mounted batch (see docs/docker.md for the full guide):
#   docker run --rm -v "$PWD:/work" facekeep compress photos/ -o compressed/
#
# Note: the AVIF/JXL codecs come from the pip wheels (pillow-avif-plugin /
# pillow-jxl-plugin) which bundle their own native libavif/libjxl — so the
# faithful image needs NO apt libavif/libjxl and stays offline. True 10/12-bit
# AVIF output (the optional external `avifenc` CLI) is deliberately NOT baked in;
# high-bit sources gracefully round down to 8-bit, exactly like a plain
# `pip install` (offline-first default unchanged). Add it yourself with
# `apt-get install -y libavif-bin` if you want the high-bit path in-container.

# ---- base: system libs + the FaceKeep package (faithful-complete) ----------
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="FaceKeep" \
      org.opencontainers.image.description="Face-aware photo compression (faithful AVIF/JXL default; optional aggressive AI-restore mode)." \
      org.opencontainers.image.source="https://github.com/wadeken/facekeep" \
      org.opencontainers.image.licenses="MIT"

# opencv-python (non-headless, as pinned by the project) links libGL and glib at
# import time; on the slim base those aren't present. These two apt packages are
# the minimal set that makes `import cv2` work. We keep the project's pinned
# opencv-python rather than swapping to -headless (the cv2 build is load-bearing
# elsewhere — see CLAUDE.md).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install the package. Only the files the build/metadata needs are copied (the
# .dockerignore trims the rest of the context): pyproject reads README for the
# long description, the dynamic version reads facekeep/__init__.py, and LICENSE
# completes the wheel. /src is kept so the `ai` target can resolve the [ai] extra
# from the same single source of truth (pyproject), not a hand-copied dep list.
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY facekeep ./facekeep
RUN pip install .

# Run as a non-root user. HOME holds the model cache (~/.cache/facekeep); mount a
# named volume there to persist AI weights across runs (see docs/docker.md).
RUN useradd --create-home --uid 10001 app
ENV HOME=/home/app
WORKDIR /work
USER app

ENTRYPOINT ["facekeep"]
CMD ["--help"]

# ---- ai (opt-in target): + aggressive-mode AI restore, CPU-only torch -------
FROM base AS ai
USER root
# Install CPU-only torch/torchvision FIRST from PyTorch's CPU index — the default
# PyPI wheels are CUDA-enabled and multi-GB. Then `[ai]` resolves its torch
# constraint against the already-installed CPU build (and pulls Real-ESRGAN /
# basicsr / GFPGAN / LPIPS). The [ai] extra stays the single source of truth for
# the dep set. mediapipe is deliberately NOT installed (its wheel ships its own
# cv2 and would clobber the pinned OpenCV — CLAUDE.md).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install "/src[ai]"
USER app

# ---- slim (DEFAULT target): faithful mode only ------------------------------
# Identical to `base`; named and placed LAST so a plain `docker build .` (no
# --target) produces the lean faithful image, matching "faithful is the default".
FROM base AS slim
