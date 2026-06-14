"""Centralized model/weights cache for FaceKeep.

One place that downloads, **checksum-verifies**, and caches model weights under a
shared directory (``~/.cache/facekeep/models``) — the same dir the YuNet detector
already uses. The aggressive-mode AI restore (Real-ESRGAN, GFPGAN) routes its
weights through :func:`ensure_weights` so they live in the documented cache (not
buried inside ``site-packages``), are validated against a known SHA-256, and fail
with a clear, ``[ai]``-pointing error when offline — instead of whatever the
underlying package throws.

Offline-first stays intact: this only runs on the *opt-in* AI restore path. A
download or checksum failure raises :class:`~facekeep.exceptions.ModelDownloadError`,
which the restore code catches to degrade gracefully (bicubic / skip-enhance).
The default faithful pipeline never touches this module.
"""

import hashlib
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .exceptions import ModelDownloadError

logger = logging.getLogger("facekeep.models")

# Shared on-disk cache for every model FaceKeep downloads. The YuNet detector
# points its own cache constant at this directory too, so there is a single
# source of truth for "where downloaded models live".
MODELS_CACHE_DIR = Path.home() / ".cache" / "facekeep" / "models"

# Minimum plausible size for a real weights file; anything smaller is almost
# certainly an error page or a truncated download, regardless of checksum.
_MIN_VALID_BYTES = 10_000


def _sha256(path: Path) -> str:
    """SHA-256 hex digest of a file, read in chunks (weights are tens of MB)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(
    url: str,
    filename: str,
    *,
    sha256: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Return a local path to a verified weights file, downloading if needed.

    Args:
        url: The ``https://`` URL to download the weights from on a cache miss.
        filename: The name to cache the file under in ``cache_dir``.
        sha256: Expected lowercase SHA-256 hex digest. When given, a cached file
            is trusted only if it matches (a mismatch is treated as a corrupt
            cache and re-downloaded once), and a fresh download is verified
            before being committed. ``None`` skips verification.
        cache_dir: Override the cache directory (defaults to ``MODELS_CACHE_DIR``).

    Returns:
        Path to the cached, verified weights file.

    Raises:
        ModelDownloadError: if the file cannot be downloaded or fails its
            checksum. The message points at the ``[ai]`` extra / offline use.
            Callers on the AI path catch this and fall back gracefully.
    """
    cache_dir = cache_dir or MODELS_CACHE_DIR
    dest = cache_dir / filename

    # Cache hit: a real-sized file that (if a checksum is known) matches it.
    if dest.exists() and dest.stat().st_size >= _MIN_VALID_BYTES:
        if sha256 is None or _sha256(dest) == sha256.lower():
            return dest
        logger.warning(
            "Cached model %s failed checksum; re-downloading.", dest.name
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model %s to %s", filename, dest)

    # Download to a temp file in the same dir, then atomically replace, so an
    # interrupted or corrupt download never leaves a half-written cache entry a
    # later run would load as valid.
    tmp = dest.with_name(dest.name + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "facekeep"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise ModelDownloadError(
            f"Could not download model {filename!r} from {url} ({e}). "
            "An internet connection is required the first time, after which the "
            "weights are cached. Install the AI extra with: pip install facekeep[ai]"
        ) from e

    if len(data) < _MIN_VALID_BYTES:
        raise ModelDownloadError(
            f"Downloaded model {filename!r} looks invalid (only {len(data)} bytes)."
        )

    if sha256 is not None:
        got = hashlib.sha256(data).hexdigest()
        if got != sha256.lower():
            raise ModelDownloadError(
                f"Model {filename!r} failed checksum verification "
                f"(expected {sha256.lower()}, got {got}). The download may be "
                "corrupt or the URL may have changed."
            )

    tmp.write_bytes(data)
    os.replace(tmp, dest)  # atomic on the same filesystem
    return dest
