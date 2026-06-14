#!/usr/bin/env python3
"""Download the FaceKeep real-photo test corpus into a local cache.

The corpus is a handful of license-clear photos (Public Domain / CC-BY) used by
``tests/test_corpus.py`` to exercise real-world detection, compression ratio,
and fidelity — things synthetic fixtures cannot. The images are deliberately
**not** committed to the repo (keeps it light, avoids redistributing CC-BY
binaries); this script fetches them on demand, and the corpus tests skip when
the cache is missing (e.g. offline / CI without network).

Usage::

    python tests/corpus/download.py            # download missing files
    python tests/corpus/download.py --force    # re-download everything
    python tests/corpus/download.py --list     # show cache dir + status

Each file's bytes are checked against the SHA256 recorded in ``manifest.json``,
so a silently changed upstream file fails loudly instead of corrupting a test.
Only the standard library is used (no ``requests`` dependency).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).with_name("manifest.json")
_UA = "facekeep-test-corpus/0.1 (https://github.com/wadeken/facekeep)"


def cache_dir() -> Path:
    """Where corpus images live. Overridable via ``FACEKEEP_CORPUS_DIR``.

    Defaults under the user cache (``~/.cache/facekeep/test-corpus``), matching
    the project's existing ``~/.cache/facekeep`` convention for downloaded
    assets (the YuNet model uses the same root).
    """
    env = os.environ.get("FACEKEEP_CORPUS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "facekeep" / "test-corpus"


def load_manifest() -> list[dict]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data["images"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(entry: dict, dest: Path) -> None:
    req = urllib.request.Request(entry["url"], headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted host)
        data = resp.read()
    got = hashlib.sha256(data).hexdigest()
    want = entry["sha256"]
    if got != want:
        raise RuntimeError(
            f"{entry['filename']}: SHA256 mismatch (upstream file changed?).\n"
            f"  expected {want}\n  got      {got}\n"
            f"  url: {entry['url']}\n"
            "If the change is legitimate, update manifest.json."
        )
    dest.write_bytes(data)


def download(force: bool = False) -> Path:
    """Fetch all missing (or all, if ``force``) corpus files. Returns the dir."""
    out = cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    for entry in load_manifest():
        dest = out / entry["filename"]
        if dest.exists() and not force:
            # Validate the cached copy; re-fetch if it got corrupted.
            if _sha256(dest) == entry["sha256"]:
                print(f"  ok    {entry['filename']} (cached)")
                continue
            print(f"  stale {entry['filename']} — re-downloading")
        print(f"  get   {entry['filename']} <- {entry['url']}")
        _download_one(entry, dest)
    print(f"Corpus ready in {out}")
    return out


def status() -> None:
    out = cache_dir()
    print(f"cache dir: {out}")
    for entry in load_manifest():
        dest = out / entry["filename"]
        if not dest.exists():
            state = "MISSING"
        elif _sha256(dest) == entry["sha256"]:
            state = "ok"
        else:
            state = "CORRUPT"
        print(f"  {state:8s} {entry['filename']:20s} "
              f"[{entry['license']}] faces={entry['faces']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download FaceKeep test corpus.")
    ap.add_argument("--force", action="store_true", help="re-download everything")
    ap.add_argument("--list", action="store_true", help="show cache status, no download")
    args = ap.parse_args(argv)

    if args.list:
        status()
        return 0
    try:
        download(force=args.force)
    except Exception as e:  # noqa: BLE001 - a CLI: report and exit nonzero
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
