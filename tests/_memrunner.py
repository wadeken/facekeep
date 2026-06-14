"""Subprocess entry point for the large-image memory test (Phase 2).

Run as a script in a *fresh* process so its peak RSS reflects only this one
compress, immune to any high-water mark the parent pytest process raised (see
the rationale in ``tests/test_large_image.py``). Not a pytest module — the
underscore prefix and absence of ``test_*`` functions keep it out of collection.

Usage:
    python tests/_memrunner.py <src_image> <out_path> <tests_dir>

Prints one JSON line to stdout: {peak, ratio, skipped, shape_match}. ``peak`` is
bytes (or null if unmeasurable). Exits non-zero on any failure so the parent can
surface stderr.
"""

import json
import sys
from pathlib import Path


def main() -> int:
    src, out, tests_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    # Make the sibling probe importable when run as a bare script (no package ctx).
    sys.path.insert(0, str(Path(tests_dir).resolve()))

    from _memprobe import peak_rss_bytes

    from facekeep import encoders, faithful
    from facekeep.config import FaceKeepConfig
    from facekeep.imageio import load

    # Pin auto-tune off: the memory ceiling pins the *default single-encode*
    # path (IMPROVEMENTS Phase 2 note); auto-tune's multi-probe profile is out of
    # scope here. Auto-tune is on by default in production.
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = False
    result = faithful.compress(src, out, cfg)
    decoded = encoders.decode(result.output_path.read_bytes())
    original = load(src).image

    print(
        json.dumps(
            {
                "peak": peak_rss_bytes(),
                "ratio": result.ratio,
                "skipped": result.skipped,
                "shape_match": list(decoded.shape) == list(original.shape),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
