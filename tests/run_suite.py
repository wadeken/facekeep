#!/usr/bin/env python3
"""Run the FaceKeep test suite one file per fresh process (the honest green signal).

A plain single-process ``pytest -q`` **segfaults** part-way through on this
project: native libs (torch / Real-ESRGAN / OpenCV / libaom) accumulate state
across tests in one interpreter and eventually tip over with a C-level fault
(``EXIT=139``). ``pytest-xdist --dist loadfile`` only spreads files across a few
long-lived workers, so each worker *still* accumulates across the several files
it owns and a worker eventually crashes (``node down: Not properly terminated``)
before the suite reaches a clean summary.

The only reliable green signal is **one fresh interpreter per test file**: every
file passes in its own process, so launching ``python -m pytest <file>`` per file
breaks the cross-file accumulation completely. This script does exactly that and
prints a per-file ledger plus a final pass/fail roll-up, so the whole suite can
be verified with a single command::

    python tests/run_suite.py                  # run every tests/test_*.py
    python tests/run_suite.py -k bit_depth     # only files whose name matches
    python tests/run_suite.py -- -rs -q        # pass extra args through to pytest

Exit code is non-zero iff any file failed, so it works as a CI gate. Only the
standard library is used. Set ``FACEKEEP_AVIFENC`` in the environment as usual to
exercise the real high-bit AVIF path; this script does not touch it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def _discover(pattern: str | None) -> list[Path]:
    files = sorted(TESTS_DIR.glob("test_*.py"))
    if pattern:
        files = [f for f in files if pattern in f.name]
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run each test file in its own process (avoids the "
        "single-interpreter native-lib segfault).",
    )
    parser.add_argument(
        "-k",
        dest="pattern",
        default=None,
        help="only run files whose name contains this substring",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="extra args forwarded to pytest (use `-- <args>` to be safe)",
    )
    args = parser.parse_args(argv)

    files = _discover(args.pattern)
    if not files:
        print("no test files matched", file=sys.stderr)
        return 1

    failures: list[str] = []
    started = time.monotonic()
    for f in files:
        rel = f.relative_to(TESTS_DIR.parent).as_posix()
        cmd = [sys.executable, "-m", "pytest", "-q", str(f), *args.pytest_args]
        t0 = time.monotonic()
        proc = subprocess.run(cmd)
        dt = time.monotonic() - t0
        if proc.returncode == 0:
            print(f"[ok]   {rel}  ({dt:.1f}s)")
        else:
            print(f"[FAIL] {rel}  (exit {proc.returncode}, {dt:.1f}s)")
            failures.append(rel)

    total = time.monotonic() - started
    print("=" * 60)
    if failures:
        print(f"FAILED {len(failures)}/{len(files)} files in {total:.1f}s:")
        for rel in failures:
            print(f"  - {rel}")
        return 1
    print(f"OK: all {len(files)} files passed in {total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
