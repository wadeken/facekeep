"""Benchmark harness — a human-readable regression report across a set of photos.

``facekeep bench <files-or-folder>`` compresses each photo (faithful and/or
aggressive) and prints a table of the numbers that matter for a *product*
regression: faithful ratio + decoded SSIM, and aggressive ``.fkeep`` ratio +
the **perceptual** quality of the reconstructed background (LPIPS, lower =
better). Saved as a baseline JSON, a later run can ``--baseline`` it and the
table shows a per-column delta — so after any change touching detection,
compression, restore, metrics, or auto-tune you can *see* whether a real-photo
number moved, instead of reading pytest red/green.

This is the **artifact** layer (ROADMAP backlog: "standalone benchmark harness
that prints a results table across a corpus"). It does not *fail* a build — that
is the job of the pytest regression locks (``test_corpus_regression.py`` and the
aggressive lock that follows). This complements them by covering the two modes
in one runnable, eyeball-able place, and by measuring aggressive mode (which the
pytest locks did not).

Honest measurement notes (the same caveats the corpus lock carries):
  * The numbers are a single-environment measurement (codec/plugin/CPU). A codec
    or plugin upgrade legitimately shifts them — re-save the baseline then.
  * Aggressive's restore-quality LPIPS is estimated against a **bicubic** restore
    by default (fast, offline, and a conservative proxy — real AI restore looks
    at least as good), the same trade-off ``compressor._search_bg_scale`` makes.
    ``ai_restore=True`` runs the real Real-ESRGAN restore instead (slow, needs
    the ``[ai]`` extra).
  * LPIPS needs the ``[ai]`` extra; when it is absent ``restore_lpips`` is left
    ``None`` (blank in the table) rather than fabricated — exactly the report's
    "blank means not measured" rule.

Everything reads through ``imageio.load`` and stays BGR internally; the only
BGR->RGB boundary is inside ``metrics`` (SSIM/LPIPS), as elsewhere.
"""

import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from . import encoders, faithful, metrics
from .aggressive import compressor
from .aggressive.format import write_fkeep
from .aggressive.restorer import Restorer
from .config import FaceKeepConfig
from .exceptions import FaceKeepError
from .imageio import load

logger = logging.getLogger("facekeep.bench")

# Baseline schema version: bump if the BenchRow fields change meaning so an old
# baseline isn't silently diffed against incompatible columns.
BASELINE_VERSION = 1

# Numeric columns that carry a per-row delta when diffed against a baseline.
# (file/mode are the key; status/error are descriptive.)
_METRIC_FIELDS = (
    "ratio",
    "ssim",
    "restore_lpips",
    "faces",
    "original_bytes",
    "output_bytes",
)


@dataclass
class BenchRow:
    """One (file, mode) measurement. Optional metrics are ``None`` when a given
    mode/install does not produce them (blank in the table, like the report's
    "not measured" rule) — never a fabricated number.

    faithful: ``ratio`` (original/encoded) + ``ssim`` (decoded vs original).
    aggressive: ``ratio`` (original/``.fkeep``) + ``restore_lpips`` (perceptual
    distance of the reconstructed image vs original; ``None`` without ``[ai]``).
    """

    file: str
    mode: str  # "faithful" | "aggressive"
    status: str = "ok"  # ok | failed
    ratio: Optional[float] = None
    ssim: Optional[float] = None
    restore_lpips: Optional[float] = None
    faces: Optional[int] = None
    original_bytes: Optional[int] = None
    output_bytes: Optional[int] = None
    error: Optional[str] = None


def _bench_faithful(src: Path, cfg: FaceKeepConfig, out_dir: Path) -> BenchRow:
    """Compress ``src`` in faithful mode and measure ratio + decoded SSIM."""
    out_stem = str(out_dir / src.stem)
    result = faithful.compress(str(src), out_stem, cfg)

    # SSIM of the decoded output vs the original (the same comparison the corpus
    # lock makes). A skip-if-larger "kept original" has ratio 1.0 and is byte-
    # identical, so SSIM is ~1.0; we still decode it honestly rather than assume.
    original = load(str(src)).image
    decoded = encoders.decode(result.output_path.read_bytes())
    ssim = metrics.ssim(original, decoded) if decoded.shape == original.shape else None

    return BenchRow(
        file=src.name,
        mode="faithful",
        ratio=result.ratio,
        ssim=ssim,
        faces=result.faces_detected,
        original_bytes=result.original_size,
        output_bytes=result.compressed_size,
    )


def _bench_aggressive(
    src: Path, cfg: FaceKeepConfig, out_dir: Path, ai_restore: bool
) -> BenchRow:
    """Compress ``src`` to a ``.fkeep``, restore it, and measure ratio + LPIPS.

    The ``.fkeep`` is written to ``out_dir`` (so its real on-disk size drives the
    ratio), restored, and the reconstruction scored against the original with
    LPIPS — the right metric for a hallucinated-but-plausible background. The
    restore is bicubic by default (fast, offline, conservative proxy); ``ai_restore``
    runs the real Real-ESRGAN path. LPIPS is left ``None`` when ``[ai]`` is absent.
    """
    photo = compressor.compress_photo(str(src), cfg)
    fkeep_path = out_dir / f"{src.stem}.fkeep"
    output_bytes = write_fkeep(photo, str(fkeep_path))

    original_bytes = src.stat().st_size
    ratio = original_bytes / output_bytes if output_bytes else None

    restorer = Restorer(cfg.aggressive)
    if ai_restore:
        restored = restorer.restore(str(fkeep_path))
    else:
        restored = restorer.preview(str(fkeep_path))

    # LPIPS wants matching dimensions; restore returns the original frame size,
    # but guard anyway. None when [ai]/lpips is unavailable (graceful, not faked).
    original = load(str(src)).image
    restore_lpips: Optional[float] = None
    if restored.shape[:2] == original.shape[:2]:
        restore_lpips = metrics.lpips_distance(original, restored)

    return BenchRow(
        file=src.name,
        mode="aggressive",
        ratio=ratio,
        restore_lpips=restore_lpips,
        faces=len(photo.faces),
        original_bytes=original_bytes,
        output_bytes=output_bytes,
    )


def run_benchmark(
    paths: List[Path],
    modes: List[str],
    config: Optional[FaceKeepConfig] = None,
    *,
    ai_restore: bool = False,
) -> List[BenchRow]:
    """Benchmark each path in each requested mode; return one row per (file, mode).

    ``modes`` is any subset of ``["faithful", "aggressive"]`` (order preserved).
    A per-file failure is isolated into a ``status="failed"`` row (with the error
    message) so one bad photo never aborts the whole run — the harness is a
    measurement tool, not a pipeline. Outputs are written to a fresh temp dir and
    discarded; only the measured numbers are kept.
    """
    config = config or FaceKeepConfig()
    rows: List[BenchRow] = []

    with tempfile.TemporaryDirectory(prefix="facekeep-bench-") as tmp:
        out_dir = Path(tmp)
        for src in paths:
            for mode in modes:
                try:
                    if mode == "faithful":
                        rows.append(_bench_faithful(src, config, out_dir))
                    elif mode == "aggressive":
                        rows.append(
                            _bench_aggressive(src, config, out_dir, ai_restore)
                        )
                    else:  # pragma: no cover - guarded by the CLI choice
                        raise ValueError(f"unknown bench mode: {mode}")
                except (FaceKeepError, OSError, ValueError) as e:
                    logger.warning("bench %s (%s) failed: %s", src.name, mode, e)
                    rows.append(
                        BenchRow(file=src.name, mode=mode, status="failed", error=str(e))
                    )

    return rows


# ---------------------------------------------------------------------------
# Baseline save / load / diff
# ---------------------------------------------------------------------------


def save_baseline(rows: List[BenchRow], path: str) -> Path:
    """Write the rows to a baseline JSON (version-tagged). Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BASELINE_VERSION,
        "rows": [asdict(r) for r in rows],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_baseline(path: str) -> List[BenchRow]:
    """Read a baseline JSON back into rows. Raises ``FaceKeepError`` on a bad file."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != BASELINE_VERSION:
            raise FaceKeepError(
                f"baseline {path} is version {version}, expected {BASELINE_VERSION}; "
                "re-save it with `facekeep bench --save-baseline`."
            )
        # Tolerate a baseline written by an older field set: keep only known keys.
        known = set(BenchRow.__annotations__)
        return [BenchRow(**{k: v for k, v in r.items() if k in known})
                for r in payload["rows"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise FaceKeepError(f"Cannot read baseline {path}: {e}") from e


def _key(row: BenchRow) -> tuple:
    return (row.file, row.mode)


def diff_baselines(
    old: List[BenchRow], new: List[BenchRow]
) -> Dict[tuple, Dict[str, Optional[float]]]:
    """Per-(file, mode) numeric delta of ``new`` minus ``old``.

    Returns ``{(file, mode): {metric: delta_or_None}}``. A metric delta is
    ``None`` when either side lacks the number (a mode that didn't measure it, a
    newly added or removed file/mode) — so the table can show "new"/"gone"/blank
    without inventing a delta. Rows present on only one side appear with all-None
    deltas (still keyed, so the caller can flag them).
    """
    old_by_key = {_key(r): r for r in old}
    new_by_key = {_key(r): r for r in new}

    diff: Dict[tuple, Dict[str, Optional[float]]] = {}
    for key in {**old_by_key, **new_by_key}:
        o = old_by_key.get(key)
        n = new_by_key.get(key)
        deltas: Dict[str, Optional[float]] = {}
        for f in _METRIC_FIELDS:
            ov = getattr(o, f, None) if o is not None else None
            nv = getattr(n, f, None) if n is not None else None
            deltas[f] = (nv - ov) if (ov is not None and nv is not None) else None
        diff[key] = deltas
    return diff


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_HEADERS = ("file", "mode", "ratio", "ssim", "rest.lpips", "faces", "status")


def _fmt_cell(value: Optional[float], spec: str) -> str:
    return "" if value is None else format(value, spec)


def _fmt_delta(d: Optional[float], spec: str) -> str:
    """Signed delta in parens, e.g. ``(+0.012)``; blank when unknown."""
    if d is None:
        return ""
    return f"({d:+{spec}})"


def format_table(
    rows: List[BenchRow], baseline: Optional[List[BenchRow]] = None
) -> str:
    """Render rows as a fixed-width text table.

    When ``baseline`` is given, each numeric cell gains a ``(±delta)`` suffix vs
    the baseline row of the same (file, mode); a row missing from the baseline is
    marked ``NEW`` in its status cell. Pure stdlib string formatting — no new
    dependency (mirrors ``report.py``'s plain-text discipline).
    """
    diff = diff_baselines(baseline, rows) if baseline is not None else {}
    base_keys = {_key(r) for r in baseline} if baseline is not None else set()

    table_rows: List[List[str]] = []
    for r in rows:
        key = _key(r)
        deltas = diff.get(key, {})
        ratio_d = _fmt_delta(deltas.get("ratio"), ".3f")
        ssim_d = _fmt_delta(deltas.get("ssim"), ".4f")
        lpips_d = _fmt_delta(deltas.get("restore_lpips"), ".4f")

        status = r.status
        if baseline is not None and key not in base_keys:
            status = f"{status} NEW"

        table_rows.append([
            r.file,
            r.mode,
            (_fmt_cell(r.ratio, ".3f") + ratio_d).strip(),
            (_fmt_cell(r.ssim, ".4f") + ssim_d).strip(),
            (_fmt_cell(r.restore_lpips, ".4f") + lpips_d).strip(),
            _fmt_cell(r.faces, "d"),
            status,
        ])

    # A baseline row that vanished from this run: surface it so a dropped
    # file/mode is visible rather than silently absent.
    if baseline is not None:
        current_keys = {_key(r) for r in rows}
        for b in baseline:
            if _key(b) not in current_keys:
                table_rows.append([b.file, b.mode, "", "", "", "", "GONE"])

    widths = [len(h) for h in _HEADERS]
    for tr in table_rows:
        for i, cell in enumerate(tr):
            widths[i] = max(widths[i], len(cell))

    def _line(cells: List[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [_line(list(_HEADERS)), _line(["-" * w for w in widths])]
    lines.extend(_line(tr) for tr in table_rows)
    return "\n".join(lines)
