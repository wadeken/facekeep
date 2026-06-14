"""Sidecar verification report — a per-file CSV ledger of a compress run.

``facekeep compress ... --report out.csv`` writes one row per input file
describing what happened: sizes, ratio, the codec quality used, face count, and
— *only when one was actually measured* — a fidelity score.

Honesty rule (the whole point of a "verification" report): the ``ssim_downscaled``
column is filled **only** when the pipeline really computed it. Faithful mode
measures a downscaled-SSIM round-trip only under ``--verify-thorough``; aggressive
mode reconstructs the image at *restore* time, not at compress time, so it has no
faithful comparison to make here. In both un-measured cases the cell is left
**blank** rather than carrying an invented number. A blank means "not measured",
not "perfect".

The report is the user's requested artifact, so it is written even under
``--dry-run`` (where every row's ``status`` is ``would-write`` and no image is
produced). Failures get a row too (``status=failed``), so the CSV is a complete
ledger of the run, not just its successes.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Column order is fixed and part of the contract (tests pin it). Keep additions
# at the end so existing consumers don't shift.
FIELDNAMES = [
    "file",
    "mode",
    "codec",
    "original_bytes",
    "output_bytes",
    "ratio",
    "quality",
    "faces",
    "ssim_downscaled",
    "status",
    "output_path",
]


@dataclass
class ReportRow:
    """One input file's outcome. Optional fields render as a blank CSV cell.

    ``None`` is deliberate, not lazy: ``ssim_downscaled`` is None unless a real
    score was measured, ``quality`` is None for aggressive mode (its bg_scale is
    not a 0-100 codec quality), and sizes/ratio are None for a failed file.
    """

    file: str
    mode: str
    status: str  # written | would-write | kept-original | skipped | failed
    codec: Optional[str] = None
    original_bytes: Optional[int] = None
    output_bytes: Optional[int] = None
    ratio: Optional[float] = None
    quality: Optional[int] = None
    faces: Optional[int] = None
    ssim_downscaled: Optional[float] = None
    output_path: Optional[str] = None

    def as_csv_dict(self) -> dict:
        """Render to the CSV column dict; None -> "" (blank cell), floats rounded."""
        return {
            "file": self.file,
            "mode": self.mode,
            "codec": _s(self.codec),
            "original_bytes": _s(self.original_bytes),
            "output_bytes": _s(self.output_bytes),
            "ratio": "" if self.ratio is None else f"{self.ratio:.4f}",
            "quality": _s(self.quality),
            "faces": _s(self.faces),
            "ssim_downscaled": (
                "" if self.ssim_downscaled is None else f"{self.ssim_downscaled:.4f}"
            ),
            "status": self.status,
            "output_path": _s(self.output_path),
        }


def _s(v: object) -> str:
    """None -> empty cell; everything else -> str."""
    return "" if v is None else str(v)


def write_report(rows: List[ReportRow], report_path: str) -> Path:
    """Write the rows to a CSV file (header + one row per file). Returns the path.

    Pure stdlib ``csv`` with ``newline=""`` (so the writer controls line endings)
    and UTF-8 (filenames may be non-ASCII). The parent directory is created if
    needed. Written regardless of ``--dry-run`` — the report itself is the
    artifact the user asked for.
    """
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_dict())
    return path
