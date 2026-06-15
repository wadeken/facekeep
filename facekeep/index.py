"""Incremental processing index — skip unchanged photos on a re-run.

``facekeep compress`` is idempotent in its *output* but not in its *cost*:
re-running it on a folder re-encodes every file from scratch, even the ones that
haven't changed since last time. For a family-photo backup that grows by a
handful of new shots a week, that is almost all wasted work.

This module is a tiny persistent cache that lets a re-run **skip a file whose
input bytes and processing settings are identical to the last successful run,
and whose output still exists on disk**. It is a pure speed feature: it never
changes the bytes of any output that *is* written — a skipped file is simply one
we already produced and don't need to produce again.

What identifies "unchanged" (all must match for a skip):

* **content hash** — SHA-256 of the *input file bytes*. The honest "did this
  file change?" test: it catches edits even when the mtime was preserved, and is
  cheap next to an AVIF/JXL encode. (Same hashing idiom used elsewhere, e.g.
  ``aggressive`` stores this in the manifest.)
* **settings fingerprint** — a short stable hash of *every* config field that
  affects the output bytes (not just the headline three). So changing
  ``--quality``, ``--codec``, ``-m``, ``--chroma``, ``--verify-thorough``, or any
  aggressive knob correctly busts the cache. The headline ``mode``/``codec``/
  ``quality`` are stored as their own columns too, but only for human-readable
  ``info``; the fingerprint is the authoritative guard.
* **output still present** — the recorded output path must still exist. If the
  user deleted the ``.avif``, we re-make it on the next run even on a cache hit.

Concurrency contract (important): this DB is opened **only in the parent
process**. ``compress`` reads it once up front to decide what to skip, then —
after the workers return — writes the new rows in input order. Worker processes
(``--jobs``) never touch it, so there is no multi-process SQLite contention and
the byte-identical-output guarantee of ``--jobs`` is preserved. ``sqlite3`` is
stdlib, so this adds no dependency.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import FaceKeepConfig

# Bump if the row schema changes incompatibly. Stored in the DB; on a mismatch
# we treat every lookup as a miss (and rewrite rows in the new shape) rather than
# crash on an old file — a stale cache only costs a re-encode, never correctness.
SCHEMA_VERSION = 1

# Default index filename, placed inside the output directory so the index travels
# with the outputs it describes (a different -o dir gets its own, no cross-talk).
INDEX_FILENAME = ".facekeep-index.sqlite"


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, hex — the content-identity key for the cache."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings_fingerprint(config: FaceKeepConfig) -> str:
    """Short stable hash of every config field that affects the output bytes.

    Two runs with the same fingerprint would produce the same output for the same
    input, so a cached output is reusable; any difference here must bust the
    cache. We include the *mode-relevant* fields only (the other mode's settings
    can't affect this output) and hash a canonical JSON form so the value is
    stable across runs and Python versions.
    """
    if config.mode == "faithful":
        relevant = {
            "mode": "faithful",
            "codec": config.faithful.codec,
            "quality": config.faithful.quality,
            "speed": config.faithful.speed,
            # Lossless flips the whole encode (bit-exact, quality ignored, may
            # redirect avif->jxl), so it is output-affecting and must bust the cache.
            "lossless": config.faithful.lossless,
            "auto_tune": config.faithful.auto_tune,
            "target_metric": config.faithful.target_metric,
            "target_value": config.faithful.target_value,
            "chroma": config.faithful.chroma,
            # 10 vs 12-bit changes the encoded bytes on a high-bit AVIF source.
            "output_bit_depth": config.faithful.output_bit_depth,
            "skip_if_larger": config.faithful.skip_if_larger,
            # verify_thorough changes what we measure but not the output bytes;
            # verify can keep/replace the output on a round-trip failure, so it is
            # output-affecting and included.
            "verify": config.faithful.verify,
            "verify_thorough": config.faithful.verify_thorough,
        }
        # Faithful uses the shared detector config directly.
        effective_detector = config.detector
    else:
        relevant = {
            "mode": "aggressive",
            "bg_scale": config.aggressive.bg_scale,
            "bg_quality": config.aggressive.bg_quality,
            # The background codec changes the stored background member/bytes
            # (background.jpg vs .avif/.jxl), so it must bust the cache.
            "bg_codec": config.aggressive.bg_codec,
            "face_quality": config.aggressive.face_quality,
            "face_codec": config.aggressive.face_codec,
            "no_face_strategy": config.aggressive.no_face_strategy,
            "no_face_bg_scale": config.aggressive.no_face_bg_scale,
            "blend_mode": config.aggressive.blend_mode,
            # Content-aware conservatism picks the effective bg_scale per image,
            # so its switch + thresholds are output-affecting (a detailed photo's
            # bytes change when these change) and must bust the cache.
            "content_aware": config.aggressive.content_aware,
            "conservative_bg_scale": config.aggressive.conservative_bg_scale,
            "text_edge_threshold": config.aggressive.text_edge_threshold,
            "small_face_ratio": config.aggressive.small_face_ratio,
            # Text protection emits region patches for localized text-like
            # clusters (vs the whole-image raise) — whether/where/how-much
            # changes the output bytes, so all three knobs must bust the cache.
            "protect_text": config.aggressive.protect_text,
            "text_region_tile_threshold": (
                config.aggressive.text_region_tile_threshold
            ),
            "text_region_max_frac": config.aggressive.text_region_max_frac,
            # Region-local conservatism picks whether a risky region is protected
            # locally (a sharp patch in the .fkeep) vs raising the whole-image
            # bg_scale, and at what resolution the patch is stored — both change
            # the output bytes, so they must bust the cache.
            "region_local": config.aggressive.region_local,
            "region_scale": config.aggressive.region_scale,
            # Hand protection adds region patches (offline C1 geometry or opt-in
            # C2 detection) — which/where/at-what-resolution changes the output
            # bytes, so these are output-affecting and must bust the cache.
            "protect_hands": config.aggressive.protect_hands,
            "protect_hands_backend": config.aggressive.protect_hands_backend,
            "hand_zone_scale": config.aggressive.hand_zone_scale,
            "hand_zone_max_frac": config.aggressive.hand_zone_max_frac,
            # C2 detection tuning changes which hands are found (and thus which
            # region patches are emitted), so it is output-affecting too.
            "hand_detect_confidence": config.aggressive.hand_detect_confidence,
            "hand_detect_max_hands": config.aggressive.hand_detect_max_hands,
            "hand_detect_long_side": config.aggressive.hand_detect_long_side,
            "hand_detect_padding": config.aggressive.hand_detect_padding,
            # Quality-targeted bg_scale chooses the effective scale per image, so
            # the target and the candidate ladder are output-affecting and must
            # bust the cache (a different target/ladder can land on a different
            # scale and thus different bytes).
            "quality_target": config.aggressive.quality_target,
            "quality_scale_candidates": list(
                config.aggressive.quality_scale_candidates
            ),
            # The residual layer adds a member (residual.jxl/.jpg) and its knobs
            # change that member's resolution/bytes, so all three are
            # output-affecting and must bust the cache.
            "residual": config.aggressive.residual,
            "residual_scale": config.aggressive.residual_scale,
            "residual_quality": config.aggressive.residual_quality,
        }
        # Aggressive overrides the shared detector (backend/confidence/min-size)
        # to protect small faces, so the fingerprint must hash what it *actually*
        # uses — otherwise changing an aggressive override wouldn't bust the cache.
        # resolved_detector is the same single source of truth the compressor uses.
        effective_detector = config.aggressive.resolved_detector(config.detector)
    # GPS-stripping changes the EXIF bytes embedded in the output (both modes),
    # so flipping it must bust the cache for an otherwise-unchanged photo.
    relevant["strip_gps"] = config.strip_gps
    # Detector settings affect which regions get 4:4:4 / where faces are stored,
    # so they are output-affecting in both modes.
    relevant["detector"] = {
        "backend": effective_detector.backend,
        "confidence": effective_detector.confidence,
        "padding": effective_detector.padding,
        "nms_iou": effective_detector.nms_iou,
        "min_size_ratio": effective_detector.min_size_ratio,
        "max_aspect_ratio": effective_detector.max_aspect_ratio,
        # ROI grows padded_bbox (aggressive crop size / faithful auto-tune
        # acceptance region), so it is output-affecting and must bust the cache.
        "roi": effective_detector.roi,
    }
    blob = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class IndexRow:
    """One cached file outcome (the fields a lookup needs + diagnostics)."""

    content_hash: str
    settings_fingerprint: str
    mode: str
    codec: Optional[str]
    quality: Optional[int]
    original_size: int
    output_path: str
    output_size: int


class ProcessIndex:
    """A SQLite-backed cache of processed files, keyed by absolute input path.

    Use as a context manager so the connection is always closed::

        with ProcessIndex(db_path) as idx:
            row = idx.lookup(abs_path)
            ...
            idx.record(abs_path, row)
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # -- lifecycle ---------------------------------------------------------- #

    def __enter__(self) -> "ProcessIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def _ensure_schema(self) -> None:
        cur = self._conn.execute("PRAGMA user_version")
        version = cur.fetchone()[0]
        if version != SCHEMA_VERSION:
            # Old or empty DB: (re)create the table fresh. A wiped cache only
            # costs re-encodes, never correctness, so this is safe.
            self._conn.execute("DROP TABLE IF EXISTS processed")
            self._conn.execute(
                """
                CREATE TABLE processed (
                    abs_path             TEXT PRIMARY KEY,
                    content_hash         TEXT NOT NULL,
                    settings_fingerprint TEXT NOT NULL,
                    mode                 TEXT NOT NULL,
                    codec                TEXT,
                    quality              INTEGER,
                    original_size        INTEGER NOT NULL,
                    output_path          TEXT NOT NULL,
                    output_size          INTEGER NOT NULL,
                    updated_at           TEXT NOT NULL
                )
                """
            )
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    # -- queries ------------------------------------------------------------ #

    @staticmethod
    def _key(abs_path: str | Path) -> str:
        # Resolve so the same file reached via different relative paths shares a
        # row. (Don't require the file to exist — resolve(strict=False).)
        return str(Path(abs_path).resolve())

    def lookup(self, abs_path: str | Path) -> Optional[IndexRow]:
        """Return the cached row for ``abs_path``, or None if not recorded."""
        row = self._conn.execute(
            "SELECT * FROM processed WHERE abs_path = ?", (self._key(abs_path),)
        ).fetchone()
        if row is None:
            return None
        return IndexRow(
            content_hash=row["content_hash"],
            settings_fingerprint=row["settings_fingerprint"],
            mode=row["mode"],
            codec=row["codec"],
            quality=row["quality"],
            original_size=row["original_size"],
            output_path=row["output_path"],
            output_size=row["output_size"],
        )

    def is_unchanged(
        self, abs_path: str | Path, content_hash: str, fingerprint: str
    ) -> Optional[IndexRow]:
        """A cache *hit* iff the row exists, the hash + fingerprint match, and the
        recorded output still exists on disk. Returns the row on a hit, else None.

        The output-existence check is deliberate: a cached row whose output the
        user has since deleted must NOT be skipped — we re-create the missing
        file. This keeps the index honest about what is actually on disk.
        """
        row = self.lookup(abs_path)
        if row is None:
            return None
        if row.content_hash != content_hash:
            return None
        if row.settings_fingerprint != fingerprint:
            return None
        if not Path(row.output_path).exists():
            return None
        return row

    # -- writes ------------------------------------------------------------- #

    def record(self, abs_path: str | Path, row: IndexRow) -> None:
        """Upsert the cached outcome for ``abs_path`` (overwrites any prior row)."""
        from datetime import datetime, timezone

        self._conn.execute(
            """
            INSERT INTO processed
                (abs_path, content_hash, settings_fingerprint, mode, codec,
                 quality, original_size, output_path, output_size, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(abs_path) DO UPDATE SET
                content_hash         = excluded.content_hash,
                settings_fingerprint = excluded.settings_fingerprint,
                mode                 = excluded.mode,
                codec                = excluded.codec,
                quality              = excluded.quality,
                original_size        = excluded.original_size,
                output_path          = excluded.output_path,
                output_size          = excluded.output_size,
                updated_at           = excluded.updated_at
            """,
            (
                self._key(abs_path),
                row.content_hash,
                row.settings_fingerprint,
                row.mode,
                row.codec,
                row.quality,
                row.original_size,
                row.output_path,
                row.output_size,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
