"""FaceKeep local web GUI (Gradio) — a thin wrapper over the library API.

Drag a photo in, pick a mode, compress, and see a before/after with the real
stats and a download of the real output file. The GUI calls the **exact same**
faithful / aggressive pipeline the CLI uses (``faithful.compress`` /
``aggressive.compressor.compress_photo`` + ``format.write_fkeep``), so the bytes
it produces are byte-identical to a ``facekeep compress`` run — it adds no
fidelity surface, only a friendlier front door (ROADMAP Phase 7).

**Gradio is an optional dependency (the ``[gui]`` extra).** It is imported
*lazily* inside :func:`build_demo` / :func:`launch`, so this module — and its
pure handler functions, which need only the core library — import fine without
it. That keeps the handlers unit-testable with no browser, and lets
``facekeep gui`` print an actionable install hint instead of crashing when the
extra is absent (the same graceful-degradation discipline as missing ``[ai]`` →
bicubic, offline → Haar). :func:`launch` runs the server **locally with sharing
and telemetry OFF** — a photo tool must not phone home or open a public tunnel
by default.

Aggressive mode's "after" is a fast **bicubic restore preview** (not the real
AI restore): full-resolution Real-ESRGAN restore is far too slow for an
interactive UI, and the preview is instant, offline, and needs no ``[ai]``.
The *download* is the real ``.fkeep`` — restore it any time with
``facekeep restore``.

A second **Compare** tab pairs an original against *any* compressed output (a
faithful ``.avif``/``.jxl``/``.webp``, or an aggressive ``.fkeep`` reconstructed
on the fly — the fast bicubic preview by default, or an opt-in real
Real-ESRGAN restore that warns it is slow and shows a spinner) and displays a
live before/after wipe slider, a difference heatmap, and SSIM/PSNR — the
interactive sibling of ``facekeep compare`` (which exports the same view as a
self-contained HTML file). It reuses the :mod:`facekeep.compare` helpers, so it
adds no new fidelity surface either.

A third **Backup** tab (ROADMAP 11.2) is the one-click folder flow: pick a
source folder and an archive folder, press one button, and every photo (and
video, when ffmpeg is available) is compressed into the archive by the **exact
CLI batch machinery** — :func:`run_backup` drives ``cli._run_batch`` per file
(the ``only_files=`` API the watch loop established), so outputs are
byte-identical to ``facekeep compress`` and re-runs skip unchanged files via
the same incremental index. It streams per-file progress (photos first, then
the serial videos), ends with a completion report (totals + a per-file table
built by the ``--report`` machinery, downloadable as CSV), states the
guardrail-2 honesty note (visually lossless ≠ bit-exact — with a lossless
toggle), and persists the last-used folders so a return visit really is one
click. Sources are never deleted or modified. Continuous watching stays with
``facekeep watch`` (the CLI automation core).
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
from typing import List

import cv2
import numpy as np

from . import compare as compare_mod, encoders, metrics, report as report_mod
from .config import PRESET_NAMES, FaceKeepConfig, apply_preset
from .exceptions import ConfigError, FaceKeepError
from .imageio import load as load_image

# Sentinel shown in the preset dropdown for "no preset, use the knobs below".
_NO_PRESET = "(custom)"


def _fmt_size(n: float) -> str:
    """Human-readable byte size (mirrors ``cli._fmt_size``)."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Down-convert an image to 8-bit for display (uint16 sources, float buffers)."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 257.0).round().astype(np.uint8)  # full-range 16->8
    return np.clip(img, 0, 255).astype(np.uint8)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR (internal) -> RGB (the only Gradio/display boundary), as uint8."""
    return cv2.cvtColor(_to_uint8(img), cv2.COLOR_BGR2RGB)


@dataclass
class CompressOutput:
    """Result of a GUI compress: the download path, before/after, and a summary."""

    output_path: str
    before_rgb: np.ndarray
    after_rgb: np.ndarray
    summary: str  # Markdown


def _build_config(
    mode: str = "faithful",
    *,
    preset: str | None = None,
    codec: str | None = None,
    quality: int | None = None,
    auto_tune: bool | None = None,
    bg_scale: float | None = None,
) -> FaceKeepConfig:
    """Build a validated config from the GUI's handful of knobs.

    Mirrors ``cli._load_config``'s precedence for the exposed fields: the preset
    expands first (and implies aggressive mode), then explicit knobs override it
    — so a hand-set value still beats the preset, exactly like the CLI. A preset
    combined with an explicit faithful mode is a loud :class:`ConfigError`, never
    a silent flip (same rule as ``--preset`` + ``-m faithful``).
    """
    config = FaceKeepConfig()
    if preset:
        if mode == "faithful":
            raise ConfigError(
                f"preset {preset!r} implies aggressive mode; it cannot be "
                "combined with faithful mode"
            )
        apply_preset(config, preset)  # sets mode=aggressive + records the name
    if mode:
        config.mode = mode
    if codec:
        config.faithful.codec = codec
    if quality is not None:
        config.faithful.quality = int(quality)
        # An explicit quality is a deliberate override: honor it directly rather
        # than letting the (default-on) auto-tune search pick something else
        # (the same rule the CLI applies for an explicit -q).
        if auto_tune is None:
            config.faithful.auto_tune = False
    if auto_tune is not None:
        config.faithful.auto_tune = bool(auto_tune)
    if bg_scale is not None:
        config.aggressive.bg_scale = float(bg_scale)
    config.validate()
    return config


def _faithful_summary(res) -> str:
    lines = [f"**Mode:** faithful — `{res.codec}`",
             f"**Faces detected:** {res.faces_detected}"]
    if res.skipped:
        lines.append(
            f"**Result:** kept the original (the encode wasn't smaller) — "
            f"{_fmt_size(res.original_size)}"
        )
    else:
        lines.append(
            f"**Size:** {_fmt_size(res.original_size)} → "
            f"{_fmt_size(res.compressed_size)} (**{res.ratio:.1f}× smaller**)"
        )
        lines.append(f"**Quality:** {res.quality_used}")
    if res.quality_score is not None:
        lines.append(f"**{res.quality_metric}:** {res.quality_score:.4f}")
    lines.append("_The output is a standard image — opening it *is* the restore._")
    return "\n\n".join(lines)


def _aggressive_summary(photo, size: int) -> str:
    ratio = photo.original_size_bytes / size if size else 0.0
    lines = [
        "**Mode:** aggressive — `.fkeep`",
        f"**Faces detected:** {len(photo.faces)}",
        f"**Size:** {_fmt_size(photo.original_size_bytes)} → {_fmt_size(size)} "
        f"(**{ratio:.1f}× smaller**)",
        f"**Background scale:** {photo.effective_bg_scale:.3f}",
    ]
    if getattr(photo, "regions", None):
        lines.append(
            f"**Protected regions:** {len(photo.regions)} kept sharp "
            "(small/distant faces, hands)"
        )
    lines.append(
        "_The 'after' is a fast **bicubic restore preview**; the real AI restore "
        "(`facekeep restore`, needs the `[ai]` extra) looks sharper. The download "
        "is the real `.fkeep` — restore it any time._"
    )
    return "\n\n".join(lines)


def compress_image(
    image_path: str,
    mode: str = "faithful",
    *,
    preset: str | None = None,
    codec: str | None = None,
    quality: int | None = None,
    auto_tune: bool | None = None,
    bg_scale: float | None = None,
    out_dir: str | None = None,
    restore_preview_quality: int = 90,
) -> CompressOutput:
    """Compress one image and return paths + before/after arrays for the GUI.

    This is the pure, browser-free core of the GUI (so it is directly unit
    testable). It runs the real pipeline, writes the real output into ``out_dir``
    (a fresh temp dir when omitted), and returns:

    * ``output_path`` — the real ``.avif``/``.jxl``/``.fkeep`` to download;
    * ``before_rgb`` — the upright original (via :func:`imageio.load`);
    * ``after_rgb`` — for faithful, the *decoded* output; for aggressive, a fast
      bicubic restore **preview** (the ``.fkeep`` itself isn't viewable);
    * ``summary`` — Markdown stats.

    Raises :class:`FaceKeepError` (or :class:`ConfigError`) on bad input/config,
    which the UI renders as an error message rather than a crash.
    """
    if not image_path:
        raise FaceKeepError("No image provided — drop a photo in first.")
    config = _build_config(mode, preset=preset, codec=codec, quality=quality,
                           auto_tune=auto_tune, bg_scale=bg_scale)

    work = Path(out_dir) if out_dir else Path(
        tempfile.mkdtemp(prefix="facekeep-gui-"))
    work.mkdir(parents=True, exist_ok=True)
    target = str(work / Path(image_path).stem)

    # "Before": the upright original through the one sanctioned reader (applies
    # EXIF orientation, so it matches the encoded output's orientation).
    before_rgb = _bgr_to_rgb(load_image(image_path).image)

    if config.mode == "faithful":
        from .faithful import compress as faithful_compress

        res = faithful_compress(image_path, target, config)
        # Decode the output we just wrote (AVIF/JXL need Pillow, not cv2) so the
        # "after" is exactly what a viewer would show.
        after_bgr = encoders.decode(Path(res.output_path).read_bytes())
        return CompressOutput(
            output_path=str(res.output_path),
            before_rgb=before_rgb,
            after_rgb=_bgr_to_rgb(after_bgr),
            summary=_faithful_summary(res),
        )

    from .aggressive.compressor import compress_photo
    from .aggressive.format import _fkeep_path, write_fkeep
    from .aggressive.restorer import Restorer

    photo = compress_photo(image_path, config)
    size = write_fkeep(photo, target, dry_run=False)
    fkeep_path = _fkeep_path(target)
    # Fast bicubic preview of the restore (no AI): instant, offline, no [ai].
    after_bgr = Restorer(config.aggressive).preview(
        str(fkeep_path), quality=restore_preview_quality)
    return CompressOutput(
        output_path=str(fkeep_path),
        before_rgb=before_rgb,
        after_rgb=_bgr_to_rgb(after_bgr),
        summary=_aggressive_summary(photo, size),
    )


@dataclass
class CompareOutput:
    """Result of a GUI compare: before/after/diff display arrays + a summary."""

    before_rgb: np.ndarray
    after_rgb: np.ndarray
    diff_rgb: np.ndarray
    summary: str  # Markdown


def compare_images(
    original_path: str,
    compressed_path: str,
    *,
    amplify: float = 8.0,
    use_ai: bool = False,
) -> CompareOutput:
    """Build a live before/after comparison for the GUI (browser-free core).

    The interactive sibling of ``facekeep compare``: it reuses the exact same
    :mod:`facekeep.compare` helpers, so it stays a thin wrapper with no new
    fidelity surface. It loads the upright original through :func:`imageio.load`,
    reconstructs the "after" the user actually gets — a faithful
    ``.avif``/``.jxl``/``.webp`` decoded, or an aggressive ``.fkeep`` restored
    with the fast **bicubic preview** by default (matching the Compress tab; a
    real AI restore is far too slow to be the interactive default) — aligns them,
    and returns RGB display arrays for a before/after slider, a difference
    heatmap, and an SSIM/PSNR summary. The single BGR->RGB boundary is
    :func:`_bgr_to_rgb`.

    ``use_ai`` opts a ``.fkeep`` into the genuine Real-ESRGAN restore (slow —
    minutes on CPU) instead of the preview, but **only when the ``[ai]`` extra is
    actually installed**: otherwise it honestly falls back to the bicubic preview
    (and says so in the summary) rather than running the same bicubic slowly and
    mislabeling it as AI. On that real-restore path ``load_after`` auto-applies
    the ``.fkeep``'s recorded preset's restore-side knobs, so the "after" matches
    ``facekeep restore`` per file (the GUI sets no restore knobs by hand, so the
    preset always wins here).

    Raises :class:`FaceKeepError` on a missing/undecodable input, which the UI
    renders as an error message rather than a crash.
    """
    if not original_path or not compressed_path:
        raise FaceKeepError(
            "Drop both an original photo and a compressed file "
            "(.avif / .jxl / .webp / .fkeep) to compare."
        )
    before = load_image(original_path).image
    # The ".fkeep" after defaults to the fast bicubic preview (instant, offline,
    # the Compress-tab convention). use_ai opts into the real Real-ESRGAN restore
    # — but only when the [ai] extra is actually installed: otherwise restore()
    # would run the same bicubic, slowly, and mislabel it, so fall back to the
    # preview honestly. (A faithful image is just decoded either way.) Imported
    # lazily so gui.py still imports without the aggressive/torch stack.
    ai = use_ai
    if ai:
        from .aggressive.restorer import realesrgan_available

        ai = realesrgan_available()
    after, kind = compare_mod.load_after(
        compressed_path, FaceKeepConfig().aggressive, preview=not ai)
    after = compare_mod.align(after, before)

    report = metrics.compare(before, after)
    diff = compare_mod.diff_map(before, after, amplify=amplify)

    orig_bytes = Path(original_path).stat().st_size
    comp_bytes = Path(compressed_path).stat().st_size
    ratio = orig_bytes / comp_bytes if comp_bytes else 0.0
    lines = [
        f"**SSIM:** {report.overall_ssim:.4f}  ·  "
        f"**PSNR:** {report.overall_psnr:.2f} dB",
        f"**Size:** {_fmt_size(orig_bytes)} → {_fmt_size(comp_bytes)} "
        f"(**{ratio:.1f}× smaller**)",
        f"_'After' = {kind}. Brighter areas in the difference view are larger "
        "deviations._",
    ]
    if use_ai and not ai:
        lines.append(
            "_Real AI restore needs the `[ai]` extra (not installed) — showed "
            "the fast bicubic preview instead._"
        )
    summary = "\n\n".join(lines)
    return CompareOutput(
        before_rgb=_bgr_to_rgb(before),
        after_rgb=_bgr_to_rgb(after),
        diff_rgb=_bgr_to_rgb(diff),
        summary=summary,
    )


# --- one-click backup (ROADMAP 11.2) ---------------------------------------

# Last-used GUI state (e.g. the Backup tab's folders) — a tiny best-effort JSON
# under the user-global facekeep cache dir (the models/detections precedent).
# Losing or corrupting it only costs prefilled defaults, never correctness.
_GUI_STATE_PATH = Path.home() / ".cache" / "facekeep" / "gui_state.json"

_BACKUP_TABLE_HEADERS = [
    "file", "status", "mode", "codec", "before", "after", "ratio",
    "quality", "faces",
]


def load_gui_state() -> dict:
    """Read the persisted GUI state (last-used folders). Best-effort: {} on any error."""
    try:
        with open(_GUI_STATE_PATH, encoding="utf-8") as fh:
            state = json.load(fh)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def save_gui_state(**updates) -> None:
    """Merge ``updates`` into the persisted GUI state. Best-effort: never raises."""
    state = load_gui_state()
    state.update(updates)
    try:
        _GUI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_GUI_STATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except OSError:
        pass


@dataclass
class BackupProgress:
    """One live progress tick: the file about to be processed + running counts."""

    done: int      # files finished so far
    total: int
    current: str   # name of the file being processed now
    kind: str      # "photo" | "video"
    ok: int = 0
    unchanged: int = 0
    failed: int = 0
    skipped: int = 0
    saved_bytes: int = 0


@dataclass
class BackupResult:
    """The completion report of a backup run (totals + the per-file ledger)."""

    files: int
    ok: int
    unchanged: int
    failed: int
    skipped: int
    total_in: int
    total_out: int
    rows: List[report_mod.ReportRow] = field(default_factory=list)
    report_path: str = ""  # the written CSV (the --report machinery's artifact)
    summary: str = ""      # Markdown


def _backup_config(lossless: bool = False) -> FaceKeepConfig:
    """The Backup tab's config: faithful defaults + the lossless toggle.

    Backup is deliberately faithful-only (the honest default for a flow whose
    output may become the user's only copy); aggressive stays on the Compress
    tab where its trade-off is explained per photo.
    """
    config = FaceKeepConfig()
    config.faithful.lossless = bool(lossless)
    config.validate()
    return config


def run_backup(
    source_dir: str,
    archive_dir: str,
    *,
    lossless: bool = False,
    include_videos: bool = True,
    report_dir: str | None = None,
):
    """One-click folder backup: the browser-free generator core of the Backup tab.

    Yields a :class:`BackupProgress` before each file and a final
    :class:`BackupResult` (always the last item). Each file runs through
    ``cli._run_batch(only_files=[f])`` — the exact CLI machinery, driven one
    file at a time so the UI gets live per-file progress — photos first, then
    the videos (which the batch already encodes serially). Outputs, the
    incremental-index skip behavior, and the per-file ledger rows are therefore
    identical to a ``facekeep compress <src> -o <archive>`` run; the only
    differences are cosmetic (no ``--jobs`` pool, per-call video ETA lines).

    The per-file :class:`report.ReportRow` ledger is also written as a CSV
    (into ``report_dir``, a fresh temp dir by default) so the UI can offer a
    download. The last-used folders are persisted (:func:`save_gui_state`) as
    soon as validation passes, so even an interrupted run remembers them.

    Raises :class:`FaceKeepError` on bad folders (missing source, source ==
    archive, nothing to back up) — before the first file is touched.
    """
    # Lazy: the CLI module hosts the shared batch machinery (_run_batch is the
    # extracted `compress` body the watch loop drives too) — not a new pipeline.
    from . import video as video_mod
    from .cli import IMAGE_EXTS, _gather, _run_batch

    if not source_dir or not str(source_dir).strip():
        raise FaceKeepError("Pick a source folder (where your photos are).")
    if not archive_dir or not str(archive_dir).strip():
        raise FaceKeepError(
            "Pick an archive folder (where the compressed copies go)."
        )
    src = Path(str(source_dir).strip())
    dst = Path(str(archive_dir).strip())
    if not src.is_dir():
        raise FaceKeepError(f"Source folder does not exist: {src}")
    try:
        same = src.resolve() == dst.resolve()
    except OSError:
        same = False
    if same:
        raise FaceKeepError(
            "The archive folder must differ from the source folder "
            "(compressed copies would land beside the originals)."
        )

    config = _backup_config(lossless)
    photos = _gather(src, IMAGE_EXTS)
    videos = (
        _gather(src, video_mod.VIDEO_EXTENSIONS)
        if include_videos and config.video.enabled else []
    )
    # Photos first, then the videos — the batch's own serial-video discipline.
    files = [(f, "photo") for f in photos] + [(f, "video") for f in videos]
    if not files:
        raise FaceKeepError(f"No photos or videos found in: {src}")
    ffmpeg_note = bool(videos) and not video_mod.ffmpeg_available()

    dst.mkdir(parents=True, exist_ok=True)
    # Persist as soon as the folders are known-good: a return visit is one
    # click even if this run is cancelled midway.
    save_gui_state(backup_source=str(src), backup_archive=str(dst),
                   backup_lossless=bool(lossless))

    ok = unchanged = failed = skipped = 0
    total_in = total_out = 0
    rows: List[report_mod.ReportRow] = []
    for i, (f, kind) in enumerate(files):
        yield BackupProgress(
            done=i, total=len(files), current=f.name, kind=kind,
            ok=ok, unchanged=unchanged, failed=failed, skipped=skipped,
            saved_bytes=max(0, total_in - total_out),
        )
        code, summary = _run_batch(
            str(src), str(dst), config, no_progress=True,
            no_videos=not include_videos, only_files=[f], collect_rows=True,
        )
        if code == 0 and summary:
            ok += summary["ok"]
            unchanged += summary["unchanged"]
            failed += summary["failed"]
            skipped += summary["skipped"]
            total_in += summary["total_in"]
            total_out += summary["total_out"]
            rows.extend(summary["rows"])
        else:  # defensive: a singleton batch that couldn't run at all
            failed += 1
            rows.append(report_mod.ReportRow(
                file=f.name, mode=config.mode, status="failed"))

    out_dir = Path(report_dir) if report_dir else Path(
        tempfile.mkdtemp(prefix="facekeep-gui-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_mod.write_report(
        rows, str(out_dir / "facekeep_backup_report.csv"))

    yield BackupResult(
        files=len(files), ok=ok, unchanged=unchanged, failed=failed,
        skipped=skipped, total_in=total_in, total_out=total_out, rows=rows,
        report_path=str(report_path),
        summary=_backup_summary(
            len(files), ok, unchanged, failed, skipped, total_in, total_out,
            lossless=lossless, ffmpeg_note=ffmpeg_note),
    )


def _backup_summary(files: int, ok: int, unchanged: int, failed: int,
                    skipped: int, total_in: int, total_out: int, *,
                    lossless: bool, ffmpeg_note: bool) -> str:
    """The completion-report Markdown (totals + the honesty notes)."""
    counts = [f"**{ok} compressed**"]
    if unchanged:
        counts.append(f"{unchanged} already backed up (unchanged)")
    if failed:
        counts.append(f"**{failed} failed**")
    if skipped:
        counts.append(f"{skipped} skipped")
    lines = [f"### Backup complete — {files} file(s)", " · ".join(counts)]
    if total_out:
        ratio = total_in / total_out
        lines.append(
            f"**Size:** {_fmt_size(total_in)} → {_fmt_size(total_out)} "
            f"(**{ratio:.1f}× smaller** — saved {_fmt_size(total_in - total_out)})"
        )
    if ffmpeg_note:
        lines.append(
            "_Videos were skipped: the external `ffmpeg` binary was not found. "
            "Photos are unaffected; install ffmpeg to compress videos too._"
        )
    lines.append(
        "_Lossless (bit-exact) encode — the archive copies are exact._"
        if lossless else
        "_Faithful output is visually lossless, **not bit-exact**. Turn on "
        "Lossless if the archive will be your only copy of irreplaceable "
        "originals._"
    )
    lines.append("_Sources were not deleted or modified._")
    return "\n\n".join(lines)


def _rows_to_table(rows: List[report_mod.ReportRow]) -> list:
    """ReportRow ledger -> display table (list of lists, human-readable sizes).

    Pure and gradio-free so it is unit-testable; column order matches
    ``_BACKUP_TABLE_HEADERS``. ``None`` renders as a blank cell (the report
    module's honesty rule: blank means "not measured", never an invented 0).
    """
    table = []
    for r in rows:
        table.append([
            r.file,
            r.status,
            r.mode,
            r.codec or "",
            "" if r.original_bytes is None else _fmt_size(r.original_bytes),
            "" if r.output_bytes is None else _fmt_size(r.output_bytes),
            "" if r.ratio is None else f"{r.ratio:.1f}x",
            "" if r.quality is None else str(r.quality),
            "" if r.faces is None else str(r.faces),
        ])
    return table


def build_demo(config: FaceKeepConfig | None = None):
    """Build the Gradio ``Blocks`` UI (gradio imported lazily — needs ``[gui]``).

    A single-image drag-and-drop front end: upload a photo, pick a mode, tune a
    few knobs, and get a before/after with stats and a download. The wiring is
    intentionally thin — all real work is :func:`compress_image`.
    """
    import gradio as gr  # lazy: only the GUI path needs the [gui] extra

    with gr.Blocks(title="FaceKeep", analytics_enabled=False) as demo:
        gr.Markdown(
            "# FaceKeep\n"
            "Face-aware photo compression that never ruins a face. Drag a photo "
            "in, pick a mode, compress.\n\n"
            "**Aggressive** (the headline) shrinks it hard — faces, hands, and "
            "fine detail stay at original quality while the benign background is "
            "rebuilt on `facekeep restore` — and downloads a smaller `.fkeep`. "
            "**Faithful** (the default) writes a standard `.avif`/`.jxl` with "
            "every pixel real that opens anywhere, no restore step."
        )
        with gr.Tab("Compress"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(type="filepath", sources=["upload"],
                                   label="Photo")
                    mode = gr.Radio(["faithful", "aggressive"], value="faithful",
                                    label="Mode")
                    with gr.Group(visible=True) as faithful_box:
                        codec = gr.Dropdown(["avif", "jxl", "webp", "both"],
                                            value="avif", label="Codec")
                        auto_tune = gr.Checkbox(
                            value=True,
                            label="Auto-tune quality (visually lossless)")
                        quality = gr.Slider(1, 100, value=82, step=1,
                                            label="Quality (used when auto-tune off)")
                    with gr.Group(visible=False) as aggressive_box:
                        preset = gr.Dropdown([_NO_PRESET, *PRESET_NAMES],
                                             value=_NO_PRESET, label="Preset")
                        bg_scale = gr.Slider(
                            0.05, 0.5, value=0.25, step=0.01,
                            label="Background scale (ignored when a preset is chosen)")
                    run = gr.Button("Compress", variant="primary")
                with gr.Column(scale=1):
                    before = gr.Image(label="Before (original)")
                    after = gr.Image(label="After (decoded output / restore preview)")
                    summary = gr.Markdown()
                    download = gr.File(label="Download output")

            def _toggle(m):
                return (gr.update(visible=m == "faithful"),
                        gr.update(visible=m == "aggressive"))

            mode.change(_toggle, inputs=mode,
                        outputs=[faithful_box, aggressive_box])

            def _run(image_path, m, cdc, at, q, pre, bg):
                # A preset owns the aggressive knobs, so ignore the bg-scale slider
                # when one is chosen (and only honor a preset in aggressive mode).
                chosen = None if (m != "aggressive" or pre == _NO_PRESET) else pre
                try:
                    out = compress_image(
                        image_path, m,
                        preset=chosen,
                        codec=cdc,
                        quality=(None if at else q),
                        auto_tune=at,
                        bg_scale=(None if chosen else bg),
                    )
                except FaceKeepError as e:
                    return None, None, f"**Error:** {e}", None
                return out.before_rgb, out.after_rgb, out.summary, out.output_path

            run.click(
                _run,
                inputs=[inp, mode, codec, auto_tune, quality, preset, bg_scale],
                outputs=[before, after, summary, download],
            )

        with gr.Tab("Compare"):
            gr.Markdown(
                "Compare an original photo against **any** compressed output — a "
                "faithful `.avif`/`.jxl`/`.webp`, or an aggressive `.fkeep` "
                "(reconstructed on the fly). Drag the slider to wipe between "
                "them; brighter areas in the difference view are larger "
                "deviations."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    cmp_orig = gr.Image(type="filepath", sources=["upload"],
                                        label="Original photo")
                    cmp_file = gr.File(
                        type="filepath",
                        label="Compressed file (.avif / .jxl / .webp / .fkeep)")
                    cmp_amplify = gr.Slider(1, 32, value=8, step=1,
                                            label="Difference amplification")
                    cmp_ai = gr.Checkbox(
                        value=False,
                        label="Real AI restore for .fkeep (slow — needs [ai])")
                    gr.Markdown(
                        "_Off = instant bicubic preview. On = the real "
                        "Real-ESRGAN restore for a `.fkeep` — sharper, but can "
                        "take **minutes** on CPU; a spinner shows while it runs._"
                    )
                    cmp_run = gr.Button("Compare", variant="primary")
                with gr.Column(scale=1):
                    cmp_slider = gr.ImageSlider(label="Before / after")
                    cmp_diff = gr.Image(label="Difference (brighter = larger)")
                    cmp_summary = gr.Markdown()

            def _run_compare(orig_path, comp_path, amp, use_ai):
                # Warn up front: a real AI restore of a .fkeep can take minutes
                # (or fall back to the preview if the [ai] extra isn't installed).
                if use_ai and comp_path and str(comp_path).lower().endswith(
                        ".fkeep"):
                    from .aggressive.restorer import realesrgan_available

                    if realesrgan_available():
                        gr.Info(
                            "Running a real AI restore — this can take several "
                            "minutes on CPU. The spinner means it's working."
                        )
                    else:
                        gr.Warning(
                            "Real AI restore needs the [ai] extra (not "
                            "installed) — showing the fast bicubic preview."
                        )
                try:
                    out = compare_images(orig_path, comp_path,
                                         amplify=float(amp), use_ai=use_ai)
                except FaceKeepError as e:
                    return None, None, f"**Error:** {e}"
                return (out.before_rgb, out.after_rgb), out.diff_rgb, out.summary

            cmp_run.click(
                _run_compare,
                inputs=[cmp_orig, cmp_file, cmp_amplify, cmp_ai],
                outputs=[cmp_slider, cmp_diff, cmp_summary],
                show_progress="full",
            )

        with gr.Tab("Backup"):
            gr.Markdown(
                "Back up a whole folder in one click: every photo (and video, "
                "when the external `ffmpeg` is installed) is compressed into "
                "the archive folder — byte-identical to `facekeep compress`. "
                "A return visit is cheap: unchanged files are skipped "
                "automatically, so you can re-run any time.\n\n"
                "**Honesty note:** faithful compression is **visually "
                "lossless, not bit-exact**. If the archive will be your "
                "**only** copy of irreplaceable originals, turn on "
                "**Lossless**. Sources are never deleted or modified.\n\n"
                "_For hands-off continuous backup, run `facekeep watch "
                "<inbox> -o <archive>` in a terminal — it keeps the folder "
                "compressed as new files arrive._"
            )
            state = load_gui_state()
            with gr.Row():
                with gr.Column(scale=1):
                    bk_src = gr.Textbox(
                        value=state.get("backup_source", ""),
                        label="Source folder (your photos)",
                        placeholder=r"e.g. C:\Users\me\Pictures\inbox")
                    bk_dst = gr.Textbox(
                        value=state.get("backup_archive", ""),
                        label="Archive folder (compressed copies)",
                        placeholder=r"e.g. D:\photo-archive")
                    bk_lossless = gr.Checkbox(
                        value=bool(state.get("backup_lossless", False)),
                        label="Lossless (bit-exact archival — larger files)")
                    bk_videos = gr.Checkbox(
                        value=True,
                        label="Include videos (slow — an overnight-batch "
                              "feature; needs ffmpeg)")
                    bk_run = gr.Button("Back up now", variant="primary")
                with gr.Column(scale=2):
                    bk_status = gr.Markdown()
                    bk_table = gr.Dataframe(
                        headers=list(_BACKUP_TABLE_HEADERS),
                        label="Per-file report", interactive=False)
                    bk_csv = gr.File(label="Download CSV report")

            def _run_backup_ui(src, dst, lossless, include_videos):
                # A generator handler: each yield updates the UI live — one
                # tick per file (photos first, then the serial videos), then
                # the completion report + table + CSV download.
                try:
                    for item in run_backup(src, dst, lossless=lossless,
                                           include_videos=include_videos):
                        if isinstance(item, BackupResult):
                            yield (item.summary, _rows_to_table(item.rows),
                                   item.report_path)
                        else:
                            counts = (f"{item.ok} ok · {item.unchanged} "
                                      f"unchanged · {item.failed} failed")
                            if item.saved_bytes:
                                counts += (f" · saved "
                                           f"{_fmt_size(item.saved_bytes)}")
                            yield (
                                f"Backing up **{item.done + 1}/{item.total}**"
                                f" — `{item.current}` ({item.kind})…\n\n"
                                f"{counts}",
                                gr.update(), gr.update(),
                            )
                except FaceKeepError as e:
                    yield f"**Error:** {e}", None, None

            bk_run.click(
                _run_backup_ui,
                inputs=[bk_src, bk_dst, bk_lossless, bk_videos],
                outputs=[bk_status, bk_table, bk_csv],
            )
    return demo


def launch(*, host: str = "127.0.0.1", port: int = 7860, share: bool = False,
           inbrowser: bool = False, **kwargs):
    """Launch the local GUI server (sharing + telemetry off by default).

    ``host`` defaults to ``127.0.0.1`` (local only — never bind all interfaces by
    default) and ``share`` to ``False`` (no public tunnel). Raises ``ImportError``
    if gradio (the ``[gui]`` extra) is not installed; the CLI translates that into
    an install hint.
    """
    demo = build_demo()
    # Pass only kwargs that are stable across gradio 4/5/6 — e.g. ``show_api``
    # was removed from ``launch`` in gradio 6, so hardcoding it crashes there.
    # Extras can still be forwarded deliberately via ``**kwargs``.
    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "share": share,
        "inbrowser": inbrowser,
        **kwargs,
    }
    return demo.launch(**launch_kwargs)
