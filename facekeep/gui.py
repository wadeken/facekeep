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
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile

import cv2
import numpy as np

from . import compare as compare_mod, encoders, metrics
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
