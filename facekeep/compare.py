"""Before/after comparison report (``facekeep compare``).

A read-only visualization tool: it loads an ORIGINAL image and a COMPRESSED
artifact — a faithful ``.avif``/``.jxl``/``.webp``, an aggressive ``.fkeep``
(restored on the fly), or an already-restored standard image — reconstructs the
"after" image the user actually gets, and writes a single **self-contained**
HTML report with a before/after slider, a difference view, and the quality
metrics.

It changes **no output pixels** — it only reads existing outputs produced by the
(unchanged) compress/restore pipelines and visualizes them, so it adds no new
fidelity surface. Per CLAUDE.md's decision rule the verification bar is therefore
pytest + ruff (no bench/eyeball).

Color order is BGR everywhere internally (OpenCV convention). The only BGR->RGB
boundary is encoding a preview into the HTML, which ``cv2.imencode`` does from
BGR directly — so there is no manual channel flip here.
"""

import base64
import copy
import html
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import imageio, metrics
from .exceptions import EncodingError, FaceKeepError

logger = logging.getLogger("facekeep.compare")

# .fkeep is the only "compressed" input that needs reconstructing to get the
# "after" pixels; every other supported input is a standard image we decode.
_FKEEP_SUFFIX = ".fkeep"

# Default longest-side cap for the *embedded preview* images so a 24MP photo
# doesn't bloat the HTML to tens of MB (base64'd twice). The metrics are always
# computed on the full-resolution arrays — only the embedded JPEGs are scaled.
_EMBED_MAX_SIDE = 1600


def _to_uint8_view(img: np.ndarray) -> np.ndarray:
    """An 8-bit BGR view for *visualization* (embedding / diff), dtype-safe.

    The metrics see the real arrays (``metrics`` normalizes by dtype); only the
    HTML preview and the difference map need a common 8-bit scale.
    """
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img.astype(np.uint32) * 255 // 65535).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def align(after: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Resize ``after`` to ``like``'s height/width if they differ.

    Faithful decode and aggressive restore both reproduce the original geometry,
    but an already-restored image passed by hand might not — and the metrics
    require matching spatial dimensions. Area-average when shrinking, cubic when
    growing (the usual quality choice for each direction).
    """
    if after.shape[:2] == like.shape[:2]:
        return after
    h, w = like.shape[:2]
    interp = cv2.INTER_AREA if (after.shape[0] > h or after.shape[1] > w) else cv2.INTER_CUBIC
    return cv2.resize(after, (w, h), interpolation=interp)


def _restore_agg_config(fkeep_path: str, agg_config, explicit_keys: frozenset):
    """``agg_config`` with the ``.fkeep``'s recorded preset's restore-side knobs.

    A ``.fkeep`` compressed with ``--preset`` records the name in its manifest
    (``settings.preset``, 1.7.0+). To make the compare "after" match what
    ``facekeep restore`` actually produces, auto-apply that preset's restore-side
    knobs (face-enhance backend/fidelity/strength) onto a *copy* of
    ``agg_config`` — unless the user explicitly set them (explicit wins, the same
    precedence as restore). This mirrors ``cli.restore._restorer_for``.

    Best-effort and tolerant by structure: an unreadable manifest (missing file
    → ``OSError``, malformed → ``FormatError``) or an absent/unknown preset
    returns ``agg_config`` unchanged, letting ``restore()`` surface any real
    problem. Returns the *same* object when there is nothing to apply, so callers
    can cheaply tell "untouched" from "preset applied".
    """
    from .aggressive.format import read_fkeep_info
    from .config import preset_restore_overrides

    try:
        manifest = read_fkeep_info(fkeep_path)
    except (FaceKeepError, OSError):
        return agg_config
    name = (manifest.get("settings") or {}).get("preset")
    overrides = preset_restore_overrides(name, explicit_keys)
    if not overrides:
        return agg_config
    agg = copy.deepcopy(agg_config)
    for dotted, value in overrides.items():
        # Restore-side preset keys all live under aggressive.* (strip the prefix).
        setattr(agg, dotted.split(".", 1)[1], value)
    return agg


def load_after(
    compressed_path: str, agg_config, *, preview: bool = False,
    explicit_keys: frozenset = frozenset(),
) -> tuple[np.ndarray, str]:
    """Reconstruct the "after" BGR image from a compressed artifact.

    Returns ``(bgr, kind)`` where ``kind`` describes how the pixels were
    obtained — ``"bicubic preview"`` / ``"restore"`` for a ``.fkeep``, or
    ``"decoded <ext>"`` for a standard image. A ``.fkeep`` is reconstructed with
    the aggressive ``Restorer`` (AI when ``[ai]`` is installed, else its bicubic
    fallback); ``preview`` forces the fast bicubic ``preview()`` path. Every
    other input is read through ``imageio.load`` (the single image-reading entry
    point, so EXIF orientation is applied and AVIF/JXL/HEIC plugins are used).

    On the real-restore path the ``.fkeep``'s recorded preset (manifest
    ``settings.preset``) auto-drives its restore-side knobs so the "after"
    matches ``facekeep restore`` per file (see :func:`_restore_agg_config`);
    ``explicit_keys`` are restore knobs the caller set by hand, which win over
    the preset. ``preview`` skips that (it never enhances faces).
    """
    p = Path(compressed_path)
    if p.suffix.lower() == _FKEEP_SUFFIX:
        # Imported lazily: the restorer pulls the aggressive subpackage (and, on
        # the AI path, torch) — a faithful-only comparison must not pay for it.
        from .aggressive.restorer import Restorer

        if preview:
            return Restorer(agg_config).preview(compressed_path, None), "bicubic preview"
        agg = _restore_agg_config(compressed_path, agg_config, explicit_keys)
        return Restorer(agg).restore(compressed_path, None), "restore"
    return imageio.load(compressed_path).image, f"decoded {p.suffix.lower()}"


def diff_map(
    before: np.ndarray, after: np.ndarray, *, amplify: float = 8.0,
    colormap: bool = True,
) -> np.ndarray:
    """A BGR difference image: per-pixel mean-absolute difference, amplified.

    Identical inputs yield an all-zero map. ``amplify`` scales the (usually
    small) differences into a visible range; with ``colormap`` the magnitude is
    rendered as an INFERNO heatmap (bright = larger difference), otherwise as a
    grayscale BGR image. Inputs must share spatial dimensions (the caller aligns
    first); dtype is normalized to an 8-bit view, so this is a *visual* diff, not
    the metric.
    """
    b = _to_uint8_view(before).astype(np.int16)
    a = _to_uint8_view(after).astype(np.int16)
    mad = np.abs(a - b).astype(np.float32).mean(axis=2)  # H x W
    scaled = np.clip(mad * amplify, 0, 255).astype(np.uint8)
    if colormap:
        return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)  # already BGR
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


def _embed(
    img_bgr: np.ndarray, *, fmt: str = "jpeg", quality: int = 85,
    max_side: Optional[int] = None,
) -> str:
    """Encode a BGR image as a ``data:`` URI for inlining into the HTML.

    JPEG for photos (smaller HTML), PNG for the sharp diff map. ``max_side``
    (when set) area-downscales the *embedded preview* so the report stays small;
    this never affects the metrics, which use the full-resolution arrays.
    """
    v = _to_uint8_view(img_bgr)
    if max_side:
        h, w = v.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            s = max_side / longest
            v = cv2.resize(
                v, (max(1, round(w * s)), max(1, round(h * s))),
                interpolation=cv2.INTER_AREA,
            )
    if fmt == "png":
        ok, buf = cv2.imencode(".png", v)
        mime = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    if not ok:
        raise EncodingError("could not encode a preview image for the HTML report")
    return f"data:{mime};base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


# --------------------------------------------------------------------------- #
# Self-contained HTML (no external assets — images are inlined as data URIs).
# CSS/JS are plain string constants (literal braces, no f-string escaping); only
# the dynamic body is interpolated.
# --------------------------------------------------------------------------- #

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; font: 14px/1.5 system-ui, sans-serif;
       background: #15171c; color: #e7e9ee; }
h1 { font-size: 18px; margin: 0 0 4px; }
.sub { color: #9aa0ad; margin: 0 0 20px; }
.panel { background: #1d2026; border: 1px solid #2a2e37; border-radius: 10px;
         padding: 16px; margin-bottom: 20px; }
.panel h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .05em;
            color: #9aa0ad; margin: 0 0 12px; }
table { border-collapse: collapse; }
th, td { text-align: left; padding: 4px 18px 4px 0; }
th { color: #c4c8d2; font-weight: 600; }
td.note { color: #8a909c; }
.ba { position: relative; max-width: 100%; user-select: none; line-height: 0; }
.ba-img { display: block; width: 100%; height: auto; }
.ba-after { position: absolute; top: 0; left: 0; }
.ba-handle { position: absolute; top: 0; bottom: 0; width: 2px; left: 50%;
             margin-left: -1px; background: #fff; pointer-events: none;
             box-shadow: 0 0 0 1px rgba(0,0,0,.4); }
.ba-range { position: absolute; left: 0; bottom: 0; width: 100%; margin: 0;
            opacity: 0; height: 40px; cursor: ew-resize; }
.tag { position: absolute; top: 8px; padding: 2px 8px; border-radius: 4px;
       background: rgba(0,0,0,.6); font-size: 12px; line-height: 1.4; }
.tag-l { left: 8px; } .tag-r { right: 8px; }
.diff img { display: block; width: 100%; height: auto; border-radius: 6px; }
"""

_SLIDER_JS = """
(function () {
  var r = document.getElementById('cmp-range');
  var a = document.getElementById('cmp-after');
  var h = document.getElementById('cmp-handle');
  function upd() {
    var v = r.value;
    a.style.clipPath = 'inset(0 ' + (100 - v) + '% 0 0)';
    h.style.left = v + '%';
  }
  r.addEventListener('input', upd);
  upd();
})();
"""


def _metrics_rows(rows: list[tuple[str, str, str]]) -> str:
    return "\n".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(val)}</td>"
        f'<td class="note">{html.escape(note)}</td></tr>'
        for name, val, note in rows
    )


def build_html(
    *, before_uri: str, after_uri: str, diff_uri: str,
    metric_rows: list[tuple[str, str, str]], meta_rows: list[tuple[str, str, str]],
    title: str, before_label: str, after_label: str, diff_caption: str,
) -> str:
    """Assemble the self-contained comparison HTML document (a pure function).

    Everything needed to render is inlined: the before/after/diff images are
    ``data:`` URIs and the CSS/JS are embedded, so the single file opens anywhere
    with no assets. Kept side-effect-free (string in, string out) so it is
    unit-tested without a browser, mirroring ``gui.py``'s pure handlers.
    """
    t = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{t}</h1>
<p class="sub">Drag the slider to wipe between original and compressed. Brighter
areas in the difference view are larger deviations.</p>

<div class="panel">
<h2>Summary</h2>
<table>{_metrics_rows(meta_rows)}</table>
</div>

<div class="panel">
<h2>Quality</h2>
<table>{_metrics_rows(metric_rows)}</table>
</div>

<div class="panel">
<h2>Before / after</h2>
<div class="ba" id="cmp-ba">
  <img class="ba-img" src="{before_uri}" alt="original">
  <img class="ba-img ba-after" id="cmp-after" src="{after_uri}" alt="compressed">
  <div class="ba-handle" id="cmp-handle"></div>
  <span class="tag tag-l">{html.escape(before_label)}</span>
  <span class="tag tag-r">{html.escape(after_label)}</span>
  <input class="ba-range" id="cmp-range" type="range" min="0" max="100" value="50"
         aria-label="comparison position">
</div>
</div>

<div class="panel diff">
<h2>Difference</h2>
<img src="{diff_uri}" alt="difference map">
<p class="sub">{html.escape(diff_caption)}</p>
</div>

<script>{_SLIDER_JS}</script>
</body>
</html>
"""


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _default_output(compressed_path: str) -> Path:
    """``<compressed-stem>_compare.html`` next to the compressed file.

    Built by appending to ``Path.stem`` (not ``with_suffix``) so a dotted input
    name like ``2024.05.20_trip.avif`` yields ``2024.05.20_trip_compare.html``,
    not a mangled name (the project-wide dotted-filename rule).
    """
    p = Path(compressed_path)
    return p.parent / f"{p.stem}_compare.html"


def render_comparison(
    original_path: str, compressed_path: str, output_path: Optional[str] = None,
    *, agg_config, preview: bool = False, amplify: float = 8.0,
    want_lpips: bool = False, want_ssimulacra2: bool = False,
    full_res: bool = False, explicit_keys: frozenset = frozenset(),
) -> dict:
    """Build and write the HTML comparison report; return a summary dict.

    Loads the original through ``imageio.load`` and the "after" through
    ``load_after``, aligns them, computes SSIM/PSNR (plus optional LPIPS /
    SSIMULACRA2 when requested *and* available — both degrade to ``None``
    gracefully), renders the difference map, inlines everything into a
    self-contained HTML file, and writes it. The returned dict carries the output
    path, the numeric metrics, and the file sizes/ratio for the CLI to print.
    """
    try:
        before = imageio.load(original_path).image
        after, after_kind = load_after(
            compressed_path, agg_config, preview=preview, explicit_keys=explicit_keys)
    except FaceKeepError:
        raise
    except Exception as e:  # noqa: BLE001 - wrap any reader/restore failure cleanly
        raise FaceKeepError(f"could not load images to compare: {e}") from e

    after = align(after, before)

    report = metrics.compare(before, after)
    lpips = metrics.lpips_distance(before, after) if want_lpips else None
    s2 = metrics.ssimulacra2_score(before, after) if want_ssimulacra2 else None

    metric_rows = [
        ("SSIM", f"{report.overall_ssim:.4f}", "higher = better; 1.0 = identical"),
        ("PSNR", f"{report.overall_psnr:.2f} dB", "higher = better"),
    ]
    if lpips is not None:
        metric_rows.append(("LPIPS", f"{lpips:.4f}", "lower = more perceptually similar"))
    if s2 is not None:
        metric_rows.append(("SSIMULACRA2", f"{s2:.2f}", "higher = better; ~90 visually lossless"))

    orig_bytes = Path(original_path).stat().st_size
    comp_bytes = Path(compressed_path).stat().st_size
    ratio = orig_bytes / comp_bytes if comp_bytes else 0.0
    h, w = before.shape[:2]
    meta_rows = [
        ("Original", html.escape(Path(original_path).name), f"{w}x{h}, {_fmt_size(orig_bytes)}"),
        ("Compressed", html.escape(Path(compressed_path).name),
         f"{_fmt_size(comp_bytes)} ({after_kind})"),
        ("Ratio", f"{ratio:.2f}x", "original size / compressed size"),
    ]

    embed_max = None if full_res else _EMBED_MAX_SIDE
    before_uri = _embed(before, fmt="jpeg", max_side=embed_max)
    after_uri = _embed(after, fmt="jpeg", max_side=embed_max)
    diff_uri = _embed(diff_map(before, after, amplify=amplify), fmt="png", max_side=embed_max)

    html_doc = build_html(
        before_uri=before_uri, after_uri=after_uri, diff_uri=diff_uri,
        metric_rows=metric_rows, meta_rows=meta_rows,
        title=f"FaceKeep compare — {Path(compressed_path).name}",
        before_label="Original", after_label=f"Compressed ({after_kind})",
        diff_caption=f"Mean absolute difference, amplified {amplify:g}x "
                     "(INFERNO heatmap; black = identical).",
    )

    out = Path(output_path) if output_path else _default_output(compressed_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_doc, encoding="utf-8")
    except OSError as e:
        raise FaceKeepError(f"could not write the report to {out}: {e}") from e

    return {
        "output_path": str(out),
        "ssim": report.overall_ssim,
        "psnr": report.overall_psnr,
        "lpips": lpips,
        "ssimulacra2": s2,
        "original_bytes": orig_bytes,
        "compressed_bytes": comp_bytes,
        "ratio": ratio,
        "after_kind": after_kind,
    }
