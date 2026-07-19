"""FaceKeep CLI.

Aggressive mode is the headline: it keeps faces/hands/detail at original quality,
downsamples only the benign background, and rebuilds it on `restore` — a ~8-12x
smaller .fkeep. Faithful mode is the default for a bare `compress`: it encodes the
whole image to a standard .avif/.jxl (every pixel real, opens anywhere, no restore
step).
"""

import copy
import importlib.util
import logging
import multiprocessing
import os
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

from . import __version__, index as index_mod, report, video as video_mod
from .config import (
    PRESET_NAMES,
    FaceKeepConfig,
    apply_preset,
    preset_restore_overrides,
)
from .detector import DetectionCache
from .exceptions import (
    ConfigError,
    FaceKeepError,
    SkipFileError,
    UnsupportedInputError,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
              ".heic", ".heif"}

# HEIC/HEIF (the default capture format on every recent iPhone) decodes only with
# the optional [heic] extra (pillow-heif). When it isn't installed we want a
# friendly, actionable hint rather than a raw "requires an extra plugin" failure
# — see the UnsupportedInputError branch in _process_one.
_HEIC_EXTS = {".heic", ".heif"}


def _heic_plugin_available() -> bool:
    """True if the optional HEIC reader (pillow-heif, the ``[heic]`` extra) imports.

    Cheap and side-effect-free (``find_spec`` only). Used to tell a *missing-plugin*
    HEIC input apart from a genuinely unreadable one: a corrupt HEIC *with* the
    plugin present is still a real failure, not a friendly skip.
    """
    try:
        return importlib.util.find_spec("pillow_heif") is not None
    except (ImportError, ValueError):
        return False


def _heic_install_hint() -> str:
    """Actionable one-liner shown when a HEIC/HEIF input needs the ``[heic]`` extra.

    ASCII-only on purpose: this prints to the terminal, and a non-ASCII dash
    renders as mojibake on a legacy Windows code page (e.g. cp950).
    """
    return ('HEIC/HEIF support needs the optional reader. Install it with: '
            'pip install "facekeep[heic]"')


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


# Live-Photo pair policy (ROADMAP 11.1). A Live Photo is a HEIC/JPG still plus
# a ~3 s .mov, linked by Apple's content-identifier metadata. Measured on this
# repo's own encode command: the container-level pairing key
# (com.apple.quicktime.content.identifier, an mdta tag) DOES survive the
# re-encode (-map_metadata 0 + use_metadata_tags), but the still-image-time
# marker lives in a mebx timed-metadata TRACK, which `-map 0:v:0 -map 0:a:0?`
# drops structurally — so a re-encoded motion side is no longer recognizable
# as a Live Photo by anything, regardless of tags. The decided default policy
# (video.preserve_live_photos): keep the pair's .mov VERBATIM (kept-original
# semantics — it is tiny, so the size cost is negligible) and compress the
# still normally (its EXIF, including the MakerNote asset identifier, rides
# into the output untouched). The pair stays fully reconstructable; what is
# honestly lost either way is Apple-Photos re-import as a *live* photo, which
# requires the original HEIC/JPEG still.
_LIVE_PHOTO_STILL_EXTS = (".heic", ".heif", ".jpg", ".jpeg")

_LIVE_PAIR_REASON = (
    "Live Photo pair - motion side kept verbatim (a re-encode would drop the "
    "still-image-time track and un-pair it)"
)


def _live_photo_sibling(video_path: Path) -> "Path | None":
    """The same-stem photo sibling that makes a ``.mov`` a Live-Photo candidate.

    Apple names both sides identically (``IMG_1234.HEIC`` + ``IMG_1234.MOV``)
    and every folder transport (iCloud for Windows, camera-upload clients, USB
    import) preserves the names. Checked on DISK, not against the current
    batch — in watch mode the two sides can arrive in different cycles. Both
    extension casings are tried for case-sensitive filesystems. A false
    positive only costs bytes (a video kept verbatim), never output
    corruption; the probe-time pairing-key check filters coincidences anyway.
    """
    for ext in _LIVE_PHOTO_STILL_EXTS:
        for cand in (video_path.with_name(video_path.stem + ext),
                     video_path.with_name(video_path.stem + ext.upper())):
            if cand.is_file():
                return cand
    return None


def _gather(path: Path, exts: set[str]) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in exts else []
    if path.is_dir():
        out = []
        for ext in exts:
            out.extend(path.glob(f"*{ext}"))
            out.extend(path.glob(f"*{ext.upper()}"))
        return sorted(set(out))
    return []


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_duration(seconds: float) -> str:
    """Clock-style duration (``M:SS`` / ``H:MM:SS``) for clip lengths/elapsed."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _fmt_eta(seconds: float) -> str:
    """Humanized rough ETA — an estimate deserves coarse units, not ``M:SS``."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _load_config(config_path, mode, codec, quality, bg_scale,
                 verify=None, verify_thorough=None, quality_target=None,
                 auto_tune=None, bit_depth=None, strip_gps=None, lossless=None,
                 residual=None, preset=None):
    config = FaceKeepConfig.load(Path(config_path) if config_path else None)
    # Preset layer: above the defaults (and any YAML-named preset), below the
    # YAML's explicitly written keys (skipped via explicit_keys) and below the
    # CLI flags applied after this block — a hand-written field always beats a
    # preset. --preset implies aggressive mode at the *CLI* level: like
    # `-m aggressive`, it overrides a YAML `mode: faithful` (a config file's
    # persistent default mode must not make the documented `--preset` flag
    # unusable — the shipped/init templates carry one), hence the `- {"mode"}`.
    # Only a same-level contradiction errors: `--preset` + `-m faithful` here,
    # and YAML `preset:` + YAML `mode: faithful` inside load()/validate().
    if preset:
        if mode == "faithful":
            raise ConfigError(
                f"--preset {preset} implies aggressive mode; it cannot be "
                "combined with -m faithful"
            )
        explicit = getattr(config, "explicit_keys", frozenset())
        apply_preset(config, preset, explicit_keys=explicit - {"mode"})
    if mode:
        config.mode = mode
    if strip_gps is not None:
        config.strip_gps = strip_gps
    if lossless is not None:
        config.faithful.lossless = lossless
    if codec:
        config.faithful.codec = codec
    if bit_depth is not None:
        config.faithful.output_bit_depth = int(bit_depth)
    if quality is not None:
        config.faithful.quality = quality
        # An explicit quality is a deliberate override: honor it directly rather
        # than letting the (default-on) auto-tune search pick something else.
        # `--auto-tune` below still wins if the user asks for it alongside `-q`.
        if auto_tune is None:
            config.faithful.auto_tune = False
    if auto_tune is not None:
        config.faithful.auto_tune = auto_tune
    if bg_scale is not None:
        config.aggressive.bg_scale = bg_scale
    if quality_target is not None:
        config.aggressive.quality_target = quality_target
    if residual is not None:
        config.aggressive.residual = residual
    if verify is not None:
        config.faithful.verify = verify
    if verify_thorough:
        config.faithful.verify_thorough = True
        config.faithful.verify = True  # thorough implies the quick check
    config.validate()
    return config


def _pool_init():
    """Worker-process initializer: pin per-process threading to 1.

    Batch parallelism runs N worker *processes*, each encoding one image. The
    codecs (libaom/JXL) and OpenCV/BLAS already spin up their own internal
    threads, so without this every worker would try to use all cores and the
    pool would massively over-subscribe the CPU (N×cores threads), often
    *slower* than serial. Pinning each worker to a single internal thread keeps
    total parallelism at ~N. Set before any heavy import does its thread-count
    probe; cv2 is pinned explicitly because it reads the count at call time.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001 - thread pinning is best-effort
        pass


def _resolve_target(f: Path, out_p: Path, single_explicit_file: bool) -> str:
    """Resolve the per-file output target the writers should use.

    ``single_explicit_file`` is True only for the one-file case where the user
    gave an explicit ``-o`` path that is not a directory (honor it verbatim);
    otherwise the writers append the codec/.fkeep extension to ``<dir>/<stem>``.
    """
    if single_explicit_file:
        return str(out_p)
    return str(out_p / f.stem)


def _marks() -> tuple:
    """Return the (good, bad) status marks, ASCII-degraded on legacy consoles.

    ``verify`` decorates its lines with check marks, but on a Windows console
    with a legacy locale codepage (e.g. cp950) ``click.echo`` raises
    ``UnicodeEncodeError`` trying to write U+2713/U+2717 — crashing the command
    after a successful verification. Degrade to ASCII only when stdout reports
    an encoding that genuinely cannot encode the glyphs; a stream without an
    encoding attribute (StringIO, test runners) accepts any str and keeps them.
    """
    enc = getattr(sys.stdout, "encoding", None)
    if enc:
        try:
            "✓✗".encode(enc)
        except (UnicodeEncodeError, LookupError):
            return "OK", "x"
    return "✓", "✗"


def _stderr_isatty() -> bool:
    """Whether stderr is an interactive terminal (the progress-bar gate).

    Read live each call (``sys.stderr`` can be swapped at runtime, e.g. by
    Click's test runner) and isolated in this tiny seam so the TTY gate is
    deterministically testable without depending on how stderr is wrapped.
    """
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):  # detached / non-stream stderr
        return False


def _maybe_progress(iterable, total: int, enabled: bool):
    """Wrap ``iterable`` in a tqdm progress bar when ``enabled``, else pass through.

    Progress is a pure UX layer over the batch loop with **no effect on
    results** — the per-file lines, totals, and ``--report`` rows are produced
    after the loop, in input order, exactly as before. The bar only conveys
    "X of N done" while work is running, which is the gap parallel mode left
    (results print only after every worker finishes, so a long folder run looks
    stalled).

    ``tqdm`` is an optional dependency: if it isn't installed we degrade
    gracefully to no bar (same as missing-AI → bicubic, offline → Haar). The bar
    is drawn on **stderr** so stdout stays clean for piping/redirection, and
    ``leave=False`` clears it on completion so it doesn't clutter the summary.
    The caller is responsible for only enabling this for a real TTY / multi-file
    run (see ``show_progress`` in ``compress``); when disabled the original
    iterable is returned untouched, so the serial/parallel loops are identical
    to before.
    """
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:  # graceful: no progress bar, the work still runs
        return iterable
    return tqdm(iterable, total=total, unit="img", desc="compress",
                leave=False, file=sys.stderr)


def _process_one(file_str: str, target: str, config: FaceKeepConfig,
                 dry_run: bool, want_report: bool,
                 detection_cache: "DetectionCache | None" = None,
                 hand_detector=None) -> dict:
    """Compress a single file and return a plain (picklable) result dict.

    This is the unit of work for batch parallelism, so it must be a top-level
    function with only picklable arguments/return (``FaceKeepConfig`` is a
    dataclass and pickles fine). It does **no** terminal output — printing and
    ``ReportRow`` construction happen back in the parent so the ordering and
    summary stay identical to a serial run. Per-file errors are caught and
    serialized into the dict (``status`` + ``error``) rather than raised, so one
    bad file never takes down the pool; the parent renders them as before.

    The returned dict always has ``mode`` and ``status``; on success it also
    carries the size/ratio/quality/faces/output-name fields the printer and the
    (optional) report need. ``want_report`` is currently informational — the
    parent decides whether to build a row — but is threaded through so a future
    change can skip extra work when no report is requested.

    ``detection_cache`` is an *open* DetectionCache or ``None``. It is only ever
    passed on the **serial** path (the parent process owns the connection); the
    parallel path always passes ``None`` because a SQLite connection is not
    picklable and worker processes must not share one. ``None`` simply detects
    normally, so the cache never changes output — it is a pure speed feature.

    ``hand_detector`` is a constructed opt-in hand detector (C2) or ``None``. Like
    the detection cache it is **serial-path-only** (the MediaPipe landmarker isn't
    picklable), so the parallel path passes ``None`` and aggressive mode falls back
    to the offline C1 geometric hand zones there. ``None`` is the offline default.
    """
    f = Path(file_str)
    result: dict = {"file": f.name, "mode": config.mode}
    try:
        if config.mode == "faithful":
            from .faithful import compress as faithful_compress

            res = faithful_compress(file_str, target, config, dry_run=dry_run,
                                    detection_cache=detection_cache)
            result.update(
                status="skipped-larger" if res.skipped else "ok",
                original_size=res.original_size,
                compressed_size=res.compressed_size,
                ratio=res.ratio,
                faces=res.faces_detected,
                codec=res.codec,
                quality=res.quality_used,
                ssim=res.quality_score,
                gain_map_carried=res.gain_map_carried,
                output_name=res.output_path.name,
            )
        else:
            from .aggressive.compressor import compress_photo
            from .aggressive.format import _fkeep_path, write_fkeep

            photo = compress_photo(file_str, config,
                                   detection_cache=detection_cache,
                                   hand_detector=hand_detector)
            size = write_fkeep(photo, target, dry_run=dry_run)
            ratio = photo.original_size_bytes / size if size else 0
            result.update(
                status="ok",
                original_size=photo.original_size_bytes,
                compressed_size=size,
                ratio=ratio,
                faces=len(photo.faces),
                output_name=_fkeep_path(target).name,
            )
    except SkipFileError as e:
        result.update(status="skipped", error=str(e))
    except UnsupportedInputError as e:
        # A HEIC/HEIF input without the [heic] extra is a friendly skip with an
        # actionable hint, not a scary FAILED — it isn't the user's fault and the
        # file isn't corrupt. Any *other* unsupported input — including a corrupt
        # HEIC with the plugin installed — stays a real failure (don't mask bugs).
        if f.suffix.lower() in _HEIC_EXTS and not _heic_plugin_available():
            result.update(status="skipped", error=_heic_install_hint())
        else:
            result.update(status="failed", error=str(e))
    except FaceKeepError as e:
        result.update(status="failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - isolate unexpected per-file failures
        result.update(status="failed-unexpected", error=str(e))
    return result


def _keep_original_video(src: Path, target: Path) -> Path:
    """Keep-the-original semantics for a skipped video (the CLI's job per 10.1).

    Mirrors faithful mode's ``encoders.copy_original``: when the run writes to
    a *different* directory (a backup destination), the original is copied
    there so the destination is complete; an in-place run leaves the source
    untouched. The copy keeps the source's real name/extension (it is the
    original, not an AV1 ``.mp4``). Never copies a file onto itself.
    """
    dest = target.parent / src.name
    if dest.exists():
        try:
            if os.path.samefile(src, dest):
                return src
        except OSError:
            pass
    elif os.path.normcase(os.path.abspath(src)) == os.path.normcase(
        os.path.abspath(dest)
    ):
        return src
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest


def _process_one_video(file_str: str, target: str, video_config,
                       dry_run: bool, detector=None,
                       face_aware_ok: bool = True,
                       live_pair: bool = False) -> dict:
    """Compress a single video and return a result dict (parent-process only).

    The video sibling of :func:`_process_one`. It always runs serially in the
    parent — videos are never submitted to the ``--jobs`` pool, because SVT-AV1
    already saturates every core on its own (fanning encodes out would
    oversubscribe the CPU and slow all of them; ROADMAP 10.3's serial-encode
    discipline).

    Dry-run honesty: a photo dry-run performs the real encode and discards it,
    but a video encode costs minutes-to-hours — running it under ``--dry-run``
    would defeat the flag. So a video dry-run only probes and reports the
    *decision* (would-encode vs keep-original), with **no size estimate**
    (status ``would-encode``, blank output size) rather than an invented one.

    Result statuses: ``ok`` (encoded, written), ``kept-original`` (skipped as
    already-efficient or not-smaller; the original is copied to a differing
    output dir — see :func:`_keep_original_video`), ``would-encode`` (dry-run
    only), plus the shared ``skipped``/``failed`` shapes. ``quality`` carries
    the SVT-AV1 CRF used and ``vmaf_p1`` the gate's 1%-low score when it ran.
    ``output_path`` is the absolute artifact path (output or kept original),
    which the index records verbatim.

    ``detector`` is the parent-built shared face detector for face-aware
    quality (10.5; ``None`` lets the library default to Haar);
    ``face_aware_ok=False`` disables face-aware for this run — the caller
    could not construct the *configured* detector, and a custom backend is
    never silently swapped for Haar (the plugin rule), so the honest fallback
    is the base target.

    ``live_pair=True`` marks a video with a same-stem photo sibling on disk
    (the caller's cheap Live-Photo pre-filter). With
    ``video_config.preserve_live_photos`` on, the probe then confirms the
    Apple pairing key (``com.apple.quicktime.content.identifier``) and a
    confirmed pair's motion side is **kept verbatim** (kept-original
    semantics, reason ``_LIVE_PAIR_REASON``) instead of re-encoded — see the
    policy note at ``_LIVE_PHOTO_STILL_EXTS``. A same-stem coincidence
    without the key encodes normally.
    """
    f = Path(file_str)
    result: dict = {"file": f.name, "mode": "video"}
    check_live = live_pair and video_config.preserve_live_photos
    try:
        if dry_run:
            info = video_mod.probe_video(f)
            reason = None
            if check_live and info.content_identifier:
                reason = _LIVE_PAIR_REASON
            elif video_config.skip_efficient:
                reason = video_mod._efficiency_skip_reason(info)
            size = info.size_bytes or f.stat().st_size
            if reason is not None:
                result.update(
                    status="kept-original", reason=reason,
                    original_size=size, compressed_size=size, ratio=1.0,
                    quality=None, output_name=f.name, output_path=str(f),
                )
            else:
                result.update(
                    status="would-encode", original_size=size,
                    quality=video_config.crf, output_name=Path(target).name,
                )
        elif check_live and video_mod.probe_video(f).content_identifier:
            # Confirmed Live-Photo pair: keep the motion side byte-identical
            # (copied into a differing output dir, like every kept-original).
            size = f.stat().st_size
            kept = _keep_original_video(f, Path(target))
            result.update(
                status="kept-original", reason=_LIVE_PAIR_REASON,
                original_size=size, compressed_size=size, ratio=1.0,
                quality=None, output_name=kept.name, output_path=str(kept),
            )
        else:
            face_target = (
                video_config.face_vmaf_target
                if face_aware_ok and video_config.face_aware else None
            )
            res = video_mod.compress_video(
                f, target,
                crf=video_config.crf, preset=video_config.preset,
                skip_efficient=video_config.skip_efficient,
                vmaf_target=video_config.vmaf_target,
                auto_tune=video_config.auto_tune,
                preserve_dolby_vision=video_config.preserve_dolby_vision,
                face_vmaf_target=face_target,
                detector=detector,
            )
            if res.skipped:
                kept = _keep_original_video(f, Path(target))
                result.update(
                    status="kept-original", reason=res.skip_reason,
                    original_size=res.original_size,
                    compressed_size=res.original_size, ratio=1.0,
                    quality=res.crf_used, output_name=kept.name,
                    output_path=str(kept), encode_seconds=res.encode_seconds,
                    faces=res.faces,
                )
            else:
                result.update(
                    status="ok", codec="av1",
                    original_size=res.original_size,
                    compressed_size=res.compressed_size, ratio=res.ratio,
                    quality=res.crf_used,
                    vmaf_p1=res.vmaf.p1 if res.vmaf is not None else None,
                    output_name=res.output_path.name,
                    output_path=str(res.output_path),
                    encode_seconds=res.encode_seconds,
                    faces=res.faces, dolby_vision=res.dolby_vision,
                )
    except SkipFileError as e:
        result.update(status="skipped", error=str(e))
    except FaceKeepError as e:
        result.update(status="failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - isolate unexpected per-file failures
        result.update(status="failed-unexpected", error=str(e))
    return result


def _print_result(res: dict, tag: str, dry_run: bool) -> None:
    """Render one worker result to the terminal (parent side, in input order)."""
    click.echo(f"[{tag}] {res['file']} ... ", nl=False)
    status = res["status"]
    if status == "skipped":
        click.echo(f"SKIP ({res['error']})")
    elif status == "skipped-unchanged":
        # Cache hit: unchanged input + same settings + output still present.
        click.echo(f"SKIP (unchanged) -> {res['output_name']}")
    elif status == "failed":
        click.echo(f"FAILED: {res['error']}", err=True)
    elif status == "failed-unexpected":
        click.echo(f"FAILED (unexpected): {res['error']}", err=True)
    elif res["mode"] == "video":
        if status == "would-encode":
            # Dry-run honesty: no encode ran, so no size estimate is printed —
            # a video is too expensive to test-encode under --dry-run.
            click.echo(
                f"WOULD ENCODE (av1 crf {res['quality']}; size unknown - "
                f"videos are not test-encoded in a dry run) -> {res['output_name']}"
            )
        elif status == "kept-original":
            verb = "WOULD KEEP ORIGINAL" if dry_run else "KEPT ORIGINAL"
            click.echo(f"{verb} ({res['reason']}) -> {res['output_name']}")
        else:  # encoded + written
            vmaf = res.get("vmaf_p1")
            vmaf_note = "" if vmaf is None else f", VMAF p1={vmaf:.1f}"
            dv_note = ", DV" if res.get("dolby_vision") else ""
            faces = res.get("faces")
            face_note = f", {faces} face(s)" if faces else ""
            click.echo(
                f"OK  {_fmt_size(res['original_size'])} -> "
                f"{_fmt_size(res['compressed_size'])} ({res['ratio']:.1f}x, "
                f"av1 crf{res['quality']}{dv_note}{face_note}{vmaf_note}, "
                f"in {_fmt_duration(res.get('encode_seconds') or 0)}) "
                f"-> {res['output_name']}"
            )
    elif res["mode"] == "faithful":
        if status == "skipped-larger":
            verb = "WOULD KEEP ORIGINAL" if dry_run else "KEPT ORIGINAL"
            click.echo(
                f"{verb} (already optimal: encode not smaller "
                f"than {_fmt_size(res['original_size'])}) -> {res['output_name']}"
            )
        else:
            verb = "WOULD WRITE" if dry_run else "OK "
            hdr_note = ", HDR" if res.get("gain_map_carried") else ""
            click.echo(
                f"{verb} {_fmt_size(res['original_size'])} -> "
                f"{_fmt_size(res['compressed_size'])} ({res['ratio']:.1f}x, "
                f"{res['faces']} face(s), {res['codec']} q{res['quality']}"
                f"{hdr_note}) -> {res['output_name']}"
            )
    else:  # aggressive success
        verb = "WOULD WRITE" if dry_run else "OK "
        tail = "[dry-run, not written]" if dry_run else "[needs restore]"
        click.echo(
            f"{verb} {_fmt_size(res['original_size'])} -> "
            f"{_fmt_size(res['compressed_size'])} ({res['ratio']:.1f}x, "
            f"{res['faces']} face(s)) {tail}"
        )


def _row_from_result(res: dict, dry_run: bool) -> report.ReportRow:
    """Build a ReportRow from a worker result (parent side)."""
    status = res["status"]
    if status in ("failed", "failed-unexpected"):
        return report.ReportRow(file=res["file"], mode=res["mode"], status="failed")
    if status == "skipped":
        return report.ReportRow(file=res["file"], mode=res["mode"], status="skipped")
    if status == "skipped-unchanged":
        # Cached: the file wasn't re-processed, but it IS on disk — report the
        # cached sizes/ratio so the ledger is complete, with status "cached".
        return report.ReportRow(
            file=res["file"], mode=res["mode"], status="cached",
            codec=res.get("codec"), original_bytes=res.get("original_size"),
            output_bytes=res.get("compressed_size"), ratio=res.get("ratio"),
            quality=res.get("quality"), output_path=res.get("output_name"),
        )
    if res["mode"] == "video":
        if status == "would-encode":
            # Dry-run: the decision is recorded, sizes stay blank (no encode
            # ran, and the report never invents a number).
            return report.ReportRow(
                file=res["file"], mode="video", status="would-write",
                codec="av1", quality=res.get("quality"),
                original_bytes=res.get("original_size"),
                output_path=res.get("output_name"),
            )
        if status == "kept-original":
            row_status = "would-keep-original" if dry_run else "kept-original"
        else:
            row_status = "written"
        return report.ReportRow(
            file=res["file"], mode="video", status=row_status,
            codec=res.get("codec"), quality=res.get("quality"),
            original_bytes=res.get("original_size"),
            output_bytes=res.get("compressed_size"), ratio=res.get("ratio"),
            faces=res.get("faces"),
            vmaf_p1=res.get("vmaf_p1"), output_path=res.get("output_name"),
        )
    if res["mode"] == "faithful":
        if status == "skipped-larger":
            row_status = "would-keep-original" if dry_run else "kept-original"
        else:
            row_status = "would-write" if dry_run else "written"
        return report.ReportRow(
            file=res["file"], mode="faithful", status=row_status,
            codec=res["codec"], original_bytes=res["original_size"],
            output_bytes=res["compressed_size"], ratio=res["ratio"],
            quality=res["quality"], faces=res["faces"],
            ssim_downscaled=res.get("ssim"), output_path=res["output_name"],
        )
    # Aggressive: no codec quality, no compress-time fidelity score (both blank).
    return report.ReportRow(
        file=res["file"], mode="aggressive",
        status="would-write" if dry_run else "written",
        original_bytes=res["original_size"], output_bytes=res["compressed_size"],
        ratio=res["ratio"], faces=res["faces"], output_path=res["output_name"],
    )


@click.group()
@click.version_option(version=__version__)
def cli():
    """FaceKeep - face-aware photo compression that never ruins a face.

    Aggressive mode (the headline) shrinks a photo library ~8-12x: faces, hands,
    and fine detail are kept at original quality while the benign background is
    downsampled and rebuilt on `facekeep restore`. The quickest way in is a
    one-word goal, e.g. `facekeep compress ./photos --preset family`.

    Faithful mode is the default for a bare `facekeep compress` (a backup tool
    shouldn't silently hand back a reconstructed background): it encodes the whole
    image to a standard .avif/.jxl - every pixel real, opens anywhere, no restore
    step. Use aggressive mode (or a preset) when shrinking the library matters
    more than a pixel-exact background.
    """


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Output path (file or directory).")
@click.option("-m", "--mode", type=click.Choice(["faithful", "aggressive"]),
              default=None,
              help="faithful (default) = standard real-pixel .avif/.jxl, no "
                   "restore step; aggressive = dramatic library-wide shrink to a "
                   ".fkeep, brought back with `facekeep restore`.")
@click.option("--preset", type=click.Choice(list(PRESET_NAMES)), default=None,
              help="Aggressive-mode preset: a one-word goal that tunes the "
                   "aggressive knobs (implies -m aggressive; any explicit "
                   "flag or YAML key still wins). ratio = smallest file, "
                   "pretty = best-looking restore, fidelity = closest to the "
                   "original (residual), family = max face/hand protection, "
                   "share = ratio + GPS strip.")
@click.option("--codec", type=click.Choice(["avif", "jxl", "webp", "both"]), default=None,
              help="Faithful-mode codec (default: avif). 'both' trial-encodes "
                   "each image with avif and jxl and keeps the smaller output. "
                   "'webp' is the maximum-compatibility fallback (opens in any "
                   "browser/old viewer; larger than avif/jxl).")
@click.option("-q", "--quality", type=int, default=None,
              help="Faithful-mode quality 0-100. Giving an explicit quality "
                   "turns auto-tune off (the quality is used directly); without "
                   "it, auto-tune picks a visually-lossless quality.")
@click.option("--auto-tune/--no-auto-tune", "auto_tune", default=None,
              help="Faithful mode: search for a visually-lossless quality "
                   "(perceptual target) instead of a fixed --quality. On by "
                   "default; --no-auto-tune uses --quality directly.")
@click.option("--bit-depth", type=click.Choice(["10", "12"]), default=None,
              help="Faithful AVIF output depth for a high-bit (16-bit) source "
                   "(default: 10). Needs the external avifenc CLI; ignored for "
                   "8-bit sources, JXL, or when avifenc is absent (falls back to "
                   "8-bit). Never widens an 8-bit source.")
@click.option("--bg-scale", type=float, default=None,
              help="Aggressive-mode background scale (default: 0.25).")
@click.option("--quality-target", type=float, default=None,
              help="Aggressive mode: auto-choose bg_scale per photo to hit this "
                   "target perceptual quality (LPIPS distance, lower = more "
                   "similar; e.g. 0.15). Needs the [ai] extra; falls back to "
                   "--bg-scale if LPIPS is unavailable.")
@click.option("--lossless/--lossy", "lossless", default=None,
              help="Faithful mode: encode mathematically lossless (bit-exact) for "
                   "archiving irreplaceable originals. Ignores --quality/auto-tune; "
                   "the file is much larger. JXL is lossless natively; lossless AVIF "
                   "needs the avifenc CLI and otherwise falls back to lossless JXL.")
@click.option("--residual/--no-residual", "residual", default=None,
              help="Aggressive mode: also store the real detail the background "
                   "downsample lost (a half-res residual layer), so restore adds "
                   "it back instead of hallucinating — background faithful-but-"
                   "lossy, larger .fkeep. Off by default.")
@click.option("--strip-gps/--keep-gps", "strip_gps", default=None,
              help="Strip the GPS (location) EXIF from the output for privacy. "
                   "Off by default (EXIF round-trips unchanged). Keeps "
                   "date/camera/orientation; applies to both modes.")
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Path to a config YAML file.")
@click.option("--verify/--no-verify", "verify", default=None,
              help="Faithful mode: decode the output and check it matches the "
                   "source (default: on).")
@click.option("--verify-thorough", is_flag=True,
              help="Faithful mode: also require a downscaled-SSIM floor on the "
                   "round-trip (implies --verify).")
@click.option("--dry-run", is_flag=True,
              help="Estimate projected sizes/ratios without writing any output "
                   "(runs the real encode/pack for photos, then discards it; "
                   "videos are probed only — encoding one costs minutes-to-"
                   "hours, so no size is estimated).")
@click.option("--report", "report_path", type=click.Path(), default=None,
              help="Write a per-file CSV report (size, ratio, quality, faces, "
                   "codec). The SSIM column is filled only when actually measured "
                   "— pair with --verify-thorough for faithful mode.")
@click.option("-j", "--jobs", type=int, default=1,
              help="Process a folder across N worker processes (default: 1 = "
                   "serial). 0 = one per CPU. Ignored for a single file. "
                   "Photos only: videos always encode serially (SVT-AV1 "
                   "already saturates the cores).")
@click.option("--no-videos", is_flag=True,
              help="Exclude videos from this run (photos only). Videos are "
                   "otherwise compressed faithfully to AV1 .mp4 — slow "
                   "(roughly 4 minutes per minute of 4K on a desktop CPU), so "
                   "this is the quick-photo-pass escape hatch. Config: "
                   "video.enabled.")
@click.option("--no-progress", is_flag=True,
              help="Disable the folder progress bar (shown by default on a "
                   "terminal for multi-file runs; auto-hidden when piped).")
@click.option("--force", is_flag=True,
              help="Re-process every file, ignoring the incremental index "
                   "(which otherwise skips unchanged photos on a re-run).")
@click.option("--index", "index_path", type=click.Path(), default=None,
              help="Path to the incremental-processing SQLite index "
                   "(default: .facekeep-index.sqlite in the output directory).")
@click.option("--no-index", is_flag=True,
              help="Disable the incremental index entirely (no DB read/write); "
                   "always process every file but record nothing.")
@click.option("--no-detect-cache", is_flag=True,
              help="Disable the detection cache (a user-global cache of face "
                   "detections, reused across re-runs). On by default for serial "
                   "runs; automatically off under --jobs.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def compress(input_path, output_path, mode, preset, codec, quality, auto_tune,
             bit_depth, bg_scale, quality_target, lossless, residual, strip_gps,
             config_path, verify, verify_thorough, dry_run, report_path, jobs,
             no_videos, no_progress, force, index_path, no_index,
             no_detect_cache, verbose):
    """Compress photo(s) and video(s). INPUT_PATH is a file or a directory.

    Faithful mode (the default) writes a standard .avif/.jxl with every pixel
    real - no restore step. For a dramatic, library-wide shrink switch to
    aggressive mode (-m aggressive, or just pick a --preset, which implies it):
    faces/hands/detail are kept sharp, the benign background is rebuilt on
    restore, and the output is a .fkeep you bring back with `facekeep restore`.

    Videos (mp4/mov/...) are compressed faithfully too: a slow offline SVT-AV1
    re-encode into a standard .mp4 that plays anywhere modern (typically 2-10x
    smaller than a phone recording at visually-lossless quality, VMAF-verified;
    HDR and A/V sync survive). It needs the external ffmpeg binary - without
    one, videos are skipped with a hint. Encoding is slow (an overnight-batch
    feature); --no-videos excludes them for a quick photo pass. Aggressive
    mode never applies to video.

    HEIC/HEIF input (e.g. iPhone photos) needs the optional [heic] extra
    (pip install "facekeep[heic]"); without it those files are skipped with a hint.
    """
    _setup_logging(verbose)
    try:
        config = _load_config(config_path, mode, codec, quality, bg_scale,
                              verify, verify_thorough, quality_target, auto_tune,
                              bit_depth, strip_gps, lossless, residual, preset)
    except FaceKeepError as e:
        click.echo(f"Config error: {e}", err=True)
        sys.exit(2)

    code, _summary = _run_batch(
        input_path, output_path, config, dry_run=dry_run,
        report_path=report_path, jobs=jobs, no_videos=no_videos,
        no_progress=no_progress, force=force, index_path=index_path,
        no_index=no_index, no_detect_cache=no_detect_cache,
    )
    if code:
        sys.exit(code)


def _run_batch(input_path, output_path, config, *, dry_run=False,
               report_path=None, jobs=1, no_videos=False, no_progress=False,
               force=False, index_path=None, no_index=False,
               no_detect_cache=False, only_files=None):
    """The shared folder/file compress machinery (photos + videos), extracted.

    This is the body ``compress`` always ran — gather, index skip, the photo
    pool, serial videos with ETA lines, index record, ordered replay/summary/
    report — as a reusable function so ``facekeep watch`` (11.1) can run the
    exact same pipeline per cycle. Behavior contract: byte-identical outputs
    and identical terminal output to the pre-extraction ``compress``.

    Returns ``(exit_code, summary)``. ``exit_code`` 0 = the batch ran (some
    files may still have failed individually); 1 = nothing to process (message
    already echoed); 2 = a usage error (message already echoed). ``summary``
    (empty on non-zero codes) carries the per-run counts the watch loop
    aggregates: ``files``/``ok``/``unchanged``/``failed``/``skipped``,
    ``total_in``/``total_out`` (bytes), and ``statuses`` (Path -> status).

    ``only_files`` restricts the run to an explicit file list instead of
    gathering from ``input_path`` (the watch loop passes the stability-checked,
    stat-prefiltered set); ``None`` gathers as before.
    """
    in_p = Path(input_path)
    include_videos = config.video.enabled and not no_videos
    if only_files is not None:
        all_videos = [f for f in only_files
                      if f.suffix.lower() in video_mod.VIDEO_EXTENSIONS]
        photo_files = sorted(f for f in only_files
                             if f.suffix.lower() in IMAGE_EXTS)
        video_files = sorted(all_videos) if include_videos else []
        excluded_videos_exist = bool(all_videos) and not include_videos
    else:
        photo_files = _gather(in_p, IMAGE_EXTS)
        video_files = _gather(in_p, video_mod.VIDEO_EXTENSIONS) if include_videos else []
        excluded_videos_exist = (
            not include_videos and bool(_gather(in_p, video_mod.VIDEO_EXTENSIONS))
        )
    if not photo_files and not video_files:
        if excluded_videos_exist:
            click.echo(
                f"Only video files found in: {input_path} - but videos are "
                "excluded (--no-videos / video.enabled: false).",
                err=True,
            )
        else:
            click.echo(
                f"No supported images or videos found in: {input_path}", err=True
            )
        return 1, {}

    # A video is *never* processed by aggressive mode (ROADMAP Phase 10:
    # per-frame AI restore is computationally absurd and temporally unstable —
    # video is faithful-only, honestly). An explicit single-video input in
    # aggressive mode is a loud error; in a folder run the photos proceed
    # aggressively and each video is skipped with the reason.
    video_skip_reason = None
    if video_files and config.mode == "aggressive":
        if in_p.is_file():
            click.echo(
                "Aggressive mode does not apply to video (video is "
                "faithful-only). Re-run without -m aggressive/--preset to "
                "compress this video.",
                err=True,
            )
            return 2, {}
        video_skip_reason = (
            "aggressive mode does not apply to video - re-run without "
            "-m aggressive/--preset to compress videos (faithful)"
        )
    # Videos need the external ffmpeg binary (the avifenc pattern). Missing:
    # a single-video input errors with the install hint; a folder run skips
    # each video with the hint and processes the photos (the HEIC precedent).
    elif video_files and not video_mod.ffmpeg_available():
        if not photo_files and in_p.is_file():
            click.echo(video_mod._MISSING_FFMPEG_HINT, err=True)
            return 2, {}
        video_skip_reason = video_mod._MISSING_FFMPEG_HINT

    video_set = set(video_files)
    files = sorted(set(photo_files) | video_set)

    out_p = Path(output_path) if output_path else (in_p if in_p.is_dir() else in_p.parent)
    if output_path and len(files) > 1 and not dry_run:
        out_p.mkdir(parents=True, exist_ok=True)

    tag = "dry-run" if dry_run else config.mode

    # Resolve each file's output target up front (same logic as before): an
    # explicit single-file -o path wins, else "<dir>/<stem>" for photos (the
    # writers append the codec/.fkeep extension) and the collision-safe
    # "<dir>/<stem>.mp4" for videos.
    single_explicit_file = (
        output_path is not None and len(files) == 1 and not Path(output_path).is_dir()
    )
    targets = {
        f: (
            str(out_p) if single_explicit_file
            else str(video_mod.output_path_for(f, out_p))
        ) if f in video_set
        else _resolve_target(f, out_p, single_explicit_file)
        for f in files
    }

    # Videos precluded from this run (aggressive mode / no ffmpeg) become
    # ready-made skip results: they take part in the replay/report like any
    # other outcome but are never probed, looked up in the index, or encoded.
    video_precluded: dict = {}
    if video_skip_reason is not None:
        for f in video_files:
            video_precluded[f] = {
                "file": f.name, "mode": "video", "status": "skipped",
                "error": video_skip_reason,
            }

    # Incremental index: decide which files we can skip because they are
    # byte-identical to the last successful run, with the same output-affecting
    # settings, AND their output still exists on disk. This is a pure speed
    # feature — it never changes the bytes of anything we *do* write. The DB is
    # opened only here, in the parent: workers stay pure (the --jobs byte-identical
    # contract is untouched), and we re-open it after the run to record results.
    #   --no-index   -> feature off entirely (no DB at all).
    #   --force      -> process everything, but still record results (refresh cache).
    #   --dry-run    -> never read or write the index (writes nothing on disk).
    use_index = not no_index and not dry_run
    db_path = None
    if use_index:
        # Default location: the *directory* the outputs land in, so the index
        # travels with them. For a single explicit -o file target, out_p IS the
        # output file path — using it directly would make ProcessIndex mkdir
        # that path as a directory and the subsequent image write would fail
        # with PermissionError (a real bug this guards against), so resolve to
        # its parent there.
        index_dir = out_p.parent if single_explicit_file else out_p
        db_path = (
            Path(index_path) if index_path else index_dir / index_mod.INDEX_FILENAME
        )
    fingerprint = index_mod.settings_fingerprint(config)
    # Videos cache under their own fingerprint (only the video: knobs): retuning
    # a photo setting must not bust a folder's cached video encodes — a video
    # re-encode costs minutes-to-hours, exactly the work the index exists to skip.
    video_fingerprint = (
        index_mod.video_settings_fingerprint(config) if video_files else None
    )

    def _fingerprint_for(f: Path) -> str:
        return video_fingerprint if f in video_set else fingerprint

    # SHA-256 of each input, computed once and reused for both the skip lookup
    # and the post-run record (so we never hash a file twice). The stat pair
    # (size, mtime_ns) is captured at the same moment — stat *before* hash, so
    # a file changing mid-read leaves a stale stat that the watch pre-filter
    # will miss on (the safe direction: re-check, never wrongly skip).
    hashes: dict = {}
    stat_map: dict = {}  # f -> (size, mtime_ns) captured at hash time
    skipped_unchanged: dict = {}  # f -> result dict for a cache hit
    candidates = [f for f in files if f not in video_precluded]
    to_process = list(candidates)
    if use_index and not force:
        try:
            with index_mod.ProcessIndex(db_path) as idx:
                remaining = []
                for f in candidates:
                    try:
                        st = f.stat()
                        stat_map[f] = (st.st_size, st.st_mtime_ns)
                        h = index_mod.hash_file(f)
                    except OSError:
                        remaining.append(f)  # unreadable now: let the pipeline report it
                        continue
                    hashes[f] = h
                    hit = idx.is_unchanged(f, h, _fingerprint_for(f))
                    if hit is not None:
                        # Hash hit with a stale/absent recorded stat (a sync
                        # client re-wrote identical bytes, or a pre-11.1 row):
                        # refresh so the next watch cycle stat-hits hash-free.
                        if (hit.input_size, hit.input_mtime_ns) != stat_map[f]:
                            idx.update_stat(f, *stat_map[f])
                        skipped_unchanged[f] = {
                            "file": f.name, "mode": hit.mode,
                            "status": "skipped-unchanged",
                            "original_size": hit.original_size,
                            "compressed_size": hit.output_size,
                            "ratio": (hit.original_size / hit.output_size
                                      if hit.output_size else 0.0),
                            "codec": hit.codec, "quality": hit.quality,
                            "output_name": Path(hit.output_path).name,
                        }
                    else:
                        remaining.append(f)
                to_process = remaining
        except sqlite3.Error as e:  # a broken cache must never block a run
            logging.getLogger("facekeep.cli").warning(
                "Ignoring unreadable index (%s); processing all files.", e
            )
            to_process = list(candidates)
            skipped_unchanged = {}

    # Split the work: photos may fan out to the --jobs pool; videos NEVER do —
    # SVT-AV1 already saturates every core on one encode, so parallel video
    # encodes would oversubscribe the CPU and slow all of them (ROADMAP 10.3's
    # serial-encode discipline). Videos run serially in the parent, after the
    # photos.
    to_process_photos = [f for f in to_process if f not in video_set]
    to_process_videos = [f for f in to_process if f in video_set]

    # How many workers? 1 (default) = serial, the original code path with no
    # pool overhead. 0 = one per CPU. Otherwise clamp to [1, photos, cpu]: never
    # spawn more workers than files, and a single file is always serial (the
    # pool/pickle cost would only slow it down). Workers run only on the photos
    # we actually need to process (cache skips are excluded; videos are serial).
    n_photos = len(to_process_photos)
    cpu = os.cpu_count() or 1
    n_workers = cpu if jobs == 0 else max(1, jobs)
    n_workers = min(n_workers, max(1, n_photos), cpu)
    parallel = n_workers > 1 and n_photos > 1

    # Show a progress bar only for a multi-photo run on an interactive terminal:
    # a single file stays terse (the bar would be noise), and a non-TTY (CI,
    # pipe, the test runner) gets no bar so stdout/stderr stay clean. The bar is
    # purely cosmetic — see _maybe_progress — and conveys liveness while workers
    # run, which input-ordered result printing alone cannot. The total is the
    # number of photos we actually process (cache-skipped files aren't work;
    # videos get their own per-file liveness/ETA lines instead of the bar).
    show_progress = (
        not no_progress and n_photos > 1 and _stderr_isatty()
    )

    # Detection cache: a user-global cache of face detections (keyed by content
    # hash + detector settings) reused across re-runs. It is a pure speed feature
    # (it never changes which faces are used or the output bytes) and is attached
    # ONLY on the serial path: a SQLite connection is not picklable and worker
    # processes must not share one, so the parallel path always detects normally.
    # Opening it is best-effort — a failure just means "detect normally".
    use_detect_cache = not no_detect_cache and not parallel
    detect_cache = None
    if use_detect_cache:
        try:
            detect_cache = DetectionCache()
        except sqlite3.Error as e:
            logging.getLogger("facekeep.cli").warning(
                "Ignoring unusable detection cache (%s); detecting normally.", e
            )
            detect_cache = None

    # Opt-in hand detector (C2) for aggressive-mode hand protection. Built once in
    # the parent and attached ONLY on the serial path — the MediaPipe landmarker is
    # not picklable, so parallel workers fall back to the offline C1 geometric hand
    # zones (same parent-only discipline as the detection cache). Construction is
    # best-effort: if the package/model is unavailable it returns None and C1 is
    # used. Skipped entirely unless aggressive mode actually requests a backend.
    hand_detector = None
    if (
        config.mode == "aggressive"
        and not parallel
        and config.aggressive.protect_hands
        and config.aggressive.protect_hands_backend is not None
    ):
        from .detector import create_hand_detector

        hand_detector = create_hand_detector(
            config.aggressive.protect_hands_backend,
            confidence=config.aggressive.hand_detect_confidence,
            num_hands=config.aggressive.hand_detect_max_hands,
            detect_long_side=config.aggressive.hand_detect_long_side,
            padding=config.aggressive.hand_detect_padding,
        )

    # Run the work, collecting one result dict per processed file. Results are
    # stored by file and replayed in input order afterwards (merged with the
    # cache skips), so the printed lines, the totals, and the --report rows are
    # identical to (and as deterministic as) a serial run regardless of
    # completion order. The progress bar advances per completed file and does not
    # touch the collected results.
    results: dict = {**video_precluded, **skipped_unchanged}
    try:
        if parallel:
            # Force the "spawn" start method for the worker pool. The default on
            # Linux is "fork", which copies the parent's address space *including
            # mutexes already locked by threads that do not exist in the child* —
            # the native thread pools in OpenCV / OpenMP (numpy, scikit-image) are
            # primed by earlier work in this process, so a forked worker can
            # deadlock on an inherited lock and never return (observed as a hung
            # `--jobs` run on Linux CI; never on Windows, which already spawns).
            # "spawn" starts a fresh interpreter per worker — portable and
            # deadlock-free — at a small extra startup cost, paid once per pool.
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_pool_init,
                mp_context=multiprocessing.get_context("spawn"),
            ) as ex:
                futures = {
                    ex.submit(_process_one, str(f), targets[f], config, dry_run,
                              bool(report_path)): f
                    for f in to_process_photos
                }
                for fut in _maybe_progress(as_completed(futures), n_photos,
                                           show_progress):
                    f = futures[fut]
                    results[f] = fut.result()
        else:
            for f in _maybe_progress(to_process_photos, n_photos, show_progress):
                results[f] = _process_one(str(f), targets[f], config, dry_run,
                                          bool(report_path),
                                          detection_cache=detect_cache,
                                          hand_detector=hand_detector)
    finally:
        if detect_cache is not None:
            detect_cache.close()

    # Face-aware video quality (10.5): build the *configured* shared detector
    # once in the parent — videos are always serial, so one instance serves the
    # whole run (the detection-cache/hand-detector parent-only discipline).
    # If the configured detector cannot be constructed (a broken custom
    # backend), face-aware is disabled for the run instead of silently
    # substituting Haar — a plugin owns its own degradation.
    video_detector = None
    video_face_aware_ok = True
    if to_process_videos and config.video.face_aware and not dry_run:
        try:
            from .detector import create_detector

            video_detector = create_detector(
                backend=config.detector.backend,
                confidence=config.detector.confidence,
                padding=config.detector.padding,
                nms_iou=config.detector.nms_iou,
                min_size_ratio=config.detector.min_size_ratio,
                max_aspect_ratio=config.detector.max_aspect_ratio,
                roi=config.detector.roi,
            )
        except Exception as e:  # noqa: BLE001 - detection never blocks videos
            logging.getLogger("facekeep.cli").warning(
                "Could not construct the configured face detector (%s); "
                "face-aware video quality disabled for this run.", e
            )
            video_face_aware_ok = False

    # Videos: serial, in the parent (see the split above). Each real encode is
    # announced up front on stderr with the clip's size/length and an ETA
    # derived from the throughput (pixels per wall-second, encode + quality
    # gate) measured on the videos already completed *this run* — the honest
    # per-machine number the ROADMAP asked for. The first video has nothing
    # measured yet and says so instead of guessing.
    # Live-Photo pair pre-filter (11.1): a .mov with a same-stem photo sibling
    # on disk is a pair *candidate*; _process_one_video confirms the Apple
    # pairing key at probe time and keeps a confirmed pair's motion side
    # verbatim (see the policy note at _LIVE_PHOTO_STILL_EXTS).
    live_pair_videos: set = set()
    if config.video.preserve_live_photos:
        live_pair_videos = {
            f for f in to_process_videos
            if f.suffix.lower() == ".mov" and _live_photo_sibling(f) is not None
        }

    done_pixels = 0.0
    done_seconds = 0.0
    for i, f in enumerate(to_process_videos, 1):
        pixels = None
        if not dry_run:
            vinfo = None
            try:
                vinfo = video_mod.probe_video(f)
            except FaceKeepError:
                pass  # unreadable/unprobeable: the compress call reports it
            will_encode = vinfo is not None and (
                not config.video.skip_efficient
                or video_mod._efficiency_skip_reason(vinfo) is None
            ) and not (
                f in live_pair_videos and vinfo.content_identifier
            )
            if will_encode:
                pixels = (vinfo.width * vinfo.height
                          * vinfo.duration_s * vinfo.fps)
                eta = (
                    f"est ~{_fmt_eta(pixels / (done_pixels / done_seconds))}"
                    if done_seconds and done_pixels and pixels
                    else "measuring encode speed on this first video"
                )
                click.echo(
                    f"[video {i}/{len(to_process_videos)}] {f.name}: "
                    f"{vinfo.width}x{vinfo.height} "
                    f"{_fmt_duration(vinfo.duration_s)} - encoding ({eta})",
                    err=True,
                )
        res = _process_one_video(str(f), targets[f], config.video, dry_run,
                                 detector=video_detector,
                                 face_aware_ok=video_face_aware_ok,
                                 live_pair=f in live_pair_videos)
        results[f] = res
        secs = res.get("encode_seconds") or 0.0
        if res["status"] == "ok" and pixels and secs > 0:
            done_pixels += pixels
            done_seconds += secs

    # Record successful outcomes back into the index so the next run can skip
    # them. Only ok/skipped-larger/kept-original are cached (a failed file must
    # retry next time); cache hits are already recorded. Writes happen here in
    # the parent, in input order, after all workers have returned — the DB is
    # never touched by a worker. --force still records (it refreshes the cache).
    if use_index:
        try:
            with index_mod.ProcessIndex(db_path) as idx:
                for f in to_process:
                    res = results[f]
                    if res["status"] not in ("ok", "skipped-larger",
                                             "kept-original"):
                        continue
                    h = hashes.get(f)
                    stat_pair = stat_map.get(f)
                    if h is None:
                        # Not hashed up front (--force / --no-index read path):
                        # capture stat at this same moment so the pair stays
                        # consistent with the hash (stat first — see above).
                        try:
                            st = f.stat()
                            stat_pair = (st.st_size, st.st_mtime_ns)
                        except OSError:
                            stat_pair = None
                        h = index_mod.hash_file(f)
                    # The path that must still exist for a future skip. Video
                    # results carry the exact artifact path (output, or the
                    # kept original); photo writers appended the codec/.fkeep
                    # extension to <dir>/<stem>, except in the single
                    # explicit-file case where the target is verbatim.
                    if "output_path" in res:
                        target_out = Path(res["output_path"])
                    elif single_explicit_file:
                        target_out = Path(targets[f])
                    else:
                        target_out = out_p / res["output_name"]
                    idx.record(f, index_mod.IndexRow(
                        content_hash=h,
                        settings_fingerprint=_fingerprint_for(f),
                        mode=res["mode"],
                        codec=res.get("codec"),
                        quality=res.get("quality"),
                        original_size=res["original_size"],
                        # Store resolved so the future existence check is
                        # cwd-independent.
                        output_path=str(target_out.resolve()),
                        output_size=res["compressed_size"],
                        input_size=stat_pair[0] if stat_pair else None,
                        input_mtime_ns=stat_pair[1] if stat_pair else None,
                    ))
        except (sqlite3.Error, OSError) as e:  # recording is best-effort
            logging.getLogger("facekeep.cli").warning(
                "Could not update index (%s); results are still written.", e
            )

    # Replay in input order: print, accumulate totals, build report rows.
    total_in = total_out = 0
    ok = 0
    unchanged = 0  # cache hits (skipped-unchanged) — counted, not re-encoded
    failed = 0
    skipped = 0
    rows = []  # per-file ReportRow ledger (only materialized when --report given)
    for f in files:
        res = results[f]
        # Videos carry their own tag: they are neither of the photo modes.
        file_tag = tag if res["mode"] != "video" else (
            "dry-run" if dry_run else "video"
        )
        _print_result(res, file_tag, dry_run)
        if res["status"] in ("ok", "skipped-larger", "kept-original"):
            total_in += res["original_size"]
            total_out += res["compressed_size"]
            ok += 1
        elif res["status"] == "would-encode":
            ok += 1  # decided, but nothing measured — no size to total
        elif res["status"] == "skipped-unchanged":
            unchanged += 1
        elif res["status"] in ("failed", "failed-unexpected"):
            failed += 1
        elif res["status"] == "skipped":
            skipped += 1
        if report_path:
            rows.append(_row_from_result(res, dry_run))

    if len(files) > 1 and (total_out or unchanged):
        prefix = "[dry-run] " if dry_run else ""
        unchanged_note = f" | {unchanged} unchanged (skipped)" if unchanged else ""
        if total_out:
            saved = "would save" if dry_run else "saved"
            click.echo(
                f"\n--- {prefix}{ok}/{len(files)} ok{unchanged_note} | "
                f"{_fmt_size(total_in)} -> {_fmt_size(total_out)} | {saved} "
                f"{_fmt_size(total_in - total_out)} ({total_in / total_out:.1f}x) ---"
            )
        else:
            # Everything was a cache hit: nothing re-encoded this run.
            click.echo(
                f"\n--- {prefix}{ok}/{len(files)} ok{unchanged_note} ---"
            )

    if report_path:
        out = report.write_report(rows, report_path)
        click.echo(f"Report written -> {out} ({len(rows)} row(s))")

    return 0, {
        "files": len(files),
        "ok": ok,
        "unchanged": unchanged,
        "failed": failed,
        "skipped": skipped,
        "total_in": total_in,
        "total_out": total_out,
        "statuses": {f: results[f]["status"] for f in files},
    }


def _scan_stats(root: Path, exts: set) -> dict:
    """One watch scan: path -> (size, mtime_ns) for the folder's media files.

    Metadata only — no file is opened or hashed, which is what keeps an idle
    watch cycle cheap on a big library. A file vanishing mid-scan (sync-client
    temp churn) is simply dropped from this cycle's snapshot.
    """
    out = {}
    for f in _gather(root, exts):
        try:
            st = f.stat()
        except OSError:
            continue
        out[f] = (st.st_size, st.st_mtime_ns)
    return out


@cli.command()
@click.argument("inbox", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output", "output_path", type=click.Path(file_okay=False),
              required=True,
              help="Archive directory the compressed copies land in (created "
                   "if missing; must differ from INBOX).")
@click.option("--interval", type=float, default=60.0, show_default=True,
              help="Seconds to sleep between scans. An idle scan touches file "
                   "metadata only (no hashing), so a short interval stays cheap "
                   "even on a big library.")
@click.option("--once", is_flag=True,
              help="Run a single pass and exit, so Task Scheduler / cron / "
                   "launchd can own the schedule without a resident process. "
                   "Exit code 1 if any file failed, else 0.")
@click.option("--settle", type=float, default=2.0, show_default=True,
              help="Seconds between the first pass's paired stability scans. A "
                   "file is processed only after its size+mtime hold still "
                   "across two consecutive scans, so a mid-sync/mid-copy file "
                   "is never half-read.")
@click.option("-m", "--mode", type=click.Choice(["faithful", "aggressive"]),
              default=None,
              help="faithful (default) = standard real-pixel .avif/.jxl; "
                   "aggressive = .fkeep (videos are then skipped - video is "
                   "faithful-only).")
@click.option("--preset", type=click.Choice(list(PRESET_NAMES)), default=None,
              help="Aggressive-mode preset (implies -m aggressive), as in "
                   "`compress`.")
@click.option("--codec", type=click.Choice(["avif", "jxl", "webp", "both"]),
              default=None, help="Faithful-mode codec (default: avif).")
@click.option("-q", "--quality", type=int, default=None,
              help="Faithful-mode quality 0-100 (disables auto-tune, as in "
                   "`compress`).")
@click.option("--lossless/--lossy", "lossless", default=None,
              help="Faithful mode: bit-exact archival encode. Use this when "
                   "the archive will be your ONLY copy of irreplaceable "
                   "originals - the default is visually lossless, not "
                   "bit-exact.")
@click.option("--strip-gps/--keep-gps", "strip_gps", default=None,
              help="Strip the GPS EXIF from outputs (as in `compress`).")
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Path to a config YAML file (facekeep.yaml is "
                   "auto-discovered).")
@click.option("-j", "--jobs", type=int, default=1,
              help="Photo worker processes per pass (0 = one per CPU); videos "
                   "always encode serially.")
@click.option("--no-videos", is_flag=True,
              help="Watch photos only (videos in the inbox are ignored).")
@click.option("--no-progress", is_flag=True,
              help="Disable the per-pass progress bar.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def watch(inbox, output_path, interval, once, settle, mode, preset, codec,
          quality, lossless, strip_gps, config_path, jobs, no_videos,
          no_progress, verbose):
    """Keep INBOX compressed into an archive folder - the automation core.

    Point it at the folder your phone photos land in (iCloud for Windows,
    OneDrive/Google Drive camera upload, Syncthing, an import folder) and
    every new photo/video is compressed into the archive automatically:
    scan -> compress new/changed files -> sleep -> repeat. Ctrl-C stops it
    cleanly; --once does a single pass for an external scheduler.

    Sources are NEVER deleted or modified - cleanup of the inbox stays your
    call. Re-runs are cheap: an unchanged file is skipped from its size+mtime
    alone (no re-reading), and a file still being synced is left alone until
    its size and mtime hold still across two scans.

    A Live Photo's paired ~3 s .mov is kept verbatim rather than re-encoded
    (re-encoding would break the pairing; the still is compressed normally).
    """
    _setup_logging(verbose)
    try:
        config = _load_config(config_path, mode, codec, quality, None,
                              strip_gps=strip_gps, lossless=lossless,
                              preset=preset)
    except FaceKeepError as e:
        click.echo(f"Config error: {e}", err=True)
        sys.exit(2)

    in_p = Path(inbox)
    out_p = Path(output_path)
    try:
        same = in_p.resolve() == out_p.resolve()
    except OSError:
        same = False
    if same:
        click.echo(
            "watch: the archive directory must differ from the inbox "
            "(outputs would land beside the sources being watched).",
            err=True,
        )
        sys.exit(2)
    out_p.mkdir(parents=True, exist_ok=True)

    # Decide video inclusion once, up front, so an excluded class of file is
    # never even scanned (idle cycles stay metadata-only over what matters)
    # and the reason is said once instead of per file per cycle.
    include_videos = config.video.enabled and not no_videos
    if include_videos and config.mode == "aggressive":
        include_videos = False
        click.echo(
            "watch: aggressive mode does not apply to video (video is "
            "faithful-only) - videos in the inbox will be skipped.",
            err=True,
        )
    elif include_videos and not video_mod.ffmpeg_available():
        include_videos = False
        click.echo(
            f"watch: videos in the inbox will be skipped - "
            f"{video_mod._MISSING_FFMPEG_HINT}",
            err=True,
        )
    video_exts = video_mod.VIDEO_EXTENSIONS if include_videos else set()
    exts = IMAGE_EXTS | video_exts

    # Guardrail 2 honesty (ROADMAP Phase 11): a backup flow must say what the
    # copy is. Faithful is visually lossless, not bit-exact; aggressive
    # reconstructs backgrounds on restore.
    if config.mode == "aggressive":
        click.echo(
            "note: aggressive mode reconstructs backgrounds on restore "
            "(plausible, not faithful). If the archive will be your only "
            "copy, faithful mode (the default) or --lossless is the honest "
            "choice."
        )
    elif not config.faithful.lossless:
        click.echo(
            "note: faithful compression is visually lossless, not bit-exact. "
            "Use --lossless if the archive will be your ONLY copy of "
            "irreplaceable originals."
        )

    photo_fp = index_mod.settings_fingerprint(config)
    video_fp = index_mod.video_settings_fingerprint(config)
    db_path = out_p / index_mod.INDEX_FILENAME

    click.echo(
        f"Watching {in_p} -> {out_p}"
        + ("" if once else f" every {interval:g}s (Ctrl-C to stop)")
        + "; sources are never deleted or modified."
    )

    prev = None  # previous scan's snapshot (path -> (size, mtime_ns))
    # Files whose last attempt failed/skipped, memoized by the stat they had:
    # they are NOT retried every cycle (a corrupt file must not burn CPU per
    # poll) — only when the file changes. In-memory only: a fresh process (or
    # each --once run) retries once.
    blocked: dict = {}
    exit_code = 0
    try:
        while True:
            stats = _scan_stats(in_p, exts)
            if prev is None:
                # Bootstrap: pair the first scan with a settle-delayed second
                # one, so the first pass can already process files that held
                # still — --once has only one pass, and loop mode shouldn't
                # sit a full interval before doing anything.
                time.sleep(settle)
                prev = stats
                stats = _scan_stats(in_p, exts)
            # Stability guard: a file is eligible only when its size+mtime are
            # identical across two consecutive scans (temp+rename AND in-place
            # sync writers both show up as a changing stat while unfinished).
            eligible = [f for f, s in stats.items() if prev.get(f) == s]
            awaiting = len(stats) - len(eligible)
            prev = stats

            blocked = {f: s for f, s in blocked.items() if f in stats}
            held = [f for f in eligible if blocked.get(f) == stats[f]]
            check = [f for f in eligible if blocked.get(f) != stats[f]]

            # Stat pre-filter (the idle-cycle cost fix): a file whose recorded
            # size+mtime+settings still match is skipped without reading a
            # byte. Only the misses go to the batch (which hashes honestly).
            todo = []
            unchanged = 0
            try:
                with index_mod.ProcessIndex(db_path) as idx:
                    for f in check:
                        size, mtime = stats[f]
                        fp = video_fp if f.suffix.lower() in video_exts else photo_fp
                        if idx.is_unchanged_stat(f, size, mtime, fp) is not None:
                            unchanged += 1
                        else:
                            todo.append(f)
            except sqlite3.Error as e:  # a broken cache must never block a run
                logging.getLogger("facekeep.cli").warning(
                    "watch: unreadable index (%s); checking all files.", e
                )
                todo, unchanged = list(check), 0

            n_ok = n_failed = n_skipped = 0
            saved = 0
            if todo:
                code, summary = _run_batch(
                    str(in_p), str(out_p), config, jobs=jobs,
                    no_videos=not include_videos, no_progress=no_progress,
                    only_files=sorted(todo),
                )
                if code == 0 and summary:
                    n_ok = summary["ok"]
                    n_failed = summary["failed"]
                    n_skipped = summary["skipped"]
                    unchanged += summary["unchanged"]
                    saved = max(0, summary["total_in"] - summary["total_out"])
                    for f, status in summary["statuses"].items():
                        if status in ("failed", "failed-unexpected", "skipped"):
                            blocked[f] = stats.get(f)
                        else:
                            blocked.pop(f, None)

            # The per-cycle summary line (ASCII-only for legacy codepages).
            parts = []
            if todo:
                done = f"{n_ok} ok"
                if n_failed:
                    done += f", {n_failed} failed"
                if n_skipped:
                    done += f", {n_skipped} skipped"
                parts.append(f"processed {len(todo)}: {done}")
                if saved:
                    parts.append(f"saved {_fmt_size(saved)}")
            else:
                parts.append("idle")
            if unchanged:
                parts.append(f"{unchanged} unchanged")
            if awaiting:
                parts.append(f"{awaiting} not yet stable (still syncing?)")
            if held:
                parts.append(
                    f"{len(held)} failed/skipped earlier "
                    "(retried when the file changes)"
                )
            if not once:
                parts.append(f"next scan in {interval:g}s")
            click.echo(f"[watch {time.strftime('%H:%M:%S')}] " + " | ".join(parts))

            if once:
                exit_code = 1 if n_failed else 0
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nwatch: stopped.")
        sys.exit(0)
    sys.exit(exit_code)


_RESTORE_EXT = {"jpg": ".jpg", "avif": ".avif", "jxl": ".jxl"}


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Output path (file or directory).")
@click.option("-f", "--format", "out_format",
              type=click.Choice(["jpg", "avif", "jxl"]), default="jpg",
              help="Output format for restored images (default: jpg). "
                   "An explicit -o file extension overrides this.")
@click.option("-q", "--quality", type=int, default=70,
              help="Quality 0-100 for avif/jxl output (default: 70). JPEG output "
                   "uses OpenCV's default unless this is set.")
@click.option("--preview", is_flag=True,
              help="Fast bicubic preview instead of AI restore.")
@click.option("--tile", type=int, default=None,
              help="Real-ESRGAN tile size in px (0 = no tiling). Smaller bounds "
                   "peak memory on large images. Default from config (512).")
@click.option("--tile-pad", type=int, default=None,
              help="Tile overlap padding to hide seams. Default from config (10).")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("-v", "--verbose", is_flag=True)
def restore(input_path, output_path, out_format, quality, preview, tile, tile_pad,
            config_path, verbose):
    """Restore aggressive-mode .fkeep file(s) to standard full-resolution images.

    The escape hatch from the .fkeep container: every photo comes back as a
    universal .jpg (default) or .avif/.jxl, so a .fkeep is never a dead end.
    Point it at a folder to un-fkeep a whole library at once.
    """
    _setup_logging(verbose)
    config = FaceKeepConfig.load(Path(config_path) if config_path else None)

    # Restore-only tiling overrides (None -> keep the config/default value).
    if tile is not None:
        config.aggressive.tile = tile
    if tile_pad is not None:
        config.aggressive.tile_pad = tile_pad
    config.validate()

    from . import encoders
    from .aggressive.format import read_fkeep_info
    from .aggressive.restorer import Restorer

    # Preset-aware restore: a .fkeep compressed with --preset records the name
    # in its manifest (settings.preset, 1.7.0+); auto-apply that preset's
    # restore-side knobs (face-enhance backend/fidelity/strength) unless the
    # user explicitly set them — explicit YAML/CLI wins, the same precedence
    # rule as compress. Restorers are cached per effective knob-set so a folder
    # of same-preset files still builds its (lazy) AI models once. Absent or
    # unknown preset names mean "no overrides" (tolerant by structure).
    explicit_keys = getattr(config, "explicit_keys", frozenset())
    base_restorer = Restorer(config.aggressive)
    _restorers = {(): base_restorer}

    def _restorer_for(f: Path) -> Restorer:
        try:
            manifest = read_fkeep_info(str(f))
        except FaceKeepError:
            return base_restorer  # unreadable: let restore() report it properly
        name = (manifest.get("settings") or {}).get("preset")
        overrides = preset_restore_overrides(name, explicit_keys)
        key = tuple(sorted(overrides.items()))
        if key not in _restorers:
            agg = copy.deepcopy(config.aggressive)
            for dotted, value in overrides.items():
                # Restore-side preset keys all live under aggressive.*
                setattr(agg, dotted.split(".", 1)[1], value)
            _restorers[key] = Restorer(agg)
        return _restorers[key]

    files = _gather(Path(input_path), {".fkeep"})
    if not files:
        click.echo(f"No .fkeep files found in: {input_path}", err=True)
        sys.exit(1)

    in_p = Path(input_path)
    out_p = Path(output_path) if output_path else (in_p if in_p.is_dir() else in_p.parent)
    explicit_file = (
        output_path is not None and len(files) == 1 and not Path(output_path).is_dir()
    )

    def _target_for(f: Path) -> str:
        # An explicit single-file -o path wins (honor its own extension);
        # otherwise derive "<stem>_restored.<ext>" — stem + literal ext keeps
        # dotted filenames intact (no Path.with_suffix on the raw name).
        if explicit_file:
            return str(out_p)
        return str(out_p / f"{f.stem}_restored.{_RESTORE_EXT[out_format][1:]}")

    targets = {f: _target_for(f) for f in files}

    # Fail fast (before writing anything) if any output needs a codec that isn't
    # installed — graceful degradation with an actionable hint, not a mid-batch
    # crash. The effective format is the resolved target's suffix.
    needed = {Path(t).suffix.lower() for t in targets.values()}
    for suffix, codec in ((".avif", "avif"), (".jxl", "jxl")):
        if suffix in needed and not encoders.codec_available(codec):
            plugin = "avif-plugin" if codec == "avif" else "jxl-plugin"
            click.echo(
                f"Cannot write {codec.upper()}: the {codec} codec is not installed. "
                f"Install it (pip install pillow-{plugin}) or restore to JPEG "
                "(the default, -f jpg).",
                err=True,
            )
            sys.exit(2)

    if output_path and len(files) > 1:
        out_p.mkdir(parents=True, exist_ok=True)

    ok = 0
    for f in files:
        try:
            verb = "preview" if preview else "restore"
            click.echo(f"[{verb}] {f.name} ... ", nl=False)
            target = targets[f]
            if preview:
                # preview skips face enhancement entirely, so the preset's
                # restore-side knobs can't matter — keep it on the base config.
                base_restorer.preview(str(f), target, quality=quality)
            else:
                _restorer_for(f).restore(str(f), target, quality=quality)
            ok += 1
            click.echo(f"OK -> {Path(target).name}")
        except FaceKeepError as e:
            click.echo(f"FAILED: {e}", err=True)

    if len(files) > 1:
        click.echo(f"\n--- {ok}/{len(files)} restored -> {out_p} ---")


@cli.command()
@click.argument("original", type=click.Path(exists=True))
@click.argument("compressed", type=click.Path(exists=True))
@click.option("--lpips", "want_lpips", is_flag=True, default=False,
              help="Also report perceptual LPIPS distance (lower = more similar). "
                   "Needs the [ai] extra; downloads small weights on first use.")
@click.option("--ssimulacra2", "want_s2", is_flag=True, default=False,
              help="Also report the SSIMULACRA2 perceptual score (higher = "
                   "better; ~90 visually lossless). Needs the [dev] extra.")
def quality(original, compressed, want_lpips, want_s2):
    """Report SSIM/PSNR between an ORIGINAL and a COMPRESSED/restored image.

    With --lpips, also reports the learned perceptual distance (LPIPS), which is
    the right acceptance metric for aggressive-mode restores — SSIM penalizes a
    plausibly-reconstructed background that *looks* right but differs pixel-wise.

    With --ssimulacra2, also reports the SSIMULACRA2 perceptual quality score
    (higher = better), a far better "visually lossless" indicator than SSIM.
    """
    import cv2

    from . import metrics

    a = cv2.imread(original, cv2.IMREAD_COLOR)
    b = cv2.imread(compressed, cv2.IMREAD_COLOR)
    if a is None or b is None:
        click.echo("Could not read one of the images.", err=True)
        sys.exit(1)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    # Decide before computing so an uninstalled extra is a clear hint, not a
    # silent blank — and so we don't trigger a weight download we won't use.
    do_lpips = want_lpips and metrics.lpips_available()
    if want_lpips and not do_lpips:
        click.echo(
            "LPIPS unavailable (the 'lpips' package is not installed). "
            "Install with: pip install facekeep[ai]",
            err=True,
        )

    report = metrics.compare(a, b, with_lpips=do_lpips)
    click.echo(f"Overall SSIM: {report.overall_ssim:.4f}")
    click.echo(f"Overall PSNR: {report.overall_psnr:.2f} dB")
    if report.lpips is not None:
        click.echo(f"LPIPS:        {report.lpips:.4f} (lower = more perceptually similar)")

    if want_s2:
        if metrics.ssimulacra2_available():
            score = metrics.ssimulacra2_score(a, b)
            if score is not None:
                click.echo(
                    f"SSIMULACRA2:  {score:.2f} (higher = better; "
                    "~90 visually lossless)"
                )
        else:
            click.echo(
                "SSIMULACRA2 unavailable (the 'ssimulacra2' package is not "
                "installed). Install with: pip install facekeep[dev]",
                err=True,
            )


@cli.command()
@click.argument("original", type=click.Path(exists=True))
@click.argument("compressed", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="HTML report path (default: <compressed>_compare.html next to "
                   "the compressed file).")
@click.option("--preview", is_flag=True,
              help="For a .fkeep, use the fast bicubic preview instead of a full "
                   "AI restore for the 'after' image.")
@click.option("--amplify", type=float, default=8.0, show_default=True,
              help="Difference-map amplification factor (visual only; metrics are "
                   "unaffected).")
@click.option("--full-res", is_flag=True,
              help="Embed full-resolution preview images (default: downscale the "
                   "embedded previews to keep the HTML small; metrics always use "
                   "full resolution either way).")
@click.option("--lpips", "want_lpips", is_flag=True, default=False,
              help="Also report perceptual LPIPS distance (lower = more similar). "
                   "Needs the [ai] extra; downloads small weights on first use.")
@click.option("--ssimulacra2", "want_s2", is_flag=True, default=False,
              help="Also report the SSIMULACRA2 perceptual score (higher = "
                   "better; ~90 visually lossless). Needs the [dev] extra.")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("-v", "--verbose", is_flag=True)
def compare(original, compressed, output_path, preview, amplify, full_res,
            want_lpips, want_s2, config_path, verbose):
    """Write an HTML before/after report comparing ORIGINAL with COMPRESSED.

    COMPRESSED may be a faithful .avif/.jxl/.webp, an aggressive .fkeep (restored
    on the fly), or an already-restored standard image. The self-contained HTML
    has a before/after slider, a difference heatmap, and SSIM/PSNR (plus optional
    LPIPS / SSIMULACRA2). It reads existing outputs only — it changes no pixels.
    """
    _setup_logging(verbose)
    from . import compare as compare_mod
    from . import metrics

    config = FaceKeepConfig.load(Path(config_path) if config_path else None)

    # Decide opt-in metrics before computing so an uninstalled extra is a clear
    # hint, not a silent blank — mirrors the `quality` command exactly.
    do_lpips = want_lpips and metrics.lpips_available()
    if want_lpips and not do_lpips:
        click.echo(
            "LPIPS unavailable (the 'lpips' package is not installed). "
            "Install with: pip install facekeep[ai]",
            err=True,
        )
    do_s2 = want_s2 and metrics.ssimulacra2_available()
    if want_s2 and not do_s2:
        click.echo(
            "SSIMULACRA2 unavailable (the 'ssimulacra2' package is not "
            "installed). Install with: pip install facekeep[dev]",
            err=True,
        )

    try:
        summary = compare_mod.render_comparison(
            original, compressed,
            Path(output_path) if output_path else None,
            agg_config=config.aggressive, preview=preview, amplify=amplify,
            want_lpips=do_lpips, want_ssimulacra2=do_s2, full_res=full_res,
            explicit_keys=getattr(config, "explicit_keys", frozenset()),
        )
    except FaceKeepError as e:
        click.echo(f"Compare failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Wrote {summary['output_path']}")
    click.echo(
        f"  {_fmt_size(summary['original_bytes'])} -> "
        f"{_fmt_size(summary['compressed_bytes'])} ({summary['ratio']:.2f}x, "
        f"{summary['after_kind']})"
    )
    click.echo(
        f"  SSIM {summary['ssim']:.4f}  PSNR {summary['psnr']:.2f} dB"
    )
    if summary.get("lpips") is not None:
        click.echo(f"  LPIPS {summary['lpips']:.4f} (lower = more similar)")
    if summary.get("ssimulacra2") is not None:
        click.echo(
            f"  SSIMULACRA2 {summary['ssimulacra2']:.2f} (higher = better)"
        )


@cli.command()
@click.argument("fkeep_path", type=click.Path(exists=True))
def info(fkeep_path):
    """Show metadata for an aggressive-mode .fkeep file."""
    from .aggressive.format import read_fkeep_info

    try:
        m = read_fkeep_info(fkeep_path)
    except FaceKeepError as e:
        click.echo(f"{e}", err=True)
        sys.exit(1)

    o = m["original"]
    click.echo(f"File:        {o['filename']}")
    click.echo(f"Original:    {o['width']}x{o['height']} ({_fmt_size(o['size_bytes'])})")
    click.echo(f"Orientation: {o.get('orientation', 1)}")
    click.echo(f"EXIF kept:   {m.get('exif_preserved', False)}")
    click.echo(f"Faces:       {len(m['faces'])}")
    regions = m.get("regions", []) or []
    if regions:
        click.echo(f"Regions:     {len(regions)} (region-local conservatism)")
    s = m["settings"]
    if s.get("preset"):
        click.echo(f"Preset:      {s['preset']}")
    click.echo(f"BG scale:    {s['bg_scale']}")
    if s.get("residual"):
        click.echo(
            f"Residual:    yes (scale {s.get('residual_scale', '?')} - "
            "background restores from real data, no AI hallucination)"
        )
    if m.get("gain_map_preserved"):
        click.echo(
            "HDR gainmap: yes (restore to .avif re-attaches it -> HDR output)"
        )
        if m.get("gain_map_params"):
            # Source-declared hdrgm application math (manifest 1.11.0+, an
            # Android Ultra HDR source) — restore re-emits it verbatim.
            gm_max = m["gain_map_params"].get("gain_map_max")
            click.echo(
                f"  hdrgm:     source parameters recorded (GainMapMax {gm_max})"
                " - re-emitted on restore"
            )
    click.echo(f"Created:     {m.get('created_at', 'N/A')}")
    click.echo(f"FaceKeep:    v{m.get('facekeep_version', 'N/A')}")


@cli.command()
@click.argument("output_path", type=click.Path(), default="facekeep.yaml")
@click.option("-f", "--force", is_flag=True,
              help="Overwrite an existing config file.")
def init(output_path, force):
    """Write a commented starter config (default: ./facekeep.yaml).

    Every key is optional and shown at its default, so the file documents the
    knobs and is safe to trim. `facekeep compress` auto-discovers `facekeep.yaml`
    in the working directory, so the written file takes effect with no extra flag.
    """
    from .config import default_config_yaml

    out = Path(output_path)
    if out.exists() and not force:
        click.echo(f"{out} already exists (use --force to overwrite).", err=True)
        sys.exit(1)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(default_config_yaml(), encoding="utf-8")
    except OSError as e:
        click.echo(f"Could not write {out}: {e}", err=True)
        sys.exit(1)
    click.echo(f"Wrote {out}")


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Interface to bind (127.0.0.1 = this machine only).")
@click.option("--port", default=7860, show_default=True, type=int,
              help="Port to serve on (gradio picks the next free one if taken).")
@click.option("--share", is_flag=True,
              help="Create a public Gradio share link. Off by default — a photo "
                   "tool should not open a public tunnel unless you ask.")
@click.option("--inbrowser", is_flag=True,
              help="Open the GUI in your default browser on launch.")
@click.option("-v", "--verbose", is_flag=True)
def gui(host, port, share, inbrowser, verbose):
    """Launch the local drag-and-drop web GUI (needs the [gui] extra).

    A thin web front end over the same pipeline as `compress`: drop a photo,
    pick a mode, see a before/after, and download the result. Runs locally with
    sharing and telemetry off by default. Install with: pip install facekeep[gui]
    """
    _setup_logging(verbose)
    from . import gui as gui_mod  # module imports fine without gradio (lazy)

    try:
        gui_mod.launch(host=host, port=port, share=share, inbrowser=inbrowser)
    except ImportError:
        # gradio is imported lazily inside launch(); a missing [gui] extra
        # surfaces here as an actionable hint, not a traceback (same graceful
        # degradation as a missing codec in `restore`).
        click.echo(
            "The GUI needs Gradio, which isn't installed.\n"
            "  Install it with:  pip install facekeep[gui]",
            err=True,
        )
        sys.exit(2)


@cli.command()
@click.argument("fkeep_path", type=click.Path(exists=True))
@click.option("--original", "original_path", type=click.Path(exists=True), default=None,
              help="Also match the manifest's stored SHA-256 against this original "
                   "file (the .fkeep alone cannot self-verify the hash — the "
                   "original pixels aren't stored).")
def verify(fkeep_path, original_path):
    """Structurally verify an aggressive-mode .fkeep file.

    Checks the container is complete and self-consistent: the ZIP opens, the
    manifest parses, every promised entry (background, each face crop + mask,
    thumbnail) is present and decodable, the counts line up, and the dimensions
    are sane. Exits non-zero if anything is wrong (or if --original doesn't match).
    """
    from .aggressive.format import FKEEP_EXTENSION, verify_fkeep

    if Path(fkeep_path).suffix.lower() != FKEEP_EXTENSION:
        click.echo(
            f"{fkeep_path}: not a .fkeep file. Faithful-mode outputs are standard "
            "images — open them to verify, or use `facekeep quality`.",
            err=True,
        )
        sys.exit(2)

    try:
        rep = verify_fkeep(fkeep_path, original_path)
    except FaceKeepError as e:
        click.echo(f"{Path(fkeep_path).name}: FAILED ({e})", err=True)
        sys.exit(1)

    status = "OK" if rep.ok else "FAILED"
    click.echo(f"{Path(fkeep_path).name}: {status}")

    # Status marks degrade to ASCII on consoles whose codepage can't encode
    # the glyphs (cp950 etc.) — see _marks(); a crash here would mask a
    # successful verification.
    good, bad = _marks()
    crops_mark = good if rep.crops_found == rep.faces_declared else bad
    masks_mark = good if rep.masks_found == rep.faces_declared else bad
    click.echo(
        f"  faces:        {rep.crops_found} crop(s), {rep.masks_found} mask(s) "
        f"(declared {rep.faces_declared}) {crops_mark}{masks_mark}"
    )
    if rep.regions_declared:
        rc_mark = good if rep.region_crops_found == rep.regions_declared else bad
        rm_mark = good if rep.region_masks_found == rep.regions_declared else bad
        click.echo(
            f"  regions:      {rep.region_crops_found} crop(s), "
            f"{rep.region_masks_found} mask(s) "
            f"(declared {rep.regions_declared}) {rc_mark}{rm_mark}"
        )
    if rep.background_size:
        bw, bh = rep.background_size
        click.echo(f"  background:   {bw}x{bh}  {good}")
    else:
        click.echo(f"  background:   missing/undecodable  {bad}")
    if rep.residual_declared:
        res_mark = f"present  {good}" if rep.residual_ok else f"missing/undecodable  {bad}"
        click.echo(f"  residual:     {res_mark}")
    if rep.gain_map_declared:
        gm_mark = f"present  {good}" if rep.gain_map_ok else f"missing/undecodable  {bad}"
        click.echo(f"  gain map:     {gm_mark}")
    click.echo(
        f"  thumbnail:    {f'present  {good}' if rep.thumbnail_ok else f'missing  {bad}'}"
    )

    if rep.stored_hash:
        if rep.hash_match is None:
            click.echo(
                f"  stored hash:  {rep.stored_hash[:12]}… (original; "
                "pass --original to match)"
            )
        else:
            verdict = "MATCHES" if rep.hash_match else "MISMATCH"
            click.echo(f"  original hash: {verdict}")

    for p in rep.problems:
        click.echo(f"  - {p}", err=True)

    sys.exit(0 if rep.ok else 1)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-m", "--mode",
              type=click.Choice(["faithful", "aggressive", "both"]),
              default="both", show_default=True,
              help="Which mode(s) to benchmark.")
@click.option("--ai-restore", is_flag=True,
              help="Score aggressive restore against the real Real-ESRGAN restore "
                   "(needs [ai]; slow). Default is a fast bicubic proxy.")
@click.option("--save-baseline", "save_baseline_path", type=click.Path(), default=None,
              help="Write the results to a baseline JSON for later comparison.")
@click.option("--baseline", "baseline_path", type=click.Path(exists=True), default=None,
              help="Compare against a saved baseline; the table shows per-column "
                   "deltas (and NEW/GONE rows).")
@click.option("--report", "report_path", type=click.Path(), default=None,
              help="Also write a CSV ledger of the rows.")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("-v", "--verbose", is_flag=True)
def bench(input_path, mode, ai_restore, save_baseline_path, baseline_path,
          report_path, config_path, verbose):
    """Benchmark compression across photos and print a regression table.

    Runs the real compress (and, for aggressive, a restore) on each photo and
    prints faithful ratio + decoded SSIM and aggressive .fkeep ratio + restore
    LPIPS (perceptual). With --baseline it diffs against a saved run so a moved
    real-photo number is visible. This is a measurement artifact, not a gate —
    the pytest regression locks fail a build.
    """
    from . import bench as bench_mod

    _setup_logging(verbose)

    files = _gather(Path(input_path), IMAGE_EXTS)
    if not files:
        click.echo(f"No images found in {input_path}", err=True)
        sys.exit(2)

    config = FaceKeepConfig.load(Path(config_path) if config_path else None)
    modes = ["faithful", "aggressive"] if mode == "both" else [mode]

    rows = bench_mod.run_benchmark(files, modes, config, ai_restore=ai_restore)

    baseline = bench_mod.load_baseline(baseline_path) if baseline_path else None
    click.echo(bench_mod.format_table(rows, baseline))

    if save_baseline_path:
        out = bench_mod.save_baseline(rows, save_baseline_path)
        click.echo(f"\nBaseline saved -> {out}")

    if report_path:
        report.write_report(
            [
                report.ReportRow(
                    file=r.file,
                    mode=r.mode,
                    status=r.status,
                    original_bytes=r.original_bytes,
                    output_bytes=r.output_bytes,
                    ratio=r.ratio,
                    faces=r.faces,
                    ssim_downscaled=r.ssim,
                )
                for r in rows
            ],
            report_path,
        )
        click.echo(f"Report written -> {report_path}")

    failed = sum(1 for r in rows if r.status == "failed")
    sys.exit(1 if failed else 0)


def main():
    cli()


if __name__ == "__main__":
    main()
