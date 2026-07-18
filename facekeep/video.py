"""Faithful video re-encode: probe -> skip-if-efficient -> SVT-AV1 -> skip-if-larger.

Phone cameras record with a *real-time hardware* encoder — it must finish each
frame in 1/30 s, so it buys quality with bitrate (measured 25-31 Mbps for 4K30
HEVC on real phones). A slow offline SVT-AV1 CRF re-encode spends the time the
phone couldn't and lands 3-14x smaller at visually-lossless VMAF (ROADMAP 10.0,
measured on the user's own clips). Same bargain as faithful photos: real pixels,
the codec's own psychovisual bit allocation, and a standard ``.mp4`` output that
plays anywhere modern — opening the file *is* the restore. Aggressive mode
deliberately does NOT port to video (per-frame AI SR is computationally absurd
and temporally unstable); faithful-only is the honest scope.

This module is the core library path (ROADMAP 10.1) plus the VMAF quality
verification layer (10.2): every written encode is scored with libvmaf by
default (order-paired frames, ``vmaf_4k`` at native resolution for 4K-class
sources) and re-encoded a CRF step lower on a miss, and an opt-in sampled CRF
auto-tune (``auto_tune=True``) searches the highest CRF that still meets the
target. ``facekeep compress`` wires it in (10.3, ``cli.py``): videos are
gathered alongside photos, configured by the ``video:`` config section, cached
in the incremental index under their own fingerprint, and always encoded
serially in the parent process (SVT-AV1 saturates the cores — ``--jobs``
applies to photos only).

ROADMAP 10.5 adds two faithful refinements. **Dolby Vision RPU carry**
(``preserve_dolby_vision``, default on): a phone DV source (both real phones
record DV profile 8.4 — an HLG base plus a per-frame tone-mapping RPU) keeps
its RPU through the re-encode as AV1 T.35 metadata OBUs (DV profile 10), so a
DV-aware display renders the output through the same Dolby pipeline as the
original — the 10.4 device finding ("the original looks more saturated") was
exactly this refinement layer missing. No ``dovi_tool`` needed: this ffmpeg's
libsvtav1 wrapper codes the RPU itself (``-dolbyvision``; the mp4 DV-AV1 box
needs ``-strict unofficial``), verified per-frame value-identical on both real
clips. A build without the option keeps today's HLG-base output, warned.
**Face-aware quality** (``face_vmaf_target``, default 95): the photo
chroma/auto-tune analog — the shared face detector runs on a few sampled
frames, and when faces are present the VMAF p1 target rises so the gate (and
the auto-tune search) hold the clip to a higher floor; family clips are what
the tool exists for. A missed face just keeps today's target; a false positive
only costs bytes — the same benign failure modes as the photo guardrails.

``ffmpeg``/``ffprobe`` are **opt-in, machine-local external binaries** — the
avifenc pattern: ``$FACEKEEP_FFMPEG`` -> PATH -> ``None``, never a Python
dependency, fully offline. Photos are unaffected when they are missing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .exceptions import VideoError

logger = logging.getLogger("facekeep.video")

# Container/stream extensions we treat as video input. Phone rolls are mp4/mov
# (+ 3gp on old Androids); the rest are common camera/desktop containers.
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".3gp", ".3g2", ".avi", ".mkv", ".webm",
    ".mts", ".m2ts", ".ts", ".wmv", ".mpg", ".mpeg",
}

# Fixed-CRF defaults (ROADMAP decision 3; auto-tune arrives in 10.2). CRF 32 is
# the conservative end of the measured 32-35 band: on both real phone clips CRF
# 35 already scored VMAF >=99.5 mean / >=96.7 min-frame, so 32 leaves margin for
# content the spike didn't cover (low light, grain, fast motion). Preset 6 is
# the measured speed/quality point (~0.25x realtime for 4K on the dev CPU).
DEFAULT_CRF = 32
DEFAULT_PRESET = 6

# Skip-if-efficient threshold, in encoded bits per pixel per frame. Measured
# anchors (ROADMAP 10.0): phone hardware HEVC ~0.10-0.13 bpp; our SVT-AV1
# CRF 32-40 outputs ~0.01-0.04 bpp. A source at/below 0.05 is already in
# "someone re-encoded this" territory — re-encoding it burns hours and adds a
# lossy generation for little size win.
_EFFICIENT_BPP = 0.05

# Codecs that are already modern+efficient: re-encoding adds a lossy generation
# for nothing regardless of bitrate. (Efficient HEVC/VP9 files are caught by
# the bpp threshold instead — a *phone* HEVC is exactly what we want to shrink.)
_EFFICIENT_CODECS = {"av1"}

# --- VMAF quality verification / auto-tune (ROADMAP 10.2) --------------------
#
# Default target, in VMAF points of the per-frame **1%-low (p1)** — NOT the
# mean. Measured basis (2026-07-16, the two real phone 4K clips, vmaf_4k at
# native resolution): the pooled mean saturates at 98.6-99.8 across CRF 30-40
# (it averages away localized damage — blind as a gate and useless as a tune
# discriminator), while p1 moves ~1.5-2.5 points per 5 CRF, monotonically, and
# is robust to a single odd frame. p1 also states the faithful promise
# directly: the *worst* moments still look good, not the average. 93 sits
# comfortably below the CRF-32 default's measured p1 (~94.6 iPhone / ~95.4
# Android — the gate stays quiet on content the 10.0 spike eyeballed as
# visually lossless) while catching a real miss (CRF 40 landed 90.7 / 92.7);
# the gate exists for the content the fixed default was never validated on
# (low light, heavy grain, fast motion).
DEFAULT_VMAF_TARGET = 93.0

# Face-aware quality (ROADMAP 10.5): the VMAF p1 target used when faces are
# detected on sampled frames (the photo chroma/auto-tune analog — faces are
# the subject, so the worst moments must clear a higher bar). Measured basis:
# at the default CRF 32 the two real phone clips landed p1 94.5 (iPhone, a
# person on camera) and 95.4 (Android, a large face) — 95 gives the borderline
# face clip exactly one gate step of extra quality (CRF 28) while a clip that
# already clears it stays single-pass. Detection failure or zero faces keeps
# the base target (never worse than today); a Haar false positive only raises
# quality/size, never corrupts — the same benign failure mode as the photo
# guardrails.
DEFAULT_FACE_VMAF_TARGET = 95.0

# libvmaf built-in models (compiled into the library; no file, no download).
_VMAF_MODEL_DEFAULT = "vmaf_v0.6.1"  # designed for 1080p viewing height
_VMAF_MODEL_4K = "vmaf_4k_v0.6.1"  # designed for 4K viewing height
# A source whose smaller dimension is at/above this scores with the 4K model
# at native resolution. Measured on the real 4K clips: the default model at
# the spike's downscale-to-1080p scored 99.5+ with a 0.02-0.12 CRF-30-vs-40
# spread (saturated), while vmaf_4k at native discriminates (see the p1 note
# above) — the 10.0 "revisit vmaf_4k" caveat, resolved.
_VMAF_4K_MIN_DIM = 1800
# Default-model sources larger than 1080p are downscaled to 1080p for scoring
# (the model's design viewing height — the 10.0 spike convention).
_VMAF_SCALE_MIN_DIM = 1080

# Post-encode gate retry policy: a miss re-encodes _GATE_CRF_STEP lower
# (measured: ~1.5-2.5 p1 points per 4-5 CRF), at most _GATE_MAX_RETRIES times
# and never below _GATE_MIN_CRF; a still-missing floor keeps the best effort
# with a warning — honest, and never an unbounded loop. The gate scores the
# WHOLE file, deliberately un-subsampled: measured, n_subsample=2/4 inflated
# this clip's p1 from 93.25 to 95.16 (the worst frames fall between samples —
# exactly what a worst-frames gate exists to see) and only halved the cost.
_GATE_CRF_STEP = 4
_GATE_MAX_RETRIES = 2
_GATE_MIN_CRF = 18

# Sampled CRF auto-tune (opt-in): binary-search this inclusive CRF range for
# the highest CRF whose sampled p1 still meets the target. Probes encode
# _SAMPLE_SPAN_S-second spans centered at _SAMPLE_POSITIONS of the duration
# (motion/light change over a clip — one sample lies); clips too short to be
# worth sampling are probed whole.
_TUNE_CRF_RANGE = (20, 55)
_SAMPLE_SPAN_S = 3.0
_SAMPLE_POSITIONS = (0.15, 0.5, 0.85)

_MISSING_FFMPEG_HINT = (
    "ffmpeg not found (set FACEKEEP_FFMPEG to the binary or put ffmpeg on "
    "PATH); video compression needs it — a build with libsvtav1, e.g. from "
    "https://ffmpeg.org/download.html. Photos are unaffected."
)


def find_ffmpeg() -> Optional[str]:
    """Locate the external ``ffmpeg`` binary, or ``None`` if unavailable.

    Video re-encoding needs an ffmpeg build with libsvtav1. Like ``avifenc`` it
    is an **opt-in, machine-local external binary** — never a Python dependency,
    and the photo pipelines never touch it. Resolution order:

    1. ``$FACEKEEP_FFMPEG`` (explicit override — an exact path to the binary),
    2. ``ffmpeg`` on the system ``PATH`` (``shutil.which``),
    3. ``None`` -> callers surface a clear install hint (photos unaffected).
    """
    env = os.environ.get("FACEKEEP_FFMPEG")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        logger.warning("FACEKEEP_FFMPEG=%s is not a file; falling back to PATH.", env)
    return shutil.which("ffmpeg")


def _find_ffprobe() -> Optional[str]:
    """Locate ``ffprobe``: sibling of the located ffmpeg first, then PATH.

    ``ffprobe`` ships beside ``ffmpeg`` in every release layout, so a single
    ``$FACEKEEP_FFMPEG`` enables both (the avifdec-beside-avifenc pattern).
    """
    ff = find_ffmpeg()
    if ff:
        sibling = Path(ff).with_name(
            "ffprobe.exe" if ff.lower().endswith(".exe") else "ffprobe"
        )
        if sibling.is_file():
            return str(sibling)
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    """Return True if both ``ffmpeg`` and ``ffprobe`` can be located."""
    return find_ffmpeg() is not None and _find_ffprobe() is not None


_libvmaf_cache: Dict[str, bool] = {}


def _ffmpeg_has_libvmaf(ffmpeg: str) -> bool:
    """True if this ffmpeg build compiled the libvmaf filter in (cached)."""
    cached = _libvmaf_cache.get(ffmpeg)
    if cached is None:
        try:
            proc = subprocess.run([ffmpeg, "-hide_banner", "-filters"],
                                  capture_output=True, text=True)
            cached = proc.returncode == 0 and " libvmaf " in proc.stdout
        except OSError:
            cached = False
        _libvmaf_cache[ffmpeg] = cached
    return cached


def vmaf_available() -> bool:
    """True when ffmpeg/ffprobe are locatable AND this ffmpeg has libvmaf.

    Not every ffmpeg build compiles libvmaf in (GPL builds from BtbN /
    ffmpeg.org do). Encoding works without it — a libvmaf-less build only
    loses the quality gate / auto-tune (skipped with a warning, never a
    crash), the usual graceful-degradation chain.
    """
    ff = find_ffmpeg()
    return ff is not None and _find_ffprobe() is not None and _ffmpeg_has_libvmaf(ff)


_dovi_encode_cache: Dict[str, bool] = {}


def _ffmpeg_supports_dovi_encode(ffmpeg: str) -> bool:
    """True if this build's libsvtav1 wrapper can code Dolby Vision RPUs.

    Detected off ``-h encoder=libsvtav1`` listing a ``dolbyvision`` option
    (present in 2025+ ffmpeg with a matching SVT-AV1 — the wrapper converts
    the decoder-exported per-frame RPU into AV1 T.35 metadata OBUs, DV
    profile 10). An older build simply lacks the option; the encode then
    proceeds without DV, warned — the HLG/HDR10 base layer is unaffected.
    Cached per binary path, like :func:`_ffmpeg_has_libvmaf`.
    """
    cached = _dovi_encode_cache.get(ffmpeg)
    if cached is None:
        try:
            proc = subprocess.run([ffmpeg, "-hide_banner", "-h",
                                   "encoder=libsvtav1"],
                                  capture_output=True, text=True)
            cached = proc.returncode == 0 and "dolbyvision" in proc.stdout
        except OSError:
            cached = False
        _dovi_encode_cache[ffmpeg] = cached
    return cached


def is_video_file(path: Union[str, Path]) -> bool:
    """Return True if the path's extension is a recognized video container."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


@dataclass
class VideoInfo:
    """Probed facts about a video's primary streams (via ffprobe JSON)."""

    path: Path
    size_bytes: int
    duration_s: float
    width: int
    height: int
    fps: float  # average frame rate — the honest number on a VFR source
    v_codec: str
    pix_fmt: str
    bit_depth: int
    v_bit_rate: int  # bps; the video stream's if declared, else the container's
    color_primaries: Optional[str]
    color_transfer: Optional[str]
    color_space: Optional[str]
    rotation: int  # display-matrix rotation in degrees (0 when none)
    a_codec: Optional[str]
    # Dolby Vision: the source's DV profile (e.g. 8 for phone 8.4 HLG-base DV)
    # when a DOVI configuration record with an RPU is present, else None.
    dovi_profile: Optional[int] = None
    # Apple Live-Photo pairing key (11.1): the container-level
    # com.apple.quicktime.content.identifier tag when present, else None. A
    # Live Photo's ~3 s .mov carries it (matching the still's EXIF MakerNote
    # asset identifier); ordinary videos don't — the CLI uses it to confirm a
    # same-stem photo sibling really is a Live-Photo pair.
    content_identifier: Optional[str] = None

    @property
    def bits_per_pixel_frame(self) -> float:
        """Encoded bits spent per pixel per frame — the efficiency yardstick."""
        pixel_rate = self.width * self.height * self.fps
        return self.v_bit_rate / pixel_rate if pixel_rate else 0.0


@dataclass
class VmafScore:
    """Pooled per-frame VMAF of one distorted-vs-reference comparison."""

    mean: float
    p1: float  # 1%-low: the score the worst 1% of frames still reach
    min: float
    frames: int
    model: str  # the libvmaf model that scored this (default vs vmaf_4k)


@dataclass
class VideoResult:
    """Result of a faithful video compression."""

    input_path: Path
    output_path: Optional[Path]  # None when skipped (nothing was written)
    original_size: int
    compressed_size: int  # 0 when skipped
    skipped: bool = False
    skip_reason: Optional[str] = None
    # Wall time of encode + quality verification (per-file ETA in 10.3).
    encode_seconds: float = 0.0
    crf_used: Optional[int] = None  # final encode's CRF (None when no encode ran)
    vmaf: Optional[VmafScore] = None  # final gate score (None when gate off/unavailable)
    # 10.5: True when the source's Dolby Vision RPU was carried into the output.
    dolby_vision: bool = False
    # 10.5: max faces the sampled-frame detection saw (None = it never ran).
    faces: Optional[int] = None

    @property
    def ratio(self) -> float:
        return self.original_size / self.compressed_size if self.compressed_size else 0.0


def _parse_fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = (stream.get(key) or "").partition("/")
        try:
            n, d = float(num), float(den or 1)
        except ValueError:
            continue
        if n > 0 and d > 0:
            return n / d
    return 0.0


def probe_video(path: Union[str, Path]) -> VideoInfo:
    """Probe a video with ffprobe and return the facts the pipeline needs.

    Raises :class:`VideoError` when ffprobe is unavailable, the file cannot be
    probed, or it has no video stream.
    """
    src = Path(path)
    ffprobe = _find_ffprobe()
    if ffprobe is None:
        raise VideoError(_MISSING_FFMPEG_HINT)
    cmd = [ffprobe, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"ffprobe failed on {src.name}: {proc.stderr.strip()[-500:]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise VideoError(f"ffprobe returned unparseable JSON for {src.name}") from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoError(f"{src.name} has no video stream")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    pix_fmt = video.get("pix_fmt") or ""
    try:
        bit_depth = int(video.get("bits_per_raw_sample") or 0)
    except (TypeError, ValueError):
        bit_depth = 0
    if not bit_depth:
        bit_depth = 12 if "12" in pix_fmt else 10 if "10" in pix_fmt else 8

    def _int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    # Live-Photo pairing key: a QuickTime mdta container tag (measured live:
    # ffprobe reports it under format.tags with its dotted name as-is).
    content_identifier = None
    for key, value in (fmt.get("tags") or {}).items():
        if key.lower() == "com.apple.quicktime.content.identifier":
            content_identifier = str(value)
            break

    rotation = 0
    dovi_profile = None
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"]))
            except (TypeError, ValueError):
                pass
        # A DOVI configuration record identifies a Dolby Vision source; only an
        # RPU-bearing one has per-frame metadata worth carrying (10.5).
        if "dv_profile" in sd and _int(sd.get("rpu_present_flag")):
            dovi_profile = _int(sd.get("dv_profile"))

    return VideoInfo(
        path=src,
        size_bytes=_int(fmt.get("size")) or (src.stat().st_size if src.exists() else 0),
        duration_s=float(fmt.get("duration") or 0.0),
        width=_int(video.get("width")),
        height=_int(video.get("height")),
        fps=_parse_fps(video),
        v_codec=video.get("codec_name") or "",
        pix_fmt=pix_fmt,
        bit_depth=bit_depth,
        v_bit_rate=_int(video.get("bit_rate")) or _int(fmt.get("bit_rate")),
        color_primaries=video.get("color_primaries"),
        color_transfer=video.get("color_transfer"),
        color_space=video.get("color_space"),
        rotation=rotation,
        a_codec=audio.get("codec_name") if audio else None,
        dovi_profile=dovi_profile,
        content_identifier=content_identifier,
    )


def _efficiency_skip_reason(info: VideoInfo) -> Optional[str]:
    """Return a skip reason when the source is already efficiently encoded.

    ROADMAP decision 4: re-encoding an already-efficient file wastes hours and
    adds a lossy generation for nothing. Two signals: the codec is already a
    modern efficient one (AV1), or the encoded bits-per-pixel-per-frame is at
    or below what our own output band looks like.
    """
    if info.v_codec.lower() in _EFFICIENT_CODECS:
        # Runtime strings stay ASCII: a cp950 console mangles em-dashes.
        return (
            f"already {info.v_codec.upper()} - re-encoding would add a lossy "
            "generation for little gain"
        )
    bpp = info.bits_per_pixel_frame
    if 0 < bpp <= _EFFICIENT_BPP:
        return (
            f"already efficient ({bpp:.3f} bits/pixel/frame <= {_EFFICIENT_BPP}) - "
            "re-encoding would add a lossy generation for little gain"
        )
    return None


def _with_video_extension(path: Path, ext: str) -> Path:
    """Return ``path`` ending with ``ext`` without mangling dotted filenames.

    The video counterpart of ``encoders._with_extension``: strips only a *known*
    video extension, so ``2024.05.20_trip.mov`` -> ``2024.05.20_trip.mp4`` but a
    dot that isn't a known extension is kept.
    """
    if path.suffix.lower() == ext:
        return path
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return path.with_suffix(ext)
    return path.parent / (path.name + ext)


def default_output_path(input_path: Union[str, Path]) -> Path:
    """Default output for a video input: same folder, ``.mp4`` extension.

    An input that is *already* ``.mp4`` would collide with itself (unlike
    photos, where jpg->avif never can), so the AV1 output gets an ``_av1``
    stem suffix instead of overwriting the source.
    """
    src = Path(input_path)
    return output_path_for(src, src.parent)


def output_path_for(input_path: Union[str, Path], out_dir: Union[str, Path]) -> Path:
    """The ``.mp4`` output path for a video landing in ``out_dir``.

    The folder-run form of :func:`default_output_path` (which is this with
    ``out_dir = src.parent``): the name keeps dotted filenames intact (only a
    *known* video extension is swapped for ``.mp4``), and an output that would
    collide with the source itself (an in-place ``.mp4`` input) gets an
    ``_av1`` stem suffix instead of overwriting it.
    """
    src = Path(input_path)
    out = Path(out_dir) / _with_video_extension(Path(src.name), ".mp4").name
    if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(src)):
        out = out.with_name(out.stem + "_av1.mp4")
    return out


def _vmaf_model_for(width: int, height: int) -> str:
    """Pick the libvmaf model for a source size (min-dim: rotation-invariant)."""
    if min(width, height) >= _VMAF_4K_MIN_DIM:
        return _VMAF_MODEL_4K
    return _VMAF_MODEL_DEFAULT


def _vmaf_frame_scores(ffmpeg: str, distorted: Path, reference: Path, *,
                       model: str, scale_to_1080: bool,
                       ref_start: Optional[float] = None,
                       ref_duration: Optional[float] = None) -> List[float]:
    """Run libvmaf and return the per-frame VMAF scores.

    Frames are paired by ORDER, not wall-clock timestamp (``settb`` +
    ``setpts=N/(30*TB)`` on both branches): libvmaf's framesync pairs on
    timestamps, and a VFR reference vs a re-encode misaligns catastrophically
    (spike-measured: a flat false 19.6 on a visually perfect encode). N-based
    setpts gives both branches identical synthetic timestamps, so frame i
    compares against frame i. ``ref_start``/``ref_duration`` trim the
    reference input for sample scoring (the distorted sample already is the
    trimmed span); ``eof_action=endall`` stops at the shorter branch so a
    boundary off-by-one frame never scores against a repeated frame.
    """
    scale = "scale=-2:1080:flags=bicubic," if scale_to_1080 else ""
    opts = (f"model=version={model}:n_threads={os.cpu_count() or 1}:"
            f"eof_action=endall:log_fmt=json:log_path=vmaf.json")
    lavfi = (f"[0:v]{scale}settb=AVTB,setpts=N/(30*TB)[d];"
             f"[1:v]{scale}settb=AVTB,setpts=N/(30*TB)[r];"
             f"[d][r]libvmaf={opts}")
    # The subprocess runs with cwd=temp dir (see below), so the input paths
    # MUST be absolute — a caller's relative path (e.g. the CLI invoked with a
    # relative -o) would otherwise resolve against the temp dir and fail with
    # "No such file or directory" (a real bug the 10.3 CLI run surfaced).
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
           "-i", os.path.abspath(distorted)]
    if ref_start:
        cmd += ["-ss", f"{ref_start:.3f}"]
    if ref_duration is not None:
        cmd += ["-t", f"{ref_duration:.3f}"]
    cmd += ["-i", os.path.abspath(reference), "-lavfi", lavfi, "-f", "null", "-"]
    # log_path stays a bare filename + cwd=temp dir: a Windows drive-letter
    # path inside a filtergraph is an escaping tarpit (the 10.0 lesson).
    with tempfile.TemporaryDirectory(prefix="facekeep_vmaf_") as td:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=td)
        if proc.returncode != 0:
            raise VideoError(
                f"VMAF scoring failed on {Path(distorted).name}: "
                f"{proc.stderr.strip()[-500:]}"
            )
        data = json.loads((Path(td) / "vmaf.json").read_text(encoding="utf-8"))
    scores = [f["metrics"]["vmaf"] for f in data.get("frames", [])]
    if not scores:
        raise VideoError(f"VMAF produced no frame scores for {Path(distorted).name}")
    return scores


def _pool_scores(scores: List[float], model: str) -> VmafScore:
    ordered = sorted(scores)
    n = len(ordered)
    return VmafScore(
        mean=sum(ordered) / n,
        p1=ordered[max(0, n // 100 - 1)],
        min=ordered[0],
        frames=n,
        model=model,
    )


def score_vmaf(distorted: Union[str, Path], reference: Union[str, Path], *,
               reference_info: Optional[VideoInfo] = None,
               ref_start: Optional[float] = None,
               ref_duration: Optional[float] = None) -> VmafScore:
    """Score ``distorted`` against ``reference`` with libvmaf, pooled per-frame.

    The model follows the *reference* size (``_vmaf_model_for``): 4K-class
    sources (min dimension >= ``_VMAF_4K_MIN_DIM``) score with ``vmaf_4k`` at
    native resolution — measured, the default model at a 1080p downscale is
    saturated blind on 4K — and everything else scores with the default model,
    downscaled to its 1080p design height when larger. Pass ``reference_info``
    when the caller already probed (skips a redundant ffprobe run).

    Raises :class:`VideoError` when ffmpeg/libvmaf are unavailable or scoring
    fails.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None or _find_ffprobe() is None:
        raise VideoError(_MISSING_FFMPEG_HINT)
    if not _ffmpeg_has_libvmaf(ffmpeg):
        raise VideoError(
            "this ffmpeg build has no libvmaf filter; VMAF scoring needs one "
            "(e.g. a GPL build from ffmpeg.org/BtbN). Encoding works without it."
        )
    info = reference_info if reference_info is not None else probe_video(reference)
    model = _vmaf_model_for(info.width, info.height)
    scale = (model == _VMAF_MODEL_DEFAULT
             and min(info.width, info.height) > _VMAF_SCALE_MIN_DIM)
    scores = _vmaf_frame_scores(
        ffmpeg, Path(distorted), Path(reference), model=model,
        scale_to_1080=scale, ref_start=ref_start, ref_duration=ref_duration,
    )
    return _pool_scores(scores, model)


def _sample_spans(duration_s: float) -> List[Tuple[float, float]]:
    """The (start, duration) spans the CRF auto-tune probes.

    Three short spans spread through the file; a clip too short for sampling
    to be worth it (or an unknown duration) is probed whole/from the head.
    """
    span = _SAMPLE_SPAN_S
    if duration_s <= 0 or duration_s <= 2 * span * len(_SAMPLE_POSITIONS):
        return [(0.0, max(duration_s, span))]
    return [(round(max(0.0, duration_s * pos - span / 2), 3), span)
            for pos in _SAMPLE_POSITIONS]


def find_crf(input_path: Union[str, Path], *, target: float = DEFAULT_VMAF_TARGET,
             preset: int = DEFAULT_PRESET,
             info: Optional[VideoInfo] = None) -> Tuple[int, VmafScore]:
    """Sampled CRF search: the highest CRF whose sampled VMAF p1 meets ``target``.

    The photo auto-tune analog (ab-av1 style): binary-search
    ``_TUNE_CRF_RANGE``, and per probed CRF encode the sample spans at the
    *real* preset/pixel format, score them order-paired against the same
    source spans, and pool all spans' frame scores into one p1. Returns
    ``(crf, sampled_score)``. When even the range's quality end misses the
    target, that end is returned with a warning — best effort, never an error.

    Raises :class:`VideoError` when ffmpeg/libvmaf are unavailable or a probe
    fails. Cost note: each probe is a real (short) encode + score, so a search
    runs ~6 of them; on a clip short enough to be probed whole this approaches
    6x the single-encode cost — auto-tune is the opt-in path for a reason.
    """
    src = Path(input_path)
    ffmpeg = find_ffmpeg()
    if ffmpeg is None or _find_ffprobe() is None:
        raise VideoError(_MISSING_FFMPEG_HINT)
    if not _ffmpeg_has_libvmaf(ffmpeg):
        raise VideoError(
            "this ffmpeg build has no libvmaf filter; the CRF auto-tune needs "
            "one (e.g. a GPL build from ffmpeg.org/BtbN)."
        )
    if info is None:
        info = probe_video(src)
    model = _vmaf_model_for(info.width, info.height)
    scale = (model == _VMAF_MODEL_DEFAULT
             and min(info.width, info.height) > _VMAF_SCALE_MIN_DIM)
    spans = _sample_spans(info.duration_s)
    ten_bit = info.bit_depth > 8

    with tempfile.TemporaryDirectory(prefix="facekeep_tune_") as td:

        def sampled_score(crf: int) -> VmafScore:
            scores: List[float] = []
            for i, (start, dur) in enumerate(spans):
                sample = Path(td) / f"sample_{crf}_{i}.mp4"
                cmd = _encode_command(ffmpeg, src, sample, crf=crf, preset=preset,
                                      ten_bit=ten_bit, start=start, duration=dur,
                                      sample=True)
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise VideoError(
                        f"auto-tune sample encode failed on {src.name}: "
                        f"{proc.stderr.strip()[-500:]}"
                    )
                scores.extend(_vmaf_frame_scores(
                    ffmpeg, sample, src, model=model, scale_to_1080=scale,
                    ref_start=start, ref_duration=dur,
                ))
            return _pool_scores(scores, model)

        lo, hi = _TUNE_CRF_RANGE
        best: Optional[Tuple[int, VmafScore]] = None
        floor_score: Optional[VmafScore] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            score = sampled_score(mid)
            logger.debug("auto-tune probe %s crf=%d: p1=%.2f (target %.1f)",
                         src.name, mid, score.p1, target)
            if score.p1 >= target:
                best = (mid, score)
                lo = mid + 1
            else:
                if mid == _TUNE_CRF_RANGE[0]:
                    floor_score = score
                hi = mid - 1

    if best is None:
        # All probes missed; the last one was necessarily the range's quality
        # end (lo never moves on a miss), so floor_score is always set here.
        floor = _TUNE_CRF_RANGE[0]
        logger.warning(
            "%s: even crf=%d misses the sampled VMAF p1 target %.1f "
            "(got %.2f); using it as the best effort.",
            src.name, floor, target, floor_score.p1,
        )
        return floor, floor_score
    return best


def _sampled_face_count(ffmpeg: str, src: Path, info: VideoInfo,
                        detector=None) -> int:
    """Max face count the detector sees across a few sampled frames (10.5).

    The video counterpart of the photo pipeline's detect step, scoped to one
    question: *does this clip contain faces?* Frames at ``_SAMPLE_POSITIONS``
    (the auto-tune spans' positions — one mechanism) are extracted upright
    (ffmpeg autorotates, matching the encode) to a temp PNG and run through
    ``detector`` (any ``FaceDetector``; ``None`` builds the default offline
    Haar). Sampling is honest-but-cheap: a face that only appears between
    samples is missed and the clip just keeps the base target — never worse
    than not looking.

    Detection must never fail the pipeline (the photo rule): any error —
    extraction, decode, the detector itself — logs a warning and counts as
    zero faces.
    """
    try:
        import cv2  # a core dependency; imported lazily so the module stays
        # stdlib-only at import time (config.py imports it for the defaults)

        if detector is None:
            from .detector import create_detector

            detector = create_detector()
        positions = ([max(0.0, info.duration_s * p) for p in _SAMPLE_POSITIONS]
                     if info.duration_s > 0 else [0.0])
        best = 0
        with tempfile.TemporaryDirectory(prefix="facekeep_faces_") as td:
            for i, t in enumerate(positions):
                frame = Path(td) / f"frame_{i}.png"
                cmd = [ffmpeg, "-y", "-hide_banner", "-nostdin",
                       "-loglevel", "error"]
                if t:
                    cmd += ["-ss", f"{t:.3f}"]
                cmd += ["-i", os.path.abspath(src), "-frames:v", "1",
                        "-update", "1", str(frame)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0 or not frame.is_file():
                    continue  # e.g. a seek past a short stream's end
                img = cv2.imread(str(frame))
                if img is None:
                    continue
                best = max(best, len(detector.detect(img)))
        return best
    except Exception as e:  # noqa: BLE001 - detection never fails the pipeline
        logger.warning(
            "%s: sampled face detection failed (%s); assuming no faces.",
            src.name, e,
        )
        return 0


def _encode_command(ffmpeg: str, src: Path, tmp: Path, *, crf: int, preset: int,
                    ten_bit: bool, start: Optional[float] = None,
                    duration: Optional[float] = None,
                    sample: bool = False, dolby_vision: bool = False) -> List[str]:
    """Build the SVT-AV1 re-encode command (the hardened 10.0 spike command).

    Deliberate policies, each a ROADMAP 10.0 finding:

    - ``-fps_mode passthrough`` keeps the source's (possibly VFR) timestamps
      verbatim. Android phones record VFR; a CFR re-time keeps the frame count
      but changes the duration -> cumulative A/V desync (~11 s/hour measured).
    - 10-bit sources encode to ``yuv420p10le`` and the VUI color tags (e.g.
      bt2020/arib-std-b67/bt2020nc HLG) ride through the decoder into SVT-AV1
      untouched, so HDR stays HDR. A Dolby Vision RPU does not survive — the
      output degrades to the HLG base layer (a tone-curve refinement loss, not
      an HDR loss; optional dovi_tool carry is ROADMAP 10.5).
    - Rotation: ffmpeg autorotates (pixels physically rotated, the display
      matrix consumed). Chosen over carrying the matrix because the result
      displays upright in *every* player, including ones that ignore matrices.
    - Tracks: first video + first audio only, audio copied bit-exact. An
      iPhone ``.mov`` also carries an APAC spatial-audio track and ``mebx``
      sensor-data tracks; APAC won't mux into ``.mp4`` and the data tracks are
      meaningless without the Apple pipeline, so both are dropped — the AAC
      track everyone actually hears is kept.
    - ``-map_metadata 0 -movflags use_metadata_tags`` carries container
      metadata (creation time, GPS location, the com.apple.quicktime.* keys)
      into the output — verified end-to-end on the real iPhone clip. Known
      cosmetic quirk: ``use_metadata_tags`` stores ``creation_time`` in both
      the mvhd atom and an mdta tag, so ffprobe shows the value twice
      (';'-joined); players/indexers read the mvhd one.

    ``sample=True`` (auto-tune probes, 10.2) encodes a video-only span meant
    solely to be VMAF-scored and thrown away: ``start``/``duration`` trim it
    (input seek — frame-accurate when transcoding), audio and metadata are
    dropped. All the video policies above stay identical, so the probe
    measures exactly what the real encode would do. (Probes never pass
    ``dolby_vision`` — the RPU is metadata the decoder ignores for pixels, so
    it cannot move a VMAF score, and a DV-less probe can't fail on an exotic
    RPU.)

    ``dolby_vision=True`` (ROADMAP 10.5; only when the *source* carries an
    RPU-bearing DOVI record — the flag hard-fails the encoder on a DV-less
    input) adds ``-dolbyvision 1``: the wrapper re-codes the decoder-exported
    per-frame RPU into AV1 T.35 metadata OBUs (DV profile 8.x -> 10.x, content
    verified value-identical on both real phone clips) at ~400 bytes/frame.
    ``-strict unofficial`` is required by the mp4 muxer for the DV-AV1
    ``dvcC``/``dvvC`` box — DV-in-AV1 mp4 signaling is not yet an official
    Dolby spec; without the box a player never engages its DV pipeline.
    """
    cmd = [ffmpeg, "-y", "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    if sample:
        cmd += ["-map", "0:v:0", "-an"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "0"]
    cmd += [
        "-fps_mode", "passthrough",
        "-c:v", "libsvtav1", "-crf", str(crf), "-preset", str(preset),
        "-svtav1-params", "tune=0",
        "-pix_fmt", "yuv420p10le" if ten_bit else "yuv420p",
    ]
    if dolby_vision:
        cmd += ["-dolbyvision", "1", "-strict", "unofficial"]
    if not sample:
        cmd += ["-c:a", "copy"]
    cmd += ["-movflags", "+faststart+use_metadata_tags", "-f", "mp4", str(tmp)]
    return cmd


def compress_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    *,
    crf: int = DEFAULT_CRF,
    preset: int = DEFAULT_PRESET,
    skip_efficient: bool = True,
    vmaf_target: Optional[float] = DEFAULT_VMAF_TARGET,
    auto_tune: bool = False,
    preserve_dolby_vision: bool = True,
    face_vmaf_target: Optional[float] = DEFAULT_FACE_VMAF_TARGET,
    detector=None,
) -> VideoResult:
    """Faithfully re-encode a phone video to SVT-AV1 in a standard ``.mp4``.

    The full path: probe -> skip-if-efficient -> [sampled face detection ->
    raised target] -> [opt-in sampled CRF auto-tune] -> encode (VFR-safe,
    HDR/VUI passthrough, DV RPU carried, metadata carried, first audio
    copied) -> VMAF quality gate -> skip-if-larger. The encode writes to a
    temp file and renames on success, so a failed or skipped run never
    leaves a partial output.

    Quality verification (ROADMAP 10.2): every written encode is VMAF-scored
    against the source by default (order-paired frames; ``vmaf_4k`` at native
    resolution for 4K-class sources) and re-encoded ``_GATE_CRF_STEP`` lower
    on a p1-below-``vmaf_target`` miss, at most ``_GATE_MAX_RETRIES`` times —
    the safety net for content the fixed CRF default was never measured on.
    ``vmaf_target=None`` disables the gate (halves the pipeline cost on 4K:
    scoring is roughly as expensive as encoding). ``auto_tune=True`` (opt-in)
    first searches the highest CRF whose *sampled* p1 meets the target
    (:func:`find_crf`), replacing the fixed ``crf``; the gate then verifies
    the whole file. Both degrade gracefully to the plain fixed-CRF encode
    (warned, unverified) when this ffmpeg build lacks libvmaf.

    Dolby Vision (ROADMAP 10.5): when the source carries an RPU-bearing DOVI
    record and ``preserve_dolby_vision`` is on (default), the per-frame RPU
    is carried into the AV1 output (profile 8.x -> 10.x; see
    :func:`_encode_command`) so a DV display renders the same tone-refined
    picture as the original. Degrades gracefully: a build without the
    ``-dolbyvision`` option — or an exotic RPU the wrapper rejects — warns
    and encodes the plain HDR base layer (the pre-10.5 output).

    Face-aware quality (ROADMAP 10.5): when ``face_vmaf_target`` is set
    (default) and VMAF is in play, the face detector runs on a few sampled
    frames first; faces present raise the effective p1 target (gate *and*
    auto-tune) to ``max(target, face_vmaf_target)`` — family clips are held
    to a higher floor, face-less footage is untouched. ``detector`` accepts
    any ``FaceDetector`` (the CLI passes the configured one); ``None`` uses
    the default offline Haar. ``face_vmaf_target=None`` disables it.

    A skip writes nothing and returns ``skipped=True`` with the reason —
    keep-the-original semantics are the caller's job (CLI integration, 10.3).

    Raises :class:`VideoError` when ffmpeg is unavailable (with an install
    hint) or the probe/encode fails.
    """
    src = Path(input_path)
    if not src.is_file():
        raise VideoError(f"input not found: {src}")
    ffmpeg = find_ffmpeg()
    if ffmpeg is None or _find_ffprobe() is None:
        raise VideoError(_MISSING_FFMPEG_HINT)

    info = probe_video(src)
    original_size = src.stat().st_size

    if skip_efficient:
        reason = _efficiency_skip_reason(info)
        if reason is not None:
            logger.info("Skipping %s: %s", src.name, reason)
            return VideoResult(
                input_path=src, output_path=None, original_size=original_size,
                compressed_size=0, skipped=True, skip_reason=reason,
            )

    want_vmaf = vmaf_target is not None or auto_tune
    have_vmaf = _ffmpeg_has_libvmaf(ffmpeg) if want_vmaf else False
    if want_vmaf and not have_vmaf:
        logger.warning(
            "%s: this ffmpeg build has no libvmaf; VMAF quality %s skipped - "
            "encoding at fixed crf=%d, unverified.",
            src.name, "gate + auto-tune" if auto_tune else "gate", crf,
        )

    # Dolby Vision carry (10.5): only ever attempted when the *source* has an
    # RPU (the flag hard-fails the encoder on a DV-less input) and this build
    # can code it; otherwise today's HLG/HDR10-base output, warned.
    use_dv = False
    if preserve_dolby_vision and info.dovi_profile is not None:
        if _ffmpeg_supports_dovi_encode(ffmpeg):
            use_dv = True
        else:
            logger.warning(
                "%s: source carries Dolby Vision (profile %d) but this ffmpeg "
                "build cannot carry the RPU (libsvtav1 without -dolbyvision); "
                "encoding the HDR base layer only.",
                src.name, info.dovi_profile,
            )

    # Face-aware target raise (10.5). Sampled detection runs only when a VMAF
    # target can actually consume the answer (gate or auto-tune, with libvmaf
    # present) — otherwise it would be pure cost.
    faces: Optional[int] = None
    gate_target = vmaf_target
    tune_target = vmaf_target if vmaf_target is not None else DEFAULT_VMAF_TARGET
    if face_vmaf_target is not None and have_vmaf:
        faces = _sampled_face_count(ffmpeg, src, info, detector)
        if faces:
            raised = max(tune_target, face_vmaf_target)
            if gate_target is not None:
                gate_target = max(gate_target, face_vmaf_target)
            if raised > tune_target:
                logger.info(
                    "%s: faces on sampled frames (max %d) - raising VMAF p1 "
                    "target %.1f -> %.1f",
                    src.name, faces, tune_target, raised,
                )
            tune_target = raised

    t0 = time.perf_counter()
    chosen_crf = crf
    if auto_tune and have_vmaf:
        chosen_crf, sampled = find_crf(src, target=tune_target, preset=preset,
                                       info=info)
        logger.info(
            "%s: auto-tune chose crf=%d (sampled VMAF p1=%.2f, target %.1f)",
            src.name, chosen_crf, sampled.p1, tune_target,
        )

    out = Path(output_path) if output_path is not None else default_output_path(src)
    if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(src)):
        raise VideoError(
            f"output would overwrite the input ({src}); pass a different output path"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")

    score: Optional[VmafScore] = None
    attempt_crf = chosen_crf
    retries = 0
    try:
        while True:
            cmd = _encode_command(ffmpeg, src, tmp, crf=attempt_crf, preset=preset,
                                  ten_bit=info.bit_depth > 8, dolby_vision=use_dv)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                if use_dv:
                    # Graceful degradation (10.5): an RPU the wrapper cannot
                    # map (e.g. an exotic/dual-layer profile) fails encoder
                    # init — retry once without DV rather than failing the
                    # file. Doesn't consume a gate retry.
                    logger.warning(
                        "%s: Dolby Vision RPU carry failed (%s); re-encoding "
                        "without it (HDR base layer kept).",
                        src.name, proc.stderr.strip()[-300:],
                    )
                    use_dv = False
                    continue
                raise VideoError(
                    f"ffmpeg encode failed on {src.name}: {proc.stderr.strip()[-1000:]}"
                )
            if gate_target is None or not have_vmaf:
                break
            score = score_vmaf(tmp, src, reference_info=info)
            if score.p1 >= gate_target:
                break
            if retries >= _GATE_MAX_RETRIES or attempt_crf <= _GATE_MIN_CRF:
                logger.warning(
                    "%s: VMAF p1=%.2f still below target %.1f at crf=%d; "
                    "keeping the best effort (mean=%.2f).",
                    src.name, score.p1, gate_target, attempt_crf, score.mean,
                )
                break
            retries += 1
            new_crf = max(attempt_crf - _GATE_CRF_STEP, _GATE_MIN_CRF)
            logger.info(
                "%s: VMAF p1=%.2f below target %.1f at crf=%d; re-encoding at "
                "crf=%d (retry %d/%d).",
                src.name, score.p1, gate_target, attempt_crf, new_crf,
                retries, _GATE_MAX_RETRIES,
            )
            attempt_crf = new_crf
        encode_seconds = time.perf_counter() - t0
        compressed_size = tmp.stat().st_size

        # Skip-if-larger: an already-tight source can beat the re-encode; keep
        # the original rather than write a bigger file (the faithful precedent).
        if compressed_size >= original_size:
            reason = (
                f"re-encode not smaller ({compressed_size} >= {original_size} "
                "bytes) - keeping the original"
            )
            logger.info("Skipping %s: %s", src.name, reason)
            return VideoResult(
                input_path=src, output_path=None, original_size=original_size,
                compressed_size=0, skipped=True, skip_reason=reason,
                encode_seconds=encode_seconds, crf_used=attempt_crf, vmaf=score,
                dolby_vision=use_dv, faces=faces,
            )
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    logger.info(
        "Compressed %s -> %s: %.1f MB -> %.1f MB (%.1fx) in %.0fs%s%s",
        src.name, out.name, original_size / 1e6, compressed_size / 1e6,
        original_size / compressed_size, encode_seconds,
        "" if score is None else
        f" (crf={attempt_crf}, VMAF p1={score.p1:.2f} mean={score.mean:.2f})",
        " [DV RPU carried]" if use_dv else "",
    )
    return VideoResult(
        input_path=src, output_path=out, original_size=original_size,
        compressed_size=compressed_size, encode_seconds=encode_seconds,
        crf_used=attempt_crf, vmaf=score, dolby_vision=use_dv, faces=faces,
    )
