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

This module is the core library path (ROADMAP 10.1). The VMAF quality gate /
auto-tune (10.2) and CLI/batch integration (10.3) build on it; nothing here is
wired into ``facekeep compress`` yet.

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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

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

    @property
    def bits_per_pixel_frame(self) -> float:
        """Encoded bits spent per pixel per frame — the efficiency yardstick."""
        pixel_rate = self.width * self.height * self.fps
        return self.v_bit_rate / pixel_rate if pixel_rate else 0.0


@dataclass
class VideoResult:
    """Result of a faithful video compression."""

    input_path: Path
    output_path: Optional[Path]  # None when skipped (nothing was written)
    original_size: int
    compressed_size: int  # 0 when skipped
    skipped: bool = False
    skip_reason: Optional[str] = None
    encode_seconds: float = 0.0  # wall time of the encode (per-file ETA in 10.3)

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

    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"]))
            except (TypeError, ValueError):
                pass

    def _int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

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
    out = _with_video_extension(src, ".mp4")
    if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(src)):
        out = out.with_name(out.stem + "_av1.mp4")
    return out


def _encode_command(ffmpeg: str, src: Path, tmp: Path, *, crf: int, preset: int,
                    ten_bit: bool) -> List[str]:
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
    """
    return [
        ffmpeg, "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "0",
        "-fps_mode", "passthrough",
        "-c:v", "libsvtav1", "-crf", str(crf), "-preset", str(preset),
        "-svtav1-params", "tune=0",
        "-pix_fmt", "yuv420p10le" if ten_bit else "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart+use_metadata_tags",
        "-f", "mp4", str(tmp),
    ]


def compress_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    *,
    crf: int = DEFAULT_CRF,
    preset: int = DEFAULT_PRESET,
    skip_efficient: bool = True,
) -> VideoResult:
    """Faithfully re-encode a phone video to SVT-AV1 in a standard ``.mp4``.

    The full path: probe -> skip-if-efficient -> encode (VFR-safe, HDR/VUI
    passthrough, metadata carried, first audio copied) -> skip-if-larger.
    The encode writes to a temp file and renames on success, so a failed or
    skipped run never leaves a partial output.

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

    out = Path(output_path) if output_path is not None else default_output_path(src)
    if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(src)):
        raise VideoError(
            f"output would overwrite the input ({src}); pass a different output path"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")

    cmd = _encode_command(ffmpeg, src, tmp, crf=crf, preset=preset,
                          ten_bit=info.bit_depth > 8)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise VideoError(
                f"ffmpeg encode failed on {src.name}: {proc.stderr.strip()[-1000:]}"
            )
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
                encode_seconds=encode_seconds,
            )
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    logger.info(
        "Compressed %s -> %s: %.1f MB -> %.1f MB (%.1fx) in %.0fs",
        src.name, out.name, original_size / 1e6, compressed_size / 1e6,
        original_size / compressed_size, encode_seconds,
    )
    return VideoResult(
        input_path=src, output_path=out, original_size=original_size,
        compressed_size=compressed_size, encode_seconds=encode_seconds,
    )
