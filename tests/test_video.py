"""Tests for the faithful video re-encode path (facekeep/video.py, ROADMAP
10.1) and its VMAF quality gate / auto-tune (10.2).

The encode/probe tests need the external ffmpeg/ffprobe binaries (with
libsvtav1 + libx264/x265 for fixture generation) and skip with a clear reason
when they are absent — the avifenc-test precedent; the real-scoring tests
additionally need libvmaf in the build (the gate/retry logic itself is covered
with stubbed scores). Fixtures are synthetic, generated at test time with
ffmpeg (the repo ships no videos); the VFR case is mandatory per ROADMAP 10.1
(the shipped-bug-grade A/V-desync gotcha).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import pytest

from facekeep import video
from facekeep.exceptions import VideoError

# Test-side bootstrap only: this dev box keeps its ffmpeg machine-local at
# .tools/ffmpeg (gitignored, documented in ROADMAP 10.0) rather than on PATH,
# so point $FACEKEEP_FFMPEG there when nothing else resolves. Product code
# resolution ($FACEKEEP_FFMPEG -> PATH -> None) is unchanged.
_TOOLS_FFMPEG = (
    Path(__file__).resolve().parents[1]
    / ".tools" / "ffmpeg" / "bin"
    / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
)
if not video.ffmpeg_available() and _TOOLS_FFMPEG.is_file():
    os.environ["FACEKEEP_FFMPEG"] = str(_TOOLS_FFMPEG)

needs_ffmpeg = pytest.mark.skipif(
    not video.ffmpeg_available(),
    reason="ffmpeg/ffprobe not found (set FACEKEEP_FFMPEG, put ffmpeg on PATH, "
    "or place a build at .tools/ffmpeg/bin) — video tests need the binaries",
)

needs_vmaf = pytest.mark.skipif(
    not video.vmaf_available(),
    reason="ffmpeg with libvmaf not found — real VMAF scoring tests need a "
    "GPL build (the gate/auto-tune logic itself is covered with stubs)",
)


# ---------------------------------------------------------------------------
# Synthetic fixture generation (only runs for tests marked needs_ffmpeg)
# ---------------------------------------------------------------------------


def _ffgen(args: list, dst: Path) -> None:
    cmd = [video.find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           *[str(a) for a in args], str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"fixture generation failed: {proc.stderr[-1000:]}"


def _ffprobe_json(path: Path, *extra: str) -> dict:
    cmd = [video._find_ffprobe(), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", *extra, str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _packet_pts(path: Path) -> list:
    """All video-packet presentation timestamps (seconds), sorted."""
    cmd = [video._find_ffprobe(), "-v", "error", "-select_streams", "v:0",
           "-show_entries", "packet=pts_time", "-print_format", "json", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return sorted(
        float(p["pts_time"]) for p in json.loads(proc.stdout)["packets"]
        if p.get("pts_time") not in (None, "N/A")
    )


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("video_fixtures")


@pytest.fixture(scope="module")
def src_cfr(fixtures: Path) -> Path:
    """A fat CFR source: lossless x264 + AAC audio + a creation_time tag.

    Lossless keeps the bitrate far above the efficiency threshold, so the
    happy-path compress genuinely re-encodes and genuinely shrinks it.
    """
    dst = fixtures / "src_cfr.mp4"
    _ffgen([
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=3",
        "-shortest", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        "-metadata", "creation_time=2024-05-20T10:00:00Z",
    ], dst)
    return dst


@pytest.fixture(scope="module")
def src_vfr(fixtures: Path) -> Path:
    """A genuinely VFR source (irregular timestamps, ~25 fps average).

    PTS_n = (n + 0.4*sin(n)) / 25 — monotonic but non-uniform, and the average
    rate deliberately differs from the generator's nominal 30 so a CFR re-time
    would measurably change the duration (the A/V-desync bug mode).
    """
    dst = fixtures / "src_vfr.mp4"
    _ffgen([
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3",
        "-vf", "setpts=(N+0.4*sin(N))/25/TB",
        "-fps_mode", "passthrough",
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-an",
    ], dst)
    return dst


@pytest.fixture(scope="module")
def src_hlg10(fixtures: Path) -> Path:
    """A 10-bit HLG source with full VUI color tags (the phone-HDR shape).

    x265 needs the tags via -x265-params — plain -color_primaries flags don't
    land in the VUI (measured in the 10.0 spike).
    """
    dst = fixtures / "src_hlg10.mp4"
    _ffgen([
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
        "-c:v", "libx265", "-preset", "ultrafast",
        "-x265-params",
        "lossless=1:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
        "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-an",
    ], dst)
    return dst


@pytest.fixture(scope="module")
def src_rotated(fixtures: Path, src_cfr: Path) -> Path:
    """src_cfr remuxed with a -90° display matrix (the phone-portrait shape)."""
    dst = fixtures / "src_rotated.mp4"
    _ffgen(["-display_rotation", "-90", "-i", src_cfr, "-c", "copy"], dst)
    return dst


@pytest.fixture(scope="module")
def av1_result(src_cfr: Path, fixtures: Path) -> video.VideoResult:
    """One real compress of the CFR source, shared by several assertions.

    Gate off: on synthetic testsrc2 content the default p1 target could
    trigger retries and make the shared fixture's CRF nondeterministic; the
    gate has its own dedicated tests.
    """
    return video.compress_video(src_cfr, fixtures / "out_cfr.mp4",
                                crf=35, preset=10, vmaf_target=None)


@pytest.fixture(scope="module")
def vfr_result(src_vfr: Path, fixtures: Path) -> video.VideoResult:
    """One real compress of the VFR source (PTS assertions + VMAF pairing)."""
    return video.compress_video(src_vfr, fixtures / "out_vfr.mp4",
                                crf=35, preset=10, vmaf_target=None)


# ---------------------------------------------------------------------------
# Pure tests (no ffmpeg needed)
# ---------------------------------------------------------------------------


def test_is_video_file():
    assert video.is_video_file("clip.MOV")
    assert video.is_video_file(Path("a/b/clip.mp4"))
    assert not video.is_video_file("photo.jpg")
    assert not video.is_video_file("archive.fkeep")


def test_default_output_naming():
    # .mov -> .mp4 in place.
    assert video.default_output_path(Path("d/clip.MOV")) == Path("d/clip.mp4")
    # Dots that aren't a known video extension are kept (the dotted-filename rule).
    assert (video.default_output_path(Path("2024.05.20_trip.mov"))
            == Path("2024.05.20_trip.mp4"))
    # An .mp4 input would collide with itself -> _av1 stem suffix.
    assert video.default_output_path(Path("d/clip.mp4")) == Path("d/clip_av1.mp4")


def _info(**overrides) -> video.VideoInfo:
    base = dict(
        path=Path("x.mp4"), size_bytes=10_000_000, duration_s=10.0,
        width=3840, height=2160, fps=30.0, v_codec="hevc",
        pix_fmt="yuv420p10le", bit_depth=10, v_bit_rate=30_000_000,
        color_primaries="bt2020", color_transfer="arib-std-b67",
        color_space="bt2020nc", rotation=0, a_codec="aac",
    )
    base.update(overrides)
    return video.VideoInfo(**base)


def test_efficiency_skip_reasons():
    # A phone HEVC at ~0.12 bpp must NOT be skipped — it is the target case.
    assert video._efficiency_skip_reason(_info()) is None
    # An AV1 source is skipped regardless of bitrate.
    assert "AV1" in video._efficiency_skip_reason(_info(v_codec="av1"))
    # A low-bpp (already re-encoded) HEVC is skipped on the bpp signal.
    low = _info(v_bit_rate=int(3840 * 2160 * 30 * 0.03))
    assert "already efficient" in video._efficiency_skip_reason(low)
    # Unknown bitrate (0) must not false-positive as efficient.
    assert video._efficiency_skip_reason(_info(v_bit_rate=0)) is None


def test_bits_per_pixel_frame():
    info = _info(v_bit_rate=int(3840 * 2160 * 30 * 0.125))
    assert info.bits_per_pixel_frame == pytest.approx(0.125)
    assert _info(fps=0.0).bits_per_pixel_frame == 0.0


def test_find_ffmpeg_env_override(monkeypatch, tmp_path):
    """$FACEKEEP_FFMPEG (a real file) wins; a bad value falls back to PATH."""
    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"stub")
    monkeypatch.setenv("FACEKEEP_FFMPEG", str(fake))
    assert video.find_ffmpeg() == str(fake)
    monkeypatch.setenv("FACEKEEP_FFMPEG", str(tmp_path / "nope.exe"))
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    assert video.find_ffmpeg() is None


def test_missing_ffmpeg_is_a_clear_hint_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.delenv("FACEKEEP_FFMPEG", raising=False)
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    assert video.ffmpeg_available() is False
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"not really a video")
    with pytest.raises(VideoError, match="FACEKEEP_FFMPEG"):
        video.compress_video(src)


def test_compress_missing_input():
    with pytest.raises(VideoError, match="not found"):
        video.compress_video(Path("does/not/exist.mp4"))


# ---------------------------------------------------------------------------
# Probe + encode tests (need the real binaries)
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_probe_reports_source_facts(src_cfr: Path):
    info = video.probe_video(src_cfr)
    assert info.v_codec == "h264"
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(30.0, abs=0.1)
    assert info.duration_s == pytest.approx(3.0, abs=0.2)
    assert info.bit_depth == 8
    assert info.a_codec == "aac"
    assert info.rotation == 0
    assert info.bits_per_pixel_frame > video._EFFICIENT_BPP  # lossless = fat


@needs_ffmpeg
def test_probe_no_video_stream(fixtures: Path):
    audio_only = fixtures / "audio_only.m4a"
    _ffgen(["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:a", "aac"], audio_only)
    with pytest.raises(VideoError, match="no video stream"):
        video.probe_video(audio_only)


@needs_ffmpeg
def test_compress_roundtrip_shrinks_and_keeps_streams(av1_result, src_cfr):
    assert not av1_result.skipped
    out = av1_result.output_path
    assert out is not None and out.exists()
    assert av1_result.compressed_size < av1_result.original_size
    assert av1_result.ratio > 1.0
    assert av1_result.encode_seconds > 0

    data = _ffprobe_json(out)
    streams = data["streams"]
    # Exactly video + audio — no data/metadata tracks ride along.
    assert [s["codec_type"] for s in streams] == ["video", "audio"]
    v = streams[0]
    assert v["codec_name"] == "av1"
    assert v["pix_fmt"] == "yuv420p"  # 8-bit source stays 8-bit
    assert streams[1]["codec_name"] == "aac"  # audio copied, not re-encoded

    # Duration survives (same timestamps, copied audio).
    src_dur = float(_ffprobe_json(src_cfr)["format"]["duration"])
    out_dur = float(data["format"]["duration"])
    assert out_dur == pytest.approx(src_dur, abs=0.1)


@needs_ffmpeg
def test_compress_carries_creation_time(av1_result, src_cfr):
    src_tags = _ffprobe_json(src_cfr)["format"].get("tags", {})
    assert "creation_time" in src_tags, "fixture lost its tag; adjust generation"
    out_tags = _ffprobe_json(av1_result.output_path)["format"].get("tags", {})
    assert out_tags.get("creation_time", "").startswith("2024-05-20T10:00:00")


@needs_ffmpeg
def test_av1_source_is_skipped(av1_result, fixtures: Path):
    """Skip-if-efficient: our own output must never be re-eaten on a re-run."""
    res = video.compress_video(av1_result.output_path, fixtures / "twice.mp4")
    assert res.skipped
    assert res.output_path is None
    assert "AV1" in res.skip_reason
    assert not (fixtures / "twice.mp4").exists()


@needs_ffmpeg
def test_vfr_timestamps_survive_verbatim(src_vfr: Path, vfr_result):
    """The mandatory VFR case: passthrough keeps every PTS, count, duration.

    A CFR re-time keeps the frame count but changes the timestamps/duration —
    the cumulative A/V-desync bug the 10.0 spike caught on a real Android clip.
    """
    src_pts = _packet_pts(src_vfr)
    # Sanity: the fixture really is VFR with an average rate away from 30.
    deltas = [b - a for a, b in zip(src_pts, src_pts[1:])]
    assert max(deltas) - min(deltas) > 1e-3, "fixture is not VFR"
    avg_fps = len(deltas) / (src_pts[-1] - src_pts[0])
    assert 24.0 < avg_fps < 26.0

    res = vfr_result
    assert not res.skipped
    out_pts = _packet_pts(res.output_path)
    assert len(out_pts) == len(src_pts)  # every frame, none dropped/duplicated
    for s, o in zip(src_pts, out_pts):
        assert o == pytest.approx(s, abs=5e-3)  # timestamps verbatim (± timescale rounding)

    src_dur = float(_ffprobe_json(src_vfr)["format"]["duration"])
    out_dur = float(_ffprobe_json(res.output_path)["format"]["duration"])
    assert out_dur == pytest.approx(src_dur, abs=0.08)


@needs_ffmpeg
def test_hlg_10bit_color_passthrough(src_hlg10: Path, fixtures: Path):
    """10-bit + HLG VUI tags survive the re-encode (HDR stays HDR)."""
    res = video.compress_video(src_hlg10, fixtures / "out_hlg.mp4",
                               crf=35, preset=10)
    assert not res.skipped
    v = _ffprobe_json(res.output_path)["streams"][0]
    assert v["codec_name"] == "av1"
    assert v["pix_fmt"] == "yuv420p10le"
    assert v.get("color_primaries") == "bt2020"
    assert v.get("color_transfer") == "arib-std-b67"
    assert v.get("color_space") == "bt2020nc"


@needs_ffmpeg
def test_rotation_autorotates_upright(src_rotated: Path, fixtures: Path):
    """Explicit rotation policy: autorotate — pixels upright, matrix consumed."""
    info = video.probe_video(src_rotated)
    assert info.rotation != 0, "fixture lost its display matrix; adjust generation"
    res = video.compress_video(src_rotated, fixtures / "out_rot.mp4",
                               crf=35, preset=10)
    out = video.probe_video(res.output_path)
    # 320x240 source displayed at -90° -> physically 240x320, no matrix left.
    assert (out.width, out.height) == (240, 320)
    assert out.rotation == 0


@needs_ffmpeg
def test_skip_if_larger_keeps_nothing(src_cfr: Path, fixtures: Path, monkeypatch):
    """A not-smaller encode is discarded: no output, no .part temp left."""
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if "libsvtav1" in cmd:  # the encode call; probes pass through
            Path(cmd[-1]).write_bytes(b"\0" * (src_cfr.stat().st_size + 1))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return real_run(cmd, **kw)

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    out = fixtures / "out_larger.mp4"
    # Gate off: the faked "encode" writes garbage bytes, which real VMAF
    # scoring would (rightly) refuse to probe.
    res = video.compress_video(src_cfr, out, vmaf_target=None)
    assert res.skipped
    assert "not smaller" in res.skip_reason
    assert res.output_path is None
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists()


@needs_ffmpeg
def test_failed_encode_cleans_temp_and_raises(src_cfr: Path, fixtures: Path,
                                              monkeypatch):
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if "libsvtav1" in cmd:
            Path(cmd[-1]).write_bytes(b"partial")
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        return real_run(cmd, **kw)

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    out = fixtures / "out_fail.mp4"
    with pytest.raises(VideoError, match="encode failed"):
        video.compress_video(src_cfr, out)
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists()


@needs_ffmpeg
def test_output_overwrite_guard(src_cfr: Path):
    with pytest.raises(VideoError, match="overwrite"):
        video.compress_video(src_cfr, src_cfr)


# ---------------------------------------------------------------------------
# VMAF quality gate / auto-tune (ROADMAP 10.2) — pure tests
# ---------------------------------------------------------------------------


def test_vmaf_model_selection():
    """4K-class sources (min dimension, so rotation-invariant) use vmaf_4k."""
    assert video._vmaf_model_for(3840, 2160) == video._VMAF_MODEL_4K
    assert video._vmaf_model_for(2160, 3840) == video._VMAF_MODEL_4K  # portrait
    assert video._vmaf_model_for(1920, 1080) == video._VMAF_MODEL_DEFAULT
    assert video._vmaf_model_for(2560, 1440) == video._VMAF_MODEL_DEFAULT
    assert video._vmaf_model_for(1280, 720) == video._VMAF_MODEL_DEFAULT


def test_sample_spans():
    # A long clip: three spans spread through it, all inside the duration.
    spans = video._sample_spans(60.0)
    assert len(spans) == len(video._SAMPLE_POSITIONS)
    for (start, dur), pos in zip(spans, video._SAMPLE_POSITIONS):
        assert dur == video._SAMPLE_SPAN_S
        assert start == pytest.approx(60.0 * pos - dur / 2, abs=0.01)
        assert 0.0 <= start and start + dur <= 60.0
    # A short clip is probed whole; an unknown duration probes the head.
    assert video._sample_spans(10.0) == [(0.0, 10.0)]
    assert video._sample_spans(0.0) == [(0.0, video._SAMPLE_SPAN_S)]


def test_pool_scores():
    scores = [90.0 + i * 0.1 for i in range(100)]  # 90.0, 90.1, ... 99.9
    pooled = video._pool_scores(scores, video._VMAF_MODEL_DEFAULT)
    assert pooled.min == 90.0
    assert pooled.p1 == 90.0  # n=100 -> the single worst frame
    assert pooled.mean == pytest.approx(94.95)
    assert pooled.frames == 100
    assert pooled.model == video._VMAF_MODEL_DEFAULT
    # Tiny frame counts degrade to the min, never an index error.
    assert video._pool_scores([50.0, 99.0], "m").p1 == 50.0


def test_vmaf_available_false_without_ffmpeg(monkeypatch):
    monkeypatch.delenv("FACEKEEP_FFMPEG", raising=False)
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    assert video.vmaf_available() is False


def test_score_vmaf_and_find_crf_require_libvmaf(monkeypatch, tmp_path):
    """ffmpeg present but built without libvmaf -> a clear error, not a crash."""
    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"stub")
    monkeypatch.setattr(video, "find_ffmpeg", lambda: str(fake))
    monkeypatch.setattr(video, "_find_ffprobe", lambda: str(fake))
    monkeypatch.setattr(video, "_ffmpeg_has_libvmaf", lambda ff: False)
    with pytest.raises(VideoError, match="libvmaf"):
        video.score_vmaf(tmp_path / "a.mp4", tmp_path / "b.mp4")
    with pytest.raises(VideoError, match="libvmaf"):
        video.find_crf(tmp_path / "b.mp4")


# ---------------------------------------------------------------------------
# Quality gate — retry logic with stubbed scores (real encodes, no libvmaf)
# ---------------------------------------------------------------------------


def _stub_score(p1: float) -> video.VmafScore:
    return video.VmafScore(mean=min(p1 + 3.0, 100.0), p1=p1, min=p1 - 2.0,
                           frames=90, model=video._VMAF_MODEL_DEFAULT)


@needs_ffmpeg
def test_gate_retries_lower_crf(src_cfr: Path, fixtures: Path, monkeypatch):
    """A p1 miss re-encodes one CRF step lower; the pass is recorded."""
    scores = iter([_stub_score(80.0), _stub_score(96.0)])
    monkeypatch.setattr(video, "_ffmpeg_has_libvmaf", lambda ff: True)
    monkeypatch.setattr(video, "score_vmaf", lambda *a, **k: next(scores))
    res = video.compress_video(src_cfr, fixtures / "out_gate_retry.mp4",
                               crf=40, preset=10, vmaf_target=93.0)
    assert not res.skipped
    assert res.crf_used == 40 - video._GATE_CRF_STEP
    assert res.vmaf is not None and res.vmaf.p1 == 96.0
    assert res.output_path.exists()


@needs_ffmpeg
def test_gate_gives_up_at_max_retries(src_cfr: Path, fixtures: Path,
                                      monkeypatch, caplog):
    """A never-passing gate keeps the best effort after the retry cap, warned."""
    calls = []
    monkeypatch.setattr(video, "_ffmpeg_has_libvmaf", lambda ff: True)
    monkeypatch.setattr(
        video, "score_vmaf",
        lambda *a, **k: calls.append(1) or _stub_score(50.0),
    )
    with caplog.at_level(logging.WARNING, logger="facekeep.video"):
        res = video.compress_video(src_cfr, fixtures / "out_gate_floor.mp4",
                                   crf=40, preset=10, vmaf_target=93.0)
    assert not res.skipped
    assert len(calls) == 1 + video._GATE_MAX_RETRIES
    assert res.crf_used == 40 - video._GATE_MAX_RETRIES * video._GATE_CRF_STEP
    assert res.vmaf.p1 == 50.0  # the honest (failing) score is still reported
    assert res.output_path.exists()
    assert "still below target" in caplog.text


@needs_ffmpeg
def test_gate_never_goes_below_crf_floor(src_cfr: Path, fixtures: Path,
                                         monkeypatch):
    """Retries clamp at _GATE_MIN_CRF and stop there even if still failing."""
    monkeypatch.setattr(video, "_ffmpeg_has_libvmaf", lambda ff: True)
    monkeypatch.setattr(video, "score_vmaf", lambda *a, **k: _stub_score(50.0))
    res = video.compress_video(src_cfr, fixtures / "out_gate_clamp.mp4",
                               crf=video._GATE_MIN_CRF + 1, preset=10,
                               vmaf_target=93.0)
    assert res.crf_used == video._GATE_MIN_CRF
    assert res.output_path.exists()


@needs_ffmpeg
def test_gate_skipped_without_libvmaf(src_cfr: Path, fixtures: Path,
                                      monkeypatch, caplog):
    """No libvmaf in the build -> warned, unverified fixed-CRF encode (also
    the auto_tune fallback) — graceful degradation, never a crash."""
    monkeypatch.setattr(video, "_ffmpeg_has_libvmaf", lambda ff: False)
    with caplog.at_level(logging.WARNING, logger="facekeep.video"):
        res = video.compress_video(src_cfr, fixtures / "out_novmaf.mp4",
                                   crf=35, preset=10, auto_tune=True)
    assert not res.skipped
    assert res.output_path.exists()
    assert res.crf_used == 35  # auto-tune skipped: the fixed CRF was used
    assert res.vmaf is None
    assert "libvmaf" in caplog.text


# ---------------------------------------------------------------------------
# Real VMAF scoring + auto-tune (need an ffmpeg build with libvmaf)
# ---------------------------------------------------------------------------


@needs_vmaf
def test_score_vmaf_real_and_monotonic(av1_result, src_cfr: Path, fixtures: Path):
    s35 = video.score_vmaf(av1_result.output_path, src_cfr)
    assert 0.0 <= s35.min <= s35.p1 <= s35.mean <= 100.0
    assert 80 <= s35.frames <= 95  # ~3 s at 30 fps
    assert s35.model == video._VMAF_MODEL_DEFAULT  # 320x240 is not 4K-class

    # More compression must score worse (the discriminator the tune relies on).
    crushed = video.compress_video(src_cfr, fixtures / "out_crushed.mp4",
                                   crf=60, preset=10, vmaf_target=None)
    s60 = video.score_vmaf(crushed.output_path, src_cfr)
    assert s60.mean < s35.mean
    assert s60.p1 < s35.p1


@needs_vmaf
def test_score_vmaf_identical_is_near_perfect(src_cfr: Path):
    s = video.score_vmaf(src_cfr, src_cfr)
    assert s.p1 >= 97.0
    assert s.mean >= 97.0


@needs_vmaf
def test_score_vmaf_vfr_pairs_frames_by_order(vfr_result, src_vfr: Path):
    """The order-pairing regression: timestamp pairing on a VFR reference
    collapses to a flat false ~20 (spike-measured on a real Android clip);
    order pairing scores the visually-good encode honestly."""
    s = video.score_vmaf(vfr_result.output_path, src_vfr)
    assert s.mean > 80.0


@needs_vmaf
def test_gate_passes_first_try_real(src_cfr: Path, fixtures: Path):
    """A modest target on a decent encode: one pass, score recorded, no retry."""
    res = video.compress_video(src_cfr, fixtures / "out_gate_real.mp4",
                               crf=35, preset=10, vmaf_target=50.0)
    assert not res.skipped
    assert res.crf_used == 35
    assert res.vmaf is not None
    assert res.vmaf.p1 >= 50.0


@needs_vmaf
def test_auto_tune_real_end_to_end(src_cfr: Path, fixtures: Path):
    lo, hi = video._TUNE_CRF_RANGE
    crf, sampled = video.find_crf(src_cfr, target=70.0, preset=10)
    assert lo <= crf <= hi
    assert sampled.frames > 0

    res = video.compress_video(src_cfr, fixtures / "out_tuned.mp4",
                               preset=10, auto_tune=True, vmaf_target=70.0)
    assert not res.skipped
    assert res.output_path.exists()
    assert res.vmaf is not None  # the gate verified the full file
    # The gate may have stepped below the searched CRF, never above it.
    assert video._GATE_MIN_CRF <= res.crf_used <= crf
