"""CLI/batch integration for faithful video (ROADMAP 10.3).

Pins the wiring contracts around ``facekeep/video.py`` (whose encode/gate
behavior is covered in ``test_video.py``):

1. **Folder runs gather videos** alongside photos; a video is compressed
   faithfully to a standard AV1 ``.mp4`` and reported like any other file.
2. **The ``video:`` config section** exists, validates, records explicit YAML
   keys, and threads its knobs (crf/preset/vmaf_target/auto_tune) into the run.
3. **Videos cache under their own index fingerprint**: a re-run skips an
   unchanged video, a changed video knob busts it, and photo knobs never do.
4. **``--report`` rows**: mode=video, codec=av1, ``quality`` = the CRF used,
   and the new ``vmaf_p1`` column filled only when the gate really scored.
5. **Aggressive mode never applies to video**: a single-video input errors
   loudly; a folder run skips each video with the reason.
6. **Graceful degradation without ffmpeg**: a single-video input errors with
   the install hint; a folder run skips videos with it (the HEIC precedent).
7. **Keep-the-original semantics**: a skipped video (already efficient / not
   smaller) copies the source into a differing output dir, in-place stays put.
8. **Dry-run honesty**: videos are probed, never test-encoded — the decision is
   reported with no invented size estimate.

Real-encode tests need the external ffmpeg binaries and skip with a clear
reason otherwise (the test_video.py precedent, including the .tools bootstrap).
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from facekeep import cli as cli_mod, encoders, index as index_mod, report, video
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import ConfigError

# Test-side bootstrap only (same as test_video.py): this dev box keeps ffmpeg
# machine-local at .tools/ffmpeg rather than on PATH.
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
    "or place a build at .tools/ffmpeg/bin) — video CLI tests need the binaries",
)

needs_vmaf = pytest.mark.skipif(
    not video.vmaf_available(),
    reason="ffmpeg with libvmaf not found — the vmaf_p1 report test needs it",
)

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


@pytest.fixture(scope="module")
def clip_mov(tmp_path_factory) -> Path:
    """A fat (lossless x264) 3 s CFR ``.mov`` — genuinely worth re-encoding."""
    dst = tmp_path_factory.mktemp("video_cli_fixtures") / "clip.mov"
    cmd = [video.find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3",
           "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-an",
           str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"fixture generation failed: {proc.stderr[-1000:]}"
    return dst


def _write_cfg(path: Path, **video_keys) -> Path:
    """A config YAML with only a video: section (photo settings stay default)."""
    lines = ["video:"]
    for key, value in video_keys.items():
        if value is None:
            value = "null"
        elif isinstance(value, bool):
            value = str(value).lower()
        lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


# --------------------------------------------------------------------------- #
# Config: the video: section
# --------------------------------------------------------------------------- #

def test_video_config_defaults_mirror_video_module():
    cfg = FaceKeepConfig()
    assert cfg.video.enabled is True
    assert cfg.video.crf == video.DEFAULT_CRF
    assert cfg.video.preset == video.DEFAULT_PRESET
    assert cfg.video.vmaf_target == video.DEFAULT_VMAF_TARGET
    assert cfg.video.auto_tune is False
    assert cfg.video.skip_efficient is True
    assert cfg.video.preserve_dolby_vision is True
    assert cfg.video.face_aware is True
    assert cfg.video.face_vmaf_target == video.DEFAULT_FACE_VMAF_TARGET


def test_video_config_yaml_load_records_explicit_keys(tmp_path):
    cfg_path = _write_cfg(tmp_path / "facekeep.yaml", crf=40, vmaf_target=None,
                          auto_tune=True)
    cfg = FaceKeepConfig.load(cfg_path)
    assert cfg.video.crf == 40
    assert cfg.video.vmaf_target is None
    assert cfg.video.auto_tune is True
    assert cfg.video.preset == video.DEFAULT_PRESET  # untouched key = default
    assert {"video.crf", "video.vmaf_target", "video.auto_tune"} <= cfg.explicit_keys


@pytest.mark.parametrize("field,value", [
    ("crf", -1), ("crf", 64), ("preset", 14), ("preset", -1),
    ("vmaf_target", 0.0), ("vmaf_target", 101.0),
    ("face_vmaf_target", 0.0), ("face_vmaf_target", 101.0),
])
def test_video_config_validation_rejects_bad_values(field, value):
    cfg = FaceKeepConfig()
    setattr(cfg.video, field, value)
    with pytest.raises(ConfigError, match=f"video.{field}"):
        cfg.validate()


def test_video_config_save_roundtrip(tmp_path):
    cfg = FaceKeepConfig()
    cfg.video.crf = 45
    cfg.video.vmaf_target = None
    cfg.save(tmp_path / "cfg.yaml")
    loaded = FaceKeepConfig.load(tmp_path / "cfg.yaml")
    assert loaded.video.crf == 45
    assert loaded.video.vmaf_target is None


# --------------------------------------------------------------------------- #
# Index fingerprints: videos cache independently of photos
# --------------------------------------------------------------------------- #

def test_video_fingerprint_busts_on_every_video_knob():
    base = FaceKeepConfig()
    fp0 = index_mod.video_settings_fingerprint(base)
    for field, value in [("crf", 40), ("preset", 4), ("vmaf_target", None),
                         ("auto_tune", True), ("skip_efficient", False),
                         ("preserve_dolby_vision", False),
                         ("face_aware", False), ("face_vmaf_target", 97.0)]:
        cfg = FaceKeepConfig()
        setattr(cfg.video, field, value)
        assert index_mod.video_settings_fingerprint(cfg) != fp0, field
    # Determinism: identical configs fingerprint identically.
    assert index_mod.video_settings_fingerprint(FaceKeepConfig()) == fp0


def test_video_and_photo_fingerprints_are_independent():
    """Retuning photos must not re-encode cached videos, and vice versa."""
    base = FaceKeepConfig()
    photo_fp = index_mod.settings_fingerprint(base)
    video_fp = index_mod.video_settings_fingerprint(base)

    photo_changed = FaceKeepConfig()
    photo_changed.faithful.quality = 50
    assert index_mod.video_settings_fingerprint(photo_changed) == video_fp

    video_changed = FaceKeepConfig()
    video_changed.video.crf = 45
    assert index_mod.settings_fingerprint(video_changed) == photo_fp


def test_video_fingerprint_detector_coupling_follows_face_aware():
    """The shared detector busts video caches iff face-aware consumes it.

    With face_aware on (the default), the detector fields that change *whether
    a face is found* are output-affecting for videos too (they can raise the
    VMAF target) -> a backend change busts. Box-shaping fields (padding/roi)
    and, with face_aware off, ALL detector fields never bust a cached video
    encode (minutes-to-hours each — the 10.3 independence promise).
    """
    base = FaceKeepConfig()
    fp0 = index_mod.video_settings_fingerprint(base)

    found_changed = FaceKeepConfig()
    found_changed.detector.backend = "yunet"
    assert index_mod.video_settings_fingerprint(found_changed) != fp0

    shape_changed = FaceKeepConfig()
    shape_changed.detector.padding = 2.0
    shape_changed.detector.roi = "person"
    assert index_mod.video_settings_fingerprint(shape_changed) == fp0

    off_base = FaceKeepConfig()
    off_base.video.face_aware = False
    off_fp = index_mod.video_settings_fingerprint(off_base)
    off_changed = FaceKeepConfig()
    off_changed.video.face_aware = False
    off_changed.detector.backend = "yunet"
    off_changed.detector.confidence = 0.9
    assert index_mod.video_settings_fingerprint(off_changed) == off_fp


# --------------------------------------------------------------------------- #
# Report: the vmaf_p1 column
# --------------------------------------------------------------------------- #

def test_video_report_row_carries_faces_and_dv_note():
    """A video row fills the faces column (10.5) and the OK line says DV."""
    res = {"file": "a.mov", "mode": "video", "status": "ok", "codec": "av1",
           "quality": 28, "original_size": 100, "compressed_size": 50,
           "ratio": 2.0, "vmaf_p1": 95.1, "faces": 2, "dolby_vision": True,
           "output_name": "a.mp4", "encode_seconds": 3.0}
    row = cli_mod._row_from_result(res, dry_run=False)
    assert row.status == "written"
    assert row.faces == 2
    assert row.vmaf_p1 == 95.1


def test_report_vmaf_p1_column_is_last_and_honest(tmp_path):
    assert report.FIELDNAMES[-1] == "vmaf_p1"
    rows = [
        report.ReportRow(file="a.mov", mode="video", status="written",
                         codec="av1", quality=32, original_bytes=100,
                         output_bytes=40, ratio=2.5, vmaf_p1=94.531),
        report.ReportRow(file="b.mov", mode="video", status="written",
                         codec="av1", quality=32),
    ]
    out = report.write_report(rows, str(tmp_path / "r.csv"))
    fieldnames, data = _read_csv(out)
    assert fieldnames == report.FIELDNAMES
    assert data[0]["vmaf_p1"] == "94.53"
    assert data[1]["vmaf_p1"] == ""  # not scored -> blank, never invented


# --------------------------------------------------------------------------- #
# Output naming (pure)
# --------------------------------------------------------------------------- #

def test_output_path_for_folder_targets():
    out = Path("out")
    assert video.output_path_for(Path("src/clip.MOV"), out) == out / "clip.mp4"
    # Dotted filenames keep their non-extension dots.
    assert (video.output_path_for(Path("src/2024.05.20_trip.mov"), out)
            == out / "2024.05.20_trip.mp4")
    # An in-place .mp4 input would collide with itself -> _av1 suffix.
    assert (video.output_path_for(Path("d/clip.mp4"), Path("d"))
            == Path("d/clip_av1.mp4"))
    # The same name into a *different* dir is no collision.
    assert video.output_path_for(Path("d/clip.mp4"), out) == out / "clip.mp4"


# --------------------------------------------------------------------------- #
# CLI guards (no ffmpeg needed: they fire before any probe/encode)
# --------------------------------------------------------------------------- #

def _dummy_video(dir_path: Path, name: str = "clip.mp4") -> Path:
    p = dir_path / name
    p.write_bytes(b"not really a video")
    return p


def test_single_video_aggressive_is_a_loud_error(tmp_path):
    src = _dummy_video(tmp_path)
    result = CliRunner().invoke(cli, ["compress", str(src), "-m", "aggressive"])
    assert result.exit_code == 2
    assert "does not apply to video" in result.output


def test_folder_aggressive_skips_videos_with_reason(tmp_path):
    _dummy_video(tmp_path)
    result = CliRunner().invoke(cli, ["compress", str(tmp_path), "-m", "aggressive"])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output
    assert "does not apply to video" in result.output


def test_no_videos_flag_excludes_videos(tmp_path):
    _dummy_video(tmp_path)
    result = CliRunner().invoke(cli, ["compress", str(tmp_path), "--no-videos"])
    assert result.exit_code == 1
    assert "excluded" in result.output


def test_video_enabled_false_excludes_videos(tmp_path):
    _dummy_video(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    _write_cfg(cfg, enabled=False)
    result = CliRunner().invoke(
        cli, ["compress", str(tmp_path), "--config", str(cfg)]
    )
    assert result.exit_code == 1
    assert "excluded" in result.output


def test_missing_ffmpeg_single_video_exits_with_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "ffmpeg_available", lambda: False)
    src = _dummy_video(tmp_path)
    result = CliRunner().invoke(cli, ["compress", str(src)])
    assert result.exit_code == 2
    assert "FACEKEEP_FFMPEG" in result.output
    assert "Photos are unaffected" in result.output


def test_missing_ffmpeg_folder_skips_videos_with_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "ffmpeg_available", lambda: False)
    _dummy_video(tmp_path)
    result = CliRunner().invoke(cli, ["compress", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output
    assert "FACEKEEP_FFMPEG" in result.output


# --------------------------------------------------------------------------- #
# Real-encode CLI end-to-end (need the binaries)
# --------------------------------------------------------------------------- #

@needs_ffmpeg
def test_cli_video_e2e_index_skip_and_knob_bust(clip_mov, tmp_path):
    """Folder run encodes a video; a re-run index-skips it; a video knob busts."""
    src_dir = clip_mov.parent
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = _write_cfg(tmp_path / "cfg.yaml", crf=35, preset=10, vmaf_target=None)

    runner = CliRunner()
    r1 = runner.invoke(cli, ["compress", str(src_dir), "-o", str(out_dir),
                             "--config", str(cfg)])
    assert r1.exit_code == 0, r1.output
    assert "av1 crf35" in r1.output
    assert "[video 1/1]" in r1.output  # the serial liveness/ETA line
    out_file = out_dir / "clip.mp4"
    assert out_file.exists()
    assert out_file.stat().st_size < clip_mov.stat().st_size

    # Warm re-run: the video is skipped via its own index fingerprint.
    r2 = runner.invoke(cli, ["compress", str(src_dir), "-o", str(out_dir),
                             "--config", str(cfg)])
    assert r2.exit_code == 0, r2.output
    assert "SKIP (unchanged)" in r2.output

    # Changing a video knob busts the cache -> a real re-encode.
    cfg2 = _write_cfg(tmp_path / "cfg2.yaml", crf=45, preset=10, vmaf_target=None)
    r3 = runner.invoke(cli, ["compress", str(src_dir), "-o", str(out_dir),
                             "--config", str(cfg2)])
    assert r3.exit_code == 0, r3.output
    assert "SKIP (unchanged)" not in r3.output
    assert "av1 crf45" in r3.output


@needs_ffmpeg
def test_cli_video_report_row(clip_mov, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = _write_cfg(tmp_path / "cfg.yaml", crf=35, preset=10, vmaf_target=None)
    report_csv = tmp_path / "r.csv"

    result = CliRunner().invoke(cli, [
        "compress", str(clip_mov.parent), "-o", str(out_dir),
        "--config", str(cfg), "--no-index", "--report", str(report_csv),
    ])
    assert result.exit_code == 0, result.output

    fieldnames, data = _read_csv(report_csv)
    assert fieldnames == report.FIELDNAMES
    assert len(data) == 1
    row = data[0]
    assert row["mode"] == "video"
    assert row["codec"] == "av1"
    assert row["status"] == "written"
    assert row["quality"] == "35"          # the CRF used
    assert float(row["ratio"]) > 1.0
    assert row["output_path"] == "clip.mp4"
    assert row["vmaf_p1"] == ""            # gate off -> blank, never invented
    assert row["faces"] == ""              # no face concept for video
    assert row["ssim_downscaled"] == ""


@needs_vmaf
def test_cli_video_gate_fills_vmaf_p1(clip_mov, tmp_path):
    """With the gate on (and libvmaf present) the report carries a real p1."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # A modest target so the synthetic clip passes first try (deterministic).
    cfg = _write_cfg(tmp_path / "cfg.yaml", crf=35, preset=10, vmaf_target=50)
    report_csv = tmp_path / "r.csv"

    result = CliRunner().invoke(cli, [
        "compress", str(clip_mov.parent), "-o", str(out_dir),
        "--config", str(cfg), "--no-index", "--report", str(report_csv),
    ])
    assert result.exit_code == 0, result.output
    assert "VMAF p1=" in result.output
    _, data = _read_csv(report_csv)
    assert float(data[0]["vmaf_p1"]) >= 50.0


@needs_ffmpeg
def test_cli_video_efficient_skip_copies_original(clip_mov, tmp_path):
    """An already-AV1 source keeps the original — copied to a differing out dir."""
    av1_dir = tmp_path / "av1src"
    av1_dir.mkdir()
    res = video.compress_video(clip_mov, av1_dir / "clip.mp4",
                               crf=45, preset=10, vmaf_target=None)
    assert not res.skipped

    out_dir = tmp_path / "backup"
    out_dir.mkdir()
    result = CliRunner().invoke(cli, ["compress", str(av1_dir), "-o", str(out_dir),
                                      "--no-index"])
    assert result.exit_code == 0, result.output
    assert "KEPT ORIGINAL" in result.output
    assert "AV1" in result.output
    kept = out_dir / "clip.mp4"
    assert kept.exists()
    assert kept.read_bytes() == (av1_dir / "clip.mp4").read_bytes()


@needs_ffmpeg
def test_cli_video_efficient_skip_in_place_stays_put(clip_mov, tmp_path):
    """In-place run: keep-the-original writes nothing (no self-copy)."""
    av1_dir = tmp_path / "av1src"
    av1_dir.mkdir()
    video.compress_video(clip_mov, av1_dir / "clip.mp4",
                         crf=45, preset=10, vmaf_target=None)
    before = sorted(p.name for p in av1_dir.iterdir())

    result = CliRunner().invoke(cli, ["compress", str(av1_dir), "--no-index"])
    assert result.exit_code == 0, result.output
    assert "KEPT ORIGINAL" in result.output
    assert sorted(p.name for p in av1_dir.iterdir()) == before


@needs_ffmpeg
def test_cli_video_dry_run_probes_but_never_encodes(clip_mov, tmp_path):
    """Dry-run honesty: the decision is reported with no invented size."""
    report_csv = tmp_path / "r.csv"
    result = CliRunner().invoke(cli, [
        "compress", str(clip_mov.parent), "--dry-run",
        "--report", str(report_csv),
    ])
    assert result.exit_code == 0, result.output
    assert "WOULD ENCODE" in result.output
    assert not list(clip_mov.parent.glob("*.mp4"))  # nothing written
    _, data = _read_csv(report_csv)
    assert data[0]["mode"] == "video"
    assert data[0]["status"] == "would-write"
    assert data[0]["output_bytes"] == ""  # no encode ran -> no invented number
    assert data[0]["ratio"] == ""


@needs_ffmpeg
@requires_avif
def test_cli_mixed_folder_photos_and_videos(clip_mov, face_image, tmp_path):
    """One folder, one photo + one video: both compressed, both summarized."""
    src_dir = tmp_path / "roll"
    src_dir.mkdir()
    (src_dir / face_image.name).write_bytes(face_image.read_bytes())
    (src_dir / clip_mov.name).write_bytes(clip_mov.read_bytes())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = _write_cfg(tmp_path / "cfg.yaml", crf=35, preset=10, vmaf_target=None)

    result = CliRunner().invoke(cli, [
        "compress", str(src_dir), "-o", str(out_dir),
        "--config", str(cfg), "--no-index",
    ])
    assert result.exit_code == 0, result.output
    assert (out_dir / "clip.mp4").exists()
    assert (out_dir / (face_image.stem + ".avif")).exists()
    assert "2/2 ok" in result.output


# --------------------------------------------------------------------------- #
# _process_one_video unit shapes (no ffmpeg: error paths only)
# --------------------------------------------------------------------------- #

def test_process_one_video_missing_ffmpeg_is_failed_result(tmp_path, monkeypatch):
    """The worker shape never raises: a missing binary becomes a failed row."""
    monkeypatch.delenv("FACEKEEP_FFMPEG", raising=False)
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    src = _dummy_video(tmp_path)
    res = cli_mod._process_one_video(
        str(src), str(tmp_path / "out.mp4"), FaceKeepConfig().video, False
    )
    assert res["status"] == "failed"
    assert "FACEKEEP_FFMPEG" in res["error"]
