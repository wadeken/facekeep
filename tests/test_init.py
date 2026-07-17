"""`facekeep init` — write a commented starter config.

A pure-UX command: it writes a commented ``facekeep.yaml`` at the shipped
defaults so a user has a documented starting point. It touches no pixels and no
output bytes; the contract these tests pin is that the written file (a) is a
default location/name, (b) round-trips cleanly through ``FaceKeepConfig.load()``
and equals a fresh default config, and (c) refuses to clobber an existing file
unless ``--force`` is given.
"""

from pathlib import Path

import yaml
from click.testing import CliRunner

from facekeep.cli import cli
from facekeep.config import FaceKeepConfig, default_config_yaml


def test_init_writes_default_named_file(tmp_path):
    """`facekeep init` with no arg writes ./facekeep.yaml in the cwd."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(cli, ["init"])
        assert res.exit_code == 0, res.output
        assert Path("facekeep.yaml").exists()
        assert "Wrote" in res.output


def test_init_custom_path(tmp_path):
    out = tmp_path / "sub" / "my.yaml"
    res = CliRunner().invoke(cli, ["init", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists()  # parent dir was created


def test_written_config_is_valid_and_equals_defaults(tmp_path):
    """The written file loads, validates, and equals a fresh default config."""
    out = tmp_path / "facekeep.yaml"
    assert CliRunner().invoke(cli, ["init", str(out)]).exit_code == 0

    cfg = FaceKeepConfig.load(out)  # load() calls validate() internally
    d = FaceKeepConfig()
    assert cfg.mode == d.mode
    assert cfg.strip_gps == d.strip_gps
    assert cfg.detector.backend == d.detector.backend
    assert cfg.detector.roi == d.detector.roi
    assert cfg.faithful.codec == d.faithful.codec
    assert cfg.faithful.auto_tune == d.faithful.auto_tune
    assert cfg.faithful.target_metric == d.faithful.target_metric
    assert cfg.faithful.output_bit_depth == d.faithful.output_bit_depth
    assert cfg.aggressive.bg_scale == d.aggressive.bg_scale
    assert cfg.aggressive.face_codec == d.aggressive.face_codec
    assert cfg.aggressive.content_aware == d.aggressive.content_aware
    assert cfg.aggressive.protect_hands == d.aggressive.protect_hands
    assert cfg.video.enabled == d.video.enabled
    assert cfg.video.crf == d.video.crf
    assert cfg.video.vmaf_target == d.video.vmaf_target


def test_written_config_is_commented(tmp_path):
    """The template carries explanatory comments (the point vs config.save())."""
    out = tmp_path / "facekeep.yaml"
    CliRunner().invoke(cli, ["init", str(out)])
    text = out.read_text(encoding="utf-8")
    assert text.lstrip().startswith("#")
    assert text.count("#") > 10  # genuinely commented, not a bare dump


def test_init_refuses_to_clobber(tmp_path):
    out = tmp_path / "facekeep.yaml"
    out.write_text("mode: aggressive\n", encoding="utf-8")
    res = CliRunner().invoke(cli, ["init", str(out)])
    assert res.exit_code == 1
    assert "already exists" in res.output
    # The existing file is untouched.
    assert out.read_text(encoding="utf-8") == "mode: aggressive\n"


def test_init_force_overwrites(tmp_path):
    out = tmp_path / "facekeep.yaml"
    out.write_text("mode: aggressive\n", encoding="utf-8")
    res = CliRunner().invoke(cli, ["init", str(out), "--force"])
    assert res.exit_code == 0, res.output
    assert FaceKeepConfig.load(out).mode == "faithful"  # overwritten with default


def test_template_yaml_keys_are_known_fields():
    """Every key in the template is a real config field (no typos / dead keys)."""
    data = yaml.safe_load(default_config_yaml())
    assert set(data) <= {"mode", "strip_gps", "detector", "faithful",
                         "aggressive", "video"}
    cfg = FaceKeepConfig()
    for section, obj in (
        ("detector", cfg.detector),
        ("faithful", cfg.faithful),
        ("aggressive", cfg.aggressive),
        ("video", cfg.video),
    ):
        for key in data.get(section, {}):
            assert hasattr(obj, key), f"unknown {section}.{key} in init template"
