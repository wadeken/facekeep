"""Aggressive-mode presets (`--preset`) — ROADMAP Phase 7.

A preset is a one-word goal that expands to a tuned bundle of ordinary config
fields, applied as a layer with a fixed precedence: dataclass defaults <
preset expansion < explicit YAML keys < explicit CLI flags. The preset *name*
is recorded in the .fkeep manifest (settings.preset, schema 1.7.0) so restore
can auto-apply the preset's restore-side knobs; the name itself is NOT
fingerprinted (the expanded fields already are).

What these tests pin (the ROADMAP item's test list):

* the registry is sane: the five contract names exist and every dotted key in
  every expansion is a real config field;
* precedence — an explicit YAML key beats a YAML *or* CLI preset, and an
  explicit CLI flag beats everything;
* a preset implies aggressive mode, and combining it with an explicit
  faithful mode is a loud error (CLI and YAML), never a silent flip;
* an unknown preset name errors listing the valid names (config and CLI);
* fingerprint equivalence: a preset and the same values set by hand produce
  the same settings_fingerprint (so the name is provably not hashed);
* the manifest records settings.preset and bumps to 1.7.0; a presetless file
  carries no new key and reads/verifies as before (backward compat);
* restore auto-applies the manifest preset's restore-side knobs, an explicit
  YAML key beats the manifest hint, and an unknown/absent name is ignored
  (tolerant by structure);
* `share` sets strip_gps; the preset's yunet request reaches create_detector
  (captured kwargs — no network); dry-run size parity holds under a preset;
  the init template documents `preset:`; `info` prints the recorded preset.

All offline: detectors are mocked where a pipeline runs, restorers are faked
where the CLI restores, and bg_codec is overridden to jpg wherever a test
doesn't specifically need a modern-codec member.
"""

import importlib.util
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from facekeep import encoders
from facekeep.aggressive import compressor as compressor_mod
from facekeep.aggressive.compressor import CompressedPhoto
from facekeep.aggressive.format import read_fkeep_info, verify_fkeep, write_fkeep
from facekeep.cli import _load_config, cli
from facekeep.config import (
    PRESET_NAMES,
    PRESETS,
    RESTORE_SIDE_PRESET_KEYS,
    FaceKeepConfig,
    _best_face_enhance_backend,
    apply_preset,
    default_config_yaml,
    preset_restore_overrides,
)
from facekeep.exceptions import ConfigError
from facekeep.index import settings_fingerprint

requires_jxl = pytest.mark.skipif(
    not encoders.codec_available("jxl"), reason="JXL encoder not installed"
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _smooth_bg(w: int = 320, h: int = 240) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.zeros((h, w, 3), np.uint8)
    bg[..., 0] = (120 + 60 * np.sin(xx / 80)).clip(0, 255)
    bg[..., 1] = (110 + 50 * np.sin((xx + yy) / 90)).clip(0, 255)
    bg[..., 2] = (90 + 40 * np.cos(yy / 70)).clip(0, 255)
    return cv2.GaussianBlur(bg, (0, 0), 2.0)


def _photo(cfg: FaceKeepConfig, w: int = 320, h: int = 240) -> CompressedPhoto:
    """A zero-face CompressedPhoto carrying ``cfg.aggressive`` (detector bypassed)."""
    return CompressedPhoto(
        original_filename="p.jpg", original_width=w * 4, original_height=h * 4,
        original_size_bytes=999, original_hash="0" * 64, original_orientation=1,
        exif=None,
        background=_smooth_bg(w, h),
        face_crops=[], face_masks=[], faces=[],
        thumbnail=np.full((128, 128, 3), 128, np.uint8),
        effective_bg_scale=0.25, config=cfg.aggressive,
    )


def _preset_cfg(name: str, bg_codec: str = "jpg") -> FaceKeepConfig:
    """A config with ``name`` applied, bg_codec overridden for plugin-free tests.

    The override is the documented precedence in action: a hand-set field after
    apply_preset beats the expansion (here it just keeps the test offline-cheap).
    """
    cfg = FaceKeepConfig()
    apply_preset(cfg, name)
    cfg.aggressive.bg_codec = bg_codec
    return cfg


def _write_fkeep(cfg: FaceKeepConfig, tmp_path: Path, stem: str) -> Path:
    out = tmp_path / stem
    write_fkeep(_photo(cfg), str(out))
    return tmp_path / f"{stem}.fkeep"


def _rewrite_manifest(fkeep: Path, mutate) -> None:
    """Rewrite manifest.json in-place (for unknown-name / old-file fixtures)."""
    with zipfile.ZipFile(fkeep) as zf:
        items = {n: zf.read(n) for n in zf.namelist()}
    m = json.loads(items["manifest.json"])
    mutate(m)
    items["manifest.json"] = json.dumps(m).encode("utf-8")
    with zipfile.ZipFile(fkeep, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, b in items.items():
            zf.writestr(n, b)


class _FakeRestorer:
    """Captures the AggressiveConfig the CLI hands each Restorer (no AI, no IO)."""

    captured: list = []

    def __init__(self, agg_cfg):
        _FakeRestorer.captured.append(agg_cfg)

    def restore(self, src, target, quality=None):
        Path(target).write_bytes(b"restored")

    def preview(self, src, target, quality=None):
        Path(target).write_bytes(b"preview")


@pytest.fixture
def fake_restorer(monkeypatch):
    _FakeRestorer.captured = []
    monkeypatch.setattr("facekeep.aggressive.restorer.Restorer", _FakeRestorer)
    return _FakeRestorer


@pytest.fixture
def no_yaml_cwd(tmp_path, monkeypatch):
    """Run from a clean cwd: the repo root ships a facekeep.yaml that
    FaceKeepConfig.load(None) would otherwise auto-discover."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _capture_create_detector(monkeypatch):
    """Patch the compressor's create_detector to capture kwargs (no network)."""
    calls = {}

    class _NoFaces:
        def detect(self, image):
            return []

    def _fake(**kwargs):
        calls["kwargs"] = kwargs
        return _NoFaces()

    monkeypatch.setattr(compressor_mod, "create_detector", _fake)
    return calls


# --------------------------------------------------------------------------- #
# A. Registry sanity
# --------------------------------------------------------------------------- #

def test_preset_names_are_the_contract():
    assert set(PRESETS) == {"ratio", "pretty", "fidelity", "family", "share"}
    assert set(PRESET_NAMES) == set(PRESETS)


def test_every_preset_key_is_a_real_field_and_validates():
    """Each expansion applies cleanly onto a fresh config and validate() passes.

    This is the test_init precedent: a preset naming a field that drifts away
    in a refactor must fail here, not at a user's terminal.
    """
    for name in PRESET_NAMES:
        cfg = FaceKeepConfig()
        apply_preset(cfg, name)  # _set_dotted raises on an unknown field
        cfg.validate()
        assert cfg.mode == "aggressive"
        assert cfg.aggressive.preset == name


def test_restore_side_keys_are_a_subset_of_real_preset_keys():
    all_keys = set()
    for builder in PRESETS.values():
        all_keys |= set(builder())
    assert RESTORE_SIDE_PRESET_KEYS <= all_keys
    # and they are exactly the pretty preset's restore knobs today
    assert RESTORE_SIDE_PRESET_KEYS == {
        "aggressive.face_enhance_backend",
        "aggressive.face_enhance_fidelity",
        "aggressive.face_enhance_strength",
    }


# --------------------------------------------------------------------------- #
# B. apply_preset semantics
# --------------------------------------------------------------------------- #

def test_apply_ratio_sets_expansion_and_implies_aggressive():
    cfg = FaceKeepConfig()
    apply_preset(cfg, "ratio")
    assert cfg.mode == "aggressive"
    assert cfg.aggressive.preset == "ratio"
    assert cfg.aggressive.bg_scale == 0.125
    assert cfg.aggressive.bg_codec == "jxl"
    assert cfg.aggressive.face_codec == "avif"
    assert cfg.aggressive.face_quality == 90
    assert cfg.aggressive.detector_backend == "yunet"
    assert cfg.strip_gps is False  # ratio does not touch privacy


def test_apply_share_is_ratio_plus_strip_gps():
    ratio, share = FaceKeepConfig(), FaceKeepConfig()
    apply_preset(ratio, "ratio")
    apply_preset(share, "share")
    assert share.strip_gps is True
    assert share.aggressive.bg_scale == ratio.aggressive.bg_scale
    assert share.aggressive.bg_codec == ratio.aggressive.bg_codec
    assert share.aggressive.face_codec == ratio.aggressive.face_codec


def test_apply_skips_explicit_keys():
    """An explicitly-written field beats the preset (the precedence contract)."""
    cfg = FaceKeepConfig()
    cfg.aggressive.bg_scale = 0.5  # "explicitly written"
    apply_preset(cfg, "ratio", explicit_keys=frozenset({"aggressive.bg_scale"}))
    assert cfg.aggressive.bg_scale == 0.5  # kept
    assert cfg.aggressive.bg_codec == "jxl"  # rest of the expansion applied


def test_apply_unknown_name_raises_listing_names():
    with pytest.raises(ConfigError, match="ratio.*pretty|Unknown preset"):
        apply_preset(FaceKeepConfig(), "speedy")


def test_apply_conflicts_with_explicit_faithful_mode():
    cfg = FaceKeepConfig()
    cfg.mode = "faithful"
    with pytest.raises(ConfigError, match="implies aggressive"):
        apply_preset(cfg, "ratio", explicit_keys=frozenset({"mode"}))


def test_pretty_resolves_best_available_backend(monkeypatch):
    """pretty = "best available enhancer": codeformer iff importable, else gfpgan.

    This is the documented preset semantic, distinct from the explicit
    face_enhance_backend rule (which never falls back silently).
    """
    real = importlib.util.find_spec

    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda n, *a, **k: object() if n == "codeformer" else real(n, *a, **k),
    )
    assert _best_face_enhance_backend() == "codeformer"

    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda n, *a, **k: None if n == "codeformer" else real(n, *a, **k),
    )
    assert _best_face_enhance_backend() == "gfpgan"
    cfg = FaceKeepConfig()
    apply_preset(cfg, "pretty")
    assert cfg.aggressive.face_enhance_backend == "gfpgan"
    assert cfg.aggressive.face_enhance_fidelity == 0.5


# --------------------------------------------------------------------------- #
# C. validate()
# --------------------------------------------------------------------------- #

def test_validate_rejects_unknown_preset_name():
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"
    cfg.aggressive.preset = "speedy"
    with pytest.raises(ConfigError, match="Unknown preset"):
        cfg.validate()


def test_validate_rejects_preset_with_faithful_mode():
    cfg = FaceKeepConfig()
    apply_preset(cfg, "ratio")
    cfg.mode = "faithful"  # e.g. a later CLI -m faithful
    with pytest.raises(ConfigError, match="implies aggressive"):
        cfg.validate()


# --------------------------------------------------------------------------- #
# D. YAML precedence (load)
# --------------------------------------------------------------------------- #

def test_yaml_preset_expands(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("preset: ratio\n", encoding="utf-8")
    cfg = FaceKeepConfig.load(p)
    assert cfg.mode == "aggressive"
    assert cfg.aggressive.preset == "ratio"
    assert cfg.aggressive.bg_scale == 0.125


def test_yaml_explicit_field_beats_yaml_preset(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "preset: ratio\naggressive:\n  bg_scale: 0.5\n", encoding="utf-8"
    )
    cfg = FaceKeepConfig.load(p)
    assert cfg.aggressive.bg_scale == 0.5  # explicit key wins
    assert cfg.aggressive.bg_codec == "jxl"  # the rest of the preset holds


def test_yaml_preset_with_faithful_mode_errors(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("mode: faithful\npreset: ratio\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="implies aggressive"):
        FaceKeepConfig.load(p)


def test_load_records_explicit_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "mode: aggressive\naggressive:\n  bg_scale: 0.5\nfaithful:\n  codec: jxl\n",
        encoding="utf-8",
    )
    cfg = FaceKeepConfig.load(p)
    assert "mode" in cfg.explicit_keys
    assert "aggressive.bg_scale" in cfg.explicit_keys
    assert "faithful.codec" in cfg.explicit_keys
    assert "aggressive.bg_codec" not in cfg.explicit_keys


def test_save_load_round_trip_preserves_preset(tmp_path):
    cfg = FaceKeepConfig()
    apply_preset(cfg, "fidelity")
    p = tmp_path / "saved.yaml"
    cfg.save(p)
    loaded = FaceKeepConfig.load(p)
    assert loaded == cfg  # dataclass equality over every field, preset included


# --------------------------------------------------------------------------- #
# E. CLI precedence + conflicts
# --------------------------------------------------------------------------- #

def test_cli_flag_beats_preset(no_yaml_cwd):
    cfg = _load_config(None, None, None, None, 0.5, preset="ratio")
    assert cfg.mode == "aggressive"
    assert cfg.aggressive.bg_scale == 0.5  # --bg-scale wins over the preset
    assert cfg.aggressive.bg_codec == "jxl"  # preset still supplies the rest
    assert cfg.aggressive.preset == "ratio"


def test_cli_preset_does_not_beat_explicit_yaml_key(tmp_path, no_yaml_cwd):
    """A CLI --preset only *chooses* the preset; an explicit YAML key outranks it."""
    p = tmp_path / "c.yaml"
    p.write_text("aggressive:\n  bg_scale: 0.5\n", encoding="utf-8")
    cfg = _load_config(str(p), None, None, None, None, preset="ratio")
    assert cfg.aggressive.bg_scale == 0.5  # YAML explicit key kept
    assert cfg.aggressive.bg_codec == "jxl"  # non-explicit fields expanded


def test_cli_preset_overrides_yaml_faithful_mode(tmp_path, no_yaml_cwd):
    """A CLI --preset, like -m aggressive, beats a YAML `mode: faithful`.

    The shipped sample and the `facekeep init` template both carry a persistent
    mode line; erroring on it would make the documented --preset flag unusable
    for exactly those users. Only same-level contradictions error (--preset +
    -m faithful on the CLI; preset: + mode: faithful inside one YAML file).
    """
    p = tmp_path / "c.yaml"
    p.write_text("mode: faithful\n", encoding="utf-8")
    cfg = _load_config(str(p), None, None, None, None, preset="ratio")
    assert cfg.mode == "aggressive"
    assert cfg.aggressive.preset == "ratio"


def test_cli_preset_works_with_the_init_template(tmp_path, no_yaml_cwd):
    """Uncommenting `preset:` in a fresh `facekeep init` file must just work."""
    p = tmp_path / "facekeep.yaml"
    text = default_config_yaml().replace("# preset: family", "preset: family")
    p.write_text(text, encoding="utf-8")
    cfg = FaceKeepConfig.load(p)
    assert cfg.mode == "aggressive"
    assert cfg.aggressive.preset == "family"
    assert cfg.aggressive.protect_hands_backend == "mediapipe"  # not in template
    # The template's own explicit lines win over the preset — the documented
    # precedence, warned right in the template ("any key you write explicitly
    # in this file still wins"). detector.roi is spelled out there as "face".
    assert cfg.detector.roi == "face"


def test_cli_preset_with_m_faithful_exits(face_image, tmp_path, no_yaml_cwd):
    res = CliRunner().invoke(
        cli, ["compress", str(face_image), "--preset", "ratio", "-m", "faithful",
              "-o", str(tmp_path / "out")],
    )
    assert res.exit_code == 2
    assert "implies aggressive" in res.output


def test_cli_unknown_preset_rejected(face_image, no_yaml_cwd):
    res = CliRunner().invoke(cli, ["compress", str(face_image), "--preset", "bogus"])
    assert res.exit_code != 0
    assert "Invalid value" in res.output  # click.Choice lists the valid names


@requires_jxl
def test_cli_preset_end_to_end_records_preset(face_image, tmp_path, no_yaml_cwd,
                                              monkeypatch):
    """compress --preset ratio writes a .fkeep whose manifest carries the name.

    The detector is mocked (the preset asks for yunet — no network in tests);
    the JXL background member is real, so this also proves the expansion's
    bg_codec reached the writer.
    """
    _capture_create_detector(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # a directory target -> "<dir>/<stem>.fkeep"
    res = CliRunner().invoke(
        cli, ["compress", str(face_image), "--preset", "ratio",
              "-o", str(out_dir), "--no-index", "--no-detect-cache"],
    )
    assert res.exit_code == 0, res.output
    fkeeps = list(out_dir.glob("*.fkeep"))
    assert len(fkeeps) == 1
    info = read_fkeep_info(str(fkeeps[0]))
    assert info["version"] == "1.8.0"
    assert info["settings"]["preset"] == "ratio"
    assert info["settings"]["bg_codec"] == "jxl"
    with zipfile.ZipFile(fkeeps[0]) as zf:
        assert "background.jxl" in zf.namelist()


def test_compressor_preset_requests_yunet(face_image, monkeypatch):
    """The preset's recall upgrade reaches create_detector (captured kwargs).

    Pins the plumbing only — the yunet->Haar offline fallback chain itself is
    already covered by the detector tests; nothing here touches the network.
    """
    calls = _capture_create_detector(monkeypatch)
    cfg = _preset_cfg("ratio")
    compressor_mod.compress_photo(str(face_image), cfg)
    assert calls["kwargs"]["backend"] == "yunet"


# --------------------------------------------------------------------------- #
# F. Fingerprint equivalence
# --------------------------------------------------------------------------- #

def test_fingerprint_preset_equals_hand_set():
    """A preset and the same values set by hand fingerprint identically.

    This proves the preset *name* is not hashed (cfg_a carries it, cfg_b does
    not) while the expanded fields — already fingerprinted — still bust the
    cache as usual.
    """
    cfg_a = FaceKeepConfig()
    apply_preset(cfg_a, "ratio")

    cfg_b = FaceKeepConfig()
    cfg_b.mode = "aggressive"
    for dotted, value in PRESETS["ratio"]().items():
        section, key = dotted.split(".", 1)
        setattr(getattr(cfg_b, section), key, value)
    assert cfg_b.aggressive.preset is None  # hand-set: no name recorded

    assert settings_fingerprint(cfg_a) == settings_fingerprint(cfg_b)
    # and the expansion genuinely moved the fingerprint vs the default
    default = FaceKeepConfig()
    default.mode = "aggressive"
    assert settings_fingerprint(cfg_a) != settings_fingerprint(default)


# --------------------------------------------------------------------------- #
# G. Manifest record + backward compatibility
# --------------------------------------------------------------------------- #

def test_manifest_records_preset(tmp_path):
    fkeep = _write_fkeep(_preset_cfg("ratio"), tmp_path, "p")
    info = read_fkeep_info(str(fkeep))
    assert info["version"] == "1.8.0"
    assert info["settings"]["preset"] == "ratio"


def test_presetless_manifest_has_no_preset_key(tmp_path):
    fkeep = _write_fkeep(FaceKeepConfig(), tmp_path, "plain")
    info = read_fkeep_info(str(fkeep))
    assert "preset" not in info["settings"]
    assert verify_fkeep(str(fkeep)).ok


def test_old_manifest_without_preset_still_verifies(tmp_path):
    """A downgraded 1.6.0 manifest (no preset key) reads and verifies unchanged."""
    fkeep = _write_fkeep(FaceKeepConfig(), tmp_path, "old")

    def _downgrade(m):
        m["version"] = "1.6.0"
        m["settings"].pop("preset", None)

    _rewrite_manifest(fkeep, _downgrade)
    info = read_fkeep_info(str(fkeep))
    assert info["version"] == "1.6.0"
    assert verify_fkeep(str(fkeep)).ok


def test_dry_run_size_parity_with_preset(tmp_path):
    """The shared _write_archive rule holds under a preset: estimate == real."""
    cfg = _preset_cfg("ratio")
    photo = _photo(cfg)
    estimated = write_fkeep(photo, str(tmp_path / "dry"), dry_run=True)
    assert not list(tmp_path.iterdir())  # dry run wrote nothing
    real = write_fkeep(photo, str(tmp_path / "real"))
    assert estimated == real


# --------------------------------------------------------------------------- #
# H. Restore auto-apply (manifest -> restore-side knobs)
# --------------------------------------------------------------------------- #

def test_restore_auto_applies_pretty_knobs(tmp_path, no_yaml_cwd, fake_restorer):
    fkeep = _write_fkeep(_preset_cfg("pretty"), tmp_path, "pretty")
    res = CliRunner().invoke(
        cli, ["restore", str(fkeep), "-o", str(tmp_path / "r.jpg")]
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "r.jpg").exists()
    # captured[0] is the base restorer; [1] is the preset-derived one used here
    assert len(fake_restorer.captured) == 2
    agg = fake_restorer.captured[1]
    assert agg.face_enhance_fidelity == 0.5
    assert agg.face_enhance_strength == 1.0
    # "best available" is re-resolved on the restoring machine
    assert agg.face_enhance_backend == _best_face_enhance_backend()


def test_restore_explicit_yaml_key_beats_manifest_preset(tmp_path, no_yaml_cwd,
                                                         fake_restorer):
    fkeep = _write_fkeep(_preset_cfg("pretty"), tmp_path, "pretty")
    conf = tmp_path / "c.yaml"
    conf.write_text(
        "aggressive:\n  face_enhance_fidelity: 0.9\n", encoding="utf-8"
    )
    res = CliRunner().invoke(
        cli, ["restore", str(fkeep), "-o", str(tmp_path / "r.jpg"),
              "--config", str(conf)],
    )
    assert res.exit_code == 0, res.output
    agg = fake_restorer.captured[1]
    assert agg.face_enhance_fidelity == 0.9  # explicit YAML key kept
    assert agg.face_enhance_backend == _best_face_enhance_backend()  # rest applied


def test_restore_presetless_uses_base_config(tmp_path, no_yaml_cwd, fake_restorer):
    fkeep = _write_fkeep(FaceKeepConfig(), tmp_path, "plain")
    res = CliRunner().invoke(
        cli, ["restore", str(fkeep), "-o", str(tmp_path / "r.jpg")]
    )
    assert res.exit_code == 0, res.output
    assert len(fake_restorer.captured) == 1  # only the base restorer was built
    assert fake_restorer.captured[0].face_enhance_fidelity == 0.7  # default


def test_restore_unknown_preset_name_is_ignored(tmp_path, no_yaml_cwd,
                                                fake_restorer):
    """A .fkeep from a *future* preset still restores on this reader."""
    fkeep = _write_fkeep(FaceKeepConfig(), tmp_path, "future")
    _rewrite_manifest(fkeep, lambda m: m["settings"].update(preset="futurism"))
    res = CliRunner().invoke(
        cli, ["restore", str(fkeep), "-o", str(tmp_path / "r.jpg")]
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "r.jpg").exists()
    assert len(fake_restorer.captured) == 1  # no overrides -> base restorer


def test_preset_restore_overrides_unit():
    assert preset_restore_overrides(None) == {}
    assert preset_restore_overrides("nope") == {}
    assert preset_restore_overrides("ratio") == {}  # no restore-side knobs
    pretty = preset_restore_overrides("pretty")
    assert set(pretty) == RESTORE_SIDE_PRESET_KEYS
    filtered = preset_restore_overrides(
        "pretty", frozenset({"aggressive.face_enhance_fidelity"})
    )
    assert "aggressive.face_enhance_fidelity" not in filtered
    assert "aggressive.face_enhance_backend" in filtered


# --------------------------------------------------------------------------- #
# I. Template + info
# --------------------------------------------------------------------------- #

def test_init_template_documents_preset():
    text = default_config_yaml()
    assert "preset:" in text
    for name in PRESET_NAMES:
        assert name in text
    # the line ships commented out: the template must still equal the defaults
    assert yaml.safe_load(text).get("preset") is None


def test_info_prints_preset(tmp_path):
    fkeep = _write_fkeep(_preset_cfg("share"), tmp_path, "s")
    res = CliRunner().invoke(cli, ["info", str(fkeep)])
    assert res.exit_code == 0, res.output
    assert "Preset:" in res.output
    assert "share" in res.output


def test_info_presetless_has_no_preset_line(tmp_path):
    fkeep = _write_fkeep(FaceKeepConfig(), tmp_path, "plain")
    res = CliRunner().invoke(cli, ["info", str(fkeep)])
    assert res.exit_code == 0, res.output
    assert "Preset:" not in res.output
