"""Tests for the perceptual SSIMULACRA2 auto-tune metric — ROADMAP Phase 5.

SSIM correlates only loosely with perception and saturates on noisy content, so
it is a weak "the eye can't tell" acceptance target. SSIMULACRA2 is a perceptual
quality metric built to detect compression artifacts the way a human would
(higher = better; ~90 visually lossless). Faithful auto-tune can now use it via
``faithful.target_metric='ssimulacra2'``, falling back to SSIM when the optional
``ssimulacra2`` package (in ``[dev]``) is absent.

Like the LPIPS tests, these never depend on the real package's numbers: a fake
scoring function is injected on the metrics module (the real ``_init_ssimulacra2``
is the only place ``ssimulacra2`` is imported, and its ImportError path is the
graceful-degradation contract). What they pin:

* package absent (fn stays None) -> ``ssimulacra2_score`` returns None, no
  exception; ``ssimulacra2_available()`` False.
* a fake fn -> identical images score higher than a divergent pair (wiring +
  direction: higher = better).
* a computation error -> None (graceful), not a crash.
* ``_to_ssimulacra2_buffer`` is the single BGR->RGB boundary (a pure-blue BGR
  image must read as blue, not red).
* ``_auto_tune_quality`` actually *uses* the selected metric (a spy proves the
  SSIMULACRA2 scorer is called, not SSIM), and falls back to SSIM when the
  package is unavailable (warns, never crashes).
* ``validate()`` rejects an unknown target_metric and accepts ssim/ssimulacra2;
  YAML round-trips.
* CLI: ``quality --ssimulacra2`` hints when unavailable and prints a score line
  with a fake fn.
* One real-package end-to-end sanity (skips when ssimulacra2 isn't installed).
"""

import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import faithful, metrics
from facekeep.cli import cli
from facekeep.config import FaceKeepConfig
from facekeep.exceptions import ConfigError


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def _reset_s2_state():
    """Save/restore the module-level SSIMULACRA2 singleton so tests don't leak."""
    saved = (metrics._ssimulacra2_fn, metrics._tried_ssimulacra2_init)
    yield
    metrics._ssimulacra2_fn, metrics._tried_ssimulacra2_init = saved


def _inject(fn):
    metrics._ssimulacra2_fn = fn
    metrics._tried_ssimulacra2_init = True  # skip the real (importing) init


def _fake_score(raise_exc=None):
    """A stand-in for ``compute_ssimulacra2(buf_a, buf_b)``.

    Reads the two in-memory PNGs back and returns a higher score the more similar
    they are (mean abs pixel diff inverted), so it pins direction (higher =
    better) without the real algorithm.
    """
    from PIL import Image

    def _fn(buf_a, buf_b):
        if raise_exc is not None:
            raise raise_exc
        a = np.asarray(Image.open(buf_a).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(buf_b).convert("RGB"), dtype=np.float64)
        return 100.0 - float(np.mean(np.abs(a - b)))

    return _fn


# --------------------------------------------------------------------------- #
# graceful degradation
# --------------------------------------------------------------------------- #

def test_score_without_package_returns_none(_reset_s2_state):
    """Package absent (fn None) -> None, no exception."""
    _inject(None)
    a = np.full((32, 32, 3), 100, np.uint8)
    b = np.full((32, 32, 3), 120, np.uint8)

    assert metrics.ssimulacra2_score(a, b) is None


def test_available_reflects_injected_state(_reset_s2_state):
    _inject(None)
    assert metrics.ssimulacra2_available() is False
    _inject(_fake_score())
    assert metrics.ssimulacra2_available() is True


def test_computation_error_returns_none(_reset_s2_state):
    _inject(_fake_score(raise_exc=ValueError("bad shape")))
    a = np.full((32, 32, 3), 100, np.uint8)
    b = np.full((32, 32, 3), 150, np.uint8)

    assert metrics.ssimulacra2_score(a, b) is None


# --------------------------------------------------------------------------- #
# wiring + direction (fake fn)
# --------------------------------------------------------------------------- #

def test_identical_scores_higher_than_divergent(_reset_s2_state):
    _inject(_fake_score())
    a = np.full((32, 32, 3), 100, np.uint8)
    far = np.full((32, 32, 3), 200, np.uint8)

    s_same = metrics.ssimulacra2_score(a, a)
    s_far = metrics.ssimulacra2_score(a, far)

    assert s_same > s_far  # higher = better, identical is best


def test_to_buffer_is_bgr_to_rgb_boundary(_reset_s2_state):
    """A pure-blue BGR image must encode as blue (RGB), not red."""
    from PIL import Image

    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[:, :, 0] = 255  # blue channel (BGR index 0)

    buf = metrics._to_ssimulacra2_buffer(bgr)
    rgb = np.asarray(Image.open(buf).convert("RGB"))

    assert rgb[0, 0, 0] == 0    # R
    assert rgb[0, 0, 2] == 255  # B  -> the BGR->RGB swap happened


def test_to_buffer_downscales_uint16(_reset_s2_state):
    """uint16 source is scaled to 8-bit (the metric loads 8-bit RGB)."""
    from PIL import Image

    bgr = np.full((4, 4, 3), 65535, np.uint16)
    buf = metrics._to_ssimulacra2_buffer(bgr)
    rgb = np.asarray(Image.open(buf).convert("RGB"))

    assert rgb.dtype == np.uint8
    assert int(rgb.max()) == 255


# --------------------------------------------------------------------------- #
# auto-tune actually uses the selected metric
# --------------------------------------------------------------------------- #

def _face_cfg(metric):
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = True
    cfg.faithful.target_metric = metric
    return cfg


def _face_region_inputs():
    """One synthetic image + a single face covering most of it."""
    from facekeep.detector import FaceRegion

    img = np.full((64, 64, 3), 128, np.uint8)
    face = FaceRegion(
        id=0, bbox=(8, 8, 56, 56), padded_bbox=(4, 4, 60, 60), confidence=0.9
    )
    return img, [face]


def test_auto_tune_uses_ssimulacra2_when_selected(_reset_s2_state, monkeypatch):
    """target_metric='ssimulacra2' -> the search calls the SSIMULACRA2 scorer."""
    _inject(_fake_score())
    img, faces = _face_region_inputs()

    s2_calls = {"n": 0}
    real = metrics.ssimulacra2_score

    def _spy(a, b):
        s2_calls["n"] += 1
        return real(a, b)

    monkeypatch.setattr(metrics, "ssimulacra2_score", _spy)
    # SSIM must NOT be the scorer here.
    monkeypatch.setattr(
        metrics, "ssim",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("SSIM used, not S2")),
    )

    cfg = _face_cfg("ssimulacra2")
    cfg.faithful.target_value = 90.0  # SSIMULACRA2 scale
    faithful._auto_tune_quality(img, faces, cfg.faithful, has_faces=True)

    assert s2_calls["n"] > 0


def test_auto_tune_falls_back_to_ssim_when_unavailable(_reset_s2_state, monkeypatch, caplog):
    """ssimulacra2 selected but unavailable -> SSIM is used, a warning, no crash."""
    _inject(None)  # unavailable
    img, faces = _face_region_inputs()

    ssim_calls = {"n": 0}
    real_ssim = metrics.ssim

    def _spy(a, b):
        ssim_calls["n"] += 1
        return real_ssim(a, b)

    monkeypatch.setattr(metrics, "ssim", _spy)

    cfg = _face_cfg("ssimulacra2")
    cfg.faithful.target_value = 90.0  # would be nonsense as an SSIM target
    with caplog.at_level("WARNING"):
        data, q = faithful._auto_tune_quality(img, faces, cfg.faithful, has_faces=True)

    assert ssim_calls["n"] > 0  # SSIM did the scoring
    assert isinstance(data, bytes) and isinstance(q, int)  # produced an encode
    assert any("falling back to SSIM" in r.message for r in caplog.records)


def test_resolve_metric_explicit_ssim(_reset_s2_state):
    """target_metric='ssim' -> SSIM scorer, configured threshold kept.

    (The *default* metric is now 'ssimulacra2'; this pins the explicit-SSIM
    path — SSIM scorer, configured target, no rebase.)
    """
    cfg = FaceKeepConfig().faithful
    cfg.target_metric = "ssim"
    cfg.target_value = 0.985
    scorer, target = faithful._resolve_tune_metric(cfg)
    assert scorer is metrics.ssim
    assert target == cfg.target_value


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #

def test_validate_rejects_unknown_metric():
    cfg = FaceKeepConfig()
    cfg.faithful.target_metric = "butteraugli"  # not yet wired
    with pytest.raises(ConfigError, match="target_metric"):
        cfg.validate()


def test_validate_accepts_known_metrics():
    for m in ("ssim", "ssimulacra2"):
        cfg = FaceKeepConfig()
        cfg.faithful.target_metric = m
        cfg.validate()  # must not raise


def test_target_metric_yaml_roundtrip(tmp_path):
    cfg = FaceKeepConfig()
    cfg.faithful.target_metric = "ssimulacra2"
    cfg.faithful.target_value = 90.0
    p = tmp_path / "c.yaml"
    cfg.save(p)
    loaded = FaceKeepConfig.load(p)
    assert loaded.faithful.target_metric == "ssimulacra2"
    assert loaded.faithful.target_value == 90.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _two_images(tmp_path):
    import cv2

    a = np.full((48, 48, 3), 100, np.uint8)
    b = np.full((48, 48, 3), 130, np.uint8)
    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(pa), a)
    cv2.imwrite(str(pb), b)
    return pa, pb


def test_quality_s2_unavailable_hints_and_still_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "ssimulacra2_available", lambda: False)
    pa, pb = _two_images(tmp_path)

    res = CliRunner().invoke(cli, ["quality", str(pa), str(pb), "--ssimulacra2"])

    assert res.exit_code == 0
    assert "Overall SSIM:" in res.output
    assert "SSIMULACRA2 unavailable" in res.output
    assert "SSIMULACRA2:  " not in res.output


def test_quality_s2_prints_score(tmp_path, monkeypatch, _reset_s2_state):
    _inject(_fake_score())
    pa, pb = _two_images(tmp_path)

    res = CliRunner().invoke(cli, ["quality", str(pa), str(pb), "--ssimulacra2"])

    assert res.exit_code == 0
    assert "SSIMULACRA2:" in res.output
    assert "higher = better" in res.output


def test_quality_without_s2_flag_omits_it(tmp_path, monkeypatch):
    def _boom():
        raise AssertionError("ssimulacra2_available must not be called without flag")

    monkeypatch.setattr(metrics, "ssimulacra2_available", _boom)
    pa, pb = _two_images(tmp_path)

    res = CliRunner().invoke(cli, ["quality", str(pa), str(pb)])

    assert res.exit_code == 0
    assert "SSIMULACRA2" not in res.output


# --------------------------------------------------------------------------- #
# real package, end-to-end (skips when not installed)
# --------------------------------------------------------------------------- #

def test_real_ssimulacra2_identical_is_high(_reset_s2_state):
    """The real metric: identical images score ~100. Skips if package absent."""
    metrics._tried_ssimulacra2_init = False  # force a real init attempt
    if not metrics.ssimulacra2_available():
        pytest.skip("ssimulacra2 package not installed")

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)

    score = metrics.ssimulacra2_score(img, img)
    assert score is not None
    assert score == pytest.approx(100.0, abs=1.0)
