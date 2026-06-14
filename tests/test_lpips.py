"""Tests for the perceptual LPIPS metric — ROADMAP Phase 4.

Aggressive mode reconstructs (hallucinates) the background on restore, so the
acceptance question is "does it look wrong," not "do the pixels match." SSIM is
the wrong tool there; LPIPS (learned perceptual distance, lower = more similar)
scores the perceptual question. It is an *evaluation* tool, never on a pipeline
default path: it pulls torch (the ``[ai]`` extra) and downloads weights on first
use, so it is opt-in and degrades gracefully when absent.

These tests never import or download ``lpips``/torch: a fake model is injected
on the metrics module (the real ``_init_lpips`` is the only place ``lpips`` is
imported, and its ImportError path is the graceful-degradation contract). What
they pin:

* ``lpips`` not installed (model stays None) -> ``lpips_distance`` returns None,
  no exception; ``lpips_available()`` is False.
* A fake model -> identical images score ~0 and a bigger difference scores
  higher (wiring + monotonicity, not the real LPIPS numbers).
* An inference-time error -> None (graceful), not a crash.
* ``compare(with_lpips=False)`` (the default) leaves ``report.lpips`` None, so
  every existing caller is unchanged; ``with_lpips=True`` fills it.
* ``_to_lpips_tensor`` produces a (1, 3, H, W) RGB tensor in [-1, 1] (the single
  BGR->RGB boundary).
* CLI: ``quality --lpips`` with no package prints a hint and still reports
  SSIM/PSNR (no crash); with a fake model it prints an LPIPS line.
"""

import numpy as np
import pytest
from click.testing import CliRunner

from facekeep import metrics
from facekeep.cli import cli


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

class _FakeLpips:
    """Stands in for ``lpips.LPIPS``.

    Called as ``model(ta, tb)`` it returns an object with ``.item()`` giving a
    plausible distance: the mean absolute difference of the two [-1, 1] tensors
    (0 for identical inputs, larger as they diverge) — enough to pin wiring and
    monotonicity without the real network.
    """

    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, ta, tb):
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        import torch

        d = torch.mean(torch.abs(ta - tb))
        return d


@pytest.fixture
def _reset_lpips_state():
    """Save/restore the module-level LPIPS singleton so tests don't leak state."""
    saved = (metrics._lpips_model, metrics._tried_lpips_init)
    yield
    metrics._lpips_model, metrics._tried_lpips_init = saved


def _inject(model):
    metrics._lpips_model = model
    metrics._tried_lpips_init = True  # skip the real (lpips-importing) init


def _needs_torch():
    pytest.importorskip("torch", reason="torch needed to exercise the fake LPIPS path")


# --------------------------------------------------------------------------- #
# graceful degradation (no torch/lpips needed)
# --------------------------------------------------------------------------- #

def test_lpips_distance_without_package_returns_none(_reset_lpips_state):
    """No lpips installed (model None) -> None, no exception."""
    _inject(None)
    a = np.full((32, 32, 3), 100, np.uint8)
    b = np.full((32, 32, 3), 120, np.uint8)

    assert metrics.lpips_distance(a, b) is None


def test_lpips_available_reflects_injected_state(_reset_lpips_state):
    _inject(None)
    assert metrics.lpips_available() is False
    _inject(object())
    assert metrics.lpips_available() is True


def test_compare_without_lpips_is_default(_reset_lpips_state):
    """compare() default (with_lpips=False) never computes LPIPS."""
    _inject(_FakeLpips())  # present, but must not be used
    a = np.full((32, 32, 3), 100, np.uint8)
    b = np.full((32, 32, 3), 120, np.uint8)

    report = metrics.compare(a, b)

    assert report.lpips is None
    assert metrics._lpips_model.calls == 0


# --------------------------------------------------------------------------- #
# wiring + monotonicity (fake model, torch tensors)
# --------------------------------------------------------------------------- #

def test_lpips_identical_images_near_zero(_reset_lpips_state):
    _needs_torch()
    _inject(_FakeLpips())
    a = np.full((32, 32, 3), 100, np.uint8)

    assert metrics.lpips_distance(a, a) == pytest.approx(0.0, abs=1e-6)


def test_lpips_bigger_difference_scores_higher(_reset_lpips_state):
    _needs_torch()
    _inject(_FakeLpips())
    a = np.full((32, 32, 3), 100, np.uint8)
    near = np.full((32, 32, 3), 110, np.uint8)
    far = np.full((32, 32, 3), 220, np.uint8)

    d_near = metrics.lpips_distance(a, near)
    d_far = metrics.lpips_distance(a, far)

    assert d_far > d_near > 0


def test_lpips_inference_error_returns_none(_reset_lpips_state):
    _needs_torch()
    _inject(_FakeLpips(raise_exc=RuntimeError("CUDA OOM")))
    a = np.full((32, 32, 3), 100, np.uint8)
    b = np.full((32, 32, 3), 150, np.uint8)

    assert metrics.lpips_distance(a, b) is None


def test_compare_with_lpips_fills_report(_reset_lpips_state):
    _needs_torch()
    _inject(_FakeLpips())
    a = np.full((32, 32, 3), 100, np.uint8)
    b = np.full((32, 32, 3), 150, np.uint8)

    report = metrics.compare(a, b, with_lpips=True)

    assert report.lpips is not None
    assert report.lpips > 0


def test_to_lpips_tensor_shape_range_and_rgb(_reset_lpips_state):
    """BGR -> (1,3,H,W) RGB tensor in [-1, 1], channels swapped correctly."""
    _needs_torch()
    # A pure-blue BGR pixel image: B=255, G=0, R=0.
    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[:, :, 0] = 255  # blue channel (BGR index 0)

    t = metrics._to_lpips_tensor(bgr)

    assert tuple(t.shape) == (1, 3, 4, 6)
    assert float(t.min()) >= -1.0 and float(t.max()) <= 1.0
    # RGB order: channel 0 (R) should be the min (-1, from 0), channel 2 (B) the
    # max (+1, from 255) — proving the BGR->RGB swap happened.
    assert float(t[0, 0].mean()) == pytest.approx(-1.0, abs=1e-6)  # R
    assert float(t[0, 2].mean()) == pytest.approx(1.0, abs=1e-6)   # B


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


def test_quality_lpips_unavailable_hints_and_still_reports(tmp_path, monkeypatch):
    """--lpips with no package: a hint on stderr, SSIM/PSNR still printed, no crash."""
    monkeypatch.setattr(metrics, "lpips_available", lambda: False)
    pa, pb = _two_images(tmp_path)

    res = CliRunner().invoke(cli, ["quality", str(pa), str(pb), "--lpips"])

    assert res.exit_code == 0
    assert "Overall SSIM:" in res.output
    assert "LPIPS unavailable" in res.output
    assert "LPIPS:" not in res.output  # no score line when unavailable


def test_quality_lpips_prints_score_with_model(tmp_path, monkeypatch, _reset_lpips_state):
    _needs_torch()
    _inject(_FakeLpips())
    pa, pb = _two_images(tmp_path)

    res = CliRunner().invoke(cli, ["quality", str(pa), str(pb), "--lpips"])

    assert res.exit_code == 0
    assert "LPIPS:" in res.output
    assert "lower = more perceptually similar" in res.output


def test_quality_without_lpips_flag_omits_it(tmp_path, monkeypatch):
    """No --lpips: never available-checks, never prints an LPIPS line."""
    def _boom():
        raise AssertionError("lpips_available must not be called without --lpips")

    monkeypatch.setattr(metrics, "lpips_available", _boom)
    pa, pb = _two_images(tmp_path)

    res = CliRunner().invoke(cli, ["quality", str(pa), str(pb)])

    assert res.exit_code == 0
    assert "Overall SSIM:" in res.output
    assert "LPIPS" not in res.output
