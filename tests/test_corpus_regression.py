"""Corpus ratio/quality *regression lock* — ROADMAP Phase 2.

This is the precise counterpart to ``test_corpus.py``. That file asserts
tolerant *floors* ("smaller than the JPEG input", "SSIM > 0.95") — the
did-not-break bar. This file pins each real photo's compression ratio and SSIM
into a **per-file band** around the measured baseline, so a change that makes
files meaningfully *bigger* or quality meaningfully *worse* fails — even while
still clearing the loose floors. It catches gradual regression, not just
collapse.

Why bands, not a single global target: the corpus inputs are already-JPEG
Commons renders, so per-image ratios differ a lot (1.23-1.94) and a one-size
target would be meaningless. The baseline numbers below are measured at the
default codec at *fixed* quality (auto-tune off — see ``_baseline_config``) and
live in ``BASELINES``; the bands are expressed *relative* to them, so updating a
baseline moves its band automatically — no magic absolute thresholds. (Auto-tune
is on by default in production, but this lock deliberately pins the fixed-quality
encode the baselines were captured at; the auto-tune path is tested separately.)

Band policy (proposal A — balanced):
  * ratio in ``baseline * [0.90, 1.15]`` — the -10% floor catches files getting
    bigger; the +15% ceiling is loose because a *better* ratio is usually fine,
    but a runaway ratio (quality crashed) is caught by the SSIM band instead.
  * SSIM in ``[baseline - 0.015, baseline + 0.010]`` (upper clamped to 1.0) —
    the lower edge catches a quality slip; the tight upper edge catches an
    *anomalous* SSIM spike, which typically means a broken decode/test path or a
    ratio collapse, not real improvement.

These bands tolerate normal lossy variation and small codec/detector version
drift while still failing on a real regression. If a codec upgrade legitimately
shifts the numbers, re-measure and update ``BASELINES`` (keep it in sync with
the table in docs/IMPROVEMENTS.md, the single source of truth).

Skips with the rest of the corpus suite when the cache is absent (offline / CI
without ``python tests/corpus/download.py``).
"""

import pytest

from facekeep import encoders, faithful, metrics
from facekeep.config import FaceKeepConfig
from facekeep.imageio import load

pytestmark = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)

# Measured at default codec/quality (FaceKeepConfig, AVIF). Single source of
# truth shared with the table in docs/IMPROVEMENTS.md — keep them in sync.
BASELINES = {
    "obama_portrait.jpg": {"ratio": 1.94, "ssim": 0.975},
    "einstein_head.jpg": {"ratio": 1.66, "ssim": 0.979},  # grayscale source
    "beatles_group.jpg": {"ratio": 1.31, "ssim": 0.992},
    "snake_river.jpg": {"ratio": 1.23, "ssim": 0.989},  # faceless landscape
    "hopetoun_falls.jpg": {"ratio": 1.25, "ssim": 0.983},  # faceless landscape
}

# Relative band widths (proposal A). Expressed against the baseline so the
# acceptance window tracks the baseline automatically.
_RATIO_LO, _RATIO_HI = 0.90, 1.15  # multiplicative
_SSIM_LO, _SSIM_HI = 0.015, 0.010  # additive (subtract / add)


def _ratio_band(baseline_ratio: float) -> tuple[float, float]:
    return baseline_ratio * _RATIO_LO, baseline_ratio * _RATIO_HI


def _ssim_band(baseline_ssim: float) -> tuple[float, float]:
    return baseline_ssim - _SSIM_LO, min(1.0, baseline_ssim + _SSIM_HI)


def _baseline_config() -> FaceKeepConfig:
    """Config at the *fixed* default quality the BASELINES were measured at.

    Auto-tune is on by default in production, but these baselines pin the
    fixed-quality encode (the numbers in the IMPROVEMENTS table were captured at
    fixed q70). Pinning ``auto_tune=False`` keeps this regression lock measuring
    exactly what it documents; the auto-tune path has its own tests.
    """
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = False
    return cfg


# Compress each corpus image once per test; cheap enough (5 small photos) and
# keeps ratio vs. SSIM in separate tests so a size regression and a quality
# regression show up as distinct failures (cf. IMPROVEMENTS "size *or* quality").


@pytest.mark.parametrize("filename", sorted(BASELINES))
def test_ratio_within_regression_band(corpus_image, filename, tmp_path):
    """Compression ratio must stay inside its per-file band.

    Lower edge = a real encode that got *bigger* (regression); upper edge = a
    ratio that ballooned (usually quality crashing — the SSIM test pins that
    side too).
    """
    src = corpus_image(filename)
    result = faithful.compress(str(src), str(tmp_path / "out"), _baseline_config())
    assert not result.skipped, f"{filename}: kept original (skip-if-larger path)"

    base = BASELINES[filename]["ratio"]
    lo, hi = _ratio_band(base)
    assert lo <= result.ratio <= hi, (
        f"{filename}: ratio {result.ratio:.3f} outside band [{lo:.3f}, {hi:.3f}] "
        f"(baseline {base:.3f})"
    )


@pytest.mark.parametrize("filename", sorted(BASELINES))
def test_ssim_within_regression_band(corpus_image, filename, tmp_path):
    """Decoded-vs-original SSIM must stay inside its per-file band.

    Lower edge = a fidelity slip; upper edge (tight, clamped to 1.0) = an
    anomalous spike, which usually signals a broken decode/test path rather than
    genuine improvement.
    """
    src = corpus_image(filename)
    result = faithful.compress(str(src), str(tmp_path / "out"), _baseline_config())
    assert not result.skipped, f"{filename}: kept original (skip-if-larger path)"

    original = load(str(src)).image
    decoded = encoders.decode(result.output_path.read_bytes())
    assert decoded.shape == original.shape  # guard against orientation/size mangling

    score = metrics.ssim(original, decoded)
    base = BASELINES[filename]["ssim"]
    lo, hi = _ssim_band(base)
    assert lo <= score <= hi, (
        f"{filename}: SSIM {score:.4f} outside band [{lo:.4f}, {hi:.4f}] "
        f"(baseline {base:.4f})"
    )
