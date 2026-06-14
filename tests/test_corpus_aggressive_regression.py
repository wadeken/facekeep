"""Aggressive-mode ratio/restore-quality *regression lock* on a real photo.

The counterpart to ``test_corpus_regression.py``, which locks *faithful* mode
(ratio + decoded SSIM). That left aggressive mode — the project's distinctive
consumer feature — with **no numeric regression guard at all**, and measured
with the wrong tool besides (SSIM, which penalizes a plausibly-reconstructed
background; aggressive must be judged perceptually). This file closes both gaps:
it pins one real photo's **aggressive `.fkeep` ratio** and its **restore LPIPS**
(learned perceptual distance, *lower = better* — the right metric for a
hallucinated-but-plausible background) into per-metric bands around a measured
baseline.

**Why a dedicated photo (``migrant_mother.jpg``), not the existing corpus.** The
other corpus images are 800px Commons renders where faces (plus padding) fill a
large fraction of the frame, so aggressive mode's downsample saves little and the
``.fkeep`` is often *larger* than the input (ratio < 1) — that is the mode used
*outside* its design point, not a meaningful thing to lock. ``migrant_mother.jpg``
is a large frame (3840x4929) where the faces are a small fraction, so the
downsample genuinely shrinks the file (ratio ~1.40) and region-local conservatism
fires (one large + one small/distant face). It is the honest "aggressive mode
doing its job" case.

**Restore is bicubic here, by design.** The autouse ``_force_bicubic_restore``
fixture (conftest) pins the restorer to the no-AI bicubic path for every
non-``real_ai`` test, so this lock measures the *bicubic* reconstruction —
offline, deterministic, and a conservative proxy (real Real-ESRGAN looks at least
as good). That matches what ``facekeep bench`` reports by default; both numbers
are reproducible (verified: ratio and LPIPS are byte-deterministic across runs).

**Gating.** Skips with the rest of the corpus suite when the cache is absent
(offline / CI without ``python tests/corpus/download.py``), *and* skips the LPIPS
assertion when the optional ``[ai]`` extra is missing (``lpips`` not installed) —
the same offline-graceful convention as the ``real_ai`` tests. The ratio band
runs whenever the corpus is present (it needs no extra).

Baselines are measured at the default config (auto-tune on, the production
default) against the cached corpus bytes, and the bands are expressed *relative*
to the baseline so updating a number moves its window automatically. If a codec/
plugin upgrade legitimately shifts the numbers, re-measure and update
``BASELINES`` (kept in sync with the IMPROVEMENTS corpus notes).
"""

import pytest

from facekeep import bench, encoders, metrics
from facekeep.config import FaceKeepConfig

pytestmark = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)

# The single photo that exercises aggressive mode at its design point. Baseline
# measured at the default FaceKeepConfig() on the cached corpus bytes:
#   ratio 1.4020, restore_lpips 0.4219 (bicubic restore), deterministic.
_FILE = "migrant_mother.jpg"
BASELINES = {
    _FILE: {"ratio": 1.402, "restore_lpips": 0.422},
}

# Relative band widths. Ratio mirrors the faithful lock's policy (multiplicative
# [-10%, +15%]). LPIPS is additive and *lower = better*, so the band is widened
# on the high side (a real quality regression raises LPIPS) and tighter on the
# low side (an anomalous drop usually means a broken restore/test path, not
# genuine improvement). The numbers are deterministic, so these are comfortable.
_RATIO_LO, _RATIO_HI = 0.90, 1.15  # multiplicative
_LPIPS_LO, _LPIPS_HI = 0.05, 0.07  # additive (subtract / add)


def _ratio_band(base: float) -> tuple[float, float]:
    return base * _RATIO_LO, base * _RATIO_HI


def _lpips_band(base: float) -> tuple[float, float]:
    return max(0.0, base - _LPIPS_LO), base + _LPIPS_HI


def _measure(path) -> bench.BenchRow:
    """Run the aggressive benchmark on one photo (bicubic restore via fixture).

    Reuses ``bench.run_benchmark`` so this lock measures exactly what
    ``facekeep bench`` reports — one source of truth for the numbers.
    """
    rows = bench.run_benchmark([path], ["aggressive"], FaceKeepConfig())
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "ok", f"{path}: aggressive bench failed: {row.error}"
    return row


def test_aggressive_ratio_within_regression_band(corpus_image):
    """The aggressive .fkeep ratio must stay inside its band (and above 1.0).

    Lower edge catches the file getting *bigger* (a protection over-firing, a
    crop bloating); the >1.0 floor pins that aggressive mode is still actually
    compressing this photo (its whole point).
    """
    src = corpus_image(_FILE)
    row = _measure(src)

    base = BASELINES[_FILE]["ratio"]
    lo, hi = _ratio_band(base)
    assert row.ratio is not None
    assert lo <= row.ratio <= hi, (
        f"{_FILE}: aggressive ratio {row.ratio:.3f} outside band "
        f"[{lo:.3f}, {hi:.3f}] (baseline {base:.3f})"
    )
    assert row.ratio > 1.0, (
        f"{_FILE}: aggressive ratio {row.ratio:.3f} <= 1.0 — the .fkeep is no "
        "smaller than the original; aggressive mode is not compressing it."
    )


def test_aggressive_restore_lpips_within_regression_band(corpus_image):
    """The restored image's perceptual distance must stay inside its band.

    LPIPS (lower = better) is the acceptance metric for a reconstructed
    background. Skips when the [ai] extra (``lpips``) is unavailable — the same
    graceful-degradation convention as the real-model tests; the ratio test
    above still guards size in that case.
    """
    if not metrics.lpips_available():
        pytest.skip(
            "LPIPS unavailable (install the [ai] extra: pip install facekeep[ai]) "
            "— the aggressive restore-quality lock needs it."
        )

    src = corpus_image(_FILE)
    row = _measure(src)

    assert row.restore_lpips is not None, (
        f"{_FILE}: LPIPS reported available but the bench produced no score."
    )
    base = BASELINES[_FILE]["restore_lpips"]
    lo, hi = _lpips_band(base)
    assert lo <= row.restore_lpips <= hi, (
        f"{_FILE}: restore LPIPS {row.restore_lpips:.4f} outside band "
        f"[{lo:.4f}, {hi:.4f}] (baseline {base:.4f}; lower = better)"
    )
