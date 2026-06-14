"""Always-on *synthetic* regression lock — ROADMAP Phase 2 (CI-specific).

The real per-file ratio/SSIM/LPIPS guards live in ``test_corpus_regression.py``
and ``test_corpus_aggressive_regression.py``. But those run on a downloaded
real-photo corpus that is **not in the repo**, so on a network-less CI that never
runs ``tests/corpus/download.py`` they **silently skip** — leaving CI with no
end-to-end "did the pipeline collapse?" floor at all.

This file is that always-on floor. It compresses a *repo-generated synthetic*
image (the shared ``face_image`` fixture — no download, no ``[ai]`` extra) and
asserts only **crash-level** invariants for both modes:

  * faithful: the encode is genuinely smaller (``ratio > 1`` — not the
    skip-if-larger path), the decode keeps the source dimensions, and the
    decoded SSIM clears a *loose* floor;
  * aggressive: the ``.fkeep`` packs, ``verify_fkeep`` passes structurally, and a
    bicubic restore comes back at the original dimensions.

**This is deliberately NOT a tight band.** Synthetic images cannot prove visual
quality (see the IMPROVEMENTS "SSIM is not visually lossless" / corpus lessons),
so the floors here only catch a *collapse* — a pipeline that stopped compressing,
mangled dimensions, or produced an unreadable/inconsistent container. A *subtle*
regression (a few % of ratio, a small SSIM slip) is intentionally invisible here
and stays the corpus locks' job. The point is purely that **CI without the corpus
still fails on a broken core**, not that this replaces the corpus measurement.

It is correctly a near-no-op on a machine that *has* the corpus downloaded (this
verification box does), where the real corpus locks actually run — this just adds
a synthetic floor that runs everywhere, including bare CI.
"""

import zipfile

import cv2
import pytest

from facekeep import encoders, faithful, metrics
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import verify_fkeep, write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.config import FaceKeepConfig
from facekeep.imageio import load

pytestmark = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)

# Collapse floors, not a band. A healthy default encode of the synthetic
# textured-face fixture measures ratio ~1.8 and decoded SSIM ~0.99; these
# thresholds sit well below that so normal lossy/codec-version drift never trips
# them — only a genuine pipeline collapse does.
_RATIO_FLOOR = 1.2  # must still actually compress (well below the ~1.8 healthy value)
_SSIM_FLOOR = 0.90  # loose: catches a mangled decode, not a subtle fidelity slip


def test_faithful_does_not_collapse(face_image, tmp_path):
    """Default faithful compress on a synthetic photo stays sane end-to-end.

    Runs at the *production default* config (auto-tune on) — unlike the corpus
    lock, which pins fixed quality to keep a stable band. Here we only want the
    always-available default path to not collapse, so the default is what to
    guard.
    """
    result = faithful.compress(str(face_image), str(tmp_path / "out"), FaceKeepConfig())

    assert not result.skipped, "kept original (skip-if-larger) — encode did not compress"
    assert result.ratio > _RATIO_FLOOR, (
        f"faithful ratio {result.ratio:.3f} <= floor {_RATIO_FLOOR} — "
        "the encode stopped compressing this photo"
    )

    original = load(str(face_image)).image
    decoded = encoders.decode(result.output_path.read_bytes())
    assert decoded.shape == original.shape, (
        f"decoded shape {decoded.shape} != source {original.shape} — "
        "orientation/size mangled"
    )

    score = metrics.ssim(original, decoded)
    assert score > _SSIM_FLOOR, (
        f"decoded SSIM {score:.4f} <= floor {_SSIM_FLOOR} — fidelity collapsed"
    )


def test_aggressive_packs_verifies_and_restores(face_image, tmp_path):
    """Aggressive mode packs a valid .fkeep that verifies and restores in shape.

    Restore is the conftest bicubic path (no AI, offline) — the structural
    floor only needs the container to be self-consistent and the reconstruction
    to come back at the right dimensions, not to hit a perceptual target (that is
    the aggressive corpus lock's job, which needs LPIPS / the [ai] extra).
    """
    cfg = FaceKeepConfig()
    cfg.mode = "aggressive"

    photo = compress_photo(str(face_image), cfg)
    out = tmp_path / "out"
    write_fkeep(photo, str(out))
    fkeep = tmp_path / "out.fkeep"
    assert fkeep.exists() and zipfile.is_zipfile(str(fkeep))

    report = verify_fkeep(str(fkeep))
    assert report.ok, f".fkeep failed structural verify: {report.problems}"

    original = cv2.imread(str(face_image))
    restored = Restorer(cfg.aggressive).preview(str(fkeep))
    assert restored.shape == original.shape, (
        f"restored shape {restored.shape} != original {original.shape}"
    )
