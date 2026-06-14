"""Real-photo corpus tests — ROADMAP Phase 2.

The rest of the suite uses synthetic, geometrically-drawn images. These tests
exercise the pipeline on *real* photographs (license-clear, downloaded on
demand — see ``tests/corpus/``) so that detection, compression ratio, and
fidelity are verified against photographic content, not just drawn ellipses.

Every test skips when the corpus cache is absent (offline / CI without the
download), matching the project's offline-graceful convention. Run
``python tests/corpus/download.py`` once to enable them locally.

Assertions use **tolerant, evidence-based bounds** (measured on the actual
corpus at capture: ratios 1.23-1.94, SSIM 0.975-0.992 at default quality) with
margin, so they catch real regressions — files getting *bigger*, fidelity
*collapsing*, a portrait detecting *no* face — without going brittle across
codec/detector versions.
"""

import pytest

from facekeep import encoders, faithful, metrics
from facekeep.config import FaceKeepConfig
from facekeep.detector import create_detector
from facekeep.imageio import load

pytestmark = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


# --- Detection on real faces ------------------------------------------------ #


@pytest.mark.parametrize(
    "filename,min_faces",
    [
        ("obama_portrait.jpg", 1),  # single clear portrait
        ("einstein_head.jpg", 1),  # single portrait, grayscale source
        ("beatles_group.jpg", 3),  # 5 faces at capture; >=3 tolerates drift
    ],
)
def test_detects_faces_on_real_portraits(corpus_image, filename, min_faces):
    """Haar must find the expected faces on real photos (not just synthetic ones).

    The whole point of the corpus: synthetic ellipse-faces prove the plumbing,
    but only real faces prove detection actually works. Lower bounds (not exact
    counts) keep this robust to detector tuning while still failing if a real
    portrait suddenly detects nothing.
    """
    image = load(str(corpus_image(filename))).image
    detector = create_detector(backend="haar")
    faces = detector.detect(image)
    assert len(faces) >= min_faces, (
        f"{filename}: expected >= {min_faces} face(s), got {len(faces)}"
    )
    # Every reported face box must be inside the image.
    h, w = image.shape[:2]
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


@pytest.mark.parametrize("filename", ["snake_river.jpg", "hopetoun_falls.jpg"])
def test_faceless_landscapes_have_no_face_storm(corpus_image, filename):
    """A faceless landscape must not explode into many false faces.

    Haar can occasionally fire on texture (a known, documented trait), so this
    is deliberately not 'exactly zero' — it guards against a regression that
    makes detection wildly over-trigger on real scenery (which would spuriously
    flip faithful mode to 4:4:4 and auto-tune). At capture both detect 0.
    """
    image = load(str(corpus_image(filename))).image
    detector = create_detector(backend="haar")
    faces = detector.detect(image)
    assert len(faces) <= 1, f"{filename}: {len(faces)} false faces on a landscape"


# --- Compression ratio & fidelity on real photos --------------------------- #

_ALL = [
    "obama_portrait.jpg",
    "einstein_head.jpg",
    "beatles_group.jpg",
    "snake_river.jpg",
    "hopetoun_falls.jpg",
]


def _fixed_quality_config() -> FaceKeepConfig:
    """Default config but with auto-tune off (fixed quality).

    Auto-tune is on by default in production; these ratio/fidelity bounds were
    measured at fixed default quality, so pinning it off keeps the bounds
    meaningful (and avoids the ~6-probe search on every corpus image). The
    auto-tune path has its own tests.
    """
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = False
    return cfg


@pytest.mark.parametrize("filename", _ALL)
def test_faithful_compresses_real_photo_smaller(corpus_image, filename, tmp_path):
    """Faithful AVIF must produce a *smaller* file than the real JPEG input.

    The core faithful-mode promise on real content. These are already-JPEG
    Commons renders, so the win is modest (measured 1.23-1.94x), not the 2.5-3x
    headline — so we assert the honest bar: the output is smaller and was a real
    encode (not the skip-if-larger keep-original path).
    """
    src = corpus_image(filename)
    result = faithful.compress(str(src), str(tmp_path / "out"), _fixed_quality_config())

    assert not result.skipped, f"{filename}: unexpectedly kept the original"
    assert result.compressed_size < result.original_size
    assert result.ratio > 1.0
    assert result.output_path.suffix == ".avif"


@pytest.mark.parametrize("filename", _ALL)
def test_faithful_fidelity_on_real_photo(corpus_image, filename, tmp_path):
    """Decoded output must stay visually close to the real source.

    SSIM floor is set well below the measured range (0.975-0.992 at default
    quality) so it flags a genuine fidelity collapse — a broken codec path, a
    color/orientation bug mangling pixels — without false-failing on normal
    lossy variation or a codec-version shift.
    """
    src = corpus_image(filename)
    result = faithful.compress(str(src), str(tmp_path / "out"), _fixed_quality_config())

    original = load(str(src)).image
    decoded = encoders.decode(result.output_path.read_bytes())
    assert decoded.shape == original.shape  # no orientation/size mangling

    score = metrics.ssim(original, decoded)
    assert score > 0.95, f"{filename}: SSIM {score:.4f} below floor"
