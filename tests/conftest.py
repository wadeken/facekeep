"""Shared pytest fixtures."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_ai: opt out of the no-AI Restorer default; the test drives the real "
        "Real-ESRGAN/GFPGAN path (slow, downloads weights, skips offline).",
    )


# --- Deterministic, offline restore by default ----------------------------- #
# Restore tests assert against the *bicubic, no-GFPGAN* path ("no AI here"). That
# must hold whether or not the [ai] extra happens to be installed on the machine:
# the offline-first convention says the default test path never touches the
# network or a heavy model. So, by default, force every Restorer onto the non-AI
# path (upsampler/face-enhancer pre-resolved to None). The real AI integration
# test opts back in with @pytest.mark.real_ai.
@pytest.fixture(autouse=True)
def _force_bicubic_restore(request, monkeypatch):
    if request.node.get_closest_marker("real_ai"):
        return  # the integration test wants the genuine AI path

    from facekeep.aggressive import restorer as _restorer

    def _no_ai_upsampler(self):
        self._tried_init = True
        self._upsampler = None

    def _no_ai_enhancer(self):
        self._tried_face_init = True
        self._face_enhancer = None

    monkeypatch.setattr(_restorer.Restorer, "_init_upsampler", _no_ai_upsampler)
    monkeypatch.setattr(_restorer.Restorer, "_init_face_enhancer", _no_ai_enhancer)

# --- Real-photo corpus (ROADMAP Phase 2) ----------------------------------- #
# The corpus images are downloaded on demand (tests/corpus/download.py), not
# committed. These fixtures locate the cache and skip when it is absent, so the
# suite stays green offline / in CI without network. Run download.py once to
# enable the corpus tests locally.

_CORPUS_ROOT = Path(__file__).parent / "corpus"


def _corpus_cache_dir() -> Path:
    """Resolve the corpus cache dir using download.py's own logic (one source)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_corpus_download", _CORPUS_ROOT / "download.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cache_dir()


@pytest.fixture(scope="session")
def corpus_manifest():
    """The corpus manifest entries (list of dicts)."""
    data = json.loads((_CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return data["images"]


@pytest.fixture
def corpus_image(corpus_manifest):
    """Return a resolver: filename -> cached Path, skipping if not downloaded.

    Skips the test (rather than failing) when an image is missing, matching the
    project's offline-graceful convention (cf. the YuNet model tests).
    """
    cache = _corpus_cache_dir()

    def _get(filename: str) -> Path:
        path = cache / filename
        if not path.exists():
            pytest.skip(
                f"corpus image {filename!r} not downloaded "
                f"(run: python tests/corpus/download.py)"
            )
        return path

    return _get


@pytest.fixture
def face_image(tmp_path):
    """A synthetic image with Haar-detectable faces on a textured background."""
    rng = np.random.default_rng(3)
    H, W = 1200, 1800
    bg = cv2.resize(
        rng.normal(128, 30, (H // 10, W // 10, 3)).astype(np.float32),
        (W, H), interpolation=cv2.INTER_CUBIC,
    )
    img = np.clip(bg, 0, 255).astype(np.uint8)

    def draw_face(cx, cy, fw):
        fh = int(fw * 1.3)
        cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, (180, 170, 165), -1)
        cv2.ellipse(img, (cx, cy - fh // 6), (fw // 2 - 5, fh // 4), 0, 0, 360,
                    (195, 185, 180), -1)
        eye_y = cy - fh // 10
        ew = fw // 7
        cv2.ellipse(img, (cx - fw // 5, eye_y), (ew, ew // 2), 0, 0, 360, (60, 55, 55), -1)
        cv2.ellipse(img, (cx + fw // 5, eye_y), (ew, ew // 2), 0, 0, 360, (60, 55, 55), -1)
        cv2.line(img, (cx, eye_y), (cx, cy + fh // 12), (150, 140, 135), max(2, fw // 40))
        cv2.ellipse(img, (cx, cy + fh // 4), (fw // 5, fh // 18), 0, 0, 180, (120, 90, 90), -1)

    draw_face(650, 560, 300)
    path = tmp_path / "family.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


@pytest.fixture
def haar_texture_image():
    """A synthetic texture that makes the raw Haar cascade misfire.

    Blocky upscaled noise produces eye-like dark clusters the Haar cascade
    latches onto: with filtering disabled it reports several false "faces"
    (seed 25 gives 4 on this size). Returned as a BGR uint8 array (no file) so
    detector tests can feed it directly. Used to show the false-positive filter
    reduces spurious detections.
    """
    rng = np.random.default_rng(25)
    H, W = 768, 1024
    small = rng.integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (5, 5), 0)


@pytest.fixture
def plain_image(tmp_path):
    """A faceless smooth-gradient image (no face-like contrast for Haar)."""
    H, W = 1000, 1500
    # Smooth two-axis gradient: no eye-like dark spots, so Haar finds nothing.
    yy = np.linspace(60, 200, H)[:, None]
    xx = np.linspace(40, 160, W)[None, :]
    base = (yy + xx) / 2.0
    img = np.stack([base, base * 0.95, base * 0.9], axis=-1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    path = tmp_path / "landscape.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path
