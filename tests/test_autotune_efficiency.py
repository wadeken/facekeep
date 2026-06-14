"""Auto-tune efficiency — ROADMAP Phase 3 (faithful mode).

Faithful auto-tune used to do a binary search of ~6 **full-image** encodes and
then *one more* full-image encode purely to re-attach EXIF/ICC the search had
stripped. The cost was dominated by encoding the whole image ~7 times.

Two changes cut that (``faithful._auto_tune_quality``):

1. **The search now probes a face-region crop, not the whole image.** The
   acceptance criterion is the face region's SSIM, so each binary-search probe
   encodes only a padded crop around the face(s). The probes shrink from
   full-resolution to a small tile; only the *final* encode is full-resolution.
2. **No separate metadata re-attach.** Probes encode without metadata (it does
   not affect SSIM); the single final full-image encode carries ``exif``/``icc``,
   so its bytes are already metadata-bearing and ``faithful.compress`` no longer
   does a wasted full re-encode.

The win is in **pixels encoded**, not call count (still ~6 probes + 1 final).
So the headline test measures total encoded *pixel area*: the crop-based search
must encode dramatically fewer pixels than the old "≈7 × full image" would, and
the single full-resolution encode must be the final one. The remaining tests
guard that the change did not cost correctness — metadata still round-trips on
both the face and no-face auto-tune paths (the face-target correctness itself is
covered in ``test_untested_paths.py``, which decodes the full-image output and
re-checks the face-region SSIM).

Bounds are expressed *relative* to the full-image pixel count (matching this
repo's regression-lock style), not magic absolutes, so they don't go flaky.
"""

import base64

import cv2
import numpy as np
import pytest
from PIL import Image

from facekeep import encoders, faithful
from facekeep.config import FaceKeepConfig

# Same compact, valid Display-P3 profile asset as tests/test_color.py /
# tests/test_error_handling.py (kept independent — no cross-test imports).
_DISPLAY_P3_ICC_B64 = (
    "AAABzGxjbXMEQAAAbW50clJHQiBYWVogAAAAAAAAAAAAAAAAYWNzcAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAPbWAAEAAAAA0y0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAJZGVzYwAAAPAAAAAyY3BydAAAASQAAAA4d3RwdAAAAVwAAAAUclhZ"
    "WgAAAXAAAAAUZ1hZWgAAAYQAAAAUYlhZWgAAAZgAAAAUclRSQwAAAawAAAAgZ1RSQwAAAawAAAAg"
    "YlRSQwAAAawAAAAgbWx1YwAAAAAAAAABAAAADGVuVVMAAAAWAAAAGABEAGkAcwBwAGwAYQB5ACAA"
    "UAAzAAAAAG1sdWMAAAAAAAAAAQAAAAxlblVTAAAAHAAAABgAUAB1AGIAbABpAGMAIABEAG8AbQBh"
    "AGkAbgAAWFlaIAAAAAAAAPbWAAEAAAAA0y1YWVogAAAAAAAAg94AAD2+////u1hZWiAAAAAAAABK"
    "vgAAsTYAAAq5WFlaIAAAAAAAACg7AAARCwAAyLlwYXJhAAAAAAADAAAAAmZmAADypwAADVkAABPQ"
    "AAAKWw=="
)
DISPLAY_P3_ICC = base64.b64decode(_DISPLAY_P3_ICC_B64)

requires_avif = pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)


def _draw_face(img, cx, cy, fw):
    """Same synthetic face recipe as conftest.face_image (Haar-detectable)."""
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


@pytest.fixture
def face_p3_jpeg(tmp_path):
    """A single-face JPEG on a smooth background, carrying a Display-P3 profile.

    conftest's ``face_image`` has faces but no ICC, and ``p3_jpeg`` has an ICC
    but no faces. The efficiency win is on the path where *both* the search and
    the (old) re-attach fire, which needs faces AND metadata together.

    The background is a smooth gradient (not textured noise): Haar finds the one
    real face and no distant false positives, so the padded face-union region
    stays a small, deterministic fraction of the frame — which is the case the
    crop-search optimization targets. (The spread-faces / false-positive case,
    where the union is large and the win shrinks, is still exercised for
    *correctness* by ``test_untested_paths.py`` on the textured fixture.)
    """
    H, W = 1200, 1600
    yy = np.linspace(70, 150, H)[:, None]
    xx = np.linspace(60, 140, W)[None, :]
    base = (yy + xx) / 2.0
    img = np.clip(np.stack([base, base * 0.96, base * 0.92], axis=-1), 0, 255).astype(
        np.uint8
    )
    _draw_face(img, 800, 600, 260)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "RGB")
    path = tmp_path / "family_p3.jpg"
    pil.save(str(path), "JPEG", quality=92, icc_profile=DISPLAY_P3_ICC)
    return path


def _image_area(path) -> int:
    """Full-resolution pixel area (H*W) of the source, via the load path."""
    from facekeep.imageio import load

    img = load(str(path)).image
    return int(img.shape[0]) * int(img.shape[1])


@pytest.fixture
def plain_p3_jpeg(tmp_path):
    """A faceless image carrying a Display-P3 ICC profile (no-face auto-tune)."""
    arr = np.full((64, 64, 3), (200, 140, 110), dtype=np.uint8)
    pil = Image.fromarray(arr, "RGB")
    path = tmp_path / "plain_p3.jpg"
    pil.save(str(path), "JPEG", quality=95, icc_profile=DISPLAY_P3_ICC)
    return path


def _track_encodes(monkeypatch):
    """Patch ``encoders.encode`` to record the pixel area of every encode.

    Returns a dict with ``calls`` (count) and ``areas`` (list of H*W per encode,
    in source order). Tracking pixel area — not just call count — is what
    exposes the crop-search win, since the call count is unchanged.
    """
    real = encoders.encode
    rec = {"calls": 0, "areas": []}

    def tracking(image, *args, **kwargs):
        rec["calls"] += 1
        rec["areas"].append(int(image.shape[0]) * int(image.shape[1]))
        return real(image, *args, **kwargs)

    monkeypatch.setattr(faithful.encoders, "encode", tracking)
    return rec


def _auto_tune_cfg():
    cfg = FaceKeepConfig()
    cfg.faithful.auto_tune = True
    return cfg


@requires_avif
def test_auto_tune_face_search_uses_small_crops(face_p3_jpeg, tmp_path, monkeypatch):
    """The face search must encode crops, with exactly one full-image encode.

    Old behavior: every probe (~6) encoded the full image, plus a full re-encode
    to re-attach metadata → total encoded area ≈ 7× the image. New behavior: the
    probes encode a small face crop and only the final returned encode is
    full-resolution. We assert (a) exactly one encode covers the full image (the
    final metadata-bearing one), (b) the rest are strictly smaller crops, and
    (c) the *total* encoded pixel volume is well under what 2 full images would
    cost — i.e. the search is no longer dominated by full-image encodes.
    """
    full_area = _image_area(face_p3_jpeg)
    rec = _track_encodes(monkeypatch)

    result = faithful.compress(str(face_p3_jpeg), str(tmp_path / "out"), _auto_tune_cfg())

    assert result.faces_detected >= 1, "fixture should yield a Haar face"
    assert not result.skipped
    assert rec["calls"] >= 2, "auto-tune should still run a multi-probe search"

    full_encodes = [a for a in rec["areas"] if a >= full_area]
    crop_encodes = [a for a in rec["areas"] if a < full_area]
    # Exactly one full-resolution encode: the final chosen-quality+metadata one.
    assert len(full_encodes) == 1, (
        f"expected exactly 1 full-image encode (the final one); "
        f"got {len(full_encodes)} of areas {rec['areas']} (full={full_area})"
    )
    # Every search probe is a strictly smaller crop — and meaningfully so (the
    # padded single-face union is well under half the frame here).
    assert crop_encodes, "search probes should encode crops, not the full image"
    assert all(a < full_area * 0.5 for a in crop_encodes), (
        f"search probes should be small crops (<50% of frame); "
        f"got {crop_encodes} (full={full_area})"
    )

    # Total encoded volume must be far below the old cost (~7× the full image:
    # 6 full probes + 1 re-attach). With a single small face the crop search is
    # ~1.9× full; assert < 3× as a robust ceiling that still proves the search is
    # no longer dominated by full-image encodes (and would fail if probes went
    # back to full resolution: that alone would be ≥ 6×).
    total = sum(rec["areas"])
    assert total < full_area * 3, (
        f"total encoded pixels {total} should be < 3x full image "
        f"({full_area * 3}); the search must not be re-encoding the full frame"
    )


@requires_avif
def test_auto_tune_no_face_path_single_full_encode(plain_p3_jpeg, tmp_path, monkeypatch):
    """No-face auto-tune is a single full-image, metadata-bearing encode.

    Pre-fix: 2 encodes (one at the configured quality + one re-attach). The
    no-bbox branch now embeds metadata in its single encode, so it is exactly 1.
    """
    rec = _track_encodes(monkeypatch)

    result = faithful.compress(str(plain_p3_jpeg), str(tmp_path / "out"), _auto_tune_cfg())

    assert result.faces_detected == 0, "fixture should be faceless"
    assert not result.skipped
    assert rec["calls"] == 1, (
        f"no-face auto-tune did {rec['calls']} encodes; expected 1 "
        f"(metadata embedded in the single encode, not re-attached)"
    )


@requires_avif
def test_auto_tune_face_path_preserves_icc(face_p3_jpeg, tmp_path):
    """The optimized face path still embeds the ICC profile in the output.

    Guards the efficiency change against silently dropping color: the bytes the
    search returns must be the metadata-bearing encode, not a stripped probe.
    """
    result = faithful.compress(str(face_p3_jpeg), str(tmp_path / "out"), _auto_tune_cfg())

    embedded = Image.open(str(result.output_path)).info.get("icc_profile")
    assert embedded == DISPLAY_P3_ICC


@requires_avif
def test_auto_tune_no_face_path_preserves_icc(plain_p3_jpeg, tmp_path):
    """The no-face auto-tune branch also still carries the ICC profile.

    This branch previously relied on the (now-removed) external re-attach block,
    so it is the one most at risk of regressing to stripped output.
    """
    result = faithful.compress(str(plain_p3_jpeg), str(tmp_path / "out"), _auto_tune_cfg())

    embedded = Image.open(str(result.output_path)).info.get("icc_profile")
    assert embedded == DISPLAY_P3_ICC
