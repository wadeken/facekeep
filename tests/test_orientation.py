"""EXIF orientation correctness across input formats.

`imageio.load()` must return upright pixels for every supported format, applying
the EXIF orientation exactly once. The non-JPEG plugin formats (HEIC/AVIF/JXL)
are the interesting cases: piexif cannot parse those containers, so orientation
must be read off the decoded Pillow image, and different plugins differ in
whether they pre-rotate the pixels — we must not double-rotate. These tests are
the regression guard for that (ROADMAP Phase 1 / Phase 2 orientation matrix).
"""

import cv2
import numpy as np
import piexif
import pytest
from PIL import Image, ImageOps

from facekeep import imageio
from facekeep.config import FaceKeepConfig
from facekeep.faithful import compress as faithful_compress

# (format-name, extension, Pillow save format, importorskip module). The plugin
# formats are skipped (not failed) where the plugin is missing, so the suite
# stays green on a minimal install while still covering everything where present.
FORMATS = [
    ("JPEG", ".jpg", "JPEG", None),
    ("HEIC", ".heic", "HEIF", "pillow_heif"),
    ("AVIF", ".avif", "AVIF", "pillow_avif"),
    ("JXL", ".jxl", "JXL", "pillow_jxl"),
]


def _exif(orientation: int) -> bytes:
    return piexif.dump(
        {"0th": {piexif.ImageIFD.Orientation: orientation},
         "Exif": {}, "1st": {}, "thumbnail": None, "GPS": {}, "Interop": {}}
    )


def _write(path, save_fmt, plugin_mod, orientation):
    """Write an upright 80x40 image (left=red, right=blue) with an EXIF tag."""
    if plugin_mod:
        pytest.importorskip(plugin_mod)
        if save_fmt == "HEIF":
            import pillow_heif
            pillow_heif.register_heif_opener()
        elif save_fmt == "AVIF":
            import pillow_avif  # noqa: F401
        elif save_fmt == "JXL":
            import pillow_jxl  # noqa: F401

    arr = np.zeros((40, 80, 3), dtype=np.uint8)  # H=40, W=80 -> landscape
    arr[:, :40] = (255, 0, 0)   # left half red (RGB)
    arr[:, 40:] = (0, 0, 255)   # right half blue (RGB)
    Image.fromarray(arr, "RGB").save(str(path), save_fmt, exif=_exif(orientation))


@pytest.mark.parametrize("name,ext,save_fmt,plugin", FORMATS)
def test_orientation_6_makes_upright_portrait(name, ext, save_fmt, plugin, tmp_path):
    # Orientation 6 = "rotate 90° CW to display": an 80x40 landscape source must
    # come back as a 40x80 portrait, applied exactly once.
    p = tmp_path / f"o6{ext}"
    _write(p, save_fmt, plugin, 6)
    loaded = imageio.load(str(p))
    assert (loaded.width, loaded.height) == (40, 80), f"{name} not upright"


@pytest.mark.parametrize("name,ext,save_fmt,plugin", FORMATS)
def test_orientation_1_unchanged(name, ext, save_fmt, plugin, tmp_path):
    p = tmp_path / f"o1{ext}"
    _write(p, save_fmt, plugin, 1)
    loaded = imageio.load(str(p))
    assert (loaded.width, loaded.height) == (80, 40), f"{name} should be untouched"


# AVIF is excluded from the content-direction check below: pillow-avif rewrites
# the EXIF orientation tag on *save* (it stores 6 as 8, a 180° difference), so a
# test fixture written via pillow-avif can't carry a faithful orientation. The
# load path itself is correct — dimensions round-trip (tested above), and the
# orientation transform table is verified directly against Pillow for all 8 tags
# (test_orientation_ops_match_pillow). A real camera AVIF has a correct tag.
CONTENT_FORMATS = [f for f in FORMATS if f[0] != "AVIF"]


@pytest.mark.parametrize("name,ext,save_fmt,plugin", CONTENT_FORMATS)
def test_orientation_6_content_direction(name, ext, save_fmt, plugin, tmp_path):
    # Content check (not just dimensions) catches wrong-direction or double
    # rotation. Ground truth (Pillow exif_transpose) for orientation 6 puts the
    # original LEFT (red) stripe at the TOP and RIGHT (blue) at the BOTTOM.
    p = tmp_path / f"o6c{ext}"
    _write(p, save_fmt, plugin, 6)
    img = imageio.load(str(p)).image  # BGR, upright
    assert img.shape[:2] == (80, 40)
    top = img[:5, :, :].reshape(-1, 3).mean(0)      # BGR
    bottom = img[-5:, :, :].reshape(-1, 3).mean(0)
    # Lossy codecs shift values slightly; assert the dominant channel, not exact.
    assert top[2] > top[0], f"{name}: top should be red (R>B)"
    assert bottom[0] > bottom[2], f"{name}: bottom should be blue (B>R)"


def test_orientation_ops_match_pillow():
    # The orientation transform table must match Pillow's canonical
    # ImageOps.exif_transpose for every tag. Done directly on arrays (no codec
    # round-trip), so it can't be skewed by a plugin mangling the EXIF tag — it
    # pins the transform direction for all 8 orientations.
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    rgb[:, :40] = (255, 0, 0)
    rgb[:, 40:] = (0, 0, 255)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for o in range(1, 9):
        op = imageio._ORIENTATION_OPS.get(o, lambda x: x)
        ours_rgb = cv2.cvtColor(op(bgr), cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb, "RGB")
        ex = pil.getexif()
        ex[0x0112] = o
        pil.info["exif"] = ex.tobytes()
        gt = np.array(ImageOps.exif_transpose(pil).convert("RGB"))

        assert ours_rgb.shape == gt.shape, f"orientation {o} shape mismatch"
        assert np.array_equal(ours_rgb, gt), f"orientation {o} differs from Pillow"


def test_exif_orientation_normalized_to_1(tmp_path):
    # The preserved EXIF must report orientation 1 (we rotated the pixels), so a
    # downstream re-embed doesn't rotate a second time. JPEG carries EXIF bytes.
    p = tmp_path / "o6.jpg"
    _write(p, "JPEG", None, 6)
    loaded = imageio.load(str(p))
    assert loaded.exif is not None
    tag = piexif.load(loaded.exif).get("0th", {}).get(piexif.ImageIFD.Orientation)
    assert tag == 1


# --- End-to-end through faithful compress() --------------------------------
#
# The tests above stop at imageio.load(). These run the full faithful pipeline
# (compress -> AVIF -> decode) for every EXIF orientation and assert the *output*
# is upright. This is the regression guard ROADMAP Phase 2 asks for ("cover all
# 8 orientations, run through compress, assert upright + EXIF preserved").
#
# Inputs are JPEG only: EXIF is controllable, there's no plugin dependency, and
# the imageio.load() layer above already covers the non-JPEG formats. EXIF
# preservation is verified as "the output decodes upright at the correct size"
# rather than by comparing the output's orientation tag, because pillow-avif
# rewrites that tag on save (see the CONTENT_FORMATS note above) — the meaningful
# invariant is that the pixels come out upright, applied exactly once.


def _decode_output(path) -> np.ndarray:
    """Decode a faithful-mode AVIF output back to a BGR array."""
    import pillow_avif  # noqa: F401
    rgb = np.array(Image.open(str(path)).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _upright_size(orientation: int) -> tuple[int, int]:
    """Ground-truth (width, height) after applying an orientation to the 80x40
    base, per Pillow's canonical exif_transpose. Computed, not hard-coded, so the
    assertion pins 'compress output == Pillow's upright size' for each tag."""
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    pil = Image.fromarray(rgb, "RGB")
    ex = pil.getexif()
    ex[0x0112] = orientation
    pil.info["exif"] = ex.tobytes()
    return ImageOps.exif_transpose(pil).size  # (width, height)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_compress_orientation_matrix_upright(orientation, tmp_path):
    # Every EXIF orientation, end-to-end: the decoded output's dimensions must
    # match Pillow's upright dimensions for that tag (portrait for 5-8, landscape
    # for 1-4). Catches missing/extra/wrong rotation in the full pipeline.
    p = tmp_path / f"o{orientation}.jpg"
    _write(p, "JPEG", None, orientation)
    res = faithful_compress(str(p), str(tmp_path / f"out{orientation}"), FaceKeepConfig())
    decoded = _decode_output(res.output_path)
    h, w = decoded.shape[:2]
    assert (w, h) == _upright_size(orientation), (
        f"orientation {orientation}: output {(w, h)} != upright {_upright_size(orientation)}"
    )


def test_compress_orientation_actually_applied(tmp_path):
    # Anti-false-green: if orientation were silently ignored, an orientation-6
    # source would come out landscape (the naive 80x40) instead of the upright
    # portrait (40x80). Assert the two differ, proving imageio.load() read the tag
    # and the pipeline applied it — so the matrix test above can't pass by luck.
    naive = (80, 40)  # source dimensions if orientation were dropped
    assert _upright_size(6) != naive  # sanity: orientation 6 *should* rotate

    p = tmp_path / "o6.jpg"
    _write(p, "JPEG", None, 6)
    res = faithful_compress(str(p), str(tmp_path / "out6"), FaceKeepConfig())
    h, w = _decode_output(res.output_path).shape[:2]
    assert (w, h) != naive, "orientation 6 was not applied (output is the naive size)"
    assert (w, h) == _upright_size(6)


def test_compress_orientation_6_content_direction(tmp_path):
    # Content check through the full pipeline (not just dimensions): for
    # orientation 6, Pillow's ground truth puts the original LEFT (red) stripe at
    # the TOP and RIGHT (blue) at the BOTTOM. Catches wrong-direction / double
    # rotation that a dimensions-only check (40x80 is symmetric to a 180° error)
    # would miss. Lossy AVIF shifts values, so assert the dominant channel only.
    p = tmp_path / "o6c.jpg"
    _write(p, "JPEG", None, 6)
    res = faithful_compress(str(p), str(tmp_path / "out6c"), FaceKeepConfig())
    img = _decode_output(res.output_path)  # BGR, upright
    assert img.shape[:2] == (80, 40)
    top = img[:5, :, :].reshape(-1, 3).mean(0)      # BGR
    bottom = img[-5:, :, :].reshape(-1, 3).mean(0)
    assert top[2] > top[0], "top should be red (R>B)"
    assert bottom[0] > bottom[2], "bottom should be blue (B>R)"
