"""Color-profile (ICC) preservation tests.

Guards the Phase 1 [CORE-GOAL]: the pipeline must carry the source ICC profile
through to the output. Wide-gamut photos (Display P3 — standard on modern
phones) come out color-shifted if the profile is dropped, which violates
"imperceptible difference". These tests use a small embedded Display-P3 profile
as a test asset (this littlecms build cannot synthesize P3 from primaries), and
assert both that the profile bytes survive and that known color patches do not
visibly shift (ΔE within tolerance).
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image, ImageCms

from facekeep import encoders, faithful, imageio
from facekeep.config import FaceKeepConfig

# A compact, valid v4 RGB ICC profile with Display-P3 primaries and a parametric
# sRGB tone curve (460 bytes; round-trips byte-identical through AVIF). Built
# offline from published Display-P3 primaries and embedded here so the test does
# not depend on the platform being able to synthesize a P3 profile.
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

# Known color patches (RGB, in the image's own P3 encoding). Use unsaturated
# warm/skin tones: fully saturated primaries get gamut-clipped on the P3->sRGB
# conversion, so they don't move and can't tell a dropped profile from a kept
# one. These tones do move (drop-shift ΔE ~10-12) while staying well-preserved
# end-to-end (ΔE <2), which makes the check both meaningful and cleanly bounded.
PATCHES = [(210, 160, 140), (180, 120, 90), (200, 140, 110)]


@pytest.fixture
def p3_jpeg(tmp_path):
    """A JPEG carrying a Display-P3 ICC profile, with known color patches."""
    h = 64
    w = h * len(PATCHES)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for i, rgb in enumerate(PATCHES):
        arr[:, i * h : (i + 1) * h] = rgb
    pil = Image.fromarray(arr, "RGB")
    path = tmp_path / "p3.jpg"
    pil.save(str(path), "JPEG", quality=98, icc_profile=DISPLAY_P3_ICC)
    return path


def _patch_centers_bgr(image_bgr):
    """Sample the center pixel of each patch (returns BGR uint8 tuples)."""
    h = image_bgr.shape[0]
    out = []
    for i in range(len(PATCHES)):
        cx = i * h + h // 2
        out.append(tuple(int(v) for v in image_bgr[h // 2, cx]))
    return out


_SRGB = ImageCms.getOpenProfile(
    io.BytesIO(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
)


def _to_srgb(rgb, icc_bytes):
    """Render an (R,G,B) patch (in its own profile's space) into sRGB.

    Comparing two colors after converting both into a common space (sRGB) tells
    us whether they look the same on screen. We use sRGB rather than Lab because
    this littlecms build refuses to build a direct RGB->LAB transform.
    """
    prof = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
    one = Image.new("RGB", (1, 1), tuple(int(v) for v in rgb))
    out = ImageCms.profileToProfile(one, prof, _SRGB, outputMode="RGB")
    return np.array(out.getpixel((0, 0)), dtype=float)


def test_load_reads_icc(p3_jpeg):
    """imageio.load() carries the source ICC profile through."""
    loaded = imageio.load(str(p3_jpeg))
    assert loaded.icc is not None
    assert loaded.icc == DISPLAY_P3_ICC


def test_load_no_icc_returns_none(plain_image):
    """A plain JPEG with no profile yields icc=None (we don't invent one)."""
    loaded = imageio.load(str(plain_image))
    assert loaded.icc is None


def test_encode_embeds_icc():
    """encode(icc=...) embeds a profile that survives an AVIF round-trip."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:] = (140, 160, 210)  # BGR
    data = encoders.encode(img, "avif", quality=90, icc=DISPLAY_P3_ICC)
    out = Image.open(io.BytesIO(data)).info.get("icc_profile")
    assert out == DISPLAY_P3_ICC


def test_encode_without_icc_embeds_nothing():
    """No icc argument -> no profile baked into the output."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:] = (140, 160, 210)
    data = encoders.encode(img, "avif", quality=90)
    out = Image.open(io.BytesIO(data)).info.get("icc_profile")
    assert not out


def test_faithful_preserves_icc(p3_jpeg, tmp_path):
    """Faithful compress carries the ICC profile into the .avif output."""
    out_path = tmp_path / "out"
    result = faithful.compress(str(p3_jpeg), str(out_path), FaceKeepConfig())
    embedded = Image.open(str(result.output_path)).info.get("icc_profile")
    assert embedded == DISPLAY_P3_ICC


def test_no_visible_color_shift(p3_jpeg, tmp_path):
    """Known patches must not shift: ΔE(source, output) within tolerance.

    Renders each patch's source color (via the source profile) and the decoded
    output color (via the *embedded* output profile) into a common sRGB space
    and compares. Small ΔE means the color is preserved end-to-end. (Measured
    max ~1.7 on these patches; 5.0 leaves margin for codec quantization.)
    """
    out_path = tmp_path / "out"
    result = faithful.compress(str(p3_jpeg), str(out_path), FaceKeepConfig())

    decoded = encoders.decode(result.output_path.read_bytes())
    out_icc = Image.open(str(result.output_path)).info.get("icc_profile")
    assert out_icc is not None

    for src_rgb, bgr in zip(PATCHES, _patch_centers_bgr(decoded)):
        out_rgb = (bgr[2], bgr[1], bgr[0])
        src_srgb = _to_srgb(src_rgb, DISPLAY_P3_ICC)
        out_srgb = _to_srgb(out_rgb, out_icc)
        delta_e = float(np.linalg.norm(src_srgb - out_srgb))
        assert delta_e < 3.0, f"patch {src_rgb}: ΔE={delta_e:.2f} too large"


def test_dropped_profile_would_shift():
    """Sanity check on the test's own logic: misreading P3 as sRGB IS a shift.

    Guards against a false-green test — it proves the ΔE comparison actually
    fires when the profile is ignored, so test_no_visible_color_shift is
    meaningful. A dropped profile means P3 pixels get shown as if sRGB.
    (Measured ~6-12 on these patches; comfortably above the <3.0 preservation
    tolerance, so the two cases never overlap.)
    """
    for patch in PATCHES:
        correct = _to_srgb(patch, DISPLAY_P3_ICC)
        as_srgb = np.array(patch, dtype=float)  # profile dropped -> read as sRGB
        delta_e = float(np.linalg.norm(correct - as_srgb))
        assert delta_e > 4.0, f"patch {patch}: drop shift only ΔE={delta_e:.1f}"


# --- Chroma subsampling fidelity (faithful mode's face-aware 4:4:4) ----------
#
# Faithful mode's one *visible* face-aware decision is chroma: with faces it
# encodes 4:4:4 (full chroma) so skin tone and lip color stay crisp; otherwise
# 4:2:0 (half chroma resolution) for size. 4:2:0 bleeds color across sharp
# chroma edges — exactly what hurts red lips against skin. These tests pin that
# 4:4:4 actually reaches the encoder and measurably preserves chroma edges, and
# that the `auto` path selects it when faces are present. (IMPROVEMENTS.md flags
# this as overlapping the color tests; without it the 4:4:4 branch is untested.)

_CHROMA_EDGE = slice(64 // 2 - 3, 64 // 2 + 3)  # cols straddling the color seam


def _chroma_edge_image():
    """A vertical red|green seam — a hard chroma edge that 4:2:0 blurs.

    Red (lips) on the left, green on the right: the luma is similar across the
    seam but the chroma flips hard, so chroma subsampling shows up as color
    bleed in the boundary columns while luma detail is largely unaffected.
    """
    h = w = 64
    img = np.zeros((h, w, 3), dtype=np.uint8)  # BGR
    img[:, : w // 2] = (40, 40, 200)  # saturated red
    img[:, w // 2 :] = (40, 200, 40)  # saturated green
    return img


def _edge_chroma_error(chroma=None, *, has_faces=False):
    """Mean abs BGR error in the seam columns after an AVIF round-trip."""
    img = _chroma_edge_image()
    kwargs = {"has_faces": has_faces} if chroma is None else {"chroma": chroma}
    decoded = encoders.decode(encoders.encode(img, "avif", quality=90, **kwargs))
    return float(
        np.abs(
            decoded[:, _CHROMA_EDGE].astype(int) - img[:, _CHROMA_EDGE].astype(int)
        ).mean()
    )


@pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)
def test_444_preserves_chroma_edge_better_than_420():
    """4:4:4 must keep a hard color edge far cleaner than 4:2:0.

    Relative assertion (not an absolute threshold) so it is robust across
    pillow-avif versions: 4:4:4 error must be a small fraction of 4:2:0's. The
    second assertion is the anti-false-green guard — it proves 4:2:0 genuinely
    bleeds here, so the comparison is meaningful and not two near-zeros.
    (Measured: 4:2:0 ~10.4, 4:4:4 ~0.33 — ~30x cleaner.)
    """
    err_420 = _edge_chroma_error("420")
    err_444 = _edge_chroma_error("444")

    assert err_420 > 2.0, f"4:2:0 should bleed at the seam, got {err_420:.2f}"
    assert err_444 < err_420 * 0.5, (
        f"4:4:4 ({err_444:.2f}) should be much cleaner than 4:2:0 ({err_420:.2f})"
    )


@pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)
def test_auto_chroma_uses_444_when_faces_present():
    """The `auto` chroma decision must pick 4:4:4 iff faces are present.

    This is the faithful-mode face-aware behavior: a face in frame should turn
    on full chroma. We assert the *effect* (seam fidelity), not an internal
    string, so it stays true even if the subsampling plumbing changes: with
    faces, auto must match explicit 4:4:4; without faces, it must match 4:2:0.
    """
    err_444 = _edge_chroma_error("444")
    err_420 = _edge_chroma_error("420")
    err_auto_faces = _edge_chroma_error(has_faces=True)
    err_auto_nofaces = _edge_chroma_error(has_faces=False)

    # With faces, auto is as clean as explicit 4:4:4 (and far cleaner than 4:2:0).
    assert err_auto_faces < err_420 * 0.5
    assert abs(err_auto_faces - err_444) < 1.0
    # Without faces, auto behaves like 4:2:0 (the size-saving default).
    assert abs(err_auto_nofaces - err_420) < 1.0


@pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)
def test_icc_survives_alongside_444_chroma():
    """ICC preservation and 4:4:4 must coexist in one encode.

    The wide-gamut (ICC) and face-aware (4:4:4) paths both write into the same
    encoder `save_kwargs`; a P3 phone photo with a face exercises both at once.
    This guards against one silently dropping the other — the realistic case
    (most P3 photos have people in them) that neither single-axis test covers.
    """
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:] = (140, 160, 210)  # BGR skin-ish tone
    data = encoders.encode(
        img, "avif", quality=90, chroma="444", icc=DISPLAY_P3_ICC
    )
    out = Image.open(io.BytesIO(data)).info.get("icc_profile")
    assert out == DISPLAY_P3_ICC


# --- Chroma verification: 4:4:4 reaches libavif + the red-lips case ----------
#
# ROADMAP Phase 5 "Chroma verification": confirm `subsampling="4:4:4"` actually
# *propagates through pillow-avif to libavif*, and test chroma bleed on a
# red-lips patch (4:4:4 vs 4:2:0). The Phase 2 tests above already pin the core
# face-aware chroma *behavior* (a red|green seam shows the bleed; `auto` -> 4:4:4
# iff faces). These two add what Phase 2 didn't:
#   1. A *propagation* proof, not just an effect on a single seam. pillow-avif
#      exposes no subsampling field on decode (verified: `info` has only
#      timestamp/duration), so we can't read the tag back. Instead we use a
#      high-frequency *chroma-only* pattern (1px-wide red/green stripes at similar
#      luma): only genuine 4:4:4 reaching libavif can preserve per-pixel chroma —
#      if the plugin silently downgraded the request to 4:2:0, the 4:4:4 round
#      trip would smear just like 4:2:0. The gap is enormous (measured ~183x), so
#      it cannot pass unless full chroma truly propagated.
#   2. The *red-lips-on-skin* case the item names by content: a red lip bar on a
#      skin-tone field — luma nearly flat across the lip edge, chroma flips hard —
#      which is exactly the real-face artifact 4:4:4 exists to prevent ("skin
#      tone and lip color stay crisp").


def _chroma_checkerboard():
    """1px-wide vertical red/green stripes — a pure high-frequency chroma signal.

    Luma is similar across stripes (both ~mid), but the chroma alternates every
    column. 4:2:0 halves chroma resolution and collapses this to mush; 4:4:4
    keeps it. So a clean 4:4:4 round-trip here means full chroma actually reached
    the encoder, not merely that "the requested string was accepted".
    """
    h = w = 64
    img = np.zeros((h, w, 3), dtype=np.uint8)  # BGR
    red = (40, 40, 200)
    green = (40, 200, 40)
    for x in range(w):
        img[:, x] = red if x % 2 == 0 else green
    return img


def _red_lips_image():
    """A red lip bar on a skin-tone field (BGR) — the named red-lips case.

    Skin background with a horizontal red bar (the "lips"). The lip edge is a
    hard chroma transition at near-constant luma, the worst case for 4:2:0.
    """
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:] = (150, 180, 210)  # skin-ish BGR
    img[20:28, 8:40] = (60, 60, 200)  # red lips
    return img


def _full_frame_chroma_error(img, chroma):
    """Mean abs BGR error over the whole frame after an AVIF round-trip."""
    decoded = encoders.decode(encoders.encode(img, "avif", quality=95, chroma=chroma))
    return float(np.abs(decoded.astype(int) - img.astype(int)).mean())


@pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)
def test_444_subsampling_reaches_libavif():
    """4:4:4 must genuinely propagate to libavif, proven on per-pixel chroma.

    pillow-avif reports no subsampling tag on decode, so we prove propagation by
    effect on a signal only true 4:4:4 can survive: 1px chroma stripes. 4:4:4
    must be far cleaner than 4:2:0 (a big ratio), and 4:2:0 must genuinely smear
    (the anti-false-green guard, so the comparison isn't two near-zeros).
    (Measured: 4:4:4 ~0.33, 4:2:0 ~61 — ~180x.)
    """
    img = _chroma_checkerboard()
    err_444 = _full_frame_chroma_error(img, "444")
    err_420 = _full_frame_chroma_error(img, "420")

    assert err_420 > 10.0, (
        f"4:2:0 should collapse 1px chroma stripes, got {err_420:.2f}"
    )
    # If the 4:4:4 request had silently fallen back to 4:2:0, this would also be
    # ~err_420; requiring it to be a small fraction proves full chroma propagated.
    assert err_444 < err_420 * 0.1, (
        f"4:4:4 ({err_444:.2f}) must preserve per-pixel chroma far better than "
        f"4:2:0 ({err_420:.2f}); a near-equal value means 4:4:4 didn't reach libavif"
    )


@pytest.mark.skipif(
    not encoders.codec_available("avif"), reason="AVIF encoder not installed"
)
def test_red_lips_chroma_preserved_by_444():
    """The named red-lips case: 4:4:4 keeps the lip edge cleaner than 4:2:0.

    A red lip bar on skin — near-flat luma, hard chroma edge. 4:2:0 bleeds color
    across the lip boundary; 4:4:4 keeps it crisp. Assert the lip-edge rows are
    measurably cleaner under 4:4:4, with a 4:2:0-genuinely-bleeds guard.
    (Measured over the lip-edge band: 4:4:4 ~0.27, 4:2:0 ~3.6.)
    """
    img = _red_lips_image()
    band = slice(18, 30)  # rows straddling the top/bottom lip edges

    def edge_err(chroma):
        decoded = encoders.decode(
            encoders.encode(img, "avif", quality=95, chroma=chroma)
        )
        return float(
            np.abs(decoded[band].astype(int) - img[band].astype(int)).mean()
        )

    err_444 = edge_err("444")
    err_420 = edge_err("420")

    assert err_420 > 1.0, f"4:2:0 should bleed at the lip edge, got {err_420:.2f}"
    assert err_444 < err_420 * 0.5, (
        f"4:4:4 ({err_444:.2f}) should keep the lip edge cleaner than "
        f"4:2:0 ({err_420:.2f})"
    )
