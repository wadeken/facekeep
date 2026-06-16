"""ICC color-profile preservation through aggressive mode (.fkeep + restore).

Aggressive mode used to write its restored JPEG via ``cv2.imwrite``, which drops
the ICC profile entirely — so a Display-P3 photo restored duller (viewers fall
back to sRGB) even though the face *pixels* were intact. This is the
aggressive-mode counterpart of the faithful-mode ICC guard in ``test_color.py``:
the source profile must now be stored in the ``.fkeep`` (as ``icc.bin``) and
re-embedded on restore to jpg / avif / jxl.

These tests reuse the small embedded Display-P3 profile and the sRGB-render ΔE
helper from ``test_color.py`` rather than re-inventing them. They are offline:
the autouse ``_force_bicubic_restore`` fixture pins the no-AI restore path, so
nothing here needs torch/weights — they exercise compress + the write path.
"""

import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

from facekeep import encoders
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import read_fkeep, read_fkeep_info, verify_fkeep, write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.config import FaceKeepConfig

# Reuse the exact P3 profile, patch colors, and sRGB-render helper the faithful
# color tests use, so this guard stays consistent with that one.
from tests.test_color import DISPLAY_P3_ICC, PATCHES, _to_srgb


@pytest.fixture
def p3_photo(tmp_path):
    """A larger P3-profiled JPEG with flat warm-tone patches (no faces needed).

    Aggressive mode's default no_face_strategy is 'conservative', so a faceless
    image still produces a valid .fkeep; the ICC logic is face-independent, so
    flat patches keep color sampling deterministic. Sized so the downsampled
    background round-trips cleanly.
    """
    h = 256
    w = h * len(PATCHES)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for i, rgb in enumerate(PATCHES):
        arr[:, i * h : (i + 1) * h] = rgb
    path = tmp_path / "p3_photo.jpg"
    Image.fromarray(arr, "RGB").save(str(path), "JPEG", quality=98,
                                     icc_profile=DISPLAY_P3_ICC)
    return path


def _patch_center_rgb(pil_img):
    """Sample each patch's center pixel from a restored PIL image (RGB tuples)."""
    w, h = pil_img.size
    out = []
    for i in range(len(PATCHES)):
        cx = i * h + h // 2
        out.append(tuple(int(v) for v in pil_img.getpixel((cx, h // 2))))
    return out


# --- compress side: ICC is captured and stored -----------------------------

def test_compress_captures_icc(p3_photo):
    """compress_photo carries the source ICC onto the CompressedPhoto."""
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    assert photo.icc == DISPLAY_P3_ICC


def test_fkeep_stores_icc_member_and_flag(p3_photo, tmp_path):
    """A P3 .fkeep has an icc.bin member, icc_preserved True, current version."""
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    out = tmp_path / "p3.fkeep"
    write_fkeep(photo, str(out))

    m = read_fkeep_info(str(out))
    # ICC landed at 1.4.0; the current schema is 1.8.0 (the high-bit bit_depth key).
    assert m["version"] == "1.8.0"
    assert m["icc_preserved"] is True
    with zipfile.ZipFile(str(out)) as z:
        assert "icc.bin" in z.namelist()
        assert z.read("icc.bin") == DISPLAY_P3_ICC
    # read_fkeep surfaces it too.
    assert read_fkeep(str(out))["icc"] == DISPLAY_P3_ICC


def test_no_profile_image_has_no_icc_member(plain_image, tmp_path):
    """A plain (no-ICC) image: no icc.bin, icc_preserved False (anti-regression)."""
    photo = compress_photo(str(plain_image), FaceKeepConfig())
    assert photo.icc is None
    out = tmp_path / "plain.fkeep"
    write_fkeep(photo, str(out))

    m = read_fkeep_info(str(out))
    assert m["icc_preserved"] is False
    with zipfile.ZipFile(str(out)) as z:
        assert "icc.bin" not in z.namelist()


# --- restore side: ICC is re-embedded on every format -----------------------

def test_restore_jpg_reembeds_icc(p3_photo, tmp_path):
    """Restoring to .jpg re-embeds the ICC (was MISSING via the old cv2 path)."""
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    fk = tmp_path / "p3.fkeep"
    write_fkeep(photo, str(fk))

    out = tmp_path / "restored.jpg"
    Restorer().restore(str(fk), str(out))
    icc = Image.open(str(out)).info.get("icc_profile")
    assert icc == DISPLAY_P3_ICC


def test_preview_jpg_reembeds_icc(p3_photo, tmp_path):
    """preview() shares _write, so the bicubic preview re-embeds ICC too."""
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    fk = tmp_path / "p3.fkeep"
    write_fkeep(photo, str(fk))

    out = tmp_path / "preview.jpg"
    Restorer().preview(str(fk), str(out))
    assert Image.open(str(out)).info.get("icc_profile") == DISPLAY_P3_ICC


def test_restore_avif_reembeds_icc_byte_identical(p3_photo, tmp_path):
    """Restoring to .avif embeds the ICC byte-for-byte (libavif preserves it).

    AVIF carries the ICC blob verbatim (the faithful color tests rely on the
    same byte-identical round-trip).
    """
    if not encoders.codec_available("avif"):
        pytest.skip("avif plugin not available")
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    fk = tmp_path / "p3.fkeep"
    write_fkeep(photo, str(fk))

    out = tmp_path / "restored.avif"
    Restorer().restore(str(fk), str(out))
    assert Image.open(str(out)).info.get("icc_profile") == DISPLAY_P3_ICC


def test_restore_jxl_reembeds_valid_icc(p3_photo, tmp_path):
    """Restoring to .jxl embeds a valid ICC profile.

    Unlike AVIF, libjxl re-serializes the ICC from its internal color model on
    save (so the bytes are not identical — a codec property, not a FaceKeep bug).
    The contract we can assert is that a *valid, parseable* RGB profile is present
    (vs the old cv2 path, which embedded nothing at all).
    """
    if not encoders.codec_available("jxl"):
        pytest.skip("jxl plugin not available")
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    fk = tmp_path / "p3.fkeep"
    write_fkeep(photo, str(fk))

    out = tmp_path / "restored.jxl"
    Restorer().restore(str(fk), str(out))
    icc = Image.open(str(out)).info.get("icc_profile")
    assert icc is not None and len(icc) > 0
    # It parses as an ICC profile (would raise on garbage).
    ImageCms.ImageCmsProfile(io.BytesIO(icc))


def test_restore_jpg_preserves_exif(tmp_path):
    """The Pillow JPEG write must not regress EXIF re-embedding."""
    # Build a P3 + EXIF source.
    h = 128
    w = h * len(PATCHES)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for i, rgb in enumerate(PATCHES):
        arr[:, i * h : (i + 1) * h] = rgb
    exif = Image.Exif()
    exif[0x010F] = "FaceKeepTest"  # Make
    src = tmp_path / "p3_exif.jpg"
    Image.fromarray(arr, "RGB").save(
        str(src), "JPEG", quality=98, icc_profile=DISPLAY_P3_ICC, exif=exif,
    )

    photo = compress_photo(str(src), FaceKeepConfig())
    assert photo.exif is not None
    fk = tmp_path / "p3.fkeep"
    write_fkeep(photo, str(fk))

    out = tmp_path / "restored.jpg"
    Restorer().restore(str(fk), str(out))
    restored = Image.open(str(out))
    assert restored.info.get("icc_profile") == DISPLAY_P3_ICC
    assert restored.getexif().get(0x010F) == "FaceKeepTest"


# --- the real point: color does not visibly shift --------------------------

def test_restored_color_matches_original(p3_photo, tmp_path):
    """With the profile preserved, restored patches render to the same sRGB color.

    The whole reason to keep ICC: a viewer renders the restored file through its
    embedded profile, so the on-screen color matches the original. Verified by
    converting both the original patch (in P3) and the restored patch (through
    the restored file's *own* embedded profile) into a common sRGB space — the ΔE
    must be small. The anti-false-green companion proves the check would catch a
    dropped profile.
    """
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    fk = tmp_path / "p3.fkeep"
    write_fkeep(photo, str(fk))
    out = tmp_path / "restored.jpg"
    Restorer().restore(str(fk), str(out))

    restored = Image.open(str(out))
    restored_icc = restored.info.get("icc_profile")
    assert restored_icc is not None
    restored_patches = _patch_center_rgb(restored)

    for orig_rgb, rest_rgb in zip(PATCHES, restored_patches):
        orig_srgb = _to_srgb(orig_rgb, DISPLAY_P3_ICC)
        rest_srgb = _to_srgb(rest_rgb, restored_icc)
        de = float(np.linalg.norm(orig_srgb - rest_srgb))
        # Background was downsampled+upscaled (bicubic here) + JPEG re-encoded, so
        # allow some pixel drift; the point is the color space is honored, not
        # bit-exactness. A dropped profile shifts these warm tones ~10-12 ΔE.
        assert de < 6.0, f"patch shifted ΔE={de:.1f} ({orig_rgb}->{rest_rgb})"


def test_dropped_profile_would_shift_restored(p3_photo, tmp_path):
    """Anti-false-green: reading the restored patches as sRGB (no profile) shifts.

    If restore had dropped the profile (the old cv2 behavior), a viewer would
    read the same pixels as sRGB. That misinterpretation must move the warm-tone
    patches measurably, otherwise the preservation test above could pass vacuously.
    """
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    fk = tmp_path / "p3.fkeep"
    write_fkeep(photo, str(fk))
    out = tmp_path / "restored.jpg"
    Restorer().restore(str(fk), str(out))

    restored = Image.open(str(out))
    restored_patches = _patch_center_rgb(restored)

    shifted = []
    for orig_rgb, rest_rgb in zip(PATCHES, restored_patches):
        orig_srgb = _to_srgb(orig_rgb, DISPLAY_P3_ICC)
        as_srgb = np.array(rest_rgb, dtype=float)  # pixels treated as sRGB
        shifted.append(float(np.linalg.norm(orig_srgb - as_srgb)))
    assert max(shifted) > 5.0, f"dropped-profile shift too small: {shifted}"


# --- backward compatibility -------------------------------------------------

def _downgrade_to_130(src_fkeep, dst_fkeep):
    """Rewrite a .fkeep as a pre-ICC 1.3.0 file (strip icc.bin, drop the flag)."""
    with zipfile.ZipFile(str(src_fkeep)) as zin, \
            zipfile.ZipFile(str(dst_fkeep), "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            if item == "icc.bin":
                continue
            data = zin.read(item)
            if item == "manifest.json":
                m = json.loads(data)
                m["version"] = "1.3.0"
                m.pop("icc_preserved", None)
                data = json.dumps(m, indent=2).encode("utf-8")
            zout.writestr(item, data)


def test_legacy_130_fkeep_reads_verifies_restores(p3_photo, tmp_path):
    """A pre-1.4.0 .fkeep with no icc.bin reads, verifies, and restores cleanly."""
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    new = tmp_path / "new.fkeep"
    write_fkeep(photo, str(new))
    legacy = tmp_path / "legacy.fkeep"
    _downgrade_to_130(new, legacy)

    # Reads with icc=None, verifies OK, restores without crashing (no ICC).
    assert read_fkeep(str(legacy))["icc"] is None
    assert verify_fkeep(str(legacy)).ok
    out = tmp_path / "legacy_restored.jpg"
    Restorer().restore(str(legacy), str(out))
    assert Image.open(str(out)).info.get("icc_profile") is None


def test_bad_icc_blob_still_writes_pixels(p3_photo, tmp_path, monkeypatch):
    """A malformed ICC blob degrades to writing pixels (no lost restore)."""
    photo = compress_photo(str(p3_photo), FaceKeepConfig())
    fk = tmp_path / "p3.fkeep"
    write_fkeep(photo, str(fk))

    # Drive _write directly with a garbage ICC blob (simpler than corrupting the
    # archive): the metadata embed should fail and fall back to writing pixels.
    r = Restorer()
    data = read_fkeep(str(fk))
    upscaled = cv2.resize(
        data["background"], (photo.original_width, photo.original_height)
    )
    result = r._composite(upscaled, data)
    out = tmp_path / "restored.jpg"
    r._write(result, str(out), data.get("exif"), icc=b"not-a-real-icc-profile")
    # Pixels were still written despite the bad profile.
    assert out.exists()
    img = Image.open(str(out))
    assert img.size == (photo.original_width, photo.original_height)
