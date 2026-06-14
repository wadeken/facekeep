"""End-to-end tests for faithful and aggressive modes."""

import cv2
import numpy as np

from facekeep import metrics
from facekeep.aggressive.compressor import compress_photo
from facekeep.aggressive.format import read_fkeep_info, write_fkeep
from facekeep.aggressive.restorer import Restorer
from facekeep.config import FaceKeepConfig
from facekeep.faithful import compress as faithful_compress


class TestFaithfulMode:
    def test_produces_standard_image(self, face_image, tmp_path):
        res = faithful_compress(str(face_image), str(tmp_path / "out"), FaceKeepConfig())
        assert res.output_path.exists()
        assert res.output_path.suffix == ".avif"
        # Output must be openable as a normal image
        from PIL import Image
        import pillow_avif  # noqa: F401
        img = Image.open(res.output_path)
        assert img.size[0] > 0

    def test_high_fidelity(self, face_image, tmp_path):
        """Faithful mode should be visually lossless (high SSIM vs original)."""
        cfg = FaceKeepConfig()
        cfg.faithful.quality = 75
        res = faithful_compress(str(face_image), str(tmp_path / "out"), cfg)

        from PIL import Image
        import pillow_avif  # noqa: F401
        decoded = cv2.cvtColor(
            np.array(Image.open(res.output_path).convert("RGB")), cv2.COLOR_RGB2BGR
        )
        original = cv2.imread(str(face_image))
        report = metrics.compare(original, decoded)
        assert report.overall_ssim > 0.95

    def test_smaller_than_original(self, face_image, tmp_path):
        res = faithful_compress(str(face_image), str(tmp_path / "out"), FaceKeepConfig())
        assert res.compressed_size < res.original_size

    def test_jxl_codec(self, face_image, tmp_path):
        from facekeep import encoders
        if not encoders.codec_available("jxl"):
            return
        cfg = FaceKeepConfig()
        cfg.faithful.codec = "jxl"
        res = faithful_compress(str(face_image), str(tmp_path / "out"), cfg)
        assert res.output_path.suffix == ".jxl"


def _tiny_incompressible_png(path) -> int:
    """Write a tiny random PNG whose AVIF re-encode is *larger* (container
    overhead dominates), so the skip-if-larger guard triggers. Returns its size.
    """
    img = np.random.default_rng(1).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path.stat().st_size


class TestSkipIfLarger:
    def test_keeps_original_when_encode_not_smaller(self, tmp_path):
        src = tmp_path / "tiny.png"
        in_size = _tiny_incompressible_png(src)

        res = faithful_compress(str(src), str(tmp_path / "out"), FaceKeepConfig())

        assert res.skipped is True
        assert res.compressed_size == res.original_size == in_size
        assert res.ratio == 1.0
        # Output exists, keeps the *source* extension, and is byte-identical.
        assert res.output_path.exists()
        assert res.output_path.suffix == ".png"
        assert res.output_path.read_bytes() == src.read_bytes()

    def test_disabled_writes_codec_output(self, tmp_path):
        src = tmp_path / "tiny.png"
        _tiny_incompressible_png(src)

        cfg = FaceKeepConfig()
        cfg.faithful.skip_if_larger = False
        res = faithful_compress(str(src), str(tmp_path / "out"), cfg)

        assert res.skipped is False
        assert res.output_path.suffix == ".avif"

    def test_normal_image_not_skipped(self, face_image, tmp_path):
        res = faithful_compress(str(face_image), str(tmp_path / "out"), FaceKeepConfig())
        assert res.skipped is False
        assert res.compressed_size < res.original_size

    def test_inplace_does_not_destroy_original(self, tmp_path):
        # If the resolved output is the source itself, the original must survive.
        src = tmp_path / "tiny.png"
        in_size = _tiny_incompressible_png(src)
        before = src.read_bytes()

        # Output target resolves to the same .png file (same stem + dir).
        res = faithful_compress(str(src), str(tmp_path / "tiny"), FaceKeepConfig())

        assert res.skipped is True
        assert src.exists() and src.stat().st_size == in_size
        assert src.read_bytes() == before


class TestAggressiveMode:
    def test_compress_restore_roundtrip(self, face_image, tmp_path):
        cfg = FaceKeepConfig()
        cfg.mode = "aggressive"
        photo = compress_photo(str(face_image), cfg)
        write_fkeep(photo, str(tmp_path / "out"))
        assert (tmp_path / "out.fkeep").exists()

        restorer = Restorer(cfg.aggressive)
        restored = restorer.preview(str(tmp_path / "out.fkeep"))
        original = cv2.imread(str(face_image))
        assert restored.shape == original.shape

    def test_manifest_records_metadata(self, face_image, tmp_path):
        cfg = FaceKeepConfig()
        cfg.mode = "aggressive"
        photo = compress_photo(str(face_image), cfg)
        write_fkeep(photo, str(tmp_path / "out"))

        info = read_fkeep_info(str(tmp_path / "out.fkeep"))
        assert info["mode"] == "aggressive"
        assert "faces" in info
        assert info["original"]["width"] > 0

    def test_fkeep_is_valid_zip(self, face_image, tmp_path):
        import zipfile
        cfg = FaceKeepConfig()
        cfg.mode = "aggressive"
        photo = compress_photo(str(face_image), cfg)
        write_fkeep(photo, str(tmp_path / "out"))
        assert zipfile.is_zipfile(str(tmp_path / "out.fkeep"))

    def test_zero_face_conservative(self, plain_image, tmp_path):
        """Faceless image with conservative strategy uses the conservative scale."""
        cfg = FaceKeepConfig()
        cfg.mode = "aggressive"
        cfg.aggressive.no_face_strategy = "conservative"
        photo = compress_photo(str(plain_image), cfg)
        if len(photo.faces) == 0:
            # No faces -> conservative bg_scale should have been applied
            assert photo.effective_bg_scale == cfg.aggressive.no_face_bg_scale
        else:
            # Detector found something -> normal scale retained
            assert photo.effective_bg_scale == cfg.aggressive.bg_scale


class TestMetrics:
    def test_identical_images_perfect_ssim(self):
        img = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        report = metrics.compare(img, img)
        assert report.overall_ssim > 0.999
