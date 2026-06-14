"""Basic FaceKeep usage examples (library API).

Run:  python examples/basic_usage.py path/to/photo.jpg
"""

import sys
from pathlib import Path

from facekeep import metrics
from facekeep.config import FaceKeepConfig
from facekeep.faithful import compress as faithful_compress


def faithful_example(image_path: str) -> None:
    """Default faithful mode: whole-image AVIF, single standard output file."""
    config = FaceKeepConfig()              # mode="faithful", codec="avif", q=70
    config.faithful.quality = 70

    result = faithful_compress(image_path, output_path="example_faithful", config=config)

    print("Faithful mode")
    print(f"  output:     {result.output_path}")
    print(f"  original:   {result.original_size / 1024:.1f} KB")
    print(f"  compressed: {result.compressed_size / 1024:.1f} KB")
    print(f"  ratio:      {result.ratio:.2f}x")
    print(f"  faces:      {result.faces_detected}")
    print(f"  codec/q:    {result.codec} q{result.quality_used}")
    print("  (output is a standard image; open it in any modern viewer)")


def quality_example(original_path: str) -> None:
    """Measure fidelity of the faithful output vs the original."""
    import cv2
    import numpy as np
    import pillow_avif  # noqa: F401
    from PIL import Image

    out = Path("example_faithful.avif")
    if not out.exists():
        return

    original = cv2.imread(original_path)
    decoded = cv2.cvtColor(
        np.array(Image.open(out).convert("RGB")), cv2.COLOR_RGB2BGR
    )
    report = metrics.compare(original, decoded)
    print("\nFidelity vs original")
    print(f"  SSIM: {report.overall_ssim:.4f}")
    print(f"  PSNR: {report.overall_psnr:.2f} dB")


def aggressive_example(image_path: str) -> None:
    """Optional aggressive mode: .fkeep container + AI/bicubic restore."""
    from facekeep.aggressive.compressor import compress_photo
    from facekeep.aggressive.format import write_fkeep
    from facekeep.aggressive.restorer import Restorer

    config = FaceKeepConfig()
    config.mode = "aggressive"
    config.aggressive.bg_scale = 0.25

    photo = compress_photo(image_path, config)
    size = write_fkeep(photo, "example_aggressive")
    print("\nAggressive mode")
    print(f"  .fkeep size: {size / 1024:.1f} KB ({photo.original_size_bytes / size:.2f}x)")

    # Restore (uses Real-ESRGAN if installed, else bicubic). preview() is bicubic.
    Restorer(config.aggressive).preview(
        "example_aggressive.fkeep", "example_restored.jpg"
    )
    print("  restored -> example_restored.jpg (preview / bicubic)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/basic_usage.py path/to/photo.jpg")
        sys.exit(1)
    image_path = sys.argv[1]

    faithful_example(image_path)
    quality_example(image_path)
    aggressive_example(image_path)


if __name__ == "__main__":
    main()
