"""FaceKeep - Face-aware photo compression for family photo backups.

Two compression modes:
  - faithful (default): whole-image AVIF/JXL encoding. Real pixels everywhere,
    no AI hallucination, no seams, output is a standard image file that opens
    anywhere. Visually lossless, ~2-4x smaller than JPEG.
  - aggressive (optional): crop faces at original quality, downsample background,
    reconstruct with AI super-resolution on restore. Extreme compression (~8-12x)
    at the cost of background fidelity.
"""

__version__ = "0.2.0"
__author__ = "FaceKeep Contributors"
