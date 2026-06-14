"""Aggressive mode: crop faces + downsample background + AI super-resolution restore.

For extreme compression (~8-12x) when background fidelity can be relaxed. The
background is discarded and reconstructed (hallucinated) on restore, so it will
look plausible but differ from the original. Faces are kept at original quality.

Output uses the .fkeep container and requires an explicit restore step.
"""
