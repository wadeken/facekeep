"""Tests for the soft mask and blending."""

import numpy as np

from facekeep.aggressive.blender import create_soft_mask


class TestSoftMask:
    def test_shape(self):
        assert create_soft_mask((100, 200), margin=16).shape == (100, 200)

    def test_center_is_one(self):
        m = create_soft_mask((100, 200), margin=16)
        assert m[50, 100] == 1.0

    def test_edges_are_zero(self):
        m = create_soft_mask((100, 200), margin=16)
        assert m[0, 0] == 0.0
        assert m[0, 100] == 0.0
        assert m[50, 0] == 0.0
        assert m[-1, -1] == 0.0

    def test_monotonic_gradient(self):
        """Alpha should increase monotonically from edge toward center."""
        m = create_soft_mask((100, 100), margin=20)
        col = m[:50, 50]  # top half of center column
        assert np.all(np.diff(col) >= 0)

    def test_no_margin_is_all_ones(self):
        assert np.all(create_soft_mask((50, 50), margin=0) == 1.0)

    def test_row_is_not_uniformly_zero(self):
        """Regression: a near-edge row must be a gradient, not all zeros."""
        m = create_soft_mask((100, 100), margin=10)
        row = m[2, :]
        assert row.max() > 0.0  # interior of the row must be > 0
