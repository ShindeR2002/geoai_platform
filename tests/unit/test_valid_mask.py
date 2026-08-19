"""
Unit tests for geoai.utils.raster_utils — valid pixel mask and reconstruction.

Validates the canonical compound valid pixel mask (Implementation Contract §2.2 F-05):
  valid = (~np.isnan(X).any(axis=1)) & (~np.all(X == 0, axis=1))

Also validates prediction raster reconstruction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from geoai.utils.raster_utils import (
    build_valid_pixel_mask,
    reconstruct_prediction_raster,
    flatten_spatial,
    restore_spatial,
    fill_nan,
    stretch_contrast,
)


class TestValidPixelMask:
    """Tests for the canonical compound valid pixel mask."""

    def test_all_valid_pixels(self):
        """Pixels with no NaN and at least one non-zero feature are valid."""
        X = np.array([
            [1.0, 2.0, 3.0],
            [0.5, 0.1, 0.9],
        ], dtype=np.float32)
        mask = build_valid_pixel_mask(X)
        assert mask.all(), "All pixels should be valid."

    def test_all_zero_pixel_is_invalid(self):
        """A pixel where ALL features are zero must be excluded."""
        X = np.array([
            [1.0, 2.0, 3.0],   # valid
            [0.0, 0.0, 0.0],   # all-zero → invalid
            [0.5, 0.0, 0.9],   # valid (not all zero)
        ], dtype=np.float32)
        mask = build_valid_pixel_mask(X)
        assert mask[0] is np.bool_(True)
        assert mask[1] is np.bool_(False), "All-zero pixel must be invalid."
        assert mask[2] is np.bool_(True)

    def test_nan_pixel_is_invalid(self):
        """A pixel with ANY NaN feature must be excluded."""
        X = np.array([
            [1.0, 2.0, 3.0],         # valid
            [1.0, np.nan, 3.0],       # NaN in feature 1 → invalid
            [np.nan, np.nan, np.nan], # all NaN → invalid
        ], dtype=np.float32)
        mask = build_valid_pixel_mask(X)
        assert mask[0] is np.bool_(True)
        assert mask[1] is np.bool_(False), "Pixel with NaN feature must be invalid."
        assert mask[2] is np.bool_(False), "All-NaN pixel must be invalid."

    def test_partial_zero_not_all_zero_is_valid(self):
        """A pixel with some zeros but not all zeros must be valid."""
        X = np.array([
            [0.0, 0.0, 1.0],  # has one non-zero → valid
        ], dtype=np.float32)
        mask = build_valid_pixel_mask(X)
        assert mask[0] is np.bool_(True)

    def test_compound_rule_both_conditions(self):
        """The mask must apply BOTH the NaN and all-zero conditions."""
        X = np.array([
            [1.0, 2.0, 3.0],    # [0] valid
            [0.0, 0.0, 0.0],    # [1] all-zero → invalid
            [1.0, np.nan, 3.0], # [2] has NaN → invalid
            [0.5, 0.6, 0.7],    # [3] valid
        ], dtype=np.float32)
        mask = build_valid_pixel_mask(X)
        expected = np.array([True, False, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_empty_matrix_returns_empty_mask(self):
        """Empty input produces empty boolean mask."""
        X = np.empty((0, 18), dtype=np.float32)
        mask = build_valid_pixel_mask(X)
        assert mask.shape == (0,)
        assert mask.dtype == bool

    def test_single_valid_pixel(self):
        """Single valid pixel returns mask of length 1 with True."""
        X = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mask = build_valid_pixel_mask(X)
        assert mask.shape == (1,)
        assert mask[0] is np.bool_(True)

    def test_mask_is_boolean(self):
        """Output mask must have boolean dtype."""
        X = np.ones((5, 3), dtype=np.float32)
        mask = build_valid_pixel_mask(X)
        assert mask.dtype == bool


class TestReconstructPredictionRaster:
    """Tests for prediction raster reconstruction (NB09 pattern)."""

    def test_valid_reconstruction(self):
        """Predictions at valid positions; NaN at invalid positions."""
        valid_mask = np.array([True, False, True, False])
        predictions = np.array([1.0, 0.0], dtype=np.float32)
        flat = reconstruct_prediction_raster(predictions, valid_mask, total_pixels=4)

        assert flat[0] == 1.0
        assert np.isnan(flat[1])
        assert flat[2] == 0.0
        assert np.isnan(flat[3])

    def test_output_dtype_is_float32(self):
        """Reconstructed raster must be float32."""
        mask = np.array([True, True, False])
        preds = np.array([1, 0], dtype=np.int32)
        flat = reconstruct_prediction_raster(preds, mask, total_pixels=3)
        assert flat.dtype == np.float32

    def test_mismatched_predictions_raises(self):
        """Mismatch between prediction count and valid_mask sum must raise."""
        mask = np.array([True, False, True])  # 2 valid pixels
        preds = np.array([1.0])  # only 1 prediction
        with pytest.raises(ValueError):
            reconstruct_prediction_raster(preds, mask, total_pixels=3)

    def test_all_valid_no_nan_in_output(self):
        """When all pixels are valid, output must contain no NaN."""
        mask = np.array([True, True, True])
        preds = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        flat = reconstruct_prediction_raster(preds, mask, total_pixels=3)
        assert not np.isnan(flat).any()


class TestFlattenRestore:
    """Tests for spatial flatten/restore round-trip."""

    def test_flatten_restore_roundtrip(self):
        """Flatten then restore must recover original spatial array."""
        original = np.random.rand(8, 8, 18).astype(np.float32)
        flat = flatten_spatial(original)
        assert flat.shape == (64, 18)
        restored = restore_spatial(flat, 8, 8)
        assert restored.shape == (8, 8, 18)
        np.testing.assert_array_equal(original, restored)

    def test_flatten_1d_restore(self):
        """Restore a 1D flat array to 2D spatial shape."""
        flat = np.arange(20, dtype=np.float32)
        restored = restore_spatial(flat, 4, 5)
        assert restored.shape == (4, 5)

    def test_flatten_requires_3d(self):
        """flatten_spatial must raise for non-3D input."""
        with pytest.raises(ValueError):
            flatten_spatial(np.zeros((10, 18), dtype=np.float32))


class TestFillNan:
    """Tests for the canonical NaN filling operation."""

    def test_nan_replaced_by_zero(self):
        """NaN values must be replaced by 0.0 (the Version 1 default)."""
        arr = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
        result = fill_nan(arr)
        assert not np.isnan(result).any()
        assert result[1] == 0.0
        assert result[3] == 0.0

    def test_non_nan_values_unchanged(self):
        """Non-NaN values must not be modified."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = fill_nan(arr)
        np.testing.assert_array_equal(arr, result)

    def test_inf_also_replaced(self):
        """Positive and negative Inf values must also be replaced."""
        arr = np.array([1.0, np.inf, -np.inf], dtype=np.float32)
        result = fill_nan(arr)
        assert not np.isinf(result).any()


class TestStretchContrast:
    """Tests for percentile-based contrast stretch."""

    def test_output_range_is_0_to_1(self):
        """Output values must be clipped to [0.0, 1.0]."""
        arr = np.linspace(0, 100, 100).astype(np.float32)
        result = stretch_contrast(arr, low_pct=2, high_pct=98)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_output_dtype_is_float32(self):
        arr = np.random.rand(20, 20).astype(np.float64)
        result = stretch_contrast(arr)
        assert result.dtype == np.float32

    def test_all_nan_returns_zeros(self):
        arr = np.full((5, 5), np.nan, dtype=np.float32)
        result = stretch_contrast(arr)
        np.testing.assert_array_equal(result, np.zeros((5, 5), dtype=np.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
