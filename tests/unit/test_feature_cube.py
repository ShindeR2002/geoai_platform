"""
Unit tests for geoai.features.feature_cube.

Validates the canonical 18-feature cube assembly:
- Correct output shape (H, W, 18)
- Correct feature ordering matches CANONICAL_FEATURE_NAMES
- Correct temporal delta computation within the cube
- Float32 dtype enforcement
- Shape mismatch detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from geoai.features.feature_cube import (
    build_feature_cube,
    get_feature_index,
    get_feature_names,
)
from geoai.core.exceptions import FeatureCountMismatchError, FeatureEngineeringError
from geoai.utils.constants import CANONICAL_FEATURE_COUNT, CANONICAL_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_epoch_arrays():
    """Return synthetic T1 and T2 arrays with distinct values for verification."""
    H, W = 10, 10
    rng = np.random.default_rng(seed=42)

    # T1: values in range [0, 1]
    red_t1 = rng.random((H, W)).astype(np.float32)
    green_t1 = rng.random((H, W)).astype(np.float32)
    blue_t1 = rng.random((H, W)).astype(np.float32)
    sar_t1 = rng.random((H, W)).astype(np.float32)
    ndvi_t1 = rng.uniform(-0.5, 0.8, (H, W)).astype(np.float32)
    ndbi_t1 = rng.uniform(-0.3, 0.5, (H, W)).astype(np.float32)
    ndwi_t1 = rng.uniform(-0.4, 0.6, (H, W)).astype(np.float32)

    # T2: values in range [1, 2] — guaranteed to differ from T1
    red_t2 = (rng.random((H, W)) + 1.0).astype(np.float32)
    green_t2 = (rng.random((H, W)) + 1.0).astype(np.float32)
    blue_t2 = (rng.random((H, W)) + 1.0).astype(np.float32)
    sar_t2 = (rng.random((H, W)) + 1.0).astype(np.float32)
    ndvi_t2 = rng.uniform(0.0, 0.9, (H, W)).astype(np.float32)
    ndbi_t2 = rng.uniform(0.0, 0.7, (H, W)).astype(np.float32)
    ndwi_t2 = rng.uniform(-0.2, 0.8, (H, W)).astype(np.float32)

    return {
        "H": H, "W": W,
        "red_t1": red_t1, "green_t1": green_t1, "blue_t1": blue_t1,
        "sar_t1": sar_t1, "ndvi_t1": ndvi_t1, "ndbi_t1": ndbi_t1,
        "ndwi_t1": ndwi_t1,
        "red_t2": red_t2, "green_t2": green_t2, "blue_t2": blue_t2,
        "sar_t2": sar_t2, "ndvi_t2": ndvi_t2, "ndbi_t2": ndbi_t2,
        "ndwi_t2": ndwi_t2,
    }


@pytest.fixture
def built_cube(synthetic_epoch_arrays):
    """Return the assembled feature cube from synthetic epoch arrays."""
    e = synthetic_epoch_arrays
    return build_feature_cube(
        red_t1=e["red_t1"], green_t1=e["green_t1"], blue_t1=e["blue_t1"],
        sar_t1=e["sar_t1"], ndvi_t1=e["ndvi_t1"], ndbi_t1=e["ndbi_t1"],
        ndwi_t1=e["ndwi_t1"],
        red_t2=e["red_t2"], green_t2=e["green_t2"], blue_t2=e["blue_t2"],
        sar_t2=e["sar_t2"], ndvi_t2=e["ndvi_t2"], ndbi_t2=e["ndbi_t2"],
        ndwi_t2=e["ndwi_t2"],
    )


# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------

class TestFeatureCubeShape:

    def test_output_shape(self, built_cube, synthetic_epoch_arrays):
        H = synthetic_epoch_arrays["H"]
        W = synthetic_epoch_arrays["W"]
        assert built_cube.shape == (H, W, 18), (
            f"Expected shape ({H}, {W}, 18), got {built_cube.shape}"
        )

    def test_feature_count_is_canonical(self, built_cube):
        assert built_cube.shape[2] == CANONICAL_FEATURE_COUNT

    def test_dtype_is_float32(self, built_cube):
        assert built_cube.dtype == np.float32, (
            f"Expected float32, got {built_cube.dtype}"
        )

    def test_no_infinite_values(self, built_cube):
        assert not np.isinf(built_cube).any(), "Feature cube contains Inf values."


# ---------------------------------------------------------------------------
# Feature ordering tests (Implementation Contract §2.2 F-02)
# ---------------------------------------------------------------------------

class TestFeatureOrdering:

    def test_red_t1_at_index_0(self, built_cube, synthetic_epoch_arrays):
        """Feature index 0 must be Red_2021."""
        np.testing.assert_array_equal(
            built_cube[:, :, 0], synthetic_epoch_arrays["red_t1"]
        )

    def test_green_t1_at_index_1(self, built_cube, synthetic_epoch_arrays):
        """Feature index 1 must be Green_2021."""
        np.testing.assert_array_equal(
            built_cube[:, :, 1], synthetic_epoch_arrays["green_t1"]
        )

    def test_sar_t1_at_index_3(self, built_cube, synthetic_epoch_arrays):
        """Feature index 3 must be SAR_2021."""
        np.testing.assert_array_equal(
            built_cube[:, :, 3], synthetic_epoch_arrays["sar_t1"]
        )

    def test_ndvi_t1_at_index_4(self, built_cube, synthetic_epoch_arrays):
        """Feature index 4 must be NDVI_2021."""
        np.testing.assert_array_equal(
            built_cube[:, :, 4], synthetic_epoch_arrays["ndvi_t1"]
        )

    def test_red_t2_at_index_7(self, built_cube, synthetic_epoch_arrays):
        """Feature index 7 must be Red_2024."""
        np.testing.assert_array_equal(
            built_cube[:, :, 7], synthetic_epoch_arrays["red_t2"]
        )

    def test_sar_t2_at_index_10(self, built_cube, synthetic_epoch_arrays):
        """Feature index 10 must be SAR_2024."""
        np.testing.assert_array_equal(
            built_cube[:, :, 10], synthetic_epoch_arrays["sar_t2"]
        )

    def test_ndvi_t2_at_index_11(self, built_cube, synthetic_epoch_arrays):
        """Feature index 11 must be NDVI_2024."""
        np.testing.assert_array_equal(
            built_cube[:, :, 11], synthetic_epoch_arrays["ndvi_t2"]
        )

    def test_delta_ndvi_at_index_14(self, built_cube, synthetic_epoch_arrays):
        """Feature index 14 must be Delta_NDVI = NDVI_T2 - NDVI_T1."""
        expected = (
            synthetic_epoch_arrays["ndvi_t2"] - synthetic_epoch_arrays["ndvi_t1"]
        )
        np.testing.assert_allclose(
            built_cube[:, :, 14], expected, atol=1e-5,
            err_msg="Delta_NDVI at index 14 does not equal NDVI_T2 - NDVI_T1."
        )

    def test_delta_ndbi_at_index_15(self, built_cube, synthetic_epoch_arrays):
        """Feature index 15 must be Delta_NDBI = NDBI_T2 - NDBI_T1."""
        expected = (
            synthetic_epoch_arrays["ndbi_t2"] - synthetic_epoch_arrays["ndbi_t1"]
        )
        np.testing.assert_allclose(built_cube[:, :, 15], expected, atol=1e-5)

    def test_delta_ndwi_at_index_16(self, built_cube, synthetic_epoch_arrays):
        """Feature index 16 must be Delta_NDWI = NDWI_T2 - NDWI_T1."""
        expected = (
            synthetic_epoch_arrays["ndwi_t2"] - synthetic_epoch_arrays["ndwi_t1"]
        )
        np.testing.assert_allclose(built_cube[:, :, 16], expected, atol=1e-5)

    def test_delta_sar_at_index_17(self, built_cube, synthetic_epoch_arrays):
        """Feature index 17 must be Delta_SAR = SAR_T2 - SAR_T1."""
        expected = (
            synthetic_epoch_arrays["sar_t2"] - synthetic_epoch_arrays["sar_t1"]
        )
        np.testing.assert_allclose(built_cube[:, :, 17], expected, atol=1e-5)

    def test_feature_names_match_canonical(self):
        """get_feature_names() must return the canonical 18-name tuple."""
        names = get_feature_names()
        assert names == CANONICAL_FEATURE_NAMES

    def test_get_feature_index_delta_ndvi(self):
        """Delta_NDVI must be at index 14."""
        assert get_feature_index("Delta_NDVI") == 14

    def test_get_feature_index_delta_sar(self):
        """Delta_SAR must be at index 17."""
        assert get_feature_index("Delta_SAR") == 17

    def test_get_feature_index_invalid_raises(self):
        """Unknown feature name must raise FeatureEngineeringError."""
        with pytest.raises(FeatureEngineeringError):
            get_feature_index("NonExistentFeature")


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestFeatureCubeErrors:

    def test_shape_mismatch_raises(self):
        """Mismatched spatial shapes must raise FeatureEngineeringError."""
        arr_10 = np.zeros((10, 10), dtype=np.float32)
        arr_20 = np.zeros((20, 10), dtype=np.float32)
        with pytest.raises(FeatureEngineeringError):
            build_feature_cube(
                red_t1=arr_10, green_t1=arr_10, blue_t1=arr_10,
                sar_t1=arr_10, ndvi_t1=arr_10, ndbi_t1=arr_10,
                ndwi_t1=arr_10,
                red_t2=arr_20,  # Different shape — must fail
                green_t2=arr_10, blue_t2=arr_10, sar_t2=arr_10,
                ndvi_t2=arr_10, ndbi_t2=arr_10, ndwi_t2=arr_10,
            )

    def test_delta_is_signed_not_absolute(self, built_cube, synthetic_epoch_arrays):
        """Deltas must be signed (T2 - T1), not absolute values."""
        e = synthetic_epoch_arrays
        # T1 ndvi > T2 ndvi in some pixels → delta should be negative there
        delta_ndvi = built_cube[:, :, 14]
        expected = e["ndvi_t2"] - e["ndvi_t1"]
        # If any negative delta exists, confirm it's preserved
        if (expected < 0).any():
            assert (delta_ndvi < 0).any(), (
                "Delta_NDVI should contain negative values when T2 < T1."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
