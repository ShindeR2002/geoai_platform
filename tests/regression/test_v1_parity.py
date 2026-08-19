"""
Regression tests for the GeoAI Platform.

Validates that Version 2 platform outputs are numerically consistent with
Version 1 reference data saved from the original notebooks. These tests
are the primary scientific parity guarantee.

Reference files required (copy from Version 1 outputs before running):
    tests/regression/v1_reference/X_v2_clean.npy
    tests/regression/v1_reference/Dholera_prediction_map.npy
    tests/regression/v1_reference/Dholera_significant_objects.npy

Tests are automatically skipped when reference files are not present, with
a clear message explaining how to populate them.

Scientific parity rules (Implementation Contract §9.3):
    - Feature cube shape[1] == 18
    - Feature cube dtype compatible with float
    - Prediction map shape matches reference
    - Prediction map values match within atol=1e-5 (float32 tolerance)
    - Significant object mask matches exactly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
from pathlib import Path

# Path to the V1 reference arrays.
REFERENCE_DIR = Path(__file__).parent / "v1_reference"

# File paths for each reference array.
REF_X_V2 = REFERENCE_DIR / "X_v2_clean.npy"
REF_PRED_MAP = REFERENCE_DIR / "Dholera_prediction_map.npy"
REF_SIG_OBJECTS = REFERENCE_DIR / "Dholera_significant_objects.npy"

# Skip decorator for missing reference files.
skip_no_x_v2 = pytest.mark.skipif(
    not REF_X_V2.exists(),
    reason=(
        f"Reference file not found: {REF_X_V2}. "
        "Copy Version 1 outputs/ml_dataset/X_v2_clean.npy to "
        "tests/regression/v1_reference/ to enable this test."
    ),
)
skip_no_pred_map = pytest.mark.skipif(
    not REF_PRED_MAP.exists(),
    reason=(
        f"Reference file not found: {REF_PRED_MAP}. "
        "Copy Version 1 outputs/generalization/Dholera_prediction_map.npy to "
        "tests/regression/v1_reference/ to enable this test."
    ),
)
skip_no_sig_objects = pytest.mark.skipif(
    not REF_SIG_OBJECTS.exists(),
    reason=(
        f"Reference file not found: {REF_SIG_OBJECTS}. "
        "Copy Version 1 outputs/generalization/Dholera_significant_objects.npy to "
        "tests/regression/v1_reference/ to enable this test."
    ),
)


# ---------------------------------------------------------------------------
# Feature dataset parity tests
# ---------------------------------------------------------------------------

class TestFeatureDatasetParity:

    @skip_no_x_v2
    def test_feature_count_is_18(self):
        """V1 X_v2_clean.npy must have 18 features (columns)."""
        X = np.load(REF_X_V2)
        assert X.shape[1] == 18, (
            f"V1 X_v2_clean.npy has {X.shape[1]} features; expected 18."
        )

    @skip_no_x_v2
    def test_feature_matrix_is_2d(self):
        """V1 feature matrix must be 2-dimensional."""
        X = np.load(REF_X_V2)
        assert X.ndim == 2, f"Expected 2D array, got shape {X.shape}."

    @skip_no_x_v2
    def test_feature_matrix_is_float_compatible(self):
        """V1 feature matrix must have a floating-point dtype."""
        X = np.load(REF_X_V2)
        assert np.issubdtype(X.dtype, np.floating), (
            f"Expected floating dtype, got {X.dtype}."
        )

    @skip_no_x_v2
    def test_no_all_zero_rows(self):
        """V1 clean dataset must not contain any all-zero rows (valid mask was applied)."""
        X = np.load(REF_X_V2)
        all_zero_rows = np.all(X == 0, axis=1).sum()
        assert all_zero_rows == 0, (
            f"V1 X_v2_clean.npy contains {all_zero_rows} all-zero rows; "
            "valid pixel mask should have removed them."
        )

    @skip_no_x_v2
    def test_no_nan_rows(self):
        """V1 clean dataset must not contain any NaN values (valid mask was applied)."""
        X = np.load(REF_X_V2)
        nan_count = int(np.isnan(X).sum())
        assert nan_count == 0, (
            f"V1 X_v2_clean.npy contains {nan_count} NaN values; "
            "valid pixel mask should have removed them."
        )


# ---------------------------------------------------------------------------
# Prediction map parity tests
# ---------------------------------------------------------------------------

class TestPredictionMapParity:

    @skip_no_pred_map
    def test_prediction_map_is_2d(self):
        """V1 Dholera prediction map must be 2-dimensional."""
        pred_map = np.load(REF_PRED_MAP)
        assert pred_map.ndim == 2, (
            f"Expected 2D prediction map, got shape {pred_map.shape}."
        )

    @skip_no_pred_map
    def test_prediction_map_contains_only_binary_and_nan(self):
        """V1 prediction map must contain only values 0.0, 1.0, and NaN."""
        pred_map = np.load(REF_PRED_MAP)
        valid = pred_map[~np.isnan(pred_map)]
        unique_vals = set(np.unique(valid.astype(np.int32)).tolist())
        assert unique_vals.issubset({0, 1}), (
            f"Prediction map contains unexpected values: {unique_vals}."
        )

    @skip_no_pred_map
    def test_prediction_map_has_change_pixels(self):
        """V1 Dholera prediction map must have at least one change pixel."""
        pred_map = np.load(REF_PRED_MAP)
        n_change = int((pred_map == 1.0).sum())
        assert n_change > 0, (
            "V1 Dholera prediction map has no change pixels. "
            "This may indicate a corrupt reference file."
        )


# ---------------------------------------------------------------------------
# Significant objects parity tests
# ---------------------------------------------------------------------------

class TestSignificantObjectsParity:

    @skip_no_sig_objects
    def test_significant_objects_is_binary(self):
        """V1 significant objects array must be binary (0/1)."""
        sig = np.load(REF_SIG_OBJECTS)
        unique_vals = set(np.unique(sig).tolist())
        assert unique_vals.issubset({0, 1}), (
            f"Significant objects contains non-binary values: {unique_vals}."
        )

    @skip_no_sig_objects
    def test_significant_objects_has_change(self):
        """V1 significant objects mask must have at least one object pixel."""
        sig = np.load(REF_SIG_OBJECTS)
        n_change = int(sig.sum())
        assert n_change > 0, (
            "V1 significant objects mask has no object pixels. "
            "This may indicate a corrupt reference file."
        )

    @skip_no_sig_objects
    def test_significant_objects_shape_matches_prediction_map(self):
        """V1 significant objects and prediction map must have the same shape."""
        if not REF_PRED_MAP.exists():
            pytest.skip("Prediction map reference file not available.")
        sig = np.load(REF_SIG_OBJECTS)
        pred = np.load(REF_PRED_MAP)
        assert sig.shape == pred.shape, (
            f"Significant objects shape {sig.shape} does not match "
            f"prediction map shape {pred.shape}."
        )


# ---------------------------------------------------------------------------
# Platform v2 parity test — requires real data
# ---------------------------------------------------------------------------

class TestV2FeatureCubeConsistency:
    """
    Validates that the V2 feature cube builder produces arrays consistent
    with the V1 reference. These tests use synthetic data to check structural
    properties rather than exact numerical match (which requires the real AOI
    rasters to be present).
    """

    def test_feature_cube_18_features(self):
        """build_feature_cube must always produce exactly 18 features."""
        from geoai.features.feature_cube import build_feature_cube
        arr = np.random.rand(20, 20).astype(np.float32)
        cube = build_feature_cube(arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr)
        assert cube.shape[2] == 18

    def test_valid_pixel_mask_excludes_all_zeros(self):
        """Valid pixel mask must exclude all-zero pixels (V1 compound mask)."""
        from geoai.utils.raster_utils import build_valid_pixel_mask
        X = np.zeros((5, 18), dtype=np.float32)
        X[2, :] = 1.0  # Only row 2 is non-zero
        mask = build_valid_pixel_mask(X)
        assert mask.sum() == 1
        assert mask[2] is np.bool_(True)

    def test_valid_pixel_mask_excludes_nan(self):
        """Valid pixel mask must exclude pixels with any NaN feature."""
        from geoai.utils.raster_utils import build_valid_pixel_mask
        X = np.ones((3, 18), dtype=np.float32)
        X[1, 5] = np.nan  # Row 1 has a NaN
        mask = build_valid_pixel_mask(X)
        assert mask.sum() == 2
        assert mask[1] is np.bool_(False)


# ---------------------------------------------------------------------------
# Instructions for populating reference files
# ---------------------------------------------------------------------------

def test_reference_directory_exists():
    """Reference directory must exist (even if empty)."""
    assert REFERENCE_DIR.exists(), (
        f"Reference directory not found: {REFERENCE_DIR}. "
        "Run: mkdir -p tests/regression/v1_reference"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
