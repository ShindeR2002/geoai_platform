"""
Unit tests for temporal delta features and pseudo-label generation.

Validates:
- Delta computations are exactly T2 - T1 (signed, not absolute)
- All four deltas computed correctly in compute_all_deltas()
- Pseudo-label change score weights and threshold from Version 1
- Threshold = mean + 1.5 * std (default)
- Output is uint8 binary mask
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from geoai.features.temporal import (
    compute_delta_ndvi,
    compute_delta_ndbi,
    compute_delta_ndwi,
    compute_delta_sar,
    compute_all_deltas,
)
from geoai.features.pseudo_labels import (
    compute_change_score,
    compute_normalised_delta,
    compute_threshold,
    generate_pseudo_labels,
)
from geoai.utils.constants import (
    DEFAULT_WEIGHT_SAR,
    DEFAULT_WEIGHT_NDVI,
    DEFAULT_WEIGHT_NDBI,
    DEFAULT_WEIGHT_NDWI,
    DEFAULT_THRESHOLD_SIGMA,
)


# ---------------------------------------------------------------------------
# Temporal delta tests
# ---------------------------------------------------------------------------

class TestTemporalDeltas:

    def test_delta_is_t2_minus_t1(self):
        """Delta must be T2 - T1 (signed difference, not absolute)."""
        t1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t2 = np.array([[2.0, 1.0], [5.0, 2.0]], dtype=np.float32)
        delta = compute_delta_ndvi(t2, t1)
        expected = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
        np.testing.assert_allclose(delta, expected, atol=1e-6)

    def test_negative_delta_preserved(self):
        """When T2 < T1 the delta must be negative."""
        t1 = np.ones((5, 5), dtype=np.float32) * 0.8
        t2 = np.ones((5, 5), dtype=np.float32) * 0.3
        delta = compute_delta_ndvi(t2, t1)
        assert (delta < 0).all(), "All delta values should be negative."
        np.testing.assert_allclose(delta, np.full((5, 5), -0.5, dtype=np.float32), atol=1e-6)

    def test_zero_delta_when_equal(self):
        """When T1 == T2 the delta must be zero."""
        arr = np.random.rand(4, 4).astype(np.float32)
        delta = compute_delta_sar(arr, arr)
        np.testing.assert_array_equal(delta, np.zeros((4, 4), dtype=np.float32))

    def test_delta_output_dtype_float32(self):
        """Delta output must be float32."""
        t = np.ones((3, 3), dtype=np.float64)
        delta = compute_delta_ndbi(t * 1.1, t)
        assert delta.dtype == np.float32

    def test_compute_all_deltas_returns_four(self):
        """compute_all_deltas must return exactly 4 arrays."""
        arr = np.ones((5, 5), dtype=np.float32)
        result = compute_all_deltas(arr, arr*2, arr, arr*2, arr, arr*2, arr, arr*2)
        assert len(result) == 4

    def test_compute_all_deltas_correct_values(self):
        """compute_all_deltas must compute each delta correctly."""
        t1 = np.ones((4, 4), dtype=np.float32)
        t2 = np.ones((4, 4), dtype=np.float32) * 3.0
        dn, db, dw, ds = compute_all_deltas(t1, t2, t1, t2, t1, t2, t1, t2)
        # All deltas should be 2.0 (T2 - T1 = 3 - 1)
        for delta, name in [(dn, "NDVI"), (db, "NDBI"), (dw, "NDWI"), (ds, "SAR")]:
            np.testing.assert_allclose(
                delta, np.full((4, 4), 2.0, dtype=np.float32), atol=1e-6,
                err_msg=f"Delta_{name} should be 2.0."
            )

    def test_delta_ordering_in_all_deltas(self):
        """compute_all_deltas result order: (delta_ndvi, delta_ndbi, delta_ndwi, delta_sar)."""
        t1 = np.zeros((3, 3), dtype=np.float32)
        # Give each delta a unique signature value
        ndvi_t2 = np.full((3, 3), 1.0, dtype=np.float32)
        ndbi_t2 = np.full((3, 3), 2.0, dtype=np.float32)
        ndwi_t2 = np.full((3, 3), 3.0, dtype=np.float32)
        sar_t2 = np.full((3, 3), 4.0, dtype=np.float32)
        dn, db, dw, ds = compute_all_deltas(t1, ndvi_t2, t1, ndbi_t2, t1, ndwi_t2, t1, sar_t2)
        assert float(dn[0, 0]) == pytest.approx(1.0), "Index 0 should be Delta_NDVI."
        assert float(db[0, 0]) == pytest.approx(2.0), "Index 1 should be Delta_NDBI."
        assert float(dw[0, 0]) == pytest.approx(3.0), "Index 2 should be Delta_NDWI."
        assert float(ds[0, 0]) == pytest.approx(4.0), "Index 3 should be Delta_SAR."


# ---------------------------------------------------------------------------
# Pseudo-label tests
# ---------------------------------------------------------------------------

class TestNormalisedDelta:

    def test_normalised_range_is_0_to_1(self):
        """Normalised delta must be in [0, 1]."""
        delta = np.array([[-3.0, 0.0, 5.0, -1.0]], dtype=np.float32)
        result = compute_normalised_delta(delta, "test")
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0 + 1e-6

    def test_maximum_absolute_value_maps_to_1(self):
        """The pixel with the maximum absolute delta value must map to 1.0."""
        delta = np.array([[1.0, -4.0, 2.0]], dtype=np.float32)
        result = compute_normalised_delta(delta, "test")
        # |delta| = [1, 4, 2]; max = 4; normalised = [0.25, 1.0, 0.5]
        assert float(result[0, 1]) == pytest.approx(1.0, abs=1e-5)

    def test_all_zero_delta_returns_zeros(self):
        """All-zero delta should return zero array (not raise)."""
        delta = np.zeros((4, 4), dtype=np.float32)
        result = compute_normalised_delta(delta, "zero_test")
        np.testing.assert_array_equal(result, np.zeros((4, 4), dtype=np.float32))


class TestChangeScore:

    def test_weights_sum_determines_max_score(self):
        """When all normalised deltas are 1.0, score = sum(weights)."""
        ones = np.ones((3, 3), dtype=np.float32)
        score = compute_change_score(ones, ones, ones, ones)
        expected = DEFAULT_WEIGHT_SAR + DEFAULT_WEIGHT_NDVI + DEFAULT_WEIGHT_NDBI + DEFAULT_WEIGHT_NDWI
        assert float(score.mean()) == pytest.approx(expected, abs=1e-5)

    def test_default_weights_sum_to_one(self):
        """Default weights must sum to 1.0."""
        total = DEFAULT_WEIGHT_SAR + DEFAULT_WEIGHT_NDVI + DEFAULT_WEIGHT_NDBI + DEFAULT_WEIGHT_NDWI
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_default_weights_match_version1(self):
        """Default weights must match Version 1 NB01 Cell 45 exactly."""
        assert DEFAULT_WEIGHT_SAR == pytest.approx(0.35)
        assert DEFAULT_WEIGHT_NDVI == pytest.approx(0.35)
        assert DEFAULT_WEIGHT_NDBI == pytest.approx(0.20)
        assert DEFAULT_WEIGHT_NDWI == pytest.approx(0.10)


class TestThreshold:

    def test_threshold_formula(self):
        """Threshold must equal mean + sigma * std."""
        score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         dtype=np.float32)
        threshold = compute_threshold(score, sigma_multiplier=1.5)
        expected = float(score.mean()) + 1.5 * float(score.std())
        assert threshold == pytest.approx(expected, abs=1e-5)

    def test_default_sigma_is_1_5(self):
        """Default sigma_multiplier must be 1.5 (Version 1 NB01 Cell 51)."""
        assert DEFAULT_THRESHOLD_SIGMA == pytest.approx(1.5)


class TestGeneratePseudoLabels:

    def test_output_is_uint8(self):
        """change_mask must be dtype uint8."""
        delta = np.random.rand(10, 10).astype(np.float32)
        mask, _, _ = generate_pseudo_labels(delta, delta, delta, delta)
        assert mask.dtype == np.uint8

    def test_output_is_binary(self):
        """change_mask must contain only 0 and 1."""
        delta = np.random.rand(20, 20).astype(np.float32)
        mask, _, _ = generate_pseudo_labels(delta, delta, delta, delta)
        unique_vals = set(np.unique(mask).tolist())
        assert unique_vals.issubset({0, 1}), f"Expected only 0/1, got {unique_vals}."

    def test_output_shape_matches_input(self):
        """Output mask shape must match the input delta shape."""
        H, W = 15, 25
        delta = np.random.rand(H, W).astype(np.float32)
        mask, _, _ = generate_pseudo_labels(delta, delta, delta, delta)
        assert mask.shape == (H, W)

    def test_returns_three_elements(self):
        """generate_pseudo_labels must return (mask, score, threshold)."""
        delta = np.random.rand(5, 5).astype(np.float32)
        result = generate_pseudo_labels(delta, delta, delta, delta)
        assert len(result) == 3

    def test_threshold_type_is_float(self):
        """Returned threshold must be a Python float."""
        delta = np.random.rand(8, 8).astype(np.float32)
        _, _, threshold = generate_pseudo_labels(delta, delta, delta, delta)
        assert isinstance(threshold, float)

    def test_shape_mismatch_raises(self):
        """Mismatched input shapes must raise FeatureEngineeringError."""
        from geoai.core.exceptions import FeatureEngineeringError
        d1 = np.ones((10, 10), dtype=np.float32)
        d2 = np.ones((20, 10), dtype=np.float32)
        with pytest.raises(FeatureEngineeringError):
            generate_pseudo_labels(d1, d2, d1, d1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
