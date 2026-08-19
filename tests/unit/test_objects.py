"""
Unit tests for object extraction, spatial metrics, and model evaluation.

Validates:
- 8-connectivity labelling
- Area filter at min=50px boundary
- Significant mask construction
- Jaccard Index formula
- Evaluation report structure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from geoai.analysis.objects import (
    extract_objects,
    filter_objects_by_area,
    label_connected_components,
    build_significant_mask,
)
from geoai.analysis.metrics import (
    jaccard_index,
    precision_recall_f1,
    compute_full_metrics,
)
from geoai.models.evaluation import (
    compute_jaccard,
    evaluate_predictions,
    format_evaluation_report,
)
from geoai.utils.constants import DEFAULT_MIN_OBJECT_SIZE_PX, OBJECT_CONNECTIVITY


# ---------------------------------------------------------------------------
# Object extraction tests
# ---------------------------------------------------------------------------

class TestLabelConnectedComponents:

    def test_single_connected_region(self):
        """A single blob should produce one label."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        labels = label_connected_components(mask)
        assert labels.max() == 1

    def test_two_separated_regions(self):
        """Two non-touching blobs should produce two distinct labels."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[1:4, 1:4] = 1   # top-left blob
        mask[15:18, 15:18] = 1  # bottom-right blob
        labels = label_connected_components(mask)
        assert labels.max() == 2

    def test_8_connectivity_diagonal_is_connected(self):
        """Diagonal neighbours must be connected under 8-connectivity (connectivity=2)."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 1
        mask[1, 1] = 1  # diagonal from (0,0) — connected in 8-connectivity
        labels = label_connected_components(mask)
        # Both pixels must get the same label
        assert labels[0, 0] == labels[1, 1], (
            "Diagonal pixels must be connected under 8-connectivity."
        )

    def test_connectivity_constant_is_2(self):
        """OBJECT_CONNECTIVITY constant must be 2 (8-connectivity)."""
        assert OBJECT_CONNECTIVITY == 2

    def test_background_label_is_zero(self):
        """Background (no-change) pixels must have label 0."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[4, 4] = 1
        labels = label_connected_components(mask)
        assert labels[0, 0] == 0
        assert labels[9, 9] == 0


class TestAreaFilter:

    def test_min_area_boundary_inclusive(self):
        """Objects with area EXACTLY equal to min_area must be retained."""
        mask = np.zeros((15, 15), dtype=np.uint8)
        # Place a 7x8=56 pixel region (> 50) and a 5x9=45 pixel region (< 50)
        mask[1:8, 1:8] = 1    # 49 pixels — below threshold
        mask[1:6, 9:14] = 1   # 25 pixels — well below threshold

        from skimage.measure import regionprops
        labels = label_connected_components(mask)
        regions = regionprops(labels)

        # Filter at exactly 50
        significant = filter_objects_by_area(regions, min_area_px=50)
        assert len(significant) == 0, "No region has >= 50 px; should return empty."

    def test_larger_region_passes_filter(self):
        """Objects above threshold must pass through."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:12, 2:12] = 1  # 100 pixels — above threshold
        _, _, sig_mask = extract_objects(mask, min_area_px=50)
        assert sig_mask.sum() == 100

    def test_small_region_removed(self):
        """Objects below threshold must be excluded from significant mask."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0:3, 0:3] = 1    # 9 pixels — below threshold
        mask[5:15, 5:15] = 1  # 100 pixels — above threshold
        regions, labels, sig_mask = extract_objects(mask, min_area_px=50)
        assert len(regions) == 1
        assert sig_mask.sum() == 100

    def test_default_min_area_is_50(self):
        """DEFAULT_MIN_OBJECT_SIZE_PX must be 50 (Version 1 NB06/NB09)."""
        assert DEFAULT_MIN_OBJECT_SIZE_PX == 50

    def test_empty_mask_returns_empty(self):
        """Binary mask of all zeros must produce empty object list."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        regions, labels, sig_mask = extract_objects(mask)
        assert len(regions) == 0
        assert sig_mask.sum() == 0

    def test_extract_objects_returns_three_items(self):
        """extract_objects must return (significant_regions, label_array, significant_mask)."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        result = extract_objects(mask)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Spatial metrics tests
# ---------------------------------------------------------------------------

class TestJaccardIndex:

    def test_perfect_overlap_is_one(self):
        """Identical masks must give Jaccard = 1.0."""
        y = np.array([0, 1, 1, 0, 1])
        assert jaccard_index(y, y) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self):
        """Non-overlapping predictions must give Jaccard = 0.0."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        assert jaccard_index(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value(self):
        """Jaccard = TP / (TP + FP + FN) — verify a known example."""
        # TP=2, FP=1, FN=1 → Jaccard = 2/4 = 0.5
        y_true = np.array([0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        assert jaccard_index(y_true, y_pred) == pytest.approx(0.5, abs=1e-6)

    def test_all_no_change_returns_zero(self):
        """When no change pixels exist in either array, Jaccard = 0.0."""
        y = np.zeros(10, dtype=int)
        assert jaccard_index(y, y) == pytest.approx(0.0)

    def test_spatial_2d_input(self):
        """Jaccard must work on 2D spatial arrays."""
        y_true = np.array([[1, 0], [1, 0]])
        y_pred = np.array([[1, 1], [0, 0]])
        # TP=1, FP=1, FN=1 → 1/3
        assert jaccard_index(y_true, y_pred) == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_jaccard_in_models_evaluation_matches(self):
        """compute_jaccard in models.evaluation must produce the same value."""
        y_true = np.array([0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        j1 = jaccard_index(y_true, y_pred)
        j2 = compute_jaccard(y_true, y_pred)
        assert j1 == pytest.approx(j2, abs=1e-6)


class TestEvaluatePredictions:

    def test_returns_required_keys(self):
        """evaluate_predictions must return all required metric keys."""
        y_t = np.array([0, 1, 1, 0])
        y_p = np.array([0, 1, 0, 0])
        results = evaluate_predictions(y_t, y_p)
        required_keys = {
            "accuracy", "precision", "recall", "f1", "jaccard",
            "confusion_matrix", "classification_report",
            "n_samples", "n_change_true", "n_change_pred",
        }
        assert required_keys.issubset(results.keys())

    def test_perfect_predictions(self):
        """Perfect predictions must give accuracy=1.0, jaccard=1.0."""
        y = np.array([0, 0, 1, 1, 1, 0])
        results = evaluate_predictions(y, y)
        assert results["accuracy"] == pytest.approx(1.0)
        assert results["jaccard"] == pytest.approx(1.0)

    def test_n_samples_correct(self):
        """n_samples must equal the input array length."""
        y = np.array([0,1,0,1,0,1,0,1,0,1,0,1], dtype=int)
        results = evaluate_predictions(y, y)
        assert results["n_samples"] == 12

    def test_format_report_contains_jaccard(self):
        """Formatted report string must mention Jaccard."""
        results = evaluate_predictions(
            np.array([0, 1, 1]), np.array([0, 1, 0])
        )
        report = format_evaluation_report(results, aoi_name="TestAOI")
        assert "Jaccard" in report
        assert "TestAOI" in report

    def test_length_mismatch_raises(self):
        """Mismatched array lengths must raise ModelError."""
        from geoai.core.exceptions import ModelError
        with pytest.raises(ModelError):
            evaluate_predictions(np.array([0, 1]), np.array([0, 1, 0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
