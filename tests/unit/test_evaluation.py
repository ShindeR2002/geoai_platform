import sys
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

# Ensure geoai package is in import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.evaluation.feature_analysis import (
    RandomForestMDIImportance,
    PermutationFeatureImportance,
    compute_correlations,
    compute_feature_statistics,
)
from geoai.evaluation.spatial_analysis import analyze_spatial_landscape
from geoai.evaluation.runtime_analysis import profile_execution_performance
from geoai.evaluation.statistical_analysis import compute_bootstrap_confidence_intervals
from geoai.evaluation.baseline_validator import validate_v1_v2_consistency
from geoai.evaluation.evaluation_manager import EvaluationManager

class MockRF:
    """Mock estimator to mimic RandomForestClassifier for tests."""
    def __init__(self) -> None:
        self.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        self.classes_ = np.array([0, 1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(X), 2))
        probs[:, 1] = 1.0
        return probs

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MockRF":
        return self


class TestFeatureAnalysis:
    """Tests for multi-method correlations and pluggable importances."""
    
    def test_mdi_importance(self):
        mock_model = MockRF()
        mdi = RandomForestMDIImportance()
        names = ["f1", "f2", "f3", "f4"]
        scores = mdi.compute_importance(mock_model, np.ones((5, 4)), np.ones(5), names)
        assert len(scores) == 4
        # Sorted descending
        assert list(scores.keys())[0] == "f4"
        assert scores["f4"] == 0.4

    def test_permutation_importance(self):
        # We need a small mock fit since sklearn's permutation_importance accesses score()
        mock_model = MockRF()
        perm = PermutationFeatureImportance(n_repeats=2, random_state=42)
        X = np.ones((10, 4))
        y = np.ones(10)
        names = ["f1", "f2", "f3", "f4"]
        
        from unittest.mock import patch
        import sklearn.inspection
        
        real_pi = sklearn.inspection.permutation_importance
        def mock_pi(*args, **kwargs):
            kwargs["n_jobs"] = 1
            return real_pi(*args, **kwargs)
            
        with patch("geoai.evaluation.feature_analysis.permutation_importance", side_effect=mock_pi):
            scores = perm.compute_importance(mock_model, X, y, names)
            
        assert len(scores) == 4

    def test_correlations_multi_method(self):
        X = np.array([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0]
        ])
        names = ["f1", "f2"]
        
        for method in ("pearson", "spearman", "kendall"):
            corr_df = compute_correlations(X, names, method=method)
            assert corr_df.shape == (2, 2)
            # Highly correlated columns
            assert np.isclose(corr_df.loc["f1", "f2"], 1.0, rtol=1e-3)

    def test_feature_statistics(self):
        X = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0]
        ])
        names = ["f1", "f2"]
        stats = compute_feature_statistics(X, names)
        assert stats["f1"]["mean"] == 2.0
        assert stats["f1"]["missing_count"] == 0
        assert stats["f2"]["mean"] == 20.0
        assert stats["f2"]["missing_count"] == 1


class TestSpatialAnalysisGracefulSkip:
    """Tests for spatial metrics and graceful missing GIS context layer skipping."""
    
    def test_graceful_missing_gis_skips(self):
        mask = np.zeros((10, 10))
        records = [
            {"area_m2": 100.0, "centroid": (2.0, 2.0)},
            {"area_m2": 200.0, "centroid": (5.0, 5.0)}
        ]
        
        # We pass a temporary empty directory as data_dir to verify it skips
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = analyze_spatial_landscape(
                significant_mask=mask,
                object_records=records,
                aoi_name="TestAOI",
                resolution_m=10.0,
                data_dir=tmp_dir
            )
            
            assert "roads" in res["skipped_gis_layers"]
            assert "water" in res["skipped_gis_layers"]
            assert res["gis_distances"]["mean_distance_to_roads_m"] is None
            assert res["gis_distances"]["mean_distance_to_water_m"] is None
            # Standard stats should still compute
            assert res["landscape"]["n_objects"] == 2
            assert res["patch_size_distribution"]["mean_size_ha"] == 0.015  # (100 + 200) / 2 = 150m2 = 0.015 ha


class TestStatisticalBootstrap:
    """Tests for bootstrap confidence interval estimation."""
    
    def test_bootstrap_cis(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        
        res = compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=20, random_seed=42)
        assert "f1" in res
        assert "iou" in res
        assert res["f1"]["ci_lower"] <= res["f1"]["ci_upper"]
        assert res["iou"]["ci_lower"] <= res["iou"]["ci_upper"]


class TestErrorAnalysis:
    """Tests for spatial error mapping diagnostics."""
    
    def test_analyze_prediction_errors(self):
        from geoai.evaluation.error_analysis import analyze_prediction_errors
        
        y_true = np.array([0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        valid_mask = np.array([False, True, True, False, True, True, False, True, False, False])
        spatial_shape = (2, 5)
        
        # valid_mask has 5 True values, matching len(y_true)
        res = analyze_prediction_errors(
            y_true=y_true,
            y_pred=y_pred,
            valid_mask=valid_mask,
            spatial_shape=spatial_shape
        )
        
        assert "confusion_pixel_counts" in res
        assert "rates" in res
        assert "error_map_2d" in res
        assert res["error_map_2d"].shape == (2, 5)
        
        # Verify the pixel mapping values
        flat_map = res["error_map_2d"].flatten()
        assert flat_map[1] == 0.0 # TN
        assert flat_map[2] == 1.0 # TP
        assert flat_map[4] == 3.0 # FN
        assert flat_map[5] == 2.0 # FP
        assert flat_map[7] == 0.0 # TN
        assert np.isnan(flat_map[0]) # Invalid


class TestStatisticalSignificanceTests:
    """Tests for paired t-test, Wilcoxon, and McNemar statistical significance tests."""
    
    def test_paired_t_test(self):
        from geoai.evaluation.statistical_analysis import run_paired_t_test
        metrics_a = np.array([0.81, 0.82, 0.83, 0.84, 0.85])
        metrics_b = np.array([0.80, 0.81, 0.82, 0.83, 0.84])
        res = run_paired_t_test(metrics_a, metrics_b)
        assert "statistic" in res
        assert "p_value" in res
        assert "significant" in res
        assert res["significant"] is True # 0.01 constant difference is highly significant

    def test_wilcoxon_test(self):
        from geoai.evaluation.statistical_analysis import run_wilcoxon_test
        metrics_a = np.array([0.81, 0.82, 0.83, 0.84, 0.85])
        metrics_b = np.array([0.80, 0.81, 0.82, 0.83, 0.84])
        res = run_wilcoxon_test(metrics_a, metrics_b)
        assert "statistic" in res
        assert "p_value" in res
        assert "significant" in res

    def test_mcnemar_test(self):
        from geoai.evaluation.statistical_analysis import run_mcnemar_test
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0] * 10)
        y_pred_a = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0] * 10)
        y_pred_b = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0] * 10)
        
        res = run_mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert "statistic" in res
        assert "p_value" in res
        assert "contingency_table" in res
        assert res["contingency_table"]["b"] == 0
        assert res["contingency_table"]["c"] == 20
