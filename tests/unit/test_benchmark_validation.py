import sys
import os
import numpy as np
import pytest
import torch
from pathlib import Path

# Ensure geoai package is in import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.evaluation.health_checker import run_preflight_checks
from geoai.evaluation.consistency_validator import run_consistency_audit, compute_array_hash
from geoai.evaluation.statistics_manager import run_comparative_significance_analysis, compute_paired_bootstrap_ci
from geoai.evaluation.profiling_manager import ProfilingManager
from geoai.evaluation.explainability_validator import run_explainability_audit
from geoai.evaluation.failure_analysis import perform_failure_analysis

class TestBenchmarkValidationSuite:
    """Unit tests for the Benchmark Validation & Reproducibility Sprint features."""

    def test_health_checker_gate(self):
        """Verify pre-flight checks raise ValueError on critical failure unless force is True."""
        X_mock = np.zeros((10, 14))
        y_mock = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        coords_mock = np.array([[i, i] for i in range(10)])
        spatial_shape = (100, 100)
        
        # Test clean run
        res = run_preflight_checks(X_mock, y_mock, coords_mock, spatial_shape, split_policy="random")
        assert res["preflight_gate"] == "PASS"
        
        # Add a NaN label
        y_mock[0] = np.nan
        with pytest.raises(ValueError):
            run_preflight_checks(X_mock, y_mock, coords_mock, spatial_shape, split_policy="random", force=False)
            
        # Test force bypass
        res_forced = run_preflight_checks(X_mock, y_mock, coords_mock, spatial_shape, split_policy="random", force=True)
        assert res_forced["preflight_gate"] == "FAIL"
        assert res_forced["nan_labels"]["status"] == "FAIL"

    def test_consistency_validator(self):
        """Verify parity hashes mismatch triggers audit FAIL status."""
        arr_a = np.array([[0, 0], [1, 1]])
        arr_b = np.array([[0, 0], [2, 2]])
        
        hash_a = compute_array_hash(arr_a)
        hash_b = compute_array_hash(arr_b)
        assert hash_a != hash_b
        
        runs = [
            {
                "model_id": "model_a",
                "train_coords": arr_a,
                "val_coords": arr_a,
                "test_coords": arr_a,
                "patch_size": 15,
                "normalization": {"mean": 0.0, "std": 1.0},
                "feature_ordering": ["f1"],
                "channel_ordering": ["f1"],
                "ignore_index": -1,
                "random_state": 42,
                "padding_strategy": "reflect"
            },
            {
                "model_id": "model_b",
                "train_coords": arr_b, # Diff split coordinates
                "val_coords": arr_a,
                "test_coords": arr_a,
                "patch_size": 15,
                "normalization": {"mean": 0.0, "std": 1.0},
                "feature_ordering": ["f1"],
                "channel_ordering": ["f1"],
                "ignore_index": -1,
                "random_state": 42,
                "padding_strategy": "reflect"
            }
        ]
        
        audit_res = run_consistency_audit(runs)
        assert audit_res["audit_status"] == "FAIL"
        assert audit_res["train_split_consistent"] == "FAIL"

    def test_statistics_manager_bootstrap(self):
        """Verify comparative statistical tests compute correctly."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred_a = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        y_pred_b = np.array([0, 0, 0, 0, 0, 1, 0, 1])
        
        res = compute_paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, n_bootstraps=50)
        assert "f1_difference" in res
        assert "mean" in res["f1_difference"]
        assert "ci_lower" in res["f1_difference"]
        
        # Test full pairwise comparative testing
        model_predictions = {"model_a": y_pred_a, "model_b": y_pred_b}
        model_seeds = {"model_a": [0.8, 0.82], "model_b": [0.75, 0.77]}
        comps = run_comparative_significance_analysis(y_true, model_predictions, model_seeds)
        assert len(comps) == 1
        assert comps[0]["model_a"] == "model_a"
        assert comps[0]["model_b"] == "model_b"

    def test_profiling_manager(self):
        """Verify Training, Inference, and Storage profiles collect system RSS and model states."""
        pm = ProfilingManager()
        
        pm.start_training()
        # Mock training duration
        pm.end_training(num_samples=100)
        
        pm.start_inference()
        pm.end_inference(num_samples=50)
        
        # Profile a simple mock model
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(10, 2)
        model = MockModule()
        
        profile = pm.generate_profile(model)
        assert "training" in profile
        assert "inference" in profile
        assert "storage" in profile
        assert profile["storage"]["total_parameters"] == 22 # 10*2 weights + 2 biases

    def test_explainability_validator_audit(self):
        """Verify explainability validator asserts deterministic runs and range bounds."""
        class MockWrapper:
            def __init__(self):
                self.model_id = "mock_model"
                class SubModel:
                    def __init__(self):
                        self.backbone_type = "lightweight"
                    def eval(self):
                        pass
                self.model = SubModel()
                self.device = torch.device("cpu")
                
            def get_attention_maps(self, x1, x2):
                # Returns mock attention map bounded [0, 1]
                return torch.ones((2, 1, 15, 15)) * 0.5
                
            def get_token_embeddings(self, x1, x2):
                return torch.ones((2, 32, 15, 15))
                
        wrapper = MockWrapper()
        x1 = torch.randn(2, 7, 15, 15)
        x2 = torch.randn(2, 7, 15, 15)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = run_explainability_audit(wrapper, x1, x2, tmp_dir)
            assert res["audit_status"] == "PASS"
            assert res["deterministic_outputs"]["status"] == "PASS"
            assert res["attention_value_range"]["status"] == "PASS"

    def test_failure_analysis_taxonomy(self):
        """Verify fail cases classification maps boundary, scale, and clusters."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.9, 0.1], [0.1, 0.9]])
        coords = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]])
        spatial_shape = (100, 100)
        
        valid_mask = np.zeros(spatial_shape, dtype=bool)
        for r, c in coords:
            valid_mask[r, c] = True
            
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = perform_failure_analysis(y_true, y_pred, y_prob, coords, spatial_shape, valid_mask, tmp_dir)
            assert "worst_fp" in res
            assert "worst_fn" in res
            assert "failure_taxonomy" in res
            assert "boundary_failures" in res["failure_taxonomy"]
            assert "small_object_failures" in res["failure_taxonomy"]
