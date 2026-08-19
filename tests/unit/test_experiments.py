import sys
import os
from pathlib import Path
import tempfile
import yaml
import numpy as np
import pytest

# Ensure geoai package is in import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.experiments.experiment import Experiment
from geoai.experiments.experiment_config import ExperimentConfig
from geoai.experiments.experiment_registry import (
    get_experiment_model,
    ModelImplementationUnavailableError,
    ResearchProvenance,
)
from geoai.experiments.leaderboard import Leaderboard
from geoai.experiments.comparison import compare_experiments
from geoai.experiments.benchmark import BenchmarkSuite
from geoai.experiments.benchmark_report import generate_benchmark_markdown_report
from geoai.experiments.ablation import AblationManager

class TestExperimentConfig:
    """Tests for ExperimentConfig YAML loading and parsing."""
    
    def test_load_config_fields(self):
        config_data = {
            "experiment_id": "test_exp_rf",
            "description": "Test description",
            "model": {
                "model_id": "rf_baseline_v1",
                "hyperparameters": {
                    "n_estimators": 50,
                    "random_state": 42
                }
            },
            "dataset": {
                "dataset_id": "dholera_sentinel_v1",
                "test_size": 0.25,
                "random_state": 99
            },
            "baseline_lock": {
                "inherit_preprocessing": True,
                "inherit_feature_engineering": True,
                "deviations": ["None"]
            },
            "ablation": {}
        }
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump(config_data, f)
            f_path = Path(f.name)
            
        try:
            config = ExperimentConfig(f_path)
            assert config.experiment_id == "test_exp_rf"
            assert config.description == "Test description"
            assert config.model_id == "rf_baseline_v1"
            assert config.hyperparameters["n_estimators"] == 50
            assert config.dataset_id == "dholera_sentinel_v1"
            assert config.test_size == 0.25
            assert config.random_state == 99
            assert config.baseline_lock["inherit_preprocessing"] is True
        finally:
            if f_path.exists():
                f_path.unlink()


class TestExperimentState:
    """Tests for Experiment lifecycle and state transitions."""
    
    def test_valid_transitions(self):
        exp = Experiment(
            experiment_id="exp_test",
            model_id="rf_enhanced_v1",
            dataset_id="dholera_sentinel_v1",
            aoi_name="Dholera",
            config={}
        )
        assert exp.get_state() == "Draft"
        
        exp.set_state("Registered")
        assert exp.get_state() == "Registered"
        
        exp.set_state("Running")
        assert exp.get_state() == "Running"
        
        exp.set_state("Completed")
        assert exp.get_state() == "Completed"
        
    def test_invalid_state_raises_value_error(self):
        exp = Experiment(
            experiment_id="exp_test",
            model_id="rf_enhanced_v1",
            dataset_id="dholera_sentinel_v1",
            aoi_name="Dholera",
            config={}
        )
        with pytest.raises(ValueError):
            exp.set_state("UnknownState")


class TestExperimentRegistry:
    """Tests for Model Capabilities and implementation availability checks."""
    
    def test_implemented_model_capabilities(self):
        model = get_experiment_model("rf_baseline_v1")
        assert model.is_implemented() is True
        caps = model.get_capabilities()
        assert "Classical ML" in caps.model_type
        assert len(caps.expected_input_channels) == 14
        
    def test_unimplemented_model_raises_on_predict(self):
        model = get_experiment_model("siamese")
        assert model.is_implemented() is False
        caps = model.get_capabilities()
        assert caps.model_type == "Siamese Networks (Siam-FullyConv)"
        assert caps.training_status == "Not Yet Implemented"
        
        X = np.ones((5, 4), dtype=np.float32)
        with pytest.raises(ModelImplementationUnavailableError):
            model.predict(X)
            
        with pytest.raises(ModelImplementationUnavailableError):
            model.predict_proba(X)


class TestComparisonAndBenchmark:
    """Tests for model metric comparisons and delta reports."""
    
    def test_metric_comparison_deltas(self):
        exp1_data = {
            "experiment_id": "baseline",
            "model_id": "rf_baseline_v1",
            "dataset_id": "dholera_sentinel_v1",
            "metrics": {
                "f1": 0.80,
                "iou": 0.70,
                "train_time_sec": 5.0,
                "memory_usage_mb": 100.0
            }
        }
        
        exp2_data = {
            "experiment_id": "candidate",
            "model_id": "rf_enhanced_v1",
            "dataset_id": "dholera_sentinel_v1",
            "metrics": {
                "f1": 0.85,
                "iou": 0.75,
                "train_time_sec": 10.0,
                "memory_usage_mb": 150.0
            }
        }
        
        res = compare_experiments(exp1_data, exp2_data)
        assert res["dataset_match"] is True
        assert res["metric_deltas"]["f1"]["delta"] == 0.05
        assert res["metric_deltas"]["iou"]["delta"] == 0.05
        assert res["resource_comparison"]["train_time_sec"]["ratio"] == 2.0
        assert res["resource_comparison"]["memory_usage_mb"]["delta"] == 50.0

    def test_benchmark_markdown_report_generation(self):
        bench_results = {
            "baseline_id": "baseline_ref",
            "baseline_data": {
                "experiment_id": "baseline_ref",
                "model_id": "rf_baseline_v1",
                "dataset_id": "dholera_sentinel_v1",
                "metrics": {"iou": 0.70, "f1": 0.80, "train_time_sec": 5.0}
            },
            "comparisons": {
                "cand_1": {
                    "exp2_model": "rf_enhanced_v1",
                    "metric_deltas": {
                        "iou": {"val2": 0.75, "delta": 0.05},
                        "f1": {"val2": 0.85, "delta": 0.05}
                    },
                    "resource_comparison": {
                        "train_time_sec": {"val1": 5.0, "val2": 10.0},
                        "memory_usage_mb": {"val1": 100.0, "val2": 150.0}
                    },
                    "dataset_match": True,
                    "seed_match": True
                }
            }
        }
        report = generate_benchmark_markdown_report(bench_results)
        assert "baseline_ref" in report
        assert "cand_1" in report
        assert "0.050000" in report
        assert "0.50x" in report


class TestLeaderboard:
    """Tests for leaderboard thread-safe addition and loading."""
    
    def test_leaderboard_file_ops(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lb = Leaderboard(leaderboard_dir=tmp_dir)
            assert lb.load() == []
            
            entry = {
                "experiment_id": "exp_1",
                "model_id": "rf_enhanced_v1",
                "aoi_name": "Dholera",
                "dataset_id": "dholera_sentinel_v1",
                "f1": 0.82,
                "iou": 0.72,
                "precision": 0.80,
                "recall": 0.84,
                "runtime_sec": 6.5,
                "memory_mb": 120.0
            }
            lb.add_entry(entry)
            
            loaded = lb.load()
            assert len(loaded) == 1
            assert loaded[0]["experiment_id"] == "exp_1"
            assert loaded[0]["iou"] == 0.720000
            
            # De-duplicate check
            lb.add_entry(entry)
            assert len(lb.load()) == 1
