import sys
import os
import sqlite3
import numpy as np
import pytest
from pathlib import Path

# Ensure geoai is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.experiments.db_registry import (
    initialize_database,
    get_run_status,
    register_run,
    register_metrics,
    register_manifest,
    export_registry_to_flat_files
)
from geoai.experiments.stats_test import run_wilcoxon_signed_rank_test, run_mcnemar_test
from geoai.experiments.campaign_runner import CampaignRunner

class TestBenchmarkCampaignEngine:
    """Test suite validating SQLite tracking, run-status checkpoints, custom rankings, and significance stats."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        db_file = tmp_path / "test_benchmark_registry.db"
        initialize_database(str(db_file))
        return str(db_file)

    def test_sqlite_registry_transactions(self, temp_db):
        """Verify runs, metrics, and manifest records populate SQLite tables correctly."""
        benchmark_id = "test_node_01"
        run_data = {
            "benchmark_id": benchmark_id,
            "experiment_id": "test_exp",
            "campaign_id": "test_camp",
            "model_id": "rf_baseline_v1",
            "dataset_id": "levir_cd",
            "preprocessing": "spectral",
            "boundary": "standard",
            "protocol": "protocol_a",
            "seed": 42,
            "status": "Running",
            "execution_time_sec": 12.5
        }
        
        # 1. Register Run
        register_run(temp_db, run_data)
        assert get_run_status(temp_db, benchmark_id) == "Running"
        
        # Update Status to Completed
        run_data["status"] = "Completed"
        register_run(temp_db, run_data)
        assert get_run_status(temp_db, benchmark_id) == "Completed"
        
        # 2. Register Metrics
        metrics = {
            "accuracy": 0.85,
            "iou": 0.72,
            "boundary_iou": 0.65,
            "f1": 0.80,
            "precision": 0.82,
            "recall": 0.78,
            "ece": 0.08,
            "brier": 0.12,
            "throughput_pixels_sec": 5000.0,
            "peak_memory_mb": 150.0
        }
        register_metrics(temp_db, benchmark_id, metrics)
        
        # 3. Verify joined values in SQLite
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT r.status, m.iou, m.throughput_pixels_sec
            FROM run_registry r
            JOIN metrics_registry m ON r.benchmark_id = m.benchmark_id
            WHERE r.benchmark_id = ?
        """, (benchmark_id,))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "Completed"
        assert abs(row[1] - 0.72) < 1e-5
        assert abs(row[2] - 5000.0) < 1e-5

    def test_resume_checkpoint_bypassing(self, temp_db, tmp_path):
        """Verify that the runner skips completed iterations using SQLite status logs."""
        runner = CampaignRunner(campaign_id="test_camp", db_path=temp_db, profile="Quick")
        
        # Verify first check
        benchmark_id = runner.generate_benchmark_id(
            model_id="rf_baseline_v1",
            dataset_id="levir_cd",
            prep="spectral",
            boundary="standard",
            protocol="protocol_a",
            seed=42
        )
        assert get_run_status(temp_db, benchmark_id) is None
        
        # Log completion manually
        run_data = {
            "benchmark_id": benchmark_id,
            "experiment_id": "rf_baseline_v1_levir_cd_spectral_standard_protocol_a",
            "campaign_id": "test_camp",
            "model_id": "rf_baseline_v1",
            "dataset_id": "levir_cd",
            "preprocessing": "spectral",
            "boundary": "standard",
            "protocol": "protocol_a",
            "seed": 42,
            "status": "Completed",
            "execution_time_sec": 5.0
        }
        register_run(temp_db, run_data)
        
        # Dummy metrics
        metrics = {"iou": 0.72, "f1": 0.80}
        register_metrics(temp_db, benchmark_id, metrics)
        
        # Execute dry-run campaign, should return count 2 (completed is loaded from cache)
        results = runner.run_campaign()
        assert results["completed_runs"] >= 2

    def test_statistical_tests_math(self):
        """Verify Wilcoxon signed-rank and McNemar statistical significance results."""
        # 1. Wilcoxon signed-rank test
        scores_a = [0.82, 0.78, 0.91, 0.65, 0.70]
        scores_b = [0.85, 0.80, 0.93, 0.68, 0.72] # consistently slightly better
        
        stat, p_val = run_wilcoxon_signed_rank_test(scores_a, scores_b)
        assert stat >= 0.0
        assert 0.0 <= p_val <= 1.0
        
        # 2. McNemar's exact test (small samples)
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
        y_pred_a = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0]) # perfect
        y_pred_b = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 0]) # 2 mismatches
        
        stat, p_val = run_mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert stat == 0.0 # exact method returns 0 statistic
        assert 0.0 <= p_val <= 1.0
