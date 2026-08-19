import sys
import os
import sqlite3
import pytest

# Ensure geoai is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.experiments.provenance import collect_provenance_metadata, get_git_commit
from geoai.experiments.db_research import (
    initialize_research_db,
    register_campaign,
    register_benchmark,
    register_experiment,
    register_run,
    register_metrics,
    register_warning,
    register_error,
    register_skipped_run,
    register_reproducibility_run,
    register_execution_log
)
from geoai.experiments.qa_system import BenchmarkQASystem
from geoai.experiments.query_engine import ResearchQueryEngine

class TestResearchPlatformUpgrade:
    """Test suite validating platform scientific reproducibility upgrade controls."""

    @pytest.fixture
    def research_db(self, tmp_path):
        db_file = tmp_path / "test_research_registry.db"
        initialize_research_db(str(db_file))
        return str(db_file)

    def test_provenance_retrieval(self):
        """Verify the provenance modules extract git commits, branch names, environment variables, and memory specs."""
        meta = collect_provenance_metadata(__file__)
        
        assert "git_commit_hash" in meta
        assert "git_branch" in meta
        assert "git_dirty" in meta
        assert "conda_env" in meta
        assert "env_variables" in meta
        assert "PYTHONHASHSEED" in meta["env_variables"]
        assert "python_version" in meta
        assert "cpu_name" in meta
        assert "ram_gb" in meta
        assert "package_versions" in meta
        assert isinstance(meta["package_versions"], dict)
        assert "torch" in meta["package_versions"]

    def test_relational_schema_integrity(self, research_db):
        """Verify SQLite foreign key constraints, cascade deletes, and logging tables."""
        camp_id = "camp_test"
        bench_id = "bench_test"
        exp_id = "exp_test"
        run_id = "run_test"
        
        # 1. Register campaign components
        register_campaign(research_db, camp_id, "Test Campaign", "Desc")
        register_benchmark(research_db, bench_id, camp_id, "Test Benchmark")
        register_experiment(research_db, exp_id, bench_id, "rf_baseline_v1", "levir_cd", "config_sha")
        register_run(research_db, run_id, exp_id, 42, "Completed", "git_sha", "GPU", 10.0)
        
        metrics = {"iou": 0.75, "f1": 0.82}
        register_metrics(research_db, run_id, metrics)
        
        # Log warnings, errors, skipped runs, and logs
        register_warning(research_db, run_id, "Test warning message")
        register_error(research_db, run_id, "Test error message", "Stacktrace text")
        register_skipped_run(research_db, "skipped_run_01", "Mock skipped reason")
        register_reproducibility_run(research_db, run_id, True, 0.75, 0.74, 0.01)
        register_execution_log(research_db, run_id, "Mock stdout output")
        
        # Check tables populated
        conn = sqlite3.connect(research_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM warnings_log")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("SELECT COUNT(*) FROM errors_log")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("SELECT COUNT(*) FROM skipped_runs")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("SELECT COUNT(*) FROM reproducibility_runs")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("SELECT COUNT(*) FROM execution_logs")
        assert cursor.fetchone()[0] == 1
        
        # 2. Delete Campaign and assert cascade deletes
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("DELETE FROM campaigns WHERE campaign_id = ?", (camp_id,))
        conn.commit()
        
        # Verify cascades deleted everything associated with run_id
        cursor.execute("SELECT COUNT(*) FROM experiments")
        assert cursor.fetchone()[0] == 0
        
        cursor.execute("SELECT COUNT(*) FROM runs")
        assert cursor.fetchone()[0] == 0
        
        cursor.execute("SELECT COUNT(*) FROM warnings_log")
        assert cursor.fetchone()[0] == 0
        
        conn.close()

    def test_qa_scoring_and_rmi(self):
        """Verify that BenchmarkQASystem checks compliance scores and calculates RMI maturity indexes."""
        runs = [
            {"preprocessing": "spectral", "seed": 42, "status": "Completed"},
            {"preprocessing": "spectral", "seed": 42, "status": "Completed"}
        ]
        score, errors = BenchmarkQASystem.evaluate_campaign_qa(
            runs_list=runs,
            expected_runs_count=2,
            feature_names=["Red_T1", "Green_T1"],
            canonical_feature_names=["Red_T1", "Green_T1"]
        )
        assert score == 100.0
        
        # Calculate RMI Score
        rmi = BenchmarkQASystem.calculate_research_maturity_index(
            qa_score=score,
            has_significance=True,
            num_datasets=5,
            ece=0.0, # perfect calibration
            has_explain=True
        )
        assert rmi == 100.0

    def test_query_engine_interface(self, research_db):
        """Verify the research query engine returns matches from SQLite records."""
        register_campaign(research_db, "camp1", "Campaign", "Desc")
        register_benchmark(research_db, "bench1", "camp1", "Benchmark")
        register_experiment(research_db, "exp1", "bench1", "rf_baseline_v1", "levir_cd", "sha")
        register_run(research_db, "run1", "exp1", 42, "Completed", "git", "GPU", 2.0)
        register_metrics(research_db, "run1", {"iou": 0.85, "ece": 0.05})
        
        engine = ResearchQueryEngine(research_db)
        
        # Test iou above
        res_iou = engine.query_iou_above(0.80)
        assert len(res_iou) == 1
        assert res_iou[0]["run_id"] == "run1"
        
        # Test dataset runs
        res_ds = engine.query_dataset_runs("levir_cd")
        assert len(res_ds) == 1
        
        # Test best model
        res_best = engine.query_best_models_per_dataset()
        assert len(res_best) == 1
        assert res_best[0]["model_id"] == "rf_baseline_v1"
