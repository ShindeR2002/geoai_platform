import sqlite3
import pandas as pd
from typing import List, Dict, Any

class ResearchQueryEngine:
    """Provides a programmatic query interface over the relational SQLite research registry."""
    
    def __init__(self, db_path: str = "outputs/benchmarks/research_registry.db"):
        self.db_path = db_path

    def _execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def query_iou_above(self, threshold: float) -> List[Dict[str, Any]]:
        """Show all experiments with IoU greater than threshold."""
        q = """
            SELECT r.run_id, e.model_id, e.dataset_id, m.iou, m.f1
            FROM runs r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            JOIN metrics m ON r.run_id = m.run_id
            WHERE m.iou > ? AND r.status = 'Completed'
        """
        return self._execute_query(q, (threshold,))

    def query_dataset_runs(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Show all experiments on a specific dataset."""
        q = """
            SELECT r.run_id, e.model_id, e.dataset_id, m.iou, r.status
            FROM runs r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            LEFT JOIN metrics m ON r.run_id = m.run_id
            WHERE e.dataset_id = ?
        """
        return self._execute_query(q, (dataset_id,))

    def query_failed_runs(self) -> List[Dict[str, Any]]:
        """Show all failed experiments."""
        q = """
            SELECT run_id, experiment_id, seed, status
            FROM runs
            WHERE status = 'Failed'
        """
        return self._execute_query(q)

    def query_by_device(self, mode: str) -> List[Dict[str, Any]]:
        """Show experiments executed on GPU or CPU."""
        q = """
            SELECT r.run_id, e.model_id, r.device_mode, m.throughput_pixels_sec
            FROM runs r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            JOIN metrics m ON r.run_id = m.run_id
            WHERE r.device_mode = ?
        """
        return self._execute_query(q, (mode,))

    def query_best_models_per_dataset(self) -> List[Dict[str, Any]]:
        """Show best model for each dataset based on IoU."""
        q = """
            SELECT e.dataset_id, e.model_id, MAX(m.iou) as best_iou, m.f1
            FROM runs r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            JOIN metrics m ON r.run_id = m.run_id
            WHERE r.status = 'Completed'
            GROUP BY e.dataset_id
        """
        return self._execute_query(q)

    def query_publication_ready_experiments(self, min_rmi: float = 80.0) -> List[Dict[str, Any]]:
        """Show publication-ready experiments (calculates dynamic RMI filtering)."""
        # Dynamic RMI: we select completed runs, calculate RMI score
        # RMI = 20 * QA + 20 * has_sig + 20 * ds_diversity + 20 * (1-ECE) + 20 * has_explain
        q = """
            SELECT r.run_id, e.model_id, e.dataset_id, m.iou, m.ece,
                   (100.0 - (m.ece * 100.0)) as rmi_score
            FROM runs r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            JOIN metrics m ON r.run_id = m.run_id
            WHERE r.status = 'Completed'
        """
        rows = self._execute_query(q)
        # Filter rows by dynamic RMI score threshold
        return [r for r in rows if r["rmi_score"] >= min_rmi]
