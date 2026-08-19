import sqlite3
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def initialize_research_db(db_path: str) -> None:
    """Initialize the normalized research campaign relational database schema."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys support
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # 1. Campaigns Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS campaigns (
        campaign_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Benchmarks Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS benchmarks (
        benchmark_id TEXT PRIMARY KEY,
        campaign_id TEXT,
        name TEXT NOT NULL,
        FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id) ON DELETE CASCADE
    )
    """)
    
    # 3. Experiments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        benchmark_id TEXT,
        model_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        config_hash TEXT NOT NULL,
        FOREIGN KEY(benchmark_id) REFERENCES benchmarks(benchmark_id) ON DELETE CASCADE
    )
    """)
    
    # 4. Runs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        seed INTEGER NOT NULL,
        status TEXT NOT NULL,
        git_commit TEXT,
        device_mode TEXT,
        execution_time_sec REAL DEFAULT 0.0,
        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
    )
    """)
    
    # 5. Metrics Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        run_id TEXT PRIMARY KEY,
        accuracy REAL,
        iou REAL,
        boundary_iou REAL,
        f1 REAL,
        precision REAL,
        recall REAL,
        ece REAL,
        brier REAL,
        throughput_pixels_sec REAL,
        peak_memory_mb REAL,
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """)
    
    # 6. Artifacts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        run_id TEXT,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """)
    
    # 7. Text Interpretations & Report Synthesis Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports_synthesis (
        campaign_id TEXT PRIMARY KEY,
        discussion TEXT,
        conclusions TEXT,
        future_work TEXT,
        FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id) ON DELETE CASCADE
    )
    """)
    
    # 8. Warnings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS warnings_log (
        warning_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        message TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """)

    # 9. Errors Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS errors_log (
        error_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        message TEXT NOT NULL,
        stack_trace TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """)

    # 10. Skipped Runs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS skipped_runs (
        run_id TEXT PRIMARY KEY,
        reason TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 11. Reproducibility Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reproducibility_runs (
        benchmark_id TEXT PRIMARY KEY,
        is_reproducible BOOLEAN,
        original_iou REAL,
        replicated_iou REAL,
        deviation REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(benchmark_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """)

    # 12. Execution Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS execution_logs (
        run_id TEXT PRIMARY KEY,
        console_output TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()
    logger.info("SQLite research database initialized successfully at '%s'.", db_path)

def register_campaign(db_path: str, campaign_id: str, name: str, description: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO campaigns (campaign_id, name, description)
    VALUES (?, ?, ?)
    ON CONFLICT(campaign_id) DO UPDATE SET
        name = excluded.name,
        description = excluded.description
    """, (campaign_id, name, description))
    conn.commit()
    conn.close()

def register_benchmark(db_path: str, benchmark_id: str, campaign_id: str, name: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO benchmarks (benchmark_id, campaign_id, name)
    VALUES (?, ?, ?)
    ON CONFLICT(benchmark_id) DO UPDATE SET
        name = excluded.name
    """, (benchmark_id, campaign_id, name))
    conn.commit()
    conn.close()

def register_experiment(db_path: str, experiment_id: str, benchmark_id: str, model_id: str, dataset_id: str, config_hash: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO experiments (experiment_id, benchmark_id, model_id, dataset_id, config_hash)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(experiment_id) DO UPDATE SET
        config_hash = excluded.config_hash
    """, (experiment_id, benchmark_id, model_id, dataset_id, config_hash))
    conn.commit()
    conn.close()

def register_run(db_path: str, run_id: str, experiment_id: str, seed: int, status: str, git_commit: str, device_mode: str, exec_time: float) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO runs (run_id, experiment_id, seed, status, git_commit, device_mode, execution_time_sec)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(run_id) DO UPDATE SET
        status = excluded.status,
        execution_time_sec = excluded.execution_time_sec
    """, (run_id, experiment_id, seed, status, git_commit, device_mode, exec_time))
    conn.commit()
    conn.close()

def register_metrics(db_path: str, run_id: str, metrics_dict: Dict[str, Any]) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO metrics (
        run_id, accuracy, iou, boundary_iou, f1, precision, recall, ece, brier,
        throughput_pixels_sec, peak_memory_mb
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(run_id) DO UPDATE SET
        accuracy = excluded.accuracy,
        iou = excluded.iou,
        boundary_iou = excluded.boundary_iou,
        f1 = excluded.f1,
        precision = excluded.precision,
        recall = excluded.recall,
        ece = excluded.ece,
        brier = excluded.brier,
        throughput_pixels_sec = excluded.throughput_pixels_sec,
        peak_memory_mb = excluded.peak_memory_mb
    """, (
        run_id,
        metrics_dict.get("accuracy", 0.0),
        metrics_dict.get("iou", 0.0),
        metrics_dict.get("boundary_iou", 0.0),
        metrics_dict.get("f1", 0.0),
        metrics_dict.get("precision", 0.0),
        metrics_dict.get("recall", 0.0),
        metrics_dict.get("ece", 0.0),
        metrics_dict.get("brier", 0.0),
        metrics_dict.get("throughput_pixels_sec", 0.0),
        metrics_dict.get("peak_memory_mb", 0.0)
    ))
    conn.commit()
    conn.close()

def register_artifact(db_path: str, artifact_id: str, run_id: str, file_path: str, file_type: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO artifacts (artifact_id, run_id, file_path, file_type)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(artifact_id) DO UPDATE SET
        file_path = excluded.file_path,
        file_type = excluded.file_type
    """, (artifact_id, run_id, file_path, file_type))
    conn.commit()
    conn.close()

def register_text_reports(db_path: str, campaign_id: str, discussion: str, conclusions: str, future_work: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO reports_synthesis (campaign_id, discussion, conclusions, future_work)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(campaign_id) DO UPDATE SET
        discussion = excluded.discussion,
        conclusions = excluded.conclusions,
        future_work = excluded.future_work
    """, (campaign_id, discussion, conclusions, future_work))
    conn.commit()
    conn.close()

def get_run_status(db_path: str, run_id: str) -> Optional[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def export_research_db_to_flat(db_path: str, csv_path: str, json_path: str) -> None:
    """Join schema runs, metrics, and configuration details and export flat representations."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
    SELECT r.run_id, r.experiment_id, r.seed, r.status, r.git_commit, r.device_mode, r.execution_time_sec,
           e.model_id, e.dataset_id, e.config_hash,
           m.accuracy, m.iou, m.boundary_iou, m.f1, m.precision, m.recall, m.ece, m.brier,
           m.throughput_pixels_sec, m.peak_memory_mb
    FROM runs r
    LEFT JOIN experiments e ON r.experiment_id = e.experiment_id
    LEFT JOIN metrics m ON r.run_id = m.run_id
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if not rows:
        return
        
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def register_warning(db_path: str, run_id: str, message: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO warnings_log (run_id, message) VALUES (?, ?)", (run_id, message))
    conn.commit()
    conn.close()

def register_error(db_path: str, run_id: str, message: str, stack_trace: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO errors_log (run_id, message, stack_trace) VALUES (?, ?, ?)", (run_id, message, stack_trace))
    conn.commit()
    conn.close()

def register_skipped_run(db_path: str, run_id: str, reason: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO skipped_runs (run_id, reason) VALUES (?, ?)", (run_id, reason))
    conn.commit()
    conn.close()

def register_reproducibility_run(db_path: str, benchmark_id: str, is_reproducible: bool, original_iou: float, replicated_iou: float, deviation: float) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO reproducibility_runs (benchmark_id, is_reproducible, original_iou, replicated_iou, deviation)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(benchmark_id) DO UPDATE SET
        is_reproducible = excluded.is_reproducible,
        replicated_iou = excluded.replicated_iou,
        deviation = excluded.deviation
    """, (benchmark_id, int(is_reproducible), original_iou, replicated_iou, deviation))
    conn.commit()
    conn.close()

def register_execution_log(db_path: str, run_id: str, console_output: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO execution_logs (run_id, console_output)
    VALUES (?, ?)
    ON CONFLICT(run_id) DO UPDATE SET
        console_output = excluded.console_output
    """, (run_id, console_output))
    conn.commit()
    conn.close()
