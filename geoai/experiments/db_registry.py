import sqlite3
import json
import csv
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

def initialize_database(db_path: str) -> None:
    """Initialize the SQLite schema for persistent campaign and run tracking."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Run Registry Table (tracks status and parameter configurations)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS run_registry (
        benchmark_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        preprocessing TEXT NOT NULL,
        boundary TEXT NOT NULL,
        protocol TEXT NOT NULL,
        seed INTEGER NOT NULL,
        status TEXT NOT NULL,
        execution_time_sec REAL DEFAULT 0.0
    )
    """)
    
    # 2. Metrics Registry Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metrics_registry (
        benchmark_id TEXT PRIMARY KEY,
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
        FOREIGN KEY (benchmark_id) REFERENCES run_registry (benchmark_id) ON DELETE CASCADE
    )
    """)
    
    # 3. System Manifest Table (reproducibility tracking)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS manifest_registry (
        benchmark_id TEXT PRIMARY KEY,
        system_info TEXT,
        package_info TEXT,
        dataset_checksum TEXT,
        FOREIGN KEY (benchmark_id) REFERENCES run_registry (benchmark_id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()
    logger.info("SQLite benchmark database initialized at '%s'.", db_path)

def get_run_status(db_path: str, benchmark_id: str) -> Optional[str]:
    """Retrieve the completion status of a benchmark run for resuming capability."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM run_registry WHERE benchmark_id = ?", (benchmark_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def register_run(db_path: str, run_data: Dict[str, Any]) -> None:
    """Insert or update a benchmark execution run entry."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO run_registry (
        benchmark_id, experiment_id, campaign_id, model_id, dataset_id,
        preprocessing, boundary, protocol, seed, status, execution_time_sec
    ) VALUES (:benchmark_id, :experiment_id, :campaign_id, :model_id, :dataset_id,
              :preprocessing, :boundary, :protocol, :seed, :status, :execution_time_sec)
    ON CONFLICT(benchmark_id) DO UPDATE SET
        status = excluded.status,
        execution_time_sec = excluded.execution_time_sec
    """, run_data)
    conn.commit()
    conn.close()

def register_metrics(db_path: str, benchmark_id: str, metrics: Dict[str, Any]) -> None:
    """Log predictions quality, calibration, and efficiency metrics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    metrics_data = {
        "benchmark_id": benchmark_id,
        "accuracy": metrics.get("accuracy", 0.0),
        "iou": metrics.get("iou", 0.0),
        "boundary_iou": metrics.get("boundary_iou", 0.0),
        "f1": metrics.get("f1", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "ece": metrics.get("ece", 0.0),
        "brier": metrics.get("brier", 0.0),
        "throughput_pixels_sec": metrics.get("throughput_pixels_sec", 0.0),
        "peak_memory_mb": metrics.get("peak_memory_mb", 0.0)
    }
    
    cursor.execute("""
    INSERT INTO metrics_registry (
        benchmark_id, accuracy, iou, boundary_iou, f1, precision, recall, ece, brier,
        throughput_pixels_sec, peak_memory_mb
    ) VALUES (:benchmark_id, :accuracy, :iou, :boundary_iou, :f1, :precision, :recall, :ece, :brier,
              :throughput_pixels_sec, :peak_memory_mb)
    ON CONFLICT(benchmark_id) DO UPDATE SET
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
    """, metrics_data)
    conn.commit()
    conn.close()

def register_manifest(db_path: str, benchmark_id: str, system_info: dict, package_info: dict, dataset_checksum: str) -> None:
    """Log system info hardware/software metadata for the run."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    manifest_data = {
        "benchmark_id": benchmark_id,
        "system_info": json.dumps(system_info),
        "package_info": json.dumps(package_info),
        "dataset_checksum": dataset_checksum
    }
    
    cursor.execute("""
    INSERT INTO manifest_registry (
        benchmark_id, system_info, package_info, dataset_checksum
    ) VALUES (:benchmark_id, :system_info, :package_info, :dataset_checksum)
    ON CONFLICT(benchmark_id) DO UPDATE SET
        system_info = excluded.system_info,
        package_info = excluded.package_info,
        dataset_checksum = excluded.dataset_checksum
    """, manifest_data)
    conn.commit()
    conn.close()

def export_registry_to_flat_files(db_path: str, csv_path: str, json_path: str) -> None:
    """Export the joined SQLite metrics table to standard CSV and JSON formats."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT r.benchmark_id, r.experiment_id, r.campaign_id, r.model_id, r.dataset_id,
           r.preprocessing, r.boundary, r.protocol, r.seed, r.status, r.execution_time_sec,
           m.accuracy, m.iou, m.boundary_iou, m.f1, m.precision, m.recall, m.ece, m.brier,
           m.throughput_pixels_sec, m.peak_memory_mb
    FROM run_registry r
    LEFT JOIN metrics_registry m ON r.benchmark_id = m.benchmark_id
    """)
    
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if not rows:
        return
        
    # Write JSON
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        
    # Write CSV
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    logger.info("Database records successfully exported to flat files: '%s' and '%s'.", csv_path, json_path)
