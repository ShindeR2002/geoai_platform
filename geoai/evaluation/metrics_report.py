import json
import csv
from pathlib import Path
import numpy as np
from typing import Dict, Any

def export_evaluation_metrics(metrics: Dict[str, Any], output_dir: Path) -> None:
    """Save full evaluation summary as JSON and flat CSV tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON Export
    # Exclude non-serializable objects (like numpy arrays)
    def clean_json_dict(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: clean_json_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_json_dict(v) for v in d]
        elif isinstance(d, np.ndarray):
            return d.tolist()
        elif isinstance(d, (np.float32, np.float64)):
            return float(d)
        elif isinstance(d, (np.int32, np.int64)):
            return int(d)
        else:
            return d
            
    clean_metrics = clean_json_dict(metrics)
    
    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2, default=str)
        
    # 2. Flat CSV Export of main key metrics
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric Name", "Value"])
        
        # Flatten dictionary to table row entries
        def write_flat_rows(prefix: str, data: Any):
            if isinstance(data, dict):
                for k, v in data.items():
                    write_flat_rows(f"{prefix}.{k}" if prefix else k, v)
            elif isinstance(data, list):
                if all(isinstance(x, (int, float)) for x in data):
                    writer.writerow([prefix, str(data)])
            elif isinstance(data, (int, float, str, bool)) or data is None:
                writer.writerow([prefix, str(data)])
                
        write_flat_rows("", clean_metrics)
