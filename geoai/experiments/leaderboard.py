import json
import csv
from pathlib import Path
from typing import List, Dict, Any

class Leaderboard:
    """Manages reading, writing, and auto-updating the global experiment leaderboard."""
    def __init__(self, leaderboard_dir: str = "outputs/experiments") -> None:
        self.leaderboard_dir = Path(leaderboard_dir)
        self.leaderboard_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.leaderboard_dir / "leaderboard.json"
        self.csv_path = self.leaderboard_dir / "leaderboard.csv"

    def load(self) -> List[Dict[str, Any]]:
        """Load entries from the JSON leaderboard file."""
        if not self.json_path.exists():
            return []
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save(self, data: List[Dict[str, Any]]) -> None:
        """Save entries to both JSON and CSV files."""
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        if not data:
            return
        
        headers = ["experiment_id", "model_id", "aoi_name", "dataset_id", "f1", "iou", "precision", "recall", "runtime_sec", "memory_mb", "date"]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def add_entry(self, entry: Dict[str, Any]) -> None:
        """Add a new experiment evaluation entry and sort by IoU descending."""
        data = self.load()
        # De-duplicate previous entries
        data = [r for r in data if r.get("experiment_id") != entry.get("experiment_id")]
        
        cleaned_entry = {
            "experiment_id": entry.get("experiment_id", ""),
            "model_id": entry.get("model_id", ""),
            "aoi_name": entry.get("aoi_name", ""),
            "dataset_id": entry.get("dataset_id", ""),
            "f1": round(float(entry.get("f1", 0.0)), 6),
            "iou": round(float(entry.get("iou", 0.0)), 6),
            "precision": round(float(entry.get("precision", 0.0)), 6),
            "recall": round(float(entry.get("recall", 0.0)), 6),
            "runtime_sec": round(float(entry.get("runtime_sec", 0.0)), 4),
            "memory_mb": round(float(entry.get("memory_mb", 0.0)), 2),
            "date": entry.get("date", datetime_now_str())
        }
        data.append(cleaned_entry)
        data.sort(key=lambda x: x.get("iou", 0.0), reverse=True)
        self.save(data)


def datetime_now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
