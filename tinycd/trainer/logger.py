"""CSV Logging module for TinyCD production framework.

This module logs epoch metrics, loss values, learning rate, and system memory stats.
"""

import csv
from pathlib import Path
from typing import Dict, Any

class CSVLogger:
    """Orchestrates writing of training history to a CSV file."""
    
    def __init__(self, output_file: Path) -> None:
        """Initialize the CSVLogger.

        Args:
            output_file (Path): Path to the target CSV file.
        """
        self.output_file = output_file
        self.headers = [
            "epoch",
            "train_loss",
            "val_loss",
            "train_precision",
            "val_precision",
            "train_recall",
            "val_recall",
            "train_f1",
            "val_f1",
            "train_iou",
            "val_iou",
            "learning_rate",
            "epoch_time",
            "gpu_memory",
            "ram_usage"
        ]
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Create output file and write header if not existing."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_file.exists():
            with open(self.output_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_epoch(self, epoch_data: Dict[str, Any]) -> None:
        """Log a single training epoch's results.

        Args:
            epoch_data (Dict[str, Any]): Dictionary containing epoch performance metrics.
        """
        row = [epoch_data.get(h, 0.0) for h in self.headers]
        # Format epoch as integer, and other floats nicely
        formatted_row = []
        for i, val in enumerate(row):
            if self.headers[i] == "epoch":
                formatted_row.append(int(val))
            elif isinstance(val, float):
                formatted_row.append(f"{val:.6f}")
            else:
                formatted_row.append(val)
                
        with open(self.output_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(formatted_row)
