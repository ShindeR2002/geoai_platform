"""Runtime Profiler for TinyCD production framework.

Measures RAM, GPU memory allocations, file sizes, parameter counts,
and inference speed performance metrics.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch

try:
    import psutil
except ImportError:
    psutil = None

class RuntimeProfiler:
    """Orchestrates runtime metric gathering and CSV serialization."""

    def __init__(self, output_file: Path) -> None:
        """Initialize RuntimeProfiler.

        Args:
            output_file (Path): Path to output CSV file.
        """
        self.output_file = output_file
        self.metrics: Dict[str, Any] = {
            "training_time": 0.0,
            "validation_time": 0.0,
            "test_time": 0.0,
            "peak_ram": 0.0,
            "peak_gpu_memory": 0.0,
            "model_size": 0.0,
            "parameter_count": 0,
            "inference_latency": 0.0,
            "samples_per_second": 0.0
        }

    def update(self, key: str, value: Any) -> None:
        """Update a specific runtime metric.

        Args:
            key (str): Metric key name.
            value (Any): Value to set.
        """
        if key in self.metrics:
            self.metrics[key] = value

    def measure_resource_usage(self) -> None:
        """Measure current RAM and GPU memory usage and update peak values."""
        # 1. Measure RAM (RSS) using psutil
        if psutil is not None:
            try:
                process = psutil.Process(os.getpid())
                current_ram = process.memory_info().rss / (1024 * 1024)  # MB
                if current_ram > self.metrics["peak_ram"]:
                    self.metrics["peak_ram"] = current_ram
            except Exception as e:
                print(f"Warning: Failed to measure RAM usage: {e}")

        # 2. Measure GPU memory using PyTorch
        if torch.cuda.is_available():
            try:
                current_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                if current_gpu > self.metrics["peak_gpu_memory"]:
                    self.metrics["peak_gpu_memory"] = current_gpu
            except Exception as e:
                print(f"Warning: Failed to measure GPU memory: {e}")

    def measure_model_size(self, checkpoint_path: Path, model: torch.nn.Module) -> None:
        """Measure size of checkpoint file on disk and parameter count.

        Args:
            checkpoint_path (Path): Path to weight checkpoint file.
            model (torch.nn.Module): PyTorch module instance.
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.metrics["parameter_count"] = total_params

        # Get size on disk in MB
        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            self.metrics["model_size"] = size_mb

    def save(self) -> None:
        """Save runtime metrics to the output CSV file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    writer.writerow([k, f"{v:.6f}"])
                else:
                    writer.writerow([k, v])
