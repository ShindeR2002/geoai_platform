import os
import time
import torch
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def get_peak_memory_rss() -> float:
    """Get peak RSS memory in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def get_peak_gpu_memory() -> float:
    """Get peak allocated CUDA memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0

class ProfilingManager:
    """System and engineering profile metrics collector."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.t0_train = 0.0
        self.train_time = 0.0
        self.train_samples = 0
        self.peak_train_ram = 0.0
        self.peak_train_gpu = 0.0
        
        self.t0_inf = 0.0
        self.inf_time = 0.0
        self.inf_samples = 0
        self.peak_inf_ram = 0.0
        self.peak_inf_gpu = 0.0

    def start_training(self):
        self.t0_train = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        self.peak_train_ram = get_peak_memory_rss()

    def end_training(self, num_samples: int):
        self.train_time = time.time() - self.t0_train
        self.train_samples = num_samples
        self.peak_train_ram = max(self.peak_train_ram, get_peak_memory_rss())
        self.peak_train_gpu = get_peak_gpu_memory()

    def start_inference(self):
        self.t0_inf = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        self.peak_inf_ram = get_peak_memory_rss()

    def end_inference(self, num_samples: int):
        self.inf_time = time.time() - self.t0_inf
        self.inf_samples = num_samples
        self.peak_inf_ram = max(self.peak_inf_ram, get_peak_memory_rss())
        self.peak_inf_gpu = get_peak_gpu_memory()

    def generate_profile(self, model: torch.nn.Module, checkpoint_path: str = None) -> Dict[str, Any]:
        # Count parameters
        total_params = 0
        trainable_params = 0
        if model is not None and hasattr(model, "parameters"):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        elif hasattr(model, "estimators_"):
            # Random Forest estimator counts
            total_params = sum(est.tree_.node_count for est in model.estimators_)
            trainable_params = total_params
            
        # Checkpoint size
        checkpoint_size_bytes = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint_size_bytes = os.path.getsize(checkpoint_path)
            
        # Throughput
        train_throughput = self.train_samples / self.train_time if self.train_time > 0 else 0.0
        inf_throughput = self.inf_samples / self.inf_time if self.inf_time > 0 else 0.0
        
        return {
            "training": {
                "time_sec": round(self.train_time, 4),
                "samples_count": self.train_samples,
                "throughput_samples_per_sec": round(train_throughput, 2),
                "peak_ram_mb": round(self.peak_train_ram, 2),
                "peak_gpu_mb": round(self.peak_train_gpu, 2)
            },
            "inference": {
                "time_sec": round(self.inf_time, 4),
                "predictions_count": self.inf_samples,
                "throughput_predictions_per_sec": round(inf_throughput, 2),
                "peak_ram_mb": round(self.peak_inf_ram, 2),
                "peak_gpu_mb": round(self.peak_inf_gpu, 2)
            },
            "storage": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "checkpoint_size_mb": round(checkpoint_size_bytes / (1024 * 1024), 4)
            }
        }

def save_profiling_report(profiles: Dict[str, Dict[str, Any]], filepath: str, csv_filepath: str, metadata: Dict[str, Any] = None) -> None:
    """Save markdown report and CSV summarizing the execution profiles of all models."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write CSV
    import csv
    with open(csv_filepath, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            "model_id", "total_parameters", "trainable_parameters", "checkpoint_size_mb",
            "train_time_sec", "train_throughput", "train_peak_ram_mb", "train_peak_gpu_mb",
            "inf_time_sec", "inf_throughput", "inf_peak_ram_mb", "inf_peak_gpu_mb"
        ])
        for model_id, prof in profiles.items():
            t_prof = prof["training"]
            i_prof = prof["inference"]
            s_prof = prof["storage"]
            writer.writerow([
                model_id, s_prof["total_parameters"], s_prof["trainable_parameters"], s_prof["checkpoint_size_mb"],
                t_prof["time_sec"], t_prof["throughput_samples_per_sec"], t_prof["peak_ram_mb"], t_prof["peak_gpu_mb"],
                i_prof["time_sec"], i_prof["throughput_predictions_per_sec"], i_prof["peak_ram_mb"], i_prof["peak_gpu_mb"]
            ])

    # Write Markdown
    with open(filepath, "w") as f:
        f.write("# GeoAI Platform — Unified Engineering Profiling Report\n\n")
        f.write(f"- **Execution Timestamp**: {timestamp}\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
        f.write("\n")
        
        f.write("## 1. Training Profile\n\n")
        f.write("| Model | Total Parameters | Trainable Parameters | Training Time (s) | Throughput (samples/s) | Peak RAM (MB) | Peak GPU (MB) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for model_id, prof in profiles.items():
            t_prof = prof["training"]
            s_prof = prof["storage"]
            f.write(f"| {model_id} | {s_prof['total_parameters']:,} | {s_prof['trainable_parameters']:,} | {t_prof['time_sec']:.2f} | {t_prof['throughput_samples_per_sec']:.2f} | {t_prof['peak_ram_mb']:.2f} | {t_prof['peak_gpu_mb']:.2f} |\n")
            
        f.write("\n## 2. Inference Profile\n\n")
        f.write("| Model | Inference Time (s) | Throughput (preds/s) | Peak RAM (MB) | Peak GPU (MB) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for model_id, prof in profiles.items():
            i_prof = prof["inference"]
            f.write(f"| {model_id} | {i_prof['time_sec']:.2f} | {i_prof['throughput_predictions_per_sec']:.2f} | {i_prof['peak_ram_mb']:.2f} | {i_prof['peak_gpu_mb']:.2f} |\n")
            
        f.write("\n## 3. Storage Profile\n\n")
        f.write("| Model | Checkpoint Size (MB) |\n")
        f.write("| :--- | :--- |\n")
        for model_id, prof in profiles.items():
            s_prof = prof["storage"]
            f.write(f"| {model_id} | {s_prof['checkpoint_size_mb']:.4f} |\n")
