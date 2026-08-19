import os
try:
    import psutil
except ImportError:
    psutil = None
from typing import Dict, Any

def profile_execution_performance(
    train_time: float = 0.0,
    inference_time: float = 0.0,
    feature_eng_time: float = 0.0,
    object_extraction_time: float = 0.0,
    export_time: float = 0.0,
    model_size_bytes: int = 0,
    total_pixels: int = 0
) -> Dict[str, Any]:
    """Assemble runtime performance, latency, memory bounds, and execution profiles."""
    # Peak memory RSS
    mem_mb = 0.0
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        pass
        
    cpu_percent = 0.0
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
    except Exception:
        pass

    # Throughput: total_pixels / inference_time
    throughput = total_pixels / inference_time if inference_time > 0 else 0.0
    
    # Disk Usage
    disk_total = 0.0
    disk_used = 0.0
    disk_percent = 0.0
    try:
        disk = psutil.disk_usage(".")
        disk_total = disk.total / (1024**3)
        disk_used = disk.used / (1024**3)
        disk_percent = disk.percent
    except Exception:
        pass

    return {
        "latency_sec": {
            "training_time": round(train_time, 4),
            "inference_time": round(inference_time, 4),
            "feature_engineering": round(feature_eng_time, 4),
            "object_extraction": round(object_extraction_time, 4),
            "export_time": round(export_time, 4),
            "total_execution_time": round(train_time + inference_time + feature_eng_time + object_extraction_time + export_time, 4)
        },
        "system": {
            "peak_memory_mb": round(mem_mb, 2),
            "cpu_usage_pct": round(cpu_percent, 2),
            "disk_total_gb": round(disk_total, 2),
            "disk_used_gb": round(disk_used, 2),
            "disk_usage_pct": round(disk_percent, 2),
            "model_size_mb": round(model_size_bytes / (1024 * 1024), 4)
        },
        "throughput": {
            "inference_pixels_per_sec": round(throughput, 2)
        }
    }
