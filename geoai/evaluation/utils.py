import os
try:
    import psutil
except ImportError:
    psutil = None
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def setup_eval_dir(eval_id: Optional[str] = None, base_dir: str = "outputs/evaluation") -> Dict[str, Path]:
    """
    Setup the isolated artifacts directory structure for an evaluation session.
    Returns a dictionary of paths mapped to the subfolders.
    """
    if not eval_id:
        eval_id = f"EVAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    eval_path = Path(base_dir) / eval_id
    eval_path.mkdir(parents=True, exist_ok=True)
    
    subdirs = ["plots", "tables", "reports", "metrics", "validation", "artifacts", "logs"]
    paths = {"root": eval_path}
    for sub in subdirs:
        sub_path = eval_path / sub
        sub_path.mkdir(parents=True, exist_ok=True)
        paths[sub] = sub_path
        
    logger.info(f"Initialized isolated evaluation directories in {eval_path.resolve()}")
    return paths

def get_process_memory_mb() -> float:
    """Get the current resident set size (RSS) memory usage of the process in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def get_disk_usage_gb(path: str = ".") -> Dict[str, float]:
    """Get total, used, and free disk space in GB for the partition containing path."""
    try:
        usage = psutil.disk_usage(path)
        return {
            "total": round(usage.total / (1024**3), 2),
            "used": round(usage.used / (1024**3), 2),
            "free": round(usage.free / (1024**3), 2),
            "percent": usage.percent
        }
    except Exception:
        return {"total": 0.0, "used": 0.0, "free": 0.0, "percent": 0.0}

def get_cpu_count() -> int:
    """Get the physical/logical CPU count."""
    return os.cpu_count() or 1
