import subprocess
import os
import platform
import sys
from typing import Dict, Any

def get_git_commit() -> str:
    """Retrieve current Git commit hash or report fallback."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "N/A - dirty/no-git"

def get_system_info() -> Dict[str, Any]:
    """Retrieve platform metadata for execution logs."""
    ram_gb = 0.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        # Fallback Windows query via wmic
        try:
            res = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            lines = res.stdout.strip().split("\n")
            if len(lines) > 1:
                bytes_ram = int(lines[1].strip())
                ram_gb = bytes_ram / (1024 ** 3)
        except Exception:
            ram_gb = 16.0  # Common execution fallback

    return {
        "python_version": sys.version.split()[0],
        "os": platform.system(),
        "os_release": platform.release(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": round(ram_gb, 2),
        "gpu": "None"
    }
