import os
import sys
import platform
import subprocess
import hashlib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_git_commit() -> str:
    """Retrieve the current Git commit hash from the repository log."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown_commit"

def get_git_branch() -> str:
    """Retrieve the current Git branch name."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown_branch"

def get_git_dirty_status() -> bool:
    """Check if the local repository has uncommitted modifications."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return len(out) > 0
    except Exception:
        return False

def get_sys_memory_gb() -> float:
    """Retrieve total physical memory size in GB (Windows-compatible)."""
    try:
        out = subprocess.check_output(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"], stderr=subprocess.DEVNULL).decode("utf-8")
        lines = out.strip().split("\n")
        if len(lines) > 1:
            bytes_val = int(lines[1].strip())
            return round(bytes_val / (1024**3), 1)
    except Exception:
        pass
    return 16.0 # Fallback default memory

def get_cpu_name() -> str:
    """Retrieve processor specifications."""
    try:
        out = subprocess.check_output(["wmic", "cpu", "get", "name"], stderr=subprocess.DEVNULL).decode("utf-8")
        lines = out.strip().split("\n")
        if len(lines) > 1:
            return lines[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"

def get_gpu_name() -> str:
    """Retrieve the primary graphics device name if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "None (CPU Execution Mode)"

def get_cuda_version() -> str:
    """Retrieve CUDA version loaded by PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda or "Unknown CUDA"
    except Exception:
        pass
    return "N/A"

def get_nvidia_driver_version() -> str:
    """Retrieve NVIDIA driver version via nvidia-smi."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return out
    except Exception:
        return "Unknown"

def get_cudnn_version() -> str:
    """Retrieve cuDNN library version loaded by PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return str(torch.backends.cudnn.version())
    except Exception:
        pass
    return "N/A"

def get_conda_env_name() -> str:
    """Get active Conda or Virtual environment name."""
    env_path = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
    if env_path:
        return env_path.split(os.sep)[-1]
    return "base"

def get_package_versions() -> Dict[str, str]:
    """Compile version information for key packages in the active workspace."""
    packages = ["torch", "torchvision", "sklearn", "xgboost", "lightgbm", "numpy", "matplotlib", "streamlit"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = "Not Installed"
    return versions

def calculate_config_hash(config_path: str) -> str:
    """Generate SHA-256 checksum of an experiment configuration file."""
    if not os.path.exists(config_path):
        return "missing_config_file"
    hasher = hashlib.sha256()
    try:
        with open(config_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Failed calculating hash for config {config_path}: {e}")
        return "hash_error"

def collect_provenance_metadata(config_path: str) -> Dict[str, Any]:
    """Retrieve system hardware details, versions, and configurations hashes."""
    env_vars = {}
    for var in ["PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"]:
        env_vars[var] = os.environ.get(var, "unset")
        
    return {
        "git_commit_hash": get_git_commit(),
        "git_branch": get_git_branch(),
        "git_dirty": get_git_dirty_status(),
        "python_version": sys.version.split()[0],
        "conda_env": get_conda_env_name(),
        "cuda_version": get_cuda_version(),
        "nvidia_driver": get_nvidia_driver_version(),
        "cudnn_version": get_cudnn_version(),
        "gpu_name": get_gpu_name(),
        "cpu_name": get_cpu_name(),
        "ram_gb": get_sys_memory_gb(),
        "operating_system": f"{platform.system()} {platform.release()}",
        "package_versions": get_package_versions(),
        "config_hash": calculate_config_hash(config_path),
        "env_variables": env_vars
    }
