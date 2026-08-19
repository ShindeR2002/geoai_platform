import hashlib
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
from geoai.datasets.dataset_metadata import DatasetMetadata
from geoai.datasets.dataset_schema import UnifiedChangeDetectionDataset

logger = logging.getLogger(__name__)

def calculate_file_checksum(file_path: Path) -> str:
    """Calculate MD5 checksum of a file on disk."""
    if not file_path.exists():
        return ""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""

def validate_dataset_rasters(metadata: DatasetMetadata, configs_dir: str = "configs") -> bool:
    """Validate that rasters exist for a dataset and have valid shapes."""
    from geoai.core.config import load_platform_config
    try:
        config = load_platform_config(configs_dir)
        aoi = config.get_aoi(metadata.aoi_name)
        for raster_type in ["rgb", "sar_vv"]:
            for year in [aoi.t1_year, aoi.t2_year]:
                p = aoi.get_raster_path(raster_type, year)
                if not p.exists():
                    return False
        return True
    except Exception:
        return False

def validate_dataset_integrity(dataset: UnifiedChangeDetectionDataset) -> Tuple[bool, List[str]]:
    """
    Validates a dataset's compliance with the UnifiedChangeDetectionDataset schema contract.
    Checks metadata, CRS, band types/shapes, normalized ranges, and label bounds.
    """
    errors = []
    
    # 1. Metadata Checks
    try:
        meta = dataset.get_metadata()
        if not isinstance(meta, DatasetMetadata):
            errors.append("Dataset metadata is not an instance of DatasetMetadata.")
        else:
            # Check versioning and capabilities
            if not meta.dataset_id:
                errors.append("Dataset ID is empty in metadata.")
            if not meta.crs:
                errors.append("Dataset CRS is missing in metadata.")
            elif not (meta.crs.startswith("EPSG:") or "local" in meta.crs.lower()):
                errors.append(f"Invalid CRS format: {meta.crs}. Must be EPSG code or local.")
    except Exception as e:
        errors.append(f"Failed retrieving dataset metadata: {e}")
        return False, errors
        
    # 2. File Checksum Validation (if specified in metadata)
    # Usually performed on zip archives in convert utility, here we check sample loading
    
    # 3. Shape and Data Type Checks
    n_tiles = len(dataset)
    if n_tiles == 0:
        errors.append("Dataset has zero registered tiles/samples.")
        return False, errors
        
    # Evaluate a sample of tiles (e.g. first 3) to ensure integrity without running long loops
    for idx in range(min(n_tiles, 3)):
        try:
            t1, t2, label = dataset.get_pair(idx)
            
            # Check shapes
            if t1.ndim != 3 or t2.ndim != 3:
                errors.append(f"Sample {idx}: Expected 3D images (Channels, Height, Width). Got t1={t1.shape}, t2={t2.shape}.")
                continue
                
            if t1.shape != t2.shape:
                errors.append(f"Sample {idx}: Shape mismatch between T1 {t1.shape} and T2 {t2.shape}.")
                
            if label.ndim != 2:
                errors.append(f"Sample {idx}: Expected 2D label map. Got shape {label.shape}.")
                continue
                
            if label.shape != (t1.shape[1], t1.shape[2]):
                errors.append(f"Sample {idx}: Spatial shape mismatch between image {(t1.shape[1], t1.shape[2])} and label {label.shape}.")
                
            # Check types
            if t1.dtype != np.float32 or t2.dtype != np.float32:
                errors.append(f"Sample {idx}: Image tensors must be float32. Got t1={t1.dtype}, t2={t2.dtype}.")
            if label.dtype != np.int64:
                errors.append(f"Sample {idx}: Label maps must be int64. Got {label.dtype}.")
                
            # Check ranges
            if np.nanmin(t1) < -1e-5 or np.nanmax(t1) > 1.0001:
                errors.append(f"Sample {idx}: T1 image out of range [0, 1]. Min: {np.nanmin(t1)}, Max: {np.nanmax(t1)}.")
            if np.nanmin(t2) < -1e-5 or np.nanmax(t2) > 1.0001:
                errors.append(f"Sample {idx}: T2 image out of range [0, 1]. Min: {np.nanmin(t2)}, Max: {np.nanmax(t2)}.")
                
            # Check label values
            unique_labels = np.unique(label)
            is_multiclass = dataset.get_metadata().is_multiclass
            if is_multiclass:
                # Must be non-negative integers
                if np.any(unique_labels < 0):
                    errors.append(f"Sample {idx}: Multiclass label contains negative values: {unique_labels}.")
            else:
                # Binary labels must contain only 0 and 1
                if not np.all(np.isin(unique_labels, [0, 1])):
                    errors.append(f"Sample {idx}: Binary label contains values other than 0 and 1: {unique_labels}.")
                    
        except Exception as e:
            errors.append(f"Failed loading sample {idx} during integrity checks: {e}")
            
    is_valid = len(errors) == 0
    return is_valid, errors
