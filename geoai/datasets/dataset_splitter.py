from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging
from pathlib import Path
from scipy.ndimage import binary_erosion
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Hardcoded deterministic 4x4 block layout allocation
# 10 blocks train, 3 blocks val, 3 blocks test
BLOCK_SPLIT_MAP = {
    0: 'train', 1: 'train', 2: 'train', 4: 'train', 5: 'train', 6: 'train', 8: 'train', 9: 'train', 10: 'train', 12: 'train',
    3: 'val', 7: 'val', 11: 'val',
    13: 'test', 14: 'test', 15: 'test'
}

@dataclass
class SplitResult:
    """Structured dataclass containing train, val, and test splits to avoid positional tuple errors."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproducibly split dataset into stratified train and test subsets."""
    strat = y if stratify else None
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )

def get_pixel_split(r: int, c: int, H: int, W: int, R: int) -> str:
    """Determine split ('train', 'val', 'test', 'buffer') for a pixel with buffer zone check."""
    bh, bw = H / 4.0, W / 4.0
    br = int(min(3, r // bh))
    bc = int(min(3, c // bw))
    center_block = br * 4 + bc
    center_split = BLOCK_SPLIT_MAP[center_block]
    
    # Check if any neighbor in the R-radius patch crosses into a different split
    for dr in range(-R, R + 1):
        nr = r + dr
        if nr < 0 or nr >= H:
            continue
        for dc in range(-R, R + 1):
            nc = c + dc
            if nc < 0 or nc >= W:
                continue
            nbr_br = int(min(3, nr // bh))
            nbr_bc = int(min(3, nc // bw))
            nbr_block = nbr_br * 4 + nbr_bc
            if BLOCK_SPLIT_MAP[nbr_block] != center_split:
                return 'buffer'
                
    return center_split

def get_pixel_coords(dataset_id: str, configs_dir: str = "configs") -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray]:
    """
    Load the feature cube for dataset_id, compute the valid mask,
    and return the original 2D coordinates for each valid pixel,
    along with the spatial shape (H, W), the valid mask, and the feature cube itself.
    """
    from geoai.datasets.dataset_registry import DatasetRegistry
    from geoai.core.config import load_platform_config
    from geoai.pipeline.stage1 import _load_all_rasters
    from geoai.features.feature_cube import build_feature_cube
    
    meta = DatasetRegistry.get_dataset_metadata(dataset_id)
    platform_config = load_platform_config(configs_dir)
    aoi = platform_config.get_aoi(meta.aoi_name)
    
    rasters = _load_all_rasters(aoi)
    red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1 = rasters[0:6]
    red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2 = rasters[6:12]
    sar_t1, sar_t2 = rasters[12:14]
    
    feature_cube = build_feature_cube(
        red_t1=red_t1, green_t1=green_t1, blue_t1=blue_t1,
        sar_t1=sar_t1, ndvi_t1=ndvi_t1, ndbi_t1=ndbi_t1, ndwi_t1=ndwi_t1,
        red_t2=red_t2, green_t2=green_t2, blue_t2=blue_t2,
        sar_t2=sar_t2, ndvi_t2=ndvi_t2, ndbi_t2=ndbi_t2, ndwi_t2=ndwi_t2,
    )
    
    H, W, _ = feature_cube.shape
    X_flat = feature_cube.reshape(-1, feature_cube.shape[2])
    non_sar_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
    not_nan = ~np.isnan(X_flat).any(axis=1)
    not_all_zero = ~np.all(X_flat[:, non_sar_indices] == 0, axis=1)
    valid_mask = not_nan & not_all_zero
    
    # Get coordinates of valid pixels
    valid_indices = np.where(valid_mask)[0]
    rows = valid_indices // W
    cols = valid_indices % W
    coords = np.stack([rows, cols], axis=1) # shape: (N_valid, 2)
    
    return coords, (H, W), valid_mask.reshape(H, W), feature_cube

def split_dataset_spatial(
    X: np.ndarray,
    y: np.ndarray,
    dataset_id: str,
    patch_size: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a Spatial Block-Splitting scheme with a spatial boundary buffer zone
    to completely prevent spatial patch overlap/leakage between splits.
    """
    R = patch_size // 2
    coords, (H, W), _, _ = get_pixel_coords(dataset_id)
    
    assert coords.shape[0] == X.shape[0], f"Coordinates count {coords.shape[0]} does not match X shape {X.shape[0]}"
    
    # 1. Construct block index grid (H, W)
    bh, bw = H / 4.0, W / 4.0
    R_idx, C_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    br = np.minimum(3, R_idx // bh).astype(int)
    bc = np.minimum(3, C_idx // bw).astype(int)
    block_ids = br * 4 + bc
    
    # 2. Map block_ids to splits integer code: 'train': 0, 'val': 1, 'test': 2
    split_to_code = {'train': 0, 'val': 1, 'test': 2}
    code_lookup = np.array([split_to_code[BLOCK_SPLIT_MAP[k]] for k in range(16)])
    split_grid = code_lookup[block_ids]
    
    # 3. Apply binary erosion to identify buffer pixels
    structuring_element = np.ones((2*R+1, 2*R+1), dtype=bool)
    final_split_grid = np.full((H, W), -1, dtype=int)
    
    for S in [0, 1, 2]:
        mask = (split_grid == S)
        eroded = binary_erosion(mask, structure=structuring_element)
        final_split_grid[eroded] = S
        
    # 4. Map back to splits array matching coords
    code_to_split = {0: 'train', 1: 'val', 2: 'test', -1: 'buffer'}
    splits = np.array([code_to_split[final_split_grid[r, c]] for r, c in coords])
    
    train_mask = (splits == 'train')
    val_mask = (splits == 'val')
    test_mask = (splits == 'test')
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info("Spatial block splits - Train: %d, Val: %d, Test: %d, Buffer: %d",
                int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()), int((splits == 'buffer').sum()))
    
    return X_train, X_test, y_train, y_test, X_val, y_val

def split_dataset_unified(
    X: np.ndarray,
    y: np.ndarray,
    dataset_id: str,
    split_policy: str = "spatial",
    patch_size: int = 15,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True
) -> SplitResult:
    """
    Centralized dataset splitting routing logic. Enforces spatial block split
    or random stratified split uniformly across classical and deep learning models.
    """
    if split_policy == "spatial":
        X_train, X_test, y_train, y_test, X_val, y_val = split_dataset_spatial(
            X, y, dataset_id, patch_size=patch_size
        )
        return SplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val
        )
    elif split_policy == "random":
        X_train, X_test, y_train, y_test = split_dataset(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        return SplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_val=None,
            y_val=None
        )
    else:
        raise ValueError(f"Unknown split_policy: {split_policy}")
