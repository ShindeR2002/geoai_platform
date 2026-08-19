"""
Preprocessing and Feature Engineering pipeline operations for the GeoAI Platform.

Implements Campaign A (Preprocessing: Refined Lee Filter) and Campaign B
(Feature Engineering: Local Variance, SAVI, and MSAVI indices).
"""

import logging
from typing import List
import numpy as np
from scipy.ndimage import convolve, uniform_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Campaign A: Preprocessing (Speckle Filtering)
# ---------------------------------------------------------------------------

def refined_lee_filter(img: np.ndarray, size: int = 7, noise_fraction: float = 0.26) -> np.ndarray:
    """Apply the Refined Lee directional speckle filter to a SAR VV image.

    This implementation uses 8 directional sub-windows within a 7x7 neighborhood
    to estimate local statistics (mean and variance) on the edge side of the pixel,
    preventing blurring across linear features and structural boundaries.

    Args:
        img: float32 array of shape (H, W) with SAR backscatter intensity or dB values.
        size: Neighborhood window size (default 7).
        noise_fraction: Standard deviation fraction of the speckle noise (default 0.26).

    Returns:
        Filtered float32 array of shape (H, W).
    """
    if img.ndim != 2:
        raise ValueError(f"Input image must be 2D, got shape {img.shape}")

    # Ensure input is float32
    img_float = img.astype(np.float32)
    H, W = img_float.shape

    # Define the 8 directional masks for a 7x7 window
    # Center is at index (3, 3)
    masks = []
    
    # 1. Horizontal Top
    m1 = np.zeros((7, 7), dtype=np.float32)
    m1[0:3, :] = 1
    m1[3, 3] = 1
    masks.append(m1)

    # 2. Horizontal Bottom
    m2 = np.zeros((7, 7), dtype=np.float32)
    m2[4:7, :] = 1
    m2[3, 3] = 1
    masks.append(m2)

    # 3. Vertical Left
    m3 = np.zeros((7, 7), dtype=np.float32)
    m3[:, 0:3] = 1
    m3[3, 3] = 1
    masks.append(m3)

    # 4. Vertical Right
    m4 = np.zeros((7, 7), dtype=np.float32)
    m4[:, 4:7] = 1
    m4[3, 3] = 1
    masks.append(m4)

    # 5. Diagonal Top-Left
    m5 = np.zeros((7, 7), dtype=np.float32)
    m5[0:3, 0:3] = 1
    m5[3, 3] = 1
    masks.append(m5)

    # 6. Diagonal Top-Right
    m6 = np.zeros((7, 7), dtype=np.float32)
    m6[0:3, 4:7] = 1
    m6[3, 3] = 1
    masks.append(m6)

    # 7. Diagonal Bottom-Left
    m7 = np.zeros((7, 7), dtype=np.float32)
    m7[4:7, 0:3] = 1
    m7[3, 3] = 1
    masks.append(m7)

    # 8. Diagonal Bottom-Right
    m8 = np.zeros((7, 7), dtype=np.float32)
    m8[4:7, 4:7] = 1
    m8[3, 3] = 1
    masks.append(m8)

    # Normalize masks so they sum to 1
    normalized_masks = [m / m.sum() for m in masks]

    # Compute means and variances for each of the 8 directions
    means = []
    vars_ = []
    for W_mask in normalized_masks:
        mu = convolve(img_float, W_mask, mode='reflect')
        var = convolve(img_float**2, W_mask, mode='reflect') - mu**2
        var = np.clip(var, 0.0, None)
        means.append(mu)
        vars_.append(var)

    means = np.stack(means, axis=0) # shape (8, H, W)
    vars_ = np.stack(vars_, axis=0) # shape (8, H, W)

    # Find the index of the direction with the minimum variance for each pixel
    min_var_idx = np.argmin(vars_, axis=0) # shape (H, W)

    # Resolve chosen mean and variance using index indexing
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    chosen_mean = means[min_var_idx, grid_y, grid_x]
    chosen_var = vars_[min_var_idx, grid_y, grid_x]

    # Calculate speckle noise variance dynamically
    noise_var = (noise_fraction * chosen_mean) ** 2

    # Apply Lee filter formula: filtered = mean + k * (img - mean)
    # where k = (var - noise_var) / var
    k = np.zeros_like(img_float)
    valid_mask = chosen_var > 1e-8
    k[valid_mask] = (chosen_var[valid_mask] - noise_var[valid_mask]) / chosen_var[valid_mask]
    k = np.clip(k, 0.0, 1.0)

    filtered = chosen_mean + k * (img_float - chosen_mean)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Campaign B: Feature Engineering (Bands & Textures)
# ---------------------------------------------------------------------------

def local_variance_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Compute local standard deviation over a sliding window (spatial texture).

    Args:
        img: float32 array of shape (H, W).
        size: Window size (default 3).

    Returns:
        Local standard deviation array of shape (H, W).
    """
    img_float = img.astype(np.float32)
    mu = uniform_filter(img_float, size=size, mode='reflect')
    var = uniform_filter(img_float**2, size=size, mode='reflect') - mu**2
    var = np.clip(var, 0.0, None)
    return np.sqrt(var).astype(np.float32)


def reconstruct_nir(red: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    """Reconstruct Near-Infrared (NIR) band algebraically from Red and NDVI.

    Formula: NIR = Red * (1 + NDVI) / (1 - NDVI)

    Args:
        red: Red reflectance array, float32 of shape (H, W).
        ndvi: NDVI array, float32 of shape (H, W).

    Returns:
        Reconstructed NIR array of shape (H, W), non-negative.
    """
    denom = 1.0 - ndvi.astype(np.float32)
    # Guard against division by zero
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    nir = red.astype(np.float32) * (1.0 + ndvi.astype(np.float32)) / denom
    return np.clip(nir, 0.0, None).astype(np.float32)


def compute_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """Compute Soil Adjusted Vegetation Index (SAVI).

    Formula: SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    Args:
        nir: NIR array, float32 of shape (H, W).
        red: Red array, float32 of shape (H, W).
        L: Soil brightness correction factor (default 0.5).

    Returns:
        SAVI array of shape (H, W).
    """
    denom = nir.astype(np.float32) + red.astype(np.float32) + L
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    savi = (nir.astype(np.float32) - red.astype(np.float32)) / denom * (1.0 + L)
    return savi.astype(np.float32)


def compute_msavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute Modified Soil Adjusted Vegetation Index (MSAVI).

    Formula: MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2

    Args:
        nir: NIR array, float32 of shape (H, W).
        red: Red array, float32 of shape (H, W).

    Returns:
        MSAVI array of shape (H, W).
    """
    nir_32 = nir.astype(np.float32)
    red_32 = red.astype(np.float32)
    
    term = (2.0 * nir_32 + 1.0) ** 2 - 8.0 * (nir_32 - red_32)
    term = np.clip(term, 0.0, None)
    
    msavi = (2.0 * nir_32 + 1.0 - np.sqrt(term)) / 2.0
    return msavi.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature Names Mapper
# ---------------------------------------------------------------------------

def get_campaign_feature_names(campaign_type: str, track: str, config: dict) -> List[str]:
    """Retrieve the dynamic list of features stacked in the dataset.

    Args:
        campaign_type: "preprocessing" or "feature_engineering".
        track: "production" or "experimental".
        config: Raw configuration dictionary.

    Returns:
        List of feature names.
    """
    from geoai.utils.constants import CANONICAL_FEATURE_NAMES
    features = list(CANONICAL_FEATURE_NAMES)
    
    if campaign_type == "feature_engineering":
        features_config = config.get("features", {})
        
        # Track A: Production
        if track == "production":
            if features_config.get("local_variance", {}).get("enabled", False):
                features.extend(["Local_Var_T1", "Local_Var_T2"])
                
        # Track B: Experimental (allows reconstructed NIR approximations)
        elif track == "experimental":
            if features_config.get("local_variance", {}).get("enabled", False):
                features.extend(["Local_Var_T1", "Local_Var_T2"])
            if features_config.get("savi", {}).get("enabled", False):
                features.extend(["SAVI_T1", "SAVI_T2"])
            if features_config.get("msavi", {}).get("enabled", False):
                features.extend(["MSAVI_T1", "MSAVI_T2"])
                
    elif campaign_type == "boundary_engineering":
        boundary_config = config.get("boundary", config.get("features", {}).get("boundary", {}))
        # Handle cases where boundary_config might be None
        if boundary_config is None:
            boundary_config = {}
        method = str(boundary_config.get("method", "sobel") or "none").lower()
        
        if method == "sobel":
            features.extend(["Sobel_Delta_NDVI", "Sobel_Delta_SAR"])
        elif method == "scharr":
            features.extend(["Scharr_Delta_NDVI", "Scharr_Delta_SAR"])
        elif method == "laplacian":
            features.extend(["Laplacian_Delta_NDVI", "Laplacian_Delta_SAR"])
        elif method == "morphological_gradient":
            features.extend(["MorphGrad_Delta_NDVI", "MorphGrad_Delta_SAR"])
        elif method == "distance_transform":
            features.extend(["EDT_Delta_NDVI", "EDT_Delta_SAR"])
        elif method == "distance_to_boundary":
            features.extend(["Dist_Nearest_Boundary"])
        elif method == "morphological":
            features.extend([
                "Internal_Boundary_Delta_NDVI", "Internal_Boundary_Delta_SAR",
                "External_Boundary_Delta_NDVI", "External_Boundary_Delta_SAR",
                "Thickness_Delta_NDVI", "Thickness_Delta_SAR"
            ])
        elif method == "boundary_refinement":
            features.extend(["Solidity", "Compactness", "Elongation", "Eccentricity", "Convexity"])
            
    return features
