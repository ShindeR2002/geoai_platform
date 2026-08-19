"""
Boundary-aware feature extractors for Milestone 8.

Implements pixel-level boundary representations (Campaign A) and object-level
boundary refinement cascades (Campaign B).
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.ndimage import (
    sobel, laplace, binary_dilation, binary_erosion,
    distance_transform_edt, generate_binary_structure
)
from skimage.feature import canny
from skimage.measure import regionprops

logger = logging.getLogger(__name__)

def compute_continuous_gradient(image: np.ndarray, method: str = "sobel") -> np.ndarray:
    """Compute continuous gradient strength using Sobel, Scharr, or Laplacian.
    
    Args:
        image: 2D float32 array.
        method: Gradient method ('sobel', 'scharr', 'laplacian').
        
    Returns:
        2D float32 array.
    """
    img_f = image.astype(np.float32)
    method = method.lower()
    
    if method == "sobel":
        gx = sobel(img_f, axis=0)
        gy = sobel(img_f, axis=1)
        magnitude = np.sqrt(gx**2 + gy**2)
        return magnitude.astype(np.float32)
        
    elif method == "scharr":
        # Scharr kernel approximations
        # Hx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]] / 16
        # Hy = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]] / 16
        from scipy.ndimage import convolve
        scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32) / 16.0
        scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32) / 16.0
        gx = convolve(img_f, scharr_x, mode='reflect')
        gy = convolve(img_f, scharr_y, mode='reflect')
        magnitude = np.sqrt(gx**2 + gy**2)
        return magnitude.astype(np.float32)
        
    elif method == "laplacian":
        magnitude = np.abs(laplace(img_f))
        return magnitude.astype(np.float32)
        
    else:
        raise ValueError(f"Unknown gradient method: {method}")

def compute_distance_transform_from_edges(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Compute continuous distance transform from Canny edge map of the image.
    
    Args:
        image: 2D float32 array.
        sigma: Standard deviation of Gaussian filter for Canny.
        
    Returns:
        2D float32 array of EDT values.
    """
    # Normalize image to [0, 1] for Canny edge detector
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        normalized = (image - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(image)
        
    # Detect edges using Canny
    edges = canny(normalized, sigma=sigma)
    
    # Compute Euclidean Distance Transform from the detected edges
    # EDT requires background (zeros) to be on the boundaries, so we pass ~edges
    edt = distance_transform_edt(~edges)
    return edt.astype(np.float32)

def compute_morphological_gradient(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Compute Morphological Gradient strength (dilation - erosion).
    
    Args:
        image: 2D float32 array.
        size: Structuring element window size.
        
    Returns:
        2D float32 array.
    """
    from scipy.ndimage import grey_dilation, grey_erosion
    img_f = image.astype(np.float32)
    dilated = grey_dilation(img_f, size=(size, size))
    eroded = grey_erosion(img_f, size=(size, size))
    grad = dilated - eroded
    return grad.astype(np.float32)

def compute_morphological_boundaries(binary_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute internal, external boundaries and morphological thickness map.
    
    Args:
        binary_mask: 2D boolean array.
        
    Returns:
        Tuple of (internal_boundary, external_boundary, thickness_map) as 2D float32.
    """
    struct = generate_binary_structure(2, 1) # 4-connectivity
    dilated = binary_dilation(binary_mask, structure=struct)
    eroded = binary_erosion(binary_mask, structure=struct)
    
    internal = binary_mask & ~eroded
    external = dilated & ~binary_mask
    
    # Thickness estimation: distance transform inside the mask
    thickness = distance_transform_edt(binary_mask)
    
    return internal.astype(np.float32), external.astype(np.float32), thickness.astype(np.float32)

def project_object_shapes(
    label_array: np.ndarray,
    valid_mask_2d: np.ndarray
) -> Dict[str, np.ndarray]:
    """Calculate solidity, compactness, elongation, and eccentricity for each region
    and project them to pixel-level grids.
    
    Args:
        label_array: 2D integer array (components label map).
        valid_mask_2d: 2D boolean array (valid land mask).
        
    Returns:
        Dict mapping metric name to 2D float32 arrays.
    """
    H, W = label_array.shape
    solidity_grid = np.zeros((H, W), dtype=np.float32)
    compactness_grid = np.zeros((H, W), dtype=np.float32)
    elongation_grid = np.zeros((H, W), dtype=np.float32)
    eccentricity_grid = np.zeros((H, W), dtype=np.float32)
    convexity_grid = np.zeros((H, W), dtype=np.float32)
    
    regions = regionprops(label_array)
    for prop in regions:
        label_id = prop.label
        mask = label_array == label_id
        
        # Solidity: ratio of pixels in the region to pixels in the convex hull
        solidity = float(prop.solidity)
        
        # Compactness: 4 * pi * Area / Perimeter^2
        perimeter = float(prop.perimeter)
        if perimeter > 0:
            compactness = (4.0 * np.pi * prop.area) / (perimeter ** 2)
        else:
            compactness = 0.0
            
        # Eccentricity: ratio of focal distance to major axis length of equivalent ellipse
        eccentricity = float(prop.eccentricity)
        
        # Elongation (approximation: major_axis_length / minor_axis_length)
        if prop.minor_axis_length > 0:
            elongation = float(prop.major_axis_length / prop.minor_axis_length)
        else:
            elongation = 1.0
            
        # Convexity: ratio of convex hull perimeter to actual perimeter
        # Perimeter of convex hull using skimage convex_image properties
        from skimage.morphology import convex_hull_image
        convex_img = convex_hull_image(prop.image)
        # Find perimeter of convex image
        from skimage.measure import perimeter as skimage_perimeter
        convex_perimeter = skimage_perimeter(convex_img)
        if perimeter > 0:
            convexity = float(convex_perimeter / perimeter)
        else:
            convexity = 1.0
            
        # Fill grids
        solidity_grid[mask] = solidity
        compactness_grid[mask] = compactness
        elongation_grid[mask] = elongation
        eccentricity_grid[mask] = eccentricity
        convexity_grid[mask] = convexity
        
    # Zero out invalid pixels
    solidity_grid[~valid_mask_2d] = 0.0
    compactness_grid[~valid_mask_2d] = 0.0
    elongation_grid[~valid_mask_2d] = 0.0
    eccentricity_grid[~valid_mask_2d] = 0.0
    convexity_grid[~valid_mask_2d] = 0.0
    
    return {
        "solidity": solidity_grid,
        "compactness": compactness_grid,
        "elongation": elongation_grid,
        "eccentricity": eccentricity_grid,
        "convexity": convexity_grid
    }


class BoundaryRefinementCascade:
    """Implements the two-pass boundary refinement cascade.
    
    Pass 1: Runs initial prediction.
    Pass 2: Computes shape metrics on connected components and projects to pixels.
    Pass 3: Runs refined prediction on the augmented feature cube.
    """
    def __init__(self, base_model: Any, refined_model: Any) -> None:
        self.base_model = base_model
        self.refined_model = refined_model
        
    def predict_refined(
        self,
        X_base_flat: np.ndarray,
        feature_cube_base: np.ndarray,
        valid_mask_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1. Run first-pass base prediction
        y_pred_base = self.base_model.predict(X_base_flat)
        
        # Reconstruct 2D prediction map
        H, W = valid_mask_2d.shape
        pred_map_2d = np.zeros((H, W), dtype=bool)
        pred_map_2d[valid_mask_2d] = y_pred_base.astype(bool)
        
        # 2. Extract connected component label array
        from scipy.ndimage import label
        label_arr, num_feats = label(pred_map_2d)
        
        # 3. Project object shapes to pixel-level grids
        shapes_dict = project_object_shapes(label_arr, valid_mask_2d)
        
        # 4. Construct augmented feature cube
        shape_bands = [
            shapes_dict["solidity"],
            shapes_dict["compactness"],
            shapes_dict["elongation"],
            shapes_dict["eccentricity"],
            shapes_dict["convexity"]
        ]
        shape_cube = np.stack(shape_bands, axis=-1)
        augmented_cube = np.concatenate([feature_cube_base, shape_cube], axis=-1)
        
        # Flatten augmented cube
        from geoai.utils.raster_utils import flatten_spatial
        X_aug_flat = flatten_spatial(augmented_cube)[valid_mask_2d.ravel()]
        
        # 5. Run second-pass refined prediction
        y_pred_refined = self.refined_model.predict(X_aug_flat)
        return y_pred_refined, y_pred_base, label_arr

