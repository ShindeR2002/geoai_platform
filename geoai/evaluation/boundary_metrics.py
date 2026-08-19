"""
Boundary quality metrics module for Milestone 8 research campaigns.
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
import skimage.measure
from skimage.morphology import convex_hull_image

def get_boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Get internal boundary pixels of a binary mask."""
    mask = mask.astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask)
    eroded = binary_erosion(mask)
    return mask & ~eroded

def compute_boundary_iou(y_true: np.ndarray, y_pred: np.ndarray, buffer_dist: int = 2) -> float:
    """Compute Boundary IoU (within buffer_dist buffer of boundaries)."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    def get_boundary_buffer(mask):
        if not np.any(mask):
            return np.zeros_like(mask)
        boundary = get_boundary_mask(mask)
        if not np.any(boundary):
            return np.zeros_like(mask)
        dist = distance_transform_edt(~boundary)
        return mask & (dist <= buffer_dist)
        
    gt_buffer = get_boundary_buffer(y_true)
    pred_buffer = get_boundary_buffer(y_pred)
    
    intersection = np.logical_and(gt_buffer, pred_buffer).sum()
    union = np.logical_or(gt_buffer, pred_buffer).sum()
    
    if union == 0:
        return 1.0 if not np.any(y_true) and not np.any(y_pred) else 0.0
    return float(intersection) / float(union)

def compute_chamfer_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Chamfer Distance between the boundaries of y_true and y_pred."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    b_true = get_boundary_mask(y_true)
    b_pred = get_boundary_mask(y_pred)
    
    n_true = np.sum(b_true)
    n_pred = np.sum(b_pred)
    
    if n_true == 0 and n_pred == 0:
        return 0.0
    
    diag = np.sqrt(y_true.shape[0]**2 + y_true.shape[1]**2)
    
    if n_true == 0 or n_pred == 0:
        return float(diag)
        
    # EDT of true boundary: distance from any pixel to b_true
    edt_true = distance_transform_edt(~b_true)
    # EDT of pred boundary: distance from any pixel to b_pred
    edt_pred = distance_transform_edt(~b_pred)
    
    # Term 1: average distance from true boundary to nearest pred boundary
    term1 = np.sum(edt_pred[b_true]) / n_true
    # Term 2: average distance from pred boundary to nearest true boundary
    term2 = np.sum(edt_true[b_pred]) / n_pred
    
    return float(term1 + term2)

def compute_hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Directed and Undirected Hausdorff Distance between boundaries."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    b_true = get_boundary_mask(y_true)
    b_pred = get_boundary_mask(y_pred)
    
    n_true = np.sum(b_true)
    n_pred = np.sum(b_pred)
    
    if n_true == 0 and n_pred == 0:
        return 0.0
        
    diag = np.sqrt(y_true.shape[0]**2 + y_true.shape[1]**2)
    
    if n_true == 0 or n_pred == 0:
        return float(diag)
        
    edt_true = distance_transform_edt(~b_true)
    edt_pred = distance_transform_edt(~b_pred)
    
    d_true_to_pred = np.max(edt_pred[b_true])
    d_pred_to_true = np.max(edt_true[b_pred])
    
    return float(max(d_true_to_pred, d_pred_to_true))

def compute_boundary_smoothness_index(mask: np.ndarray) -> float:
    """Calculate the average Boundary Smoothness Index (BSI) of objects in mask.
    
    BSI = P_convex_hull / P_actual.
    """
    mask = mask.astype(bool)
    if not np.any(mask):
        return 1.0
        
    # Label connected components
    labeled, num_features = skimage.measure.label(mask, return_num=True)
    if num_features == 0:
        return 1.0
        
    props = skimage.measure.regionprops(labeled)
    bsi_vals = []
    
    for prop in props:
        perimeter = float(prop.perimeter)
        if perimeter == 0:
            continue
            
        try:
            # Generate convex hull of the component image
            ch_img = convex_hull_image(prop.image)
            ch_perimeter = skimage.measure.perimeter(ch_img)
            bsi = float(ch_perimeter / perimeter)
            # Clip BSI to [0.0, 1.0] since convex hull perimeter should be <= actual perimeter
            bsi_vals.append(np.clip(bsi, 0.0, 1.0))
        except Exception:
            bsi_vals.append(1.0)
            
    if not bsi_vals:
        return 1.0
    return float(np.mean(bsi_vals))

def compute_average_thickness(mask: np.ndarray) -> float:
    """Compute global average boundary thickness: 2 * Area / Perimeter."""
    mask = mask.astype(bool)
    area = np.sum(mask)
    if area == 0:
        return 0.0
    perimeter = skimage.measure.perimeter(mask)
    if perimeter == 0:
        return 0.0
    return float(2.0 * area / perimeter)

def compute_boundary_campaign_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all boundary campaign metrics for 2D truth and prediction maps."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    biou = compute_boundary_iou(y_true, y_pred, buffer_dist=2)
    chamfer = compute_chamfer_distance(y_true, y_pred)
    hausdorff = compute_hausdorff_distance(y_true, y_pred)
    
    bsi_true = compute_boundary_smoothness_index(y_true)
    bsi_pred = compute_boundary_smoothness_index(y_pred)
    bsi_diff = abs(bsi_true - bsi_pred)
    
    thick_true = compute_average_thickness(y_true)
    thick_pred = compute_average_thickness(y_pred)
    thick_diff = abs(thick_true - thick_pred)
    
    return {
        "boundary_iou": biou,
        "chamfer_distance": chamfer,
        "hausdorff_distance": hausdorff,
        "bsi_true": bsi_true,
        "bsi_pred": bsi_pred,
        "bsi_diff": bsi_diff,
        "thickness_true": thick_true,
        "thickness_pred": thick_pred,
        "thickness_diff": thick_diff
    }
