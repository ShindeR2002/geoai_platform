"""
Advanced spatial error analysis, prediction uncertainty, and taxonomy classification
for the GeoAI Platform evaluation framework.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.ndimage import label, uniform_filter, binary_dilation

logger = logging.getLogger(__name__)

def analyze_prediction_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
    spatial_shape: Tuple[int, int],
    X: np.ndarray = None,
    feature_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Categorize errors on pixel level, compute spatial error density, boundary errors,
    reconstruct TP/FP/FN/TN maps, and group pixel errors into classified error objects.
    """
    H, W = spatial_shape
    total_pixels = H * W
    
    y_t = y_true.astype(int)
    y_p = y_pred.astype(int)
    
    # 1. Pixel confusion stats
    tp_indices = (y_t == 1) & (y_p == 1)
    tn_indices = (y_t == 0) & (y_p == 0)
    fp_indices = (y_t == 0) & (y_p == 1)
    fn_indices = (y_t == 1) & (y_p == 0)
    
    tp_count = tp_indices.sum()
    tn_count = tn_indices.sum()
    fp_count = fp_indices.sum()
    fn_count = fn_indices.sum()
    
    # 2. Reconstruct separated 2D maps
    tp_map_flat = np.zeros(total_pixels, dtype=np.float32)
    fp_map_flat = np.zeros(total_pixels, dtype=np.float32)
    fn_map_flat = np.zeros(total_pixels, dtype=np.float32)
    tn_map_flat = np.zeros(total_pixels, dtype=np.float32)
    error_map_flat = np.full(total_pixels, np.nan, dtype=np.float32)
    
    tp_map_flat[valid_mask] = tp_indices.astype(np.float32)
    fp_map_flat[valid_mask] = fp_indices.astype(np.float32)
    fn_map_flat[valid_mask] = fn_indices.astype(np.float32)
    tn_map_flat[valid_mask] = tn_indices.astype(np.float32)
    
    cat = np.zeros_like(y_t, dtype=np.float32)
    cat[tn_indices] = 0.0  # TN
    cat[tp_indices] = 1.0  # TP
    cat[fp_indices] = 2.0  # FP
    cat[fn_indices] = 3.0  # FN
    error_map_flat[valid_mask] = cat
    
    tp_map_2d = tp_map_flat.reshape(H, W)
    fp_map_2d = fp_map_flat.reshape(H, W)
    fn_map_2d = fn_map_flat.reshape(H, W)
    tn_map_2d = tn_map_flat.reshape(H, W)
    error_map_2d = error_map_flat.reshape(H, W)
    
    # 3. Compute Error Density Map (15x15 uniform window of all errors: FP + FN)
    error_pixels_2d = (fp_map_2d > 0) | (fn_map_2d > 0)
    error_density_map = uniform_filter(error_pixels_2d.astype(np.float32), size=15, mode='reflect') * 100.0
    
    # 4. Compute Boundary Error Statistics
    # How many of the errors (FP/FN) occur within a 2-pixel buffer of the true boundaries
    true_change_2d = np.zeros(total_pixels, dtype=bool)
    true_change_2d[valid_mask] = (y_t == 1)
    true_change_2d = true_change_2d.reshape(H, W)
    
    # Extract boundary by finding pixels where dilated true change differs from true change
    dilated_true = binary_dilation(true_change_2d, structure=np.ones((5, 5)))
    boundary_mask = dilated_true & ~true_change_2d
    
    total_errors = fp_count + fn_count
    boundary_error_count = int(np.logical_and(error_pixels_2d, boundary_mask).sum())
    boundary_error_ratio = float(boundary_error_count / total_errors) if total_errors > 0 else 0.0
    
    # 5. Extract largest FP and FN patches
    fp_labeled, fp_num = label(fp_map_2d > 0)
    fp_sizes = []
    if fp_num > 0:
        slices = np.bincount(fp_labeled.ravel())
        for idx in range(1, fp_num + 1):
            if idx < len(slices):
                fp_sizes.append((idx, int(slices[idx])))
        fp_sizes.sort(key=lambda x: x[1], reverse=True)
        
    fn_labeled, fn_num = label(fn_map_2d > 0)
    fn_sizes = []
    if fn_num > 0:
        slices = np.bincount(fn_labeled.ravel())
        for idx in range(1, fn_num + 1):
            if idx < len(slices):
                fn_sizes.append((idx, int(slices[idx])))
        fn_sizes.sort(key=lambda x: x[1], reverse=True)
        
    # 6. Error Taxonomy Classifier
    # Segment all false prediction pixels (FP and FN) into error components and categorize them
    taxonomy_stats = {
        "Boundary Error": 0,
        "Missed Small Object": 0,
        "Merged Objects": 0,
        "Fragmented Objects": 0,
        "Noise": 0,
        "Vegetation Confusion": 0,
        "Construction Confusion": 0,
        "Shadow Confusion": 0,
        "Unknown": 0
    }
    
    # Pre-calculate object masks for overlap checks
    true_labeled, true_num = label(true_change_2d)
    pred_change_2d = np.zeros(total_pixels, dtype=bool)
    pred_change_2d[valid_mask] = (y_p == 1)
    pred_change_2d = pred_change_2d.reshape(H, W)
    pred_labeled, pred_num = label(pred_change_2d)
    
    # Extract feature lookup if features are provided
    delta_ndvi_flat = None
    delta_sar_flat = None
    delta_ndwi_flat = None
    if X is not None and feature_names is not None:
        if "Delta_NDVI" in feature_names:
            delta_ndvi_flat = X[:, feature_names.index("Delta_NDVI")]
        if "Delta_SAR" in feature_names:
            delta_sar_flat = X[:, feature_names.index("Delta_SAR")]
        if "Delta_NDWI" in feature_names:
            delta_ndwi_flat = X[:, feature_names.index("Delta_NDWI")]
            
    # Classify False Positive Objects (Predicted Change, but Ground Truth is No Change)
    for p_id in range(1, pred_num + 1):
        p_mask = pred_labeled == p_id
        p_size = int(p_mask.sum())
        
        # Check true objects overlapping this predicted object
        overlapping_true_ids = np.unique(true_labeled[p_mask])
        overlapping_true_ids = overlapping_true_ids[overlapping_true_ids > 0] # exclude background
        
        # 1. No overlap with true change
        if len(overlapping_true_ids) == 0:
            # Check spectral triggers if features exist
            avg_ndvi = 0.0
            avg_sar = 0.0
            avg_ndwi = 0.0
            if X is not None:
                # Find valid mask indices for this patch
                patch_valid_indices = p_mask.ravel()[valid_mask]
                if patch_valid_indices.any():
                    if delta_ndvi_flat is not None:
                        avg_ndvi = float(np.mean(delta_ndvi_flat[patch_valid_indices]))
                    if delta_sar_flat is not None:
                        avg_sar = float(np.mean(delta_sar_flat[patch_valid_indices]))
                    if delta_ndwi_flat is not None:
                        avg_ndwi = float(np.mean(delta_ndwi_flat[patch_valid_indices]))
                        
            if p_size < 50:
                taxonomy_stats["Noise"] += 1
            elif abs(avg_ndvi) > 0.15:
                taxonomy_stats["Vegetation Confusion"] += 1
            elif abs(avg_sar) > 0.15:
                taxonomy_stats["Construction Confusion"] += 1
            elif abs(avg_ndwi) > 0.15:
                taxonomy_stats["Shadow Confusion"] += 1
            else:
                taxonomy_stats["Noise"] += 1
                
        # 2. Overlaps with exactly one true object
        elif len(overlapping_true_ids) == 1:
            t_id = overlapping_true_ids[0]
            t_mask = true_labeled == t_id
            intersection = np.logical_and(p_mask, t_mask).sum()
            union = np.logical_or(p_mask, t_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            
            if iou < 0.5:
                taxonomy_stats["Boundary Error"] += 1
            else:
                pass # This is mostly a correct prediction
                
        # 3. Overlaps with multiple true objects (Merged)
        else:
            taxonomy_stats["Merged Objects"] += 1
            
    # Classify False Negative Objects (True Change, but Predicted is No Change)
    for t_id in range(1, true_num + 1):
        t_mask = true_labeled == t_id
        t_size = int(t_mask.sum())
        
        overlapping_pred_ids = np.unique(pred_labeled[t_mask])
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]
        
        # 1. Missed entirely
        if len(overlapping_pred_ids) == 0:
            if t_size < 100:
                taxonomy_stats["Missed Small Object"] += 1
            else:
                # Check spectral properties of this missed patch
                avg_ndvi = 0.0
                avg_sar = 0.0
                if X is not None:
                    patch_valid_indices = t_mask.ravel()[valid_mask]
                    if patch_valid_indices.any():
                        if delta_ndvi_flat is not None:
                            avg_ndvi = float(np.mean(delta_ndvi_flat[patch_valid_indices]))
                        if delta_sar_flat is not None:
                            avg_sar = float(np.mean(delta_sar_flat[patch_valid_indices]))
                            
                if abs(avg_ndvi) > 0.15:
                    taxonomy_stats["Vegetation Confusion"] += 1
                elif abs(avg_sar) > 0.15:
                    taxonomy_stats["Construction Confusion"] += 1
                else:
                    taxonomy_stats["Unknown"] += 1
                    
        # 2. Split into multiple predictions (Fragmented)
        elif len(overlapping_pred_ids) > 1:
            taxonomy_stats["Fragmented Objects"] += 1

    return {
        "confusion_pixel_counts": {
            "tp": int(tp_count),
            "tn": int(tn_count),
            "fp": int(fp_count),
            "fn": int(fn_count)
        },
        "rates": {
            "false_positive_rate": round(float(fp_count / (fp_count + tn_count)) if (fp_count + tn_count) > 0 else 0.0, 6),
            "false_negative_rate": round(float(fn_count / (fn_count + tp_count)) if (fn_count + tp_count) > 0 else 0.0, 6),
        },
        "boundary_error_ratio": round(boundary_error_ratio, 6),
        "largest_fp_patches_px": [size for _, size in fp_sizes[:5]],
        "largest_fn_patches_px": [size for _, size in fn_sizes[:5]],
        "taxonomy": taxonomy_stats,
        "tp_map_2d": tp_map_2d,
        "fp_map_2d": fp_map_2d,
        "fn_map_2d": fn_map_2d,
        "tn_map_2d": tn_map_2d,
        "error_map_2d": error_map_2d,
        "error_density_map": error_density_map
    }


def compute_rf_prediction_uncertainty(
    clf: Any,
    X: np.ndarray,
    valid_mask: np.ndarray,
    spatial_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """
    Compute Random Forest prediction uncertainty (vote entropy & vote variance) using tree disagreement.
    
    Vote Entropy: H(x) = - sum ( p_c * log2(p_c) ) where p_c is the ratio of trees voting for class c.
    Vote Variance: class probability variance across trees.
    """
    H, W = spatial_shape
    total_pixels = H * W
    
    # 1. Extract individual estimator predictions
    # Shape: (n_estimators, n_samples)
    tree_preds = []
    
    # Support both wrapped and unwrapped random forests
    rf_model = clf._rf if hasattr(clf, "_rf") and clf._rf is not None else clf
    
    if not hasattr(rf_model, "estimators_"):
        # Fallback if model is not an ensemble or not trained
        logger.warning("Model does not expose estimators_ — returning zero uncertainty maps.")
        return {
            "vote_entropy_2d": np.zeros(spatial_shape, dtype=np.float32),
            "vote_variance_2d": np.zeros(spatial_shape, dtype=np.float32)
        }
        
    for tree in rf_model.estimators_:
        preds = tree.predict(X) # binary 0 or 1
        tree_preds.append(preds)
        
    tree_preds = np.stack(tree_preds, axis=0) # shape: (n_estimators, n_samples)
    n_trees = tree_preds.shape[0]
    
    # Compute class probabilities (proportions of trees voting for class 1)
    p_change = tree_preds.sum(axis=0) / n_trees
    p_no_change = 1.0 - p_change
    
    # 2. Compute Vote Entropy
    # Handle log2(0) by using np.clip or np.where
    p1 = np.clip(p_change, 1e-8, 1.0)
    p0 = np.clip(p_no_change, 1e-8, 1.0)
    entropy = - (p1 * np.log2(p1) + p0 * np.log2(p0))
    
    # 3. Compute Vote Variance (class probability variance)
    # For binary inputs (0 and 1), variance is p * (1 - p)
    variance = p_change * p_no_change
    
    # Reconstruct 2D maps
    entropy_flat = np.zeros(total_pixels, dtype=np.float32)
    variance_flat = np.zeros(total_pixels, dtype=np.float32)
    
    entropy_flat[valid_mask] = entropy
    variance_flat[valid_mask] = variance
    
    return {
        "vote_entropy_2d": entropy_flat.reshape(H, W),
        "vote_variance_2d": variance_flat.reshape(H, W),
        "entropy_flat": entropy,
        "variance_flat": variance
    }
