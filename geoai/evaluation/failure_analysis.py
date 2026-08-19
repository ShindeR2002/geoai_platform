import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from scipy.ndimage import label, uniform_filter, binary_dilation

logger = logging.getLogger(__name__)

def perform_failure_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray, # (N, 2)
    coords: np.ndarray,
    spatial_shape: Tuple[int, int],
    valid_mask: np.ndarray, # 2D boolean mask
    output_dir: str,
    feature_cube: np.ndarray = None
) -> Dict[str, Any]:
    """
    Categorize predictions into TP, TN, FP, FN, and segment error pixels into:
    - Boundary failures
    - Small-object failures (< 100 pixels)
    - Large-object failures (>= 100 pixels)
    - Spatial cluster failures
    Save neighborhood plots for TP, TN, FP, FN, and most uncertain pixels.
    """
    H, W = spatial_shape
    os.makedirs(output_dir, exist_ok=True)
    
    # y_prob has shape (N, 2). Prob of change is y_prob[:, 1].
    prob_change = y_prob[:, 1]
    
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    # 1. Select key pixels
    results = {}
    
    # Highest confidence TP (true=1, pred=1, max prob_change)
    if tp_mask.any():
        idx_tp = np.where(tp_mask)[0][np.argmax(prob_change[tp_mask])]
        tp_pixel = coords[idx_tp]
        results["highest_confidence_tp"] = {"coords": tp_pixel.tolist(), "prob_change": float(prob_change[idx_tp])}
    else:
        results["highest_confidence_tp"] = None
        
    # Highest confidence TN (true=0, pred=0, min prob_change)
    if tn_mask.any():
        idx_tn = np.where(tn_mask)[0][np.argmin(prob_change[tn_mask])]
        tn_pixel = coords[idx_tn]
        results["highest_confidence_tn"] = {"coords": tn_pixel.tolist(), "prob_change": float(prob_change[idx_tn])}
    else:
        results["highest_confidence_tn"] = None
        
    # Worst FP (true=0, pred=1, max prob_change)
    if fp_mask.any():
        idx_fp = np.where(fp_mask)[0][np.argmax(prob_change[fp_mask])]
        fp_pixel = coords[idx_fp]
        results["worst_fp"] = {"coords": fp_pixel.tolist(), "prob_change": float(prob_change[idx_fp])}
    else:
        results["worst_fp"] = None
        
    # Worst FN (true=1, pred=0, min prob_change)
    if fn_mask.any():
        idx_fn = np.where(fn_mask)[0][np.argmin(prob_change[fn_mask])]
        fn_pixel = coords[idx_fn]
        results["worst_fn"] = {"coords": fn_pixel.tolist(), "prob_change": float(prob_change[idx_fn])}
    else:
        results["worst_fn"] = None
        
    # Most uncertain (prob_change closest to 0.5)
    idx_unc = np.argmin(np.abs(prob_change - 0.5))
    unc_pixel = coords[idx_unc]
    results["most_uncertain"] = {"coords": unc_pixel.tolist(), "prob_change": float(prob_change[idx_unc])}

    # 2. Map errors spatially
    errors_flat = (y_pred != y_true)
    errors_2d = np.zeros((H, W), dtype=bool)
    for idx, (r, c) in enumerate(coords):
        errors_2d[r, c] = errors_flat[idx]
    
    y_true_2d = np.zeros((H, W), dtype=bool)
    for idx, (r, c) in enumerate(coords):
        y_true_2d[r, c] = (y_true[idx] == 1)
    
    # A. Boundary failures (within 2-pixel buffer of ground truth boundaries)
    dilated_true = binary_dilation(y_true_2d, structure=np.ones((5, 5)))
    boundary_mask = dilated_true & ~y_true_2d
    boundary_errors = errors_2d & boundary_mask
    boundary_error_count = int(boundary_errors.sum())
    
    # B. Connected component size checks (small-object vs large-object failures)
    error_labeled, num_features = label(errors_2d)
    sizes = np.bincount(error_labeled.ravel())
    
    small_object_error_count = 0
    large_object_error_count = 0
    
    # label 0 is background
    for label_idx in range(1, len(sizes)):
        if sizes[label_idx] < 100:
            small_object_error_count += int(sizes[label_idx])
        else:
            large_object_error_count += int(sizes[label_idx])
            
    # C. Spatial cluster failures (local density in 15x15 uniform window > 30%)
    local_density = uniform_filter(errors_2d.astype(np.float32), size=15)
    cluster_errors = (local_density > 0.3) & errors_2d
    cluster_error_count = int(cluster_errors.sum())
    
    results["failure_taxonomy"] = {
        "total_errors": int(errors_flat.sum()),
        "boundary_failures": boundary_error_count,
        "small_object_failures": small_object_error_count,
        "large_object_failures": large_object_error_count,
        "spatial_cluster_failures": cluster_error_count
    }
    
    # 3. Save neighborhood visualizations for key pixels (T1, T2, y_true, y_pred)
    if feature_cube is not None:
        key_cases = {
            "highest_confidence_tp": results["highest_confidence_tp"],
            "highest_confidence_tn": results["highest_confidence_tn"],
            "worst_fp": results["worst_fp"],
            "worst_fn": results["worst_fn"],
            "most_uncertain": results["most_uncertain"]
        }
        
        # Build 2D predictions map
        y_pred_2d = np.zeros((H, W), dtype=np.uint8)
        for idx, (r, c) in enumerate(coords):
            y_pred_2d[r, c] = y_pred[idx]
        
        for name, info in key_cases.items():
            if info is None:
                continue
            r, c = info["coords"]
            
            # Slice 15x15 patch
            R = 7
            r0, r1 = max(0, r - R), min(H, r + R + 1)
            c0, c1 = max(0, c - R), min(W, c + R + 1)
            
            # Slice features (Red band for T1 and T2 to inspect visual context)
            t1_slice = feature_cube[r0:r1, c0:c1, 0] # T1 Red
            t2_slice = feature_cube[r0:r1, c0:c1, 7] # T2 Red
            gt_slice = y_true_2d[r0:r1, c0:c1].astype(np.uint8)
            pred_slice = y_pred_2d[r0:r1, c0:c1]
            
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes[0, 0].imshow(t1_slice, cmap='gray')
            axes[0, 0].set_title("T1 Red Feature Patch")
            
            axes[0, 1].imshow(t2_slice, cmap='gray')
            axes[0, 1].set_title("T2 Red Feature Patch")
            
            axes[1, 0].imshow(gt_slice, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1, 0].set_title("Ground Truth Change")
            
            axes[1, 1].imshow(pred_slice, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1, 1].set_title(f"Prediction (prob={info['prob_change']:.4f})")
            
            # Mark the center pixel
            for ax in axes.ravel():
                ax.plot(c - c0, r - r0, 'bx', markersize=10, markeredgewidth=2)
                
            fig.suptitle(f"Failure Analysis Case: {name.replace('_', ' ').title()}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150)
            plt.close()
            
    return results

def generate_failure_report(results: Dict[str, Any], filepath: str) -> None:
    """Generate Markdown report for failure analysis."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, "w") as f:
        f.write("# GeoAI Platform — Failure Analysis Report\n\n")
        f.write(f"- **Execution Timestamp**: {timestamp}\n\n")
        
        f.write("## 1. Spatial & Object Failure Taxonomy\n\n")
        tax = results.get("failure_taxonomy", {})
        f.write("| Error Category | Pixel Count | Percentage of Total Errors |\n")
        f.write("| :--- | :--- | :--- |\n")
        tot = tax.get("total_errors", 1) or 1
        f.write(f"| Total Errors | {tax.get('total_errors')} | 100.00% |\n")
        f.write(f"| Boundary Failures | {tax.get('boundary_failures')} | {tax.get('boundary_failures')/tot*100:.2f}% |\n")
        f.write(f"| Small-Object Failures (<100px) | {tax.get('small_object_failures')} | {tax.get('small_object_failures')/tot*100:.2f}% |\n")
        f.write(f"| Large-Object Failures (>=100px) | {tax.get('large_object_failures')} | {tax.get('large_object_failures')/tot*100:.2f}% |\n")
        f.write(f"| Spatial Cluster Failures | {tax.get('spatial_cluster_failures')} | {tax.get('spatial_cluster_failures')/tot*100:.2f}% |\n\n")
        
        f.write("## 2. Key Diagnostic Cases\n\n")
        cases = ["highest_confidence_tp", "highest_confidence_tn", "worst_fp", "worst_fn", "most_uncertain"]
        for case in cases:
            info = results.get(case)
            f.write(f"### {case.replace('_', ' ').title()}\n")
            if info:
                f.write(f"- **Center Coordinate**: Row={info['coords'][0]}, Col={info['coords'][1]}\n")
                f.write(f"- **Prediction Probability (Change)**: {info['prob_change']:.4f}\n")
                f.write(f"![{case}](example_predictions/{case}.png)\n\n")
            else:
                f.write("- Case not observed or missing features.\n\n")
                
        f.write("\n## Scientific Failure Mitigation\n\n")
        f.write("> [!NOTE]\n")
        f.write("> Analysis indicates whether errors are dominated by high-frequency spatial noise (small-object errors) or edge classification alignments (boundary errors). Boundary failures are typically addressed via edge-refinement morphology.\n")
