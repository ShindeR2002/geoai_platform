#!/usr/bin/env python3
"""
GeoAI Platform — Campaign B: Boundary Refinement Cascade Benchmark Orchestrator.
"""

import os
import sys
import time
import json
import logging
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoai.experiments.experiment_runner import ExperimentRunner
from geoai.experiments.experiment_config import ExperimentConfig
from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import split_dataset
from geoai.preprocessing.pipeline import get_campaign_feature_names
from geoai.evaluation.statistical_analysis import run_mcnemar_test
from geoai.evaluation.boundary_metrics import compute_boundary_campaign_metrics, compute_boundary_iou
from geoai.core.config import load_platform_config
from geoai.pipeline.stage1 import _load_all_rasters
from geoai.features.feature_cube import build_feature_cube
from geoai.features.boundary import BoundaryRefinementCascade
from geoai.utils.raster_utils import flatten_spatial
from geoai.experiments.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    root_dir = Path(__file__).resolve().parents[1]
    configs_dir = root_dir / "configs"
    outputs_dir = root_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    base_config_path = configs_dir / "experiments" / "boundary_baseline.yaml"
    refined_config_path = configs_dir / "experiments" / "boundary_refinement.yaml"
    
    runner = ExperimentRunner(outputs_dir=str(root_dir / "outputs" / "experiments"), configs_dir=str(configs_dir))
    
    # 1. Ensure models are trained
    logger.info("Ensuring base model (boundary_baseline) is trained...")
    runner.run(base_config_path)
    
    logger.info("Ensuring refined model (boundary_refinement) is trained...")
    runner.run(refined_config_path)
    
    # 2. Load models
    base_model_path = root_dir / "outputs" / "experiments" / "boundary_baseline" / "model.pkl"
    refined_model_path = root_dir / "outputs" / "experiments" / "boundary_refinement" / "model.pkl"
    
    base_model = joblib.load(base_model_path)
    refined_model = joblib.load(refined_model_path)
    
    # 3. Load dataset to get X and y for the test split
    config_base = ExperimentConfig(base_config_path)
    dataset_base = DatasetRegistry.load_dataset(
        config_base.dataset_id,
        configs_dir=str(configs_dir),
        preprocessing_config=config_base.preprocessing,
        features_config=config_base.features,
        campaign_type=config_base.campaign_type,
        track=config_base.track
    )
    y = dataset_base.y
    
    # Load platform configuration
    platform_config = load_platform_config(str(configs_dir))
    aoi = platform_config.get_aoi("Dholera")
    rasters = _load_all_rasters(aoi)
    red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1 = rasters[0:6]
    red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2 = rasters[6:12]
    sar_t1, sar_t2 = rasters[12:14]
    
    # Build the 18-feature baseline cube
    feature_cube_base = build_feature_cube(
        red_t1=red_t1, green_t1=green_t1, blue_t1=blue_t1,
        sar_t1=sar_t1, ndvi_t1=ndvi_t1, ndbi_t1=ndbi_t1, ndwi_t1=ndwi_t1,
        red_t2=red_t2, green_t2=green_t2, blue_t2=blue_t2,
        sar_t2=sar_t2, ndvi_t2=ndvi_t2, ndbi_t2=ndbi_t2, ndwi_t2=ndwi_t2,
    )
    
    # Compute exact canonical valid pixel mask
    from geoai.features.dataset import prepare_training_dataset
    _, _, valid_mask_flat, _ = prepare_training_dataset(feature_cube_base, np.zeros_like(ndvi_t1))
    valid_mask_2d = valid_mask_flat.reshape(ndvi_t1.shape)
    
    X_base_flat = flatten_spatial(feature_cube_base)[valid_mask_2d.ravel()]
    
    # 4. Instantiate cascade and run prediction
    logger.info("Executing two-pass BoundaryRefinementCascade...")
    cascade = BoundaryRefinementCascade(base_model, refined_model)
    
    t0 = time.time()
    y_pred_refined, y_pred_base, label_arr = cascade.predict_refined(
        X_base_flat=X_base_flat,
        feature_cube_base=feature_cube_base,
        valid_mask_2d=valid_mask_2d
    )
    inference_time = time.time() - t0
    logger.info("Cascade prediction completed in %.4f seconds.", inference_time)
    
    # Reconstruct 2D arrays
    y_true_2d = np.zeros(valid_mask_2d.shape, dtype=bool)
    y_true_2d[valid_mask_2d] = y.astype(bool)
    
    y_pred_base_2d = np.zeros(valid_mask_2d.shape, dtype=bool)
    y_pred_base_2d[valid_mask_2d] = y_pred_base.astype(bool)
    
    y_pred_refined_2d = np.zeros(valid_mask_2d.shape, dtype=bool)
    y_pred_refined_2d[valid_mask_2d] = y_pred_refined.astype(bool)
    
    # Evaluate classification metrics on the test split
    # For a fair comparison of classification performance, let's use the split_dataset index
    # Split the dataset indices deterministically to calculate standard test metrics (IoU, F1)
    indices = np.arange(len(y))
    _, test_idx, _, _ = split_dataset(
        indices, y,
        test_size=config_base.test_size,
        random_state=config_base.random_state,
        stratify=True
    )
    
    y_test = y[test_idx]
    y_pred_base_test = y_pred_base[test_idx]
    y_pred_refined_test = y_pred_refined[test_idx]
    
    # Base classification metrics
    base_metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred_base_test,
        y_prob=None,
        train_time=0.0,
        inference_time=0.0,
        model_size_bytes=base_model_path.stat().st_size,
        memory_usage_mb=0.0
    )
    
    # Refined classification metrics
    refined_metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred_refined_test,
        y_prob=None,
        train_time=0.0,
        inference_time=inference_time,
        model_size_bytes=refined_model_path.stat().st_size,
        memory_usage_mb=0.0
    )
    
    # Compute spatial boundary quality metrics
    base_boundary = compute_boundary_campaign_metrics(y_true_2d, y_pred_base_2d)
    refined_boundary = compute_boundary_campaign_metrics(y_true_2d, y_pred_refined_2d)
    
    # Perform McNemar's test for significance on the test split
    mcnemar = run_mcnemar_test(y_test, y_pred_base_test, y_pred_refined_test)
    
    results = {
        "boundary_baseline": {
            "iou": base_metrics.get("iou", 0.0),
            "f1": base_metrics.get("f1", 0.0),
            "precision": base_metrics.get("precision", 0.0),
            "recall": base_metrics.get("recall", 0.0),
            "boundary_iou": base_boundary["boundary_iou"],
            "chamfer_distance": base_boundary["chamfer_distance"],
            "hausdorff_distance": base_boundary["hausdorff_distance"],
            "bsi_true": base_boundary["bsi_true"],
            "bsi_pred": base_boundary["bsi_pred"],
            "bsi_diff": base_boundary["bsi_diff"],
            "thickness_true": base_boundary["thickness_true"],
            "thickness_pred": base_boundary["thickness_pred"],
            "thickness_diff": base_boundary["thickness_diff"],
            "status": "Completed"
        },
        "boundary_refinement_cascade": {
            "iou": refined_metrics.get("iou", 0.0),
            "f1": refined_metrics.get("f1", 0.0),
            "precision": refined_metrics.get("precision", 0.0),
            "recall": refined_metrics.get("recall", 0.0),
            "boundary_iou": refined_boundary["boundary_iou"],
            "chamfer_distance": refined_boundary["chamfer_distance"],
            "hausdorff_distance": refined_boundary["hausdorff_distance"],
            "bsi_true": refined_boundary["bsi_true"],
            "bsi_pred": refined_boundary["bsi_pred"],
            "bsi_diff": refined_boundary["bsi_diff"],
            "thickness_true": refined_boundary["thickness_true"],
            "thickness_pred": refined_boundary["thickness_pred"],
            "thickness_diff": refined_boundary["thickness_diff"],
            "mcnemar_p_value": mcnemar["p_value"],
            "statistically_significant": mcnemar["significant"],
            "inference_time_sec": inference_time,
            "status": "Completed"
        }
    }
    
    # Save to summary JSON file
    summary_path = outputs_dir / "boundary_refinement_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Summary report saved to %s", summary_path)
    
    # Write Markdown summary report
    _write_markdown_summary(results, summary_path.with_suffix(".md"))
    
    logger.info("Campaign B Benchmark Complete.")

def _write_markdown_summary(results: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Campaign B — Boundary Refinement Cascade Benchmark Report\n\n")
        f.write("This report evaluates the two-pass object-level morphological boundary refinement cascade on its ability to refine initial object boundaries.\n\n")
        
        f.write("| Configuration | Pixel IoU | Boundary IoU | Chamfer Dist | Hausdorff Dist | BSI Diff | Thickness Diff | McNemar p | Significant? |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for k in ["boundary_baseline", "boundary_refinement_cascade"]:
            v = results.get(k, {})
            if v.get("status") != "Completed":
                f.write(f"| `{k}` | Failed | | | | | | | |\n")
                continue
            p_val_str = f"{v.get('mcnemar_p_value', 1.0):.4e}" if k != "boundary_baseline" else "N/A"
            sig_str = "Yes" if v.get("statistically_significant", False) else "No" if k != "boundary_baseline" else "Baseline"
            f.write(f"| `{k}` | {v['iou']:.4f} | {v['boundary_iou']:.4f} | {v['chamfer_distance']:.4f} | {v['hausdorff_distance']:.4f} | {v['bsi_diff']:.4f} | {v['thickness_diff']:.4f} | {p_val_str} | {sig_str} |\n")
            
        f.write("\n## Conclusions & Scientific Findings\n")
        base = results["boundary_baseline"]
        refined = results["boundary_refinement_cascade"]
        f.write(f"- The `boundary_refinement_cascade` achieved a Boundary IoU of **{refined['boundary_iou']:.4f}** (baseline: **{base['boundary_iou']:.4f}**).\n")
        f.write(f"- Chamfer Distance was changed to **{refined['chamfer_distance']:.4f}** (baseline: **{base['chamfer_distance']:.4f}**).\n")
        f.write(f"- directed Hausdorff Distance was changed to **{refined['hausdorff_distance']:.4f}** (baseline: **{base['hausdorff_distance']:.4f}**).\n")
        f.write(f"- Boundary Smoothness Index difference to true changed to **{refined['bsi_diff']:.4f}** (baseline: **{base['bsi_diff']:.4f}**).\n")
        f.write(f"- Average Boundary Thickness difference to true changed to **{refined['thickness_diff']:.4f}** (baseline: **{base['thickness_diff']:.4f}**).\n")
        if refined.get("statistically_significant", False):
            f.write("- The cascade refinement yields a statistically significant change in predictions (McNemar p-value < 0.05).\n")
        else:
            f.write("- The cascade refinement does not show a statistically significant shift relative to baseline predictions on this partition.\n")

if __name__ == "__main__":
    main()
