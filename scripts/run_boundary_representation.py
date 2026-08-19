#!/usr/bin/env python3
"""
GeoAI Platform — Campaign A: Boundary Representation Benchmark Orchestrator.
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
from geoai.evaluation.boundary_metrics import compute_boundary_campaign_metrics
from geoai.core.config import load_platform_config
from geoai.pipeline.stage1 import _load_all_rasters
from geoai.features.feature_cube import build_feature_cube

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    root_dir = Path(__file__).resolve().parents[1]
    configs_dir = root_dir / "configs"
    outputs_dir = root_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    configs_map = {
        "boundary_baseline": "configs/experiments/boundary_baseline.yaml",
        "boundary_sobel": "configs/experiments/boundary_sobel.yaml",
        "boundary_scharr": "configs/experiments/boundary_scharr.yaml",
        "boundary_laplacian": "configs/experiments/boundary_laplacian.yaml",
        "boundary_morphological_gradient": "configs/experiments/boundary_morphological_gradient.yaml",
        "boundary_distance_transform": "configs/experiments/boundary_distance_transform.yaml"
    }
    
    runner = ExperimentRunner(outputs_dir=str(root_dir / "outputs" / "experiments"), configs_dir=str(configs_dir))
    
    results = {}
    predictions_cache = {}
    
    # Load platform configuration and valid_mask_2d
    platform_config = load_platform_config(str(configs_dir))
    # We use Dholera AOI as it is standard in the dataset_id "dholera_sentinel_v1"
    aoi = platform_config.get_aoi("Dholera")
    rasters = _load_all_rasters(aoi)
    red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1 = rasters[0:6]
    red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2 = rasters[6:12]
    sar_t1, sar_t2 = rasters[12:14]
    
    # Build baseline feature cube to compute exact valid mask
    feature_cube_base = build_feature_cube(
        red_t1=red_t1, green_t1=green_t1, blue_t1=blue_t1,
        sar_t1=sar_t1, ndvi_t1=ndvi_t1, ndbi_t1=ndbi_t1, ndwi_t1=ndwi_t1,
        red_t2=red_t2, green_t2=green_t2, blue_t2=blue_t2,
        sar_t2=sar_t2, ndvi_t2=ndvi_t2, ndbi_t2=ndbi_t2, ndwi_t2=ndwi_t2,
    )
    from geoai.features.dataset import prepare_training_dataset
    _, _, valid_mask_flat, _ = prepare_training_dataset(feature_cube_base, np.zeros_like(ndvi_t1))
    valid_mask_2d = valid_mask_flat.reshape(ndvi_t1.shape)
    
    for exp_id, rel_path in configs_map.items():
        config_path = root_dir / rel_path
        if not config_path.exists():
            logger.warning("Config path %s not found. Skipping.", config_path)
            continue
            
        logger.info("=" * 60)
        logger.info("Executing Campaign A Experiment: %s", exp_id)
        logger.info("=" * 60)
        
        try:
            model_pkl_path = root_dir / "outputs" / "experiments" / exp_id / "model.pkl"
            metrics_json_path = root_dir / "outputs" / "experiments" / exp_id / "metrics.json"
            
            if model_pkl_path.exists() and metrics_json_path.exists():
                logger.info("Experiment %s already completed. Skipping retraining.", exp_id)
            else:
                logger.info("Executing training for Experiment: %s", exp_id)
                runner.run(config_path)
            
            # Load stored metrics
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                
            # Reconstruct the test split predictions for McNemar
            config = ExperimentConfig(config_path)
            dataset = DatasetRegistry.load_dataset(
                config.dataset_id,
                configs_dir=str(configs_dir),
                preprocessing_config=config.preprocessing,
                features_config=config.features,
                campaign_type=config.campaign_type,
                track=config.track
            )
            X = dataset.X
            y = dataset.y
            
            # Subsampling features
            from geoai.experiments.experiment_registry import get_experiment_model
            model_wrapper = get_experiment_model(config.model_id)
            expected_channels = list(model_wrapper.get_capabilities().expected_input_channels)
            
            dataset_feature_names = get_campaign_feature_names(
                campaign_type=config.campaign_type,
                track=config.track,
                config=config.to_dict()
            )
            
            # Add extra boundary features to expected_channels
            from geoai.utils.constants import CANONICAL_FEATURE_NAMES
            extra_features = [f for f in dataset_feature_names if f not in CANONICAL_FEATURE_NAMES]
            for f in extra_features:
                if f not in expected_channels:
                    expected_channels.append(f)
            
            col_indices = [
                dataset_feature_names.index(channel)
                for channel in expected_channels
                if channel in dataset_feature_names
            ]
            
            if col_indices:
                X = X[:, col_indices]
                
            X_train, X_test, y_train, y_test = split_dataset(
                X, y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=True
            )
            
            clf = joblib.load(model_pkl_path)
            y_pred_test = clf.predict(X_test)
            
            predictions_cache[exp_id] = {
                "y_pred_test": y_pred_test,
                "y_true_test": y_test
            }
            
            # Compute full-AOI spatial prediction to evaluate spatial boundary metrics
            logger.info("Computing full-AOI spatial predictions for boundary metrics...")
            y_pred_full = clf.predict(X)
            
            y_true_2d = np.zeros(valid_mask_2d.shape, dtype=bool)
            y_true_2d[valid_mask_2d] = y.astype(bool)
            
            y_pred_2d = np.zeros(valid_mask_2d.shape, dtype=bool)
            y_pred_2d[valid_mask_2d] = y_pred_full.astype(bool)
            
            boundary_metrics = compute_boundary_campaign_metrics(y_true_2d, y_pred_2d)
            
            results[exp_id] = {
                "iou": metrics.get("iou", 0.0),
                "f1": metrics.get("f1", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "boundary_iou": boundary_metrics["boundary_iou"],
                "chamfer_distance": boundary_metrics["chamfer_distance"],
                "hausdorff_distance": boundary_metrics["hausdorff_distance"],
                "bsi_true": boundary_metrics["bsi_true"],
                "bsi_pred": boundary_metrics["bsi_pred"],
                "bsi_diff": boundary_metrics["bsi_diff"],
                "thickness_true": boundary_metrics["thickness_true"],
                "thickness_pred": boundary_metrics["thickness_pred"],
                "thickness_diff": boundary_metrics["thickness_diff"],
                "train_time_sec": metrics.get("train_time_sec", 0.0),
                "inference_time_sec": metrics.get("inference_time_sec", 0.0),
                "memory_usage_mb": metrics.get("memory_usage_mb", 0.0),
                "status": "Completed"
            }
            logger.info("Boundary metrics for %s: %s", exp_id, boundary_metrics)
            
        except Exception as e:
            logger.exception("Experiment %s failed: %s", exp_id, e)
            results[exp_id] = {
                "status": "Failed",
                "error": str(e)
            }
            
    # Compute McNemar tests relative to boundary_baseline
    significance_results = {}
    if "boundary_baseline" in predictions_cache:
        base_data = predictions_cache["boundary_baseline"]
        for exp_id, pred_data in predictions_cache.items():
            if exp_id == "boundary_baseline":
                continue
            mcnemar = run_mcnemar_test(pred_data["y_true_test"], base_data["y_pred_test"], pred_data["y_pred_test"])
            significance_results[exp_id] = mcnemar
            if exp_id in results:
                results[exp_id]["mcnemar_p_value"] = mcnemar["p_value"]
                results[exp_id]["statistically_significant"] = mcnemar["significant"]
                
    # Save the summary JSON file
    summary_path = outputs_dir / "boundary_representation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Summary report saved to %s", summary_path)
    
    # Generate Leaderboard Plot
    _plot_leaderboard(results, outputs_dir)
    
    # Generate Markdown summary report
    _write_markdown_summary(results, summary_path.with_suffix(".md"))
    
    logger.info("Campaign A Benchmark Complete.")

def _plot_leaderboard(results: dict, outputs_dir: Path):
    valid_results = {k: v for k, v in results.items() if v.get("status") == "Completed"}
    if not valid_results:
        return
        
    names = list(valid_results.keys())
    b_ious = [valid_results[n]["boundary_iou"] for n in names]
    
    # Sort
    idx = np.argsort(b_ious)
    sorted_names = [names[i] for i in idx]
    sorted_b_ious = [b_ious[i] for i in idx]
    
    plt.figure(figsize=(10, 5))
    bars = plt.barh(sorted_names, sorted_b_ious, color='#4F46E5')
    plt.xlim(0.0, 1.0)
    plt.xlabel("Boundary IoU (2px buffer)")
    plt.title("Campaign A: Boundary Representation Leaderboard\n(Which continuous representation reduces boundary errors best?)")
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width:.4f}", 
                 va='center', ha='left', fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig(outputs_dir / "boundary_representation_leaderboard.png", dpi=150)
    plt.close()

def _write_markdown_summary(results: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Campaign A — Boundary Representation Benchmark Report\n\n")
        f.write("This report evaluates various pixel-level boundary representation methods on their ability to reduce object boundary errors.\n\n")
        
        f.write("| Method | Pixel IoU | Boundary IoU | Chamfer Dist | Hausdorff Dist | BSI Diff | Thickness Diff | McNemar p | Significant? |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for k, v in results.items():
            if v.get("status") != "Completed":
                f.write(f"| `{k}` | Failed | | | | | | | |\n")
                continue
            p_val_str = f"{v.get('mcnemar_p_value', 1.0):.4e}" if k != "boundary_baseline" else "N/A"
            sig_str = "Yes" if v.get("statistically_significant", False) else "No" if k != "boundary_baseline" else "Baseline"
            f.write(f"| `{k}` | {v['iou']:.4f} | {v['boundary_iou']:.4f} | {v['chamfer_distance']:.4f} | {v['hausdorff_distance']:.4f} | {v['bsi_diff']:.4f} | {v['thickness_diff']:.4f} | {p_val_str} | {sig_str} |\n")
            
        f.write("\n## Conclusions & Scientific Findings\n")
        completed = [k for k, v in results.items() if v.get("status") == "Completed"]
        if len(completed) > 1:
            best_method = max((k for k in completed if k != "boundary_baseline"), key=lambda x: results[x]["boundary_iou"])
            base = results["boundary_baseline"]
            best = results[best_method]
            f.write(f"- The best performing boundary representation method is `{best_method}` with a Boundary IoU of **{best['boundary_iou']:.4f}** (compared to baseline **{base['boundary_iou']:.4f}**).\n")
            f.write(f"- Chamfer Distance was reduced to **{best['chamfer_distance']:.4f}** (compared to baseline **{base['chamfer_distance']:.4f}**).\n")
            f.write(f"- directed Hausdorff Distance was changed to **{best['hausdorff_distance']:.4f}** (baseline: **{base['hausdorff_distance']:.4f}**).\n")
            if best.get("statistically_significant", False):
                f.write("- The improvement is statistically significant based on McNemar's test.\n")

if __name__ == "__main__":
    main()
