#!/usr/bin/env python3
"""
GeoAI Platform — Milestone 10: Deep Learning Baseline Research Campaign Orchestrator.
Trains and evaluates FC-EF, FC-Siam-Conc, FC-Siam-Diff, and Lightweight Siamese CNN baselines
under Protocol A (Independent AOI) and Protocol B (Cross-AOI Transfer) with spatial block splits.
"""

import os
import sys
import json
import logging
import time
import shutil
from pathlib import Path
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.core.config import load_platform_config
from geoai.experiments.experiment_runner import ExperimentRunner
from geoai.experiments.experiment_registry import get_experiment_model
from geoai.models.baselines.dl_wrapper import split_dataset_spatial, get_pixel_coords
from geoai.evaluation.boundary_metrics import compute_boundary_campaign_metrics
from geoai.evaluation.evaluation_manager import _compute_calibration_metrics
from geoai.experiments.metrics import compute_metrics
from geoai.utils.raster_utils import flatten_spatial

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_experiment_yaml(model_id: str, dataset_id: str, patch_size: int, epochs: int, output_path: Path):
    """Generate experiment config YAML dynamically."""
    config_dict = {
        "experiment_id": f"dl_{model_id}_{dataset_id}",
        "description": f"Deep Learning Baseline {model_id} on {dataset_id}",
        "model": {
            "model_id": model_id,
            "hyperparameters": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": epochs,
                "patience": 5,
                "random_state": 42
            }
        },
        "dataset": {
            "dataset_id": dataset_id,
            "test_size": 0.20,
            "random_state": 42
        },
        "features": {
            "patch_size": patch_size
        },
        "baseline_lock": {
            "inherit_preprocessing": True,
            "inherit_feature_engineering": True,
            "deviations": [
                f"Deep Learning baseline model ({model_id})"
            ]
        },
        "ablation": {}
    }
    
    import yaml
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, indent=2)

def evaluate_cross_aoi(model_wrapper, tgt_id: str, configs_dir: Path, patch_size: int):
    """Evaluate a trained model wrapper on a target dataset under Protocol B."""
    logger.info("Evaluating cross-AOI transfer -> Target: %s", tgt_id)
    
    tgt_dataset = DatasetRegistry.load_dataset(tgt_id, configs_dir=str(configs_dir))
    X_tgt = tgt_dataset.X
    y_tgt = tgt_dataset.y
    
    # Extract expects channels
    expected_channels = model_wrapper.get_capabilities().expected_input_channels
    dataset_feature_names = list(model_wrapper.get_feature_names())
    col_indices = [
        dataset_feature_names.index(channel)
        for channel in expected_channels
        if channel in dataset_feature_names
    ]
    if col_indices:
        X_tgt = X_tgt[:, col_indices]
        
    # Spatial block split on target to get the correct test set
    _, X_test_tgt, _, y_test_tgt, _, _ = split_dataset_spatial(
        X_tgt, y_tgt, tgt_id, patch_size=patch_size
    )
    
    # Configure model wrapper for target evaluation coordinates mapping
    model_wrapper.set_dataset_info(dataset_id=tgt_id, config=None)
    model_wrapper.row_bytes_to_coords = None # Reset cache to rebuild for target dataset
    
    t0_inf = time.time()
    y_pred_tgt = model_wrapper.predict(X_test_tgt)
    y_prob_tgt = model_wrapper.predict_proba(X_test_tgt)[:, 1]
    inf_time = time.time() - t0_inf
    
    # Compute classification metrics
    metrics = compute_metrics(
        y_true=y_test_tgt,
        y_pred=y_pred_tgt,
        y_prob=y_prob_tgt,
        train_time=0.0,
        inference_time=inf_time,
        model_size_bytes=0,
        memory_usage_mb=0.0
    )
    
    # Spatial boundary evaluation on the full target AOI feature cube
    platform_config = load_platform_config(str(configs_dir))
    tgt_meta = DatasetRegistry.get_dataset_metadata(tgt_id)
    aoi_tgt = platform_config.get_aoi(tgt_meta.aoi_name)
    
    from geoai.pipeline.stage1 import _load_all_rasters
    from geoai.features.feature_cube import build_feature_cube
    from geoai.features.temporal import compute_all_deltas
    from geoai.features.pseudo_labels import generate_pseudo_labels
    from geoai.features.dataset import prepare_training_dataset
    
    rasters_tgt = _load_all_rasters(aoi_tgt)
    red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1 = rasters_tgt[0:6]
    red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2 = rasters_tgt[6:12]
    sar_t1, sar_t2 = rasters_tgt[12:14]
    
    feature_cube_tgt = build_feature_cube(
        red_t1=red_t1, green_t1=green_t1, blue_t1=blue_t1,
        sar_t1=sar_t1, ndvi_t1=ndvi_t1, ndbi_t1=ndbi_t1, ndwi_t1=ndwi_t1,
        red_t2=red_t2, green_t2=green_t2, blue_t2=blue_t2,
        sar_t2=sar_t2, ndvi_t2=ndvi_t2, ndbi_t2=ndbi_t2, ndwi_t2=ndwi_t2,
    )
    
    delta_ndvi, delta_ndbi, delta_ndwi, delta_sar = compute_all_deltas(
        ndvi_t1=ndvi_t1, ndvi_t2=ndvi_t2,
        ndbi_t1=ndbi_t1, ndbi_t2=ndbi_t2,
        ndwi_t1=ndwi_t1, ndwi_t2=ndwi_t2,
        sar_t1=sar_t1, sar_t2=sar_t2
    )
    
    change_mask_tgt, _, _ = generate_pseudo_labels(
        delta_sar=delta_sar,
        delta_ndvi=delta_ndvi,
        delta_ndbi=delta_ndbi,
        delta_ndwi=delta_ndwi,
        pseudo_label_config=platform_config.processing.pseudo_labels,
    )
    
    _, y_tgt_full, valid_mask_flat, _ = prepare_training_dataset(feature_cube_tgt, change_mask_tgt)
    valid_mask_2d = valid_mask_flat.reshape(ndvi_t1.shape)
    
    X_tgt_full = flatten_spatial(feature_cube_tgt)[valid_mask_2d.ravel()]
    if col_indices:
        X_tgt_full = X_tgt_full[:, col_indices]
        
    y_pred_full = model_wrapper.predict(X_tgt_full)
    
    y_true_2d = np.zeros(valid_mask_2d.shape, dtype=bool)
    y_true_2d[valid_mask_2d] = y_tgt_full.astype(bool)
    
    y_pred_2d = np.zeros(valid_mask_2d.shape, dtype=bool)
    y_pred_2d[valid_mask_2d] = y_pred_full.astype(bool)
    
    boundary_metrics = compute_boundary_campaign_metrics(y_true_2d, y_pred_2d)
    
    results = {
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
        "thickness_diff": boundary_metrics["thickness_diff"]
    }
    
    # Calibration metrics
    y_prob_tgt_full = model_wrapper.predict_proba(X_tgt_full)[:, 1]
    cal = _compute_calibration_metrics(y_tgt_full, y_prob_tgt_full)
    results["ece"] = cal["ece"]
    results["mce"] = cal["mce"]
    results["brier"] = cal["brier"]
    
    return results

def main():
    root_dir = Path(__file__).resolve().parents[1]
    configs_dir = root_dir / "configs"
    outputs_dir = root_dir / "outputs" / "deep_learning"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Configuration variables
    # Allow overriding epochs through environment variable (e.g. DL_EPOCHS=15)
    epochs = int(os.environ.get("DL_EPOCHS", 5))
    patch_size = int(os.environ.get("DL_PATCH_SIZE", 15))
    
    datasets = ["ps10_sentinel_v1", "dholera_sentinel_v1"]
    models = ["fc_ef", "fc_siam_conc", "fc_siam_diff", "lightweight_siam_cnn"]
    
    runner = ExperimentRunner(outputs_dir=str(root_dir / "outputs" / "experiments"), configs_dir=str(configs_dir))
    
    transfer_results = {}
    
    for src in datasets:
        transfer_results[src] = {}
        for model_id in models:
            logger.info("==================================================")
            logger.info("Training Model: %s on Dataset: %s", model_id, src)
            logger.info("==================================================")
            
            # Write dynamic experiment config file
            exp_yaml_path = configs_dir / "experiments" / f"tmp_dl_run_{model_id}_{src}.yaml"
            generate_experiment_yaml(model_id, src, patch_size, epochs, exp_yaml_path)
            
            # Execute training using ExperimentRunner
            try:
                experiment = runner.run(exp_yaml_path)
                
                # Check execution state
                if experiment.get_state() != "Completed":
                    logger.error("Experiment failed for model: %s on dataset: %s", model_id, src)
                    continue
                    
                # Load the fitted wrapper model
                model_wrapper = get_experiment_model(model_id)
                model_wrapper.set_dataset_info(dataset_id=src, config=None)
                
                # Load the best model checkpoint weights
                checkpoint_path = root_dir / "outputs" / "experiments" / f"dl_{model_id}_{src}" / "checkpoints" / "best_model.pt"
                model_wrapper.load(checkpoint_path)
                
                # Run evaluations across target datasets
                transfer_results[src][model_id] = {}
                for tgt in datasets:
                    if src == tgt:
                        # Protocol A: Evaluate on source test set
                        # Read the stored evaluation metrics from metrics.json
                        metrics_path = root_dir / "outputs" / "experiments" / f"dl_{model_id}_{src}" / "metrics.json"
                        with open(metrics_path, "r") as f:
                            src_metrics = json.load(f)
                            
                        # Extract metrics for protocol A
                        # Re-run target boundary metrics on target for consistency
                        res = evaluate_cross_aoi(model_wrapper, tgt, configs_dir, patch_size)
                        # We mix classification metrics from experiment metrics.json and spatial metrics
                        res["iou"] = src_metrics.get("iou", 0.0)
                        res["f1"] = src_metrics.get("f1", 0.0)
                        res["precision"] = src_metrics.get("precision", 0.0)
                        res["recall"] = src_metrics.get("recall", 0.0)
                        
                        transfer_results[src][model_id][tgt] = res
                    else:
                        # Protocol B: Evaluate cross-AOI transfer
                        res = evaluate_cross_aoi(model_wrapper, tgt, configs_dir, patch_size)
                        transfer_results[src][model_id][tgt] = res
                        
            except Exception as e:
                logger.exception("Error executing campaign run: model=%s, dataset=%s: %s", model_id, src, e)
            finally:
                # Cleanup temp config YAML
                if exp_yaml_path.exists():
                    exp_yaml_path.unlink()
                    
    # Save campaign summary to JSON
    summary_path = outputs_dir / "dl_baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(transfer_results, f, indent=2)
    logger.info("Deep learning campaign summary saved to %s", summary_path)
    
    # Generate Markdown report
    _write_markdown_summary(transfer_results, datasets, models, outputs_dir / "dl_baseline_summary.md")
    logger.info("Campaign completed successfully.")

def _write_markdown_summary(results: dict, datasets: list, models: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Deep Learning Baselines (Milestone 10) Research Campaign Report\n\n")
        f.write("This report presents the scientific evaluation of PyTorch deep learning baseline models under Protocol A (Independent AOI) and Protocol B (Cross-AOI Transfer) using Spatial Block-Splitting and Buffer Zones.\n\n")
        
        for src in datasets:
            f.write(f"## Source Domain: `{src}`\n\n")
            f.write("| Model ID | Target Domain | Pixel IoU | F1 Score | Boundary IoU | Chamfer Distance | ECE |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for model_id in models:
                if src not in results or model_id not in results[src]:
                    continue
                for tgt in datasets:
                    res = results[src][model_id][tgt]
                    f.write(f"| `{model_id}` | `{tgt}` | {res['iou']:.4f} | {res['f1']:.4f} | {res['boundary_iou']:.4f} | {res['chamfer_distance']:.2f} | {res['ece']:.4f} |\n")
            f.write("\n")

if __name__ == "__main__":
    main()
