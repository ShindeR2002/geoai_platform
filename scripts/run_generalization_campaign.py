#!/usr/bin/env python3
"""
GeoAI Platform — Milestone 9: Cross-AOI Generalization & Domain Robustness Campaign Orchestrator.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import split_dataset
from geoai.core.config import load_platform_config
from geoai.pipeline.stage1 import _load_all_rasters
from geoai.features.feature_cube import build_feature_cube
from geoai.evaluation.generalization import GeneralizationEvaluator
from geoai.evaluation.boundary_metrics import compute_boundary_campaign_metrics
from geoai.evaluation.evaluation_manager import _compute_calibration_metrics
from geoai.experiments.metrics import compute_metrics
from geoai.utils.raster_utils import flatten_spatial

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_transfer(src_id: str, tgt_id: str, configs_dir: Path):
    """Train on source dataset and evaluate on target dataset (classification and spatial boundary metrics)."""
    logger.info("Executing evaluation: %s -> %s", src_id, tgt_id)
    
    # Load source & target datasets
    src_dataset = DatasetRegistry.load_dataset(src_id, configs_dir=str(configs_dir))
    tgt_dataset = DatasetRegistry.load_dataset(tgt_id, configs_dir=str(configs_dir))
    
    # Stratified split for training
    X_train_src, _, y_train_src, _ = split_dataset(
        src_dataset.X, src_dataset.y,
        test_size=0.20,
        random_state=42,
        stratify=True
    )
    
    # Stratified split for testing (eval on target's test split)
    _, X_test_tgt, _, y_test_tgt = split_dataset(
        tgt_dataset.X, tgt_dataset.y,
        test_size=0.20,
        random_state=42,
        stratify=True
    )
    
    # Train standard baseline model
    clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    t0_fit = time.time()
    clf.fit(X_train_src, y_train_src)
    fit_time = time.time() - t0_fit
    
    # Predict on target test split
    y_pred_tgt = clf.predict(X_test_tgt)
    y_prob_tgt = clf.predict_proba(X_test_tgt)[:, 1]
    
    # Compute test classification metrics
    metrics = compute_metrics(
        y_true=y_test_tgt,
        y_pred=y_pred_tgt,
        y_prob=y_prob_tgt,
        train_time=fit_time,
        inference_time=0.0,
        model_size_bytes=0,
        memory_usage_mb=0.0
    )
    
    # Spatial boundary evaluation on the full target AOI feature cube
    platform_config = load_platform_config(str(configs_dir))
    tgt_meta = DatasetRegistry.get_dataset_metadata(tgt_id)
    aoi_tgt = platform_config.get_aoi(tgt_meta.aoi_name)
    
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
    
    from geoai.features.dataset import prepare_training_dataset
    from geoai.features.temporal import compute_all_deltas
    from geoai.features.pseudo_labels import generate_pseudo_labels
    
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
    y_pred_full = clf.predict(X_tgt_full)
    
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
    
    # Calculate calibration
    y_prob_tgt_full = clf.predict_proba(X_tgt_full)[:, 1]
    cal = _compute_calibration_metrics(y_tgt_full, y_prob_tgt_full)
    results["ece"] = cal["ece"]
    results["mce"] = cal["mce"]
    results["brier"] = cal["brier"]
    
    return results

def main():
    root_dir = Path(__file__).resolve().parents[1]
    configs_dir = root_dir / "configs"
    outputs_dir = root_dir / "outputs" / "generalization"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = GeneralizationEvaluator(configs_dir=str(configs_dir), outputs_dir=str(outputs_dir))
    
    # 1. Fetch registered datasets
    datasets = DatasetRegistry.list_datasets()
    logger.info("Registered datasets in catalog: %s", datasets)
    
    # Protocol A & B Matrix
    transfer_results = {}
    
    # Let's align features (we use canonical 18 features)
    from geoai.utils.constants import CANONICAL_FEATURE_NAMES
    feature_names = list(CANONICAL_FEATURE_NAMES)
    
    # Pre-load datasets to compute similarity matrices
    dataset_matrices = {}
    for d_id in datasets:
        dataset = DatasetRegistry.load_dataset(d_id, configs_dir=str(configs_dir))
        dataset_matrices[d_id] = dataset.X
        
    similarity_matrix = {}
    domain_shift_projections = {}
    
    for src in datasets:
        transfer_results[src] = {}
        similarity_matrix[src] = {}
        for tgt in datasets:
            # Run Protocols A (src == tgt) and B (src != tgt)
            res = evaluate_transfer(src, tgt, configs_dir)
            transfer_results[src][tgt] = res
            
            # Compute Dataset Similarity & Domain Shift visualization
            X_src = dataset_matrices[src]
            X_tgt = dataset_matrices[tgt]
            
            sim_score, sim_label, sim_details = evaluator.compute_dataset_similarity(X_src, X_tgt, feature_names)
            similarity_matrix[src][tgt] = {
                "similarity_score": sim_score,
                "similarity_label": sim_label,
                "details": sim_details
            }
            
            # Generate visualization projections
            meta_src = DatasetRegistry.get_dataset_metadata(src)
            meta_tgt = DatasetRegistry.get_dataset_metadata(tgt)
            proj = evaluator.generate_domain_shift_projections(
                X_src, X_tgt,
                src_name=meta_src.aoi_name,
                tgt_name=meta_tgt.aoi_name
            )
            domain_shift_projections[f"{src}_{tgt}"] = proj
            
    # Protocol C (Leave-One-AOI-Out) Fallback
    loao_results = {}
    loao_active = len(datasets) >= 3
    loao_message = ""
    
    if loao_active:
        logger.info("Executing Protocol C (Leave-One-AOI-Out)...")
        # Placeholder for LOAO if 3 or more datasets are added
        for held_out in datasets:
            # Combine all other datasets
            X_train_list = []
            y_train_list = []
            for train_id in datasets:
                if train_id == held_out:
                    continue
                # Split to extract train portion
                ds = DatasetRegistry.load_dataset(train_id, configs_dir=str(configs_dir))
                X_tr, _, y_tr, _ = split_dataset(ds.X, ds.y, test_size=0.20, random_state=42, stratify=True)
                X_train_list.append(X_tr)
                y_train_list.append(y_tr)
                
            X_combined_train = np.concatenate(X_train_list, axis=0)
            y_combined_train = np.concatenate(y_train_list, axis=0)
            
            # Target test split
            tgt_ds = DatasetRegistry.load_dataset(held_out, configs_dir=str(configs_dir))
            _, X_test_tgt, _, y_test_tgt = split_dataset(tgt_ds.X, tgt_ds.y, test_size=0.20, random_state=42, stratify=True)
            
            clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
            clf.fit(X_combined_train, y_combined_train)
            y_pred = clf.predict(X_test_tgt)
            
            # Simple IoU / F1 metric for LOAO
            intersection = ((y_test_tgt == 1) & (y_pred == 1)).sum()
            union = ((y_test_tgt == 1) | (y_pred == 1)).sum()
            iou = intersection / union if union > 0 else 0.0
            
            loao_results[held_out] = {
                "iou": float(iou),
                "status": "Completed"
            }
    else:
        loao_message = "Leave-One-AOI-Out validation requires at least three registered AOIs."
        logger.warning(loao_message)
        
    # Calculate generalization gap, difficulty index, and robustness score
    aoi_difficulty = {}
    for tgt in datasets:
        # Difficulty = average target performance (Protocol B) across all source models
        tgt_scores = [transfer_results[src][tgt]["iou"] for src in datasets]
        # Binned Difficulty Index = 1.0 - mean(iou)
        aoi_difficulty[tgt] = float(1.0 - np.mean(tgt_scores))
        
    generalization_gap = {}
    for src in datasets:
        generalization_gap[src] = {}
        for tgt in datasets:
            if src == tgt:
                gap = 0.0
            else:
                gap = transfer_results[src][src]["iou"] - transfer_results[src][tgt]["iou"]
            generalization_gap[src][tgt] = float(gap)
            
    summary_results = {
        "datasets": datasets,
        "transfer_results": transfer_results,
        "similarity_matrix": similarity_matrix,
        "generalization_gap": generalization_gap,
        "aoi_difficulty": aoi_difficulty,
        "domain_shift_projections": domain_shift_projections,
        "loao": {
            "active": loao_active,
            "message": loao_message,
            "results": loao_results
        }
    }
    
    # Save to JSON
    summary_path = root_dir / "outputs" / "generalization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    logger.info("Generalization summary saved to %s", summary_path)
    
    # Write Markdown summary
    _write_markdown_summary(summary_results, root_dir / "outputs" / "generalization_summary.md")
    
    # Generate Thesis report
    from geoai.evaluation.publication_report import generate_generalization_report
    generate_generalization_report(summary_results, root_dir / "outputs" / "generalization_report.md")
    logger.info("Generalization campaign completed successfully.")

def _write_markdown_summary(results: dict, path: Path):
    datasets = results["datasets"]
    transfer = results["transfer_results"]
    similarity = results["similarity_matrix"]
    difficulty = results["aoi_difficulty"]
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Cross-AOI Generalization & Domain Robustness Benchmark Report\n\n")
        f.write("This report presents generalization gaps, transferability scores, and domain shift similarity measures.\n\n")
        
        # 1. Similarity Table
        f.write("## 1. Pairwise Dataset Similarity Grid\n\n")
        f.write("| Source AOI | Target AOI | Similarity Score | Qualitative Label | Mean PSI | Mean JSD |\n")
        f.write("| :--- | :--- | :---: | :--- | :---: | :---: |\n")
        for src in datasets:
            for tgt in datasets:
                sim = similarity[src][tgt]
                f.write(f"| `{src}` | `{tgt}` | {sim['similarity_score']:.2f} | **{sim['similarity_label']}** | {sim['details']['mean_psi']:.4f} | {sim['details']['mean_jsd']:.4f} |\n")
                
        # 2. Transfer Matrix Table
        f.write("\n## 2. Cross-AOI Jaccard IoU Transfer Matrix\n\n")
        header = "| Source \\ Target | " + " | ".join([f"`{d}`" for d in datasets]) + " |\n"
        separator = "| :--- | " + " | ".join([":---:" for _ in datasets]) + " |\n"
        f.write(header)
        f.write(separator)
        for src in datasets:
            row_str = f"| `{src}` | " + " | ".join([f"{transfer[src][tgt]['iou']:.4f}" for tgt in datasets]) + " |\n"
            f.write(row_str)
            
        # 3. Difficulty Rankings
        f.write("\n## 3. AOI Difficulty Indices (Lower IoU implies Higher Difficulty)\n\n")
        f.write("| AOI ID | Difficulty Index | Avg Transfer IoU |\n")
        f.write("| :--- | :---: | :---: |\n")
        for d_id in sorted(datasets, key=lambda x: difficulty[x], reverse=True):
            avg_iou = 1.0 - difficulty[d_id]
            f.write(f"| `{d_id}` | {difficulty[d_id]:.4f} | {avg_iou:.4f} |\n")
            
        # 4. LOAO Summary
        f.write("\n## 4. Protocol C: Leave-One-AOI-Out Generalization\n\n")
        loao = results["loao"]
        if loao["active"]:
            f.write("| Held-Out Target | Leave-One-Out IoU |\n")
            f.write("| :--- | :---: |\n")
            for k, v in loao["results"].items():
                f.write(f"| `{k}` | {v['iou']:.4f} |\n")
        else:
            f.write(f"> ⚠️ **Bypassed**: {loao['message']}\n")

if __name__ == "__main__":
    main()
