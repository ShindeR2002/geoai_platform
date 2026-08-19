#!/usr/bin/env python3
"""
GeoAI Platform — Preprocessing & Feature Engineering Benchmark Campaigns Orchestrator.

Executes:
1. Campaign A: Preprocessing Benchmark (Fixed 18 features)
2. Campaign B: Feature Engineering Benchmark
   - Track A (Production: physical bands only)
   - Track B (Experimental: reconstructed NIR approximations)

Performs scientific validation, statistical significance testing,
computes tradeoffs, generates plots, and writes Markdown & LaTeX summary reports.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    root_dir = Path(__file__).resolve().parents[1]
    configs_dir = root_dir / "configs"
    outputs_base_dir = root_dir / "outputs" / "preprocessing_benchmark"
    reports_dir = outputs_base_dir / "reports"
    figures_dir = outputs_base_dir / "figures"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 9 configuration files mapping
    configs_map = {
        # Campaign A: Preprocessing (Fixed 18 features)
        "preproc_baseline": "configs/experiments/preproc_baseline.yaml",
        "preproc_refined_lee": "configs/experiments/preproc_refined_lee.yaml",
        
        # Campaign B: Feature Engineering (Track A - Production)
        "feat_prod_baseline": "configs/experiments/feat_prod_baseline.yaml",
        "feat_prod_local_variance": "configs/experiments/feat_prod_local_variance.yaml",
        
        # Campaign B: Feature Engineering (Track B - Experimental)
        "feat_exp_savi": "configs/experiments/feat_exp_savi.yaml",
        "feat_exp_msavi": "configs/experiments/feat_exp_msavi.yaml",
        "feat_exp_savi_msavi": "configs/experiments/feat_exp_savi_msavi.yaml",
        "feat_exp_combined": "configs/experiments/feat_exp_combined.yaml",
        "feat_lee_savi_msavi": "configs/experiments/feat_lee_savi_msavi.yaml",
    }
    
    runner = ExperimentRunner(outputs_dir=str(root_dir / "outputs" / "experiments"), configs_dir=str(configs_dir))
    
    results = {}
    
    # 1. Run all experiments
    for exp_id, rel_path in configs_map.items():
        config_path = root_dir / rel_path
        if not config_path.exists():
            logger.warning("Config path %s not found. Skipping.", config_path)
            continue
            
        logger.info("=" * 60)
        logger.info("Executing Experiment: %s", exp_id)
        logger.info("=" * 60)
        
        try:
            model_pkl_path = root_dir / "outputs" / "experiments" / exp_id / "model.pkl"
            metrics_json_path = root_dir / "outputs" / "experiments" / exp_id / "metrics.json"
            
            if model_pkl_path.exists() and metrics_json_path.exists():
                logger.info("Experiment %s already completed (found model.pkl and metrics.json). Skipping retraining.", exp_id)
            else:
                logger.info("Executing training for Experiment: %s", exp_id)
                # Run experiment using standard runner
                runner.run(config_path)
            
            # Load stored metrics
            if metrics_json_path.exists():
                with open(metrics_json_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                logger.error("metrics.json not found for %s", exp_id)
                metrics = {}
                
            results[exp_id] = {
                "config_path": str(config_path),
                "metrics": metrics,
                "status": "Completed"
            }
        except Exception as e:
            logger.exception("Experiment %s failed with exception", exp_id)
            results[exp_id] = {
                "config_path": str(config_path),
                "metrics": {},
                "status": "Failed",
                "error": str(e)
            }
            
    # Save raw results dictionary
    with open(outputs_base_dir / "raw_campaign_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # 2. Re-load test sets and compute predictions for statistical significance testing (McNemar's test)
    logger.info("Computing statistical significance tests (McNemar's test)...")
    
    # Define baseline mapping
    # Preprocessing uses preproc_baseline
    # Feature Engineering uses feat_prod_baseline
    baselines = {
        "preprocessing": "preproc_baseline",
        "feature_engineering_production": "feat_prod_baseline",
        "feature_engineering_experimental": "feat_prod_baseline"
    }
    
    predictions_cache = {}
    
    for exp_id, res in results.items():
        if res["status"] != "Completed":
            continue
            
        config = ExperimentConfig(Path(res["config_path"]))
        model_pkl_path = root_dir / "outputs" / "experiments" / exp_id / "model.pkl"
        
        if not model_pkl_path.exists():
            continue
            
        try:
            # Reconstruct the test split deterministically
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
            
            # Slice columns
            expected_channels = list(config.to_dict().get("model", {}).get("model_id", ""))
            # Get model Capabilities to fetch expected channel names
            # Re-implement Capabilities extraction
            from geoai.experiments.experiment_registry import get_experiment_model
            model_wrapper = get_experiment_model(config.model_id)
            expected_channels = list(model_wrapper.get_capabilities().expected_input_channels)
            
            if config.campaign_type == "feature_engineering":
                extra_features = []
                features_config = config.features
                if config.track == "production" or config.track == "experimental":
                    if features_config.get("local_variance", {}).get("enabled", False):
                        extra_features.extend(["Local_Var_T1", "Local_Var_T2"])
                if config.track == "experimental":
                    if features_config.get("savi", {}).get("enabled", False):
                        extra_features.extend(["SAVI_T1", "SAVI_T2"])
                    if features_config.get("msavi", {}).get("enabled", False):
                        extra_features.extend(["MSAVI_T1", "MSAVI_T2"])
                for f in extra_features:
                    if f not in expected_channels:
                        expected_channels.append(f)
                        
            dataset_feature_names = get_campaign_feature_names(
                campaign_type=config.campaign_type,
                track=config.track,
                config=config.to_dict()
            )
            
            col_indices = [
                dataset_feature_names.index(channel)
                for channel in expected_channels
                if channel in dataset_feature_names
            ]
            
            if col_indices:
                X = X[:, col_indices]
                
            _, X_test, _, y_test = split_dataset(
                X, y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=True
            )
            
            # Load model and predict
            clf = joblib.load(model_pkl_path)
            y_pred = clf.predict(X_test)
            
            predictions_cache[exp_id] = {
                "y_pred": y_pred,
                "y_true": y_test
            }
        except Exception as e:
            logger.error("Could not compute predictions for significance test on %s: %s", exp_id, e)

    # Calculate McNemar's tests
    significance_results = {}
    for exp_id, pred_data in predictions_cache.items():
        config = ExperimentConfig(Path(results[exp_id]["config_path"]))
        
        # Resolve baseline exp_id
        if config.campaign_type == "preprocessing":
            baseline_id = baselines["preprocessing"]
        else:
            if config.track == "production":
                baseline_id = baselines["feature_engineering_production"]
            else:
                baseline_id = baselines["feature_engineering_experimental"]
                
        if baseline_id == exp_id:
            # Baseline compared to itself is skip
            continue
            
        if baseline_id in predictions_cache:
            base_data = predictions_cache[baseline_id]
            mcnemar = run_mcnemar_test(pred_data["y_true"], base_data["y_pred"], pred_data["y_pred"])
            significance_results[exp_id] = mcnemar
            
    # 3. Generate leaderboards & plots
    logger.info("Generating scientific benchmark plots...")
    _generate_campaign_plots(results, figures_dir)
    
    # 4. Generate Reports
    logger.info("Generating Markdown and LaTeX reports...")
    _generate_markdown_report(results, significance_results, reports_dir)
    _generate_latex_report(results, significance_results, reports_dir)
    
    logger.info("Campaign Orchestration Complete. Outputs written to %s.", outputs_base_dir)


def _generate_campaign_plots(results: dict, figures_dir: Path):
    """Draw performance and trade-off plots for Campaign A and Campaign B."""
    # Split results into campaigns
    campaign_a = {}
    campaign_b_prod = {}
    campaign_b_exp = {}
    
    for exp_id, res in results.items():
        if res["status"] != "Completed":
            continue
        config = ExperimentConfig(Path(res["config_path"]))
        metrics = res["metrics"]
        
        data = {
            "name": exp_id,
            "iou": metrics.get("iou", 0.0),
            "f1": metrics.get("f1", 0.0),
            "train_time": metrics.get("train_time_sec", 0.0),
            "inf_time": metrics.get("inference_time_sec", 0.0),
            "ram": metrics.get("memory_usage_mb", 0.0)
        }
        
        if config.campaign_type == "preprocessing":
            campaign_a[exp_id] = data
        elif config.campaign_type == "feature_engineering":
            if config.track == "production":
                campaign_b_prod[exp_id] = data
            else:
                campaign_b_exp[exp_id] = data

    # 1. Campaign A: Preprocessing Leaderboard
    if campaign_a:
        names = list(campaign_a.keys())
        ious = [campaign_a[n]["iou"] for n in names]
        
        plt.figure(figsize=(8, 4))
        bars = plt.barh(names, ious, color=['#4F46E5', '#0EA5E9'])
        plt.xlim(0.0, 1.0)
        plt.xlabel("Jaccard IoU")
        plt.title("Campaign A: Preprocessing Leaderboard\n(Research Question: Does speckle filtering improve data quality?)")
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width:.4f}", 
                     va='center', ha='left', fontweight='bold')
                     
        plt.tight_layout()
        plt.savefig(figures_dir / "campaign_a_leaderboard.png", dpi=150)
        plt.close()

    # 2. Campaign B: Feature Engineering Leaderboard
    if campaign_b_prod or campaign_b_exp:
        names_p = list(campaign_b_prod.keys())
        ious_p = [campaign_b_prod[n]["iou"] for n in names_p]
        colors_p = ['#059669'] * len(names_p)
        
        names_e = list(campaign_b_exp.keys())
        ious_e = [campaign_b_exp[n]["iou"] for n in names_e]
        colors_e = ['#DC2626'] * len(names_e)
        
        all_names = names_p + names_e
        all_ious = ious_p + ious_e
        all_colors = colors_p + colors_e
        
        # Sort by IoU
        sorted_indices = np.argsort(all_ious)
        sorted_names = [all_names[i] for i in sorted_indices]
        sorted_ious = [all_ious[i] for i in sorted_indices]
        sorted_colors = [all_colors[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 5))
        bars = plt.barh(sorted_names, sorted_ious, color=sorted_colors)
        plt.xlim(0.0, 1.0)
        plt.xlabel("Jaccard IoU")
        plt.title("Campaign B: Feature Engineering Leaderboard\n(Green: Production [Physical Bands], Red: Experimental [NIR Reconstructions])")
        
        # Draw legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#059669', label='Production (Track A)'),
            Patch(facecolor='#DC2626', label='Experimental (Track B - Reconstructed NIR)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width:.4f}", 
                     va='center', ha='left', fontweight='bold')
                     
        plt.tight_layout()
        plt.savefig(figures_dir / "campaign_b_leaderboard.png", dpi=150)
        plt.close()

    # 3. Campaign B: Resource vs Accuracy Trade-off
    if campaign_b_prod or campaign_b_exp:
        plt.figure(figsize=(8, 6))
        
        for n, d in campaign_b_prod.items():
            plt.scatter(d["inf_time"], d["iou"], color='#059669', s=150, zorder=5, label='Track A (Production)')
            plt.text(d["inf_time"] + 0.01, d["iou"], n, fontsize=9, va='center')
            
        for n, d in campaign_b_exp.items():
            plt.scatter(d["inf_time"], d["iou"], color='#DC2626', s=150, zorder=5, label='Track B (Experimental)')
            plt.text(d["inf_time"] + 0.01, d["iou"], n, fontsize=9, va='center')
            
        plt.xlabel("Inference Time (seconds)")
        plt.ylabel("Jaccard IoU")
        plt.title("Accuracy vs. Latency Trade-off (Campaign B)")
        
        # Remove duplicate legends
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(figures_dir / "campaign_b_tradeoff.png", dpi=150)
        plt.close()


def _generate_markdown_report(results: dict, significance: dict, reports_dir: Path):
    """Write the scientific summary report in Markdown format."""
    path = reports_dir / "preprocessing_summary.md"
    
    # Identify best/fastest/trade-offs
    campaign_a = []
    campaign_b_prod = []
    campaign_b_exp = []
    
    for exp_id, res in results.items():
        if res["status"] != "Completed":
            continue
        config = ExperimentConfig(Path(res["config_path"]))
        m = res["metrics"]
        item = {
            "id": exp_id,
            "desc": config.description,
            "iou": m.get("iou", 0.0),
            "f1": m.get("f1", 0.0),
            "train": m.get("train_time_sec", 0.0),
            "inf": m.get("inference_time_sec", 0.0),
            "ram": m.get("memory_usage_mb", 0.0),
            "size": m.get("model_size_bytes", 0) / 1024.0, # KB
            "throughput": (3300 * 3300) / m.get("inference_time_sec", 1.0) if m.get("inference_time_sec", 0.0) > 0 else 0.0
        }
        if config.campaign_type == "preprocessing":
            campaign_a.append(item)
        elif config.campaign_type == "feature_engineering":
            if config.track == "production":
                campaign_b_prod.append(item)
            else:
                campaign_b_exp.append(item)
                
    # Sort
    campaign_a.sort(key=lambda x: x["iou"], reverse=True)
    campaign_b_prod.sort(key=lambda x: x["iou"], reverse=True)
    campaign_b_exp.sort(key=lambda x: x["iou"], reverse=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Preprocessing & Feature Engineering Benchmark Campaigns Report\n\n")
        f.write("This document summarizes the scientific benchmark findings for both Campaign A (Preprocessing Quality) and Campaign B (Feature Augmentation).\n\n")
        
        # Meta Header
        f.write("## Experiment Metadata\n")
        f.write("- **Data Source**: Sentinel-2 L2A (EO) and Sentinel-1 GRD (SAR)\n")
        f.write("- **Native Spectral Bands Used**: Red, Green, Blue, NDVI, NDBI, NDWI (Optical) + VV (SAR)\n")
        f.write("- **Reconstructed Spectral Bands**: NIR (strictly limited to Track B - Experimental)\n")
        f.write("- **Hardware Platform**: CPU-based execution environment\n\n")
        
        # ----------------------------------------------------
        # Campaign A
        # ----------------------------------------------------
        f.write("--- \n\n")
        f.write("## Campaign A — Preprocessing Benchmark\n")
        f.write("**Research Question**: *Does improving the quality of existing input measurements (without changing the model or expanding the feature space) improve Sentinel-1/2 change detection performance?*\n\n")
        
        f.write("| Experiment ID | Jaccard IoU | F1 Score | Training Time (s) | Inference Latency (s) | RAM (MB) | McNemar p-value | Significance |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for item in campaign_a:
            exp_id = item["id"]
            sig = significance.get(exp_id, {"p_value": 1.0, "significant": False})
            p_val = f"{sig['p_value']:.4e}" if exp_id != "preproc_baseline" else "N/A"
            sig_text = "Yes" if sig["significant"] else "No" if exp_id != "preproc_baseline" else "Baseline"
            f.write(f"| `{exp_id}` | {item['iou']:.5f} | {item['f1']:.5f} | {item['train']:.3f} | {item['inf']:.3f} | {item['ram']:.1f} | {p_val} | {sig_text} |\n")
            
        f.write("\n**Interpretation & Conclusions (Campaign A)**:\n")
        if len(campaign_a) > 1:
            best_a = campaign_a[0]
            base_a = next(x for x in campaign_a if x["id"] == "preproc_baseline")
            delta_iou = best_a["iou"] - base_a["iou"]
            f.write(f"- **Best Preprocessing Configuration**: `{best_a['id']}` (IoU: {best_a['iou']:.5f}, F1: {best_a['f1']:.5f}).\n")
            f.write(f"- **Quality Impact**: `{best_a['id']}` yields a **{delta_iou*100:+.3f}%** change in Jaccard IoU relative to raw baseline.\n")
            
            sig_best = significance.get(best_a["id"], {"significant": False})
            if sig_best["significant"]:
                f.write("- **Statistical Significance**: The difference is statistically significant (McNemar p-value < 0.05), indicating a true physical improvement in signal representation.\n")
            else:
                f.write("- **Statistical Significance**: The difference is NOT statistically significant, suggesting the preprocessing benefits are marginal on this dataset partition.\n")
        
        # ----------------------------------------------------
        # Campaign B
        # ----------------------------------------------------
        f.write("\n--- \n\n")
        f.write("## Campaign B — Feature Engineering Benchmark\n")
        f.write("**Research Question**: *Do additional derived spatial, spectral, or texture variables improve model performance beyond the current feature set?*\n\n")
        
        f.write("### Track A — Production Feature Engineering (Physical Bands Only)\n")
        f.write("These configurations use only native physical measurements on disk. No reconstructed bands are utilized. **Suitable for production deployment.**\n\n")
        
        f.write("| Experiment ID | Jaccard IoU | F1 Score | Training Time (s) | Inference Latency (s) | RAM (MB) | McNemar p-value | Significance |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for item in campaign_b_prod:
            exp_id = item["id"]
            sig = significance.get(exp_id, {"p_value": 1.0, "significant": False})
            p_val = f"{sig['p_value']:.4e}" if exp_id != "feat_prod_baseline" else "N/A"
            sig_text = "Yes" if sig["significant"] else "No" if exp_id != "feat_prod_baseline" else "Baseline"
            f.write(f"| `{exp_id}` | {item['iou']:.5f} | {item['f1']:.5f} | {item['train']:.3f} | {item['inf']:.3f} | {item['ram']:.1f} | {p_val} | {sig_text} |\n")
            
        f.write("\n### Track B — Experimental Feature Engineering (NIR Reconstruction Approximations)\n")
        f.write("> [!WARNING]\n")
        f.write("> **Experimental Approximations**: The configurations below rely on a mathematically reconstructed Near-Infrared (NIR) band. They are included strictly for exploratory research and must NOT be deployed in production or mixed with physical results.\n\n")
        
        f.write("| Experiment ID | Jaccard IoU | F1 Score | Training Time (s) | Inference Latency (s) | RAM (MB) | McNemar p-value | Significance |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for item in campaign_b_exp:
            exp_id = item["id"]
            sig = significance.get(exp_id, {"p_value": 1.0, "significant": False})
            p_val = f"{sig['p_value']:.4e}" if exp_id != "feat_prod_baseline" else "N/A"
            sig_text = "Yes" if sig["significant"] else "No" if exp_id != "feat_prod_baseline" else "Baseline"
            f.write(f"| `{exp_id}` | {item['iou']:.5f} | {item['f1']:.5f} | {item['train']:.3f} | {item['inf']:.3f} | {item['ram']:.1f} | {p_val} | {sig_text} |\n")
            
        f.write("\n**Interpretation & Conclusions (Campaign B)**:\n")
        if campaign_b_prod:
            best_p = campaign_b_prod[0]
            f.write(f"- **Best Production Feature (Track A)**: `{best_p['id']}` (IoU: {best_p['iou']:.5f}), providing physically grounded spatial texture representation.\n")
        if campaign_b_exp:
            best_e = campaign_b_exp[0]
            f.write(f"- **Best Experimental Feature (Track B)**: `{best_e['id']}` (IoU: {best_e['iou']:.5f}, F1: {best_e['f1']:.5f}), demonstrating the mathematical utility of soil-adjusted indices.\n")
            
        # ----------------------------------------------------
        # Dataset Limitations & Recommendations
        # ----------------------------------------------------
        f.write("\n--- \n\n")
        f.write("## Data Limitations & Recommendations\n\n")
        f.write("### Identified Data Limitations\n")
        f.write("The current dataset catalog stores precomputed spectral indices and 3-band RGB imagery rather than raw multi-band Sentinel-2 L2A images. Consequently, physical Near-Infrared (Band 8) is missing, forcing SAVI and MSAVI to depend on algebraic band reconstruction: ")
        f.write("$$\\text{NIR} = \\text{Red} \\cdot \\frac{1 + \\text{NDVI}}{1 - \\text{NDVI}}$$\n")
        f.write("This reconstruction is an approximation that can introduce propagation errors, especially in pixels where NDVI values approach 1.0.\n\n")
        
        f.write("### Project Recommendations\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Future Dataset Expansion (Band 8 / NIR Preservation)**\n")
        f.write("> Future catalog expansions must preserve all original Sentinel-2 spectral bands (especially Band 8 / NIR) rather than storing only precomputed indices on disk. This will let the platform calculate physical, sensor-grounded SAVI, MSAVI, EVI, and other texture features, moving them from Track B (Experimental) to Track A (Production).\n")


def _generate_latex_report(results: dict, significance: dict, reports_dir: Path):
    """Write publication-ready LaTeX tables of the benchmarks."""
    path = reports_dir / "preprocessing_summary.tex"
    
    # Separate campaigns
    campaign_a = []
    campaign_b_prod = []
    campaign_b_exp = []
    
    for exp_id, res in results.items():
        if res["status"] != "Completed":
            continue
        config = ExperimentConfig(Path(res["config_path"]))
        m = res["metrics"]
        item = {
            "id": exp_id.replace("_", "\\_"),
            "iou": m.get("iou", 0.0),
            "f1": m.get("f1", 0.0),
            "train": m.get("train_time_sec", 0.0),
            "inf": m.get("inference_time_sec", 0.0),
            "ram": m.get("memory_usage_mb", 0.0),
            "p_val": significance.get(exp_id, {"p_value": 1.0})["p_value"] if exp_id not in ("preproc_baseline", "feat_prod_baseline") else 1.0
        }
        if config.campaign_type == "preprocessing":
            campaign_a.append(item)
        elif config.campaign_type == "feature_engineering":
            if config.track == "production":
                campaign_b_prod.append(item)
            else:
                campaign_b_exp.append(item)
                
    campaign_a.sort(key=lambda x: x["iou"], reverse=True)
    campaign_b_prod.sort(key=lambda x: x["iou"], reverse=True)
    campaign_b_exp.sort(key=lambda x: x["iou"], reverse=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("% LaTeX tables for GeoAI Preprocessing & Feature Engineering Benchmark Campaigns\n\n")
        
        # Table A: Preprocessing
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Campaign A: Preprocessing Benchmark Results (Fixed Feature Space)}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Experiment ID & Jaccard IoU & F1-Score & Train Time (s) & Inf Latency (s) & McNemar $p$-value \\\\\n")
        f.write("\\hline\n")
        for item in campaign_a:
            p_str = f"{item['p_val']:.3e}" if "baseline" not in item["id"] else "N/A"
            f.write(f"{item['id']} & {item['iou']:.5f} & {item['f1']:.5f} & {item['train']:.3f} & {item['inf']:.3f} & {p_str} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:campaign_a_preproc}\n")
        f.write("\\end{table}\n\n")
        
        # Table B: Feature Engineering
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Campaign B: Feature Engineering Results (Production vs Experimental Approximations)}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Experiment ID & Jaccard IoU & F1-Score & Train Time (s) & Inf Latency (s) & Track Type \\\\\n")
        f.write("\\hline\n")
        f.write("\\multicolumn{6}{l}{\\textit{Track A: Production (Physical Bands Only)}} \\\\\n")
        for item in campaign_b_prod:
            f.write(f"{item['id']} & {item['iou']:.5f} & {item['f1']:.5f} & {item['train']:.3f} & {item['inf']:.3f} & Production \\\\\n")
        f.write("\\hline\n")
        f.write("\\multicolumn{6}{l}{\\textit{Track B: Experimental (NIR Reconstruction Approximations)}} \\\\\n")
        for item in campaign_b_exp:
            f.write(f"{item['id']} & {item['iou']:.5f} & {item['f1']:.5f} & {item['train']:.3f} & {item['inf']:.3f} & Experimental \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:campaign_b_features}\n")
        f.write("\\end{table}\n")


if __name__ == "__main__":
    main()
