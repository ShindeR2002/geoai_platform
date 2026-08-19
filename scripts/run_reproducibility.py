import os
import sys
import argparse
import numpy as np
import pandas as pd
import hashlib
import time
import datetime
import joblib
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Ensure geoai package is in import path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from geoai.experiments.experiment_registry import get_experiment_model
from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import split_dataset_unified, get_pixel_coords
from geoai.core.config import load_platform_config
from geoai.models.baselines.dl_wrapper import DLBaseModelWrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 2025]

def get_classical_classifier(model_id: str, seed: int, mode: str):
    n_est = 10 if mode == "quick" else 100
    if model_id in ("rf_baseline_v1", "rf_enhanced_v1"):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=n_est, random_state=seed, n_jobs=-1, class_weight="balanced")
    elif model_id == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=n_est, random_state=seed, n_jobs=-1)
    elif model_id == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=n_est, learning_rate=0.1, max_depth=6, random_state=seed, n_jobs=-1, eval_metric="logloss")
    elif model_id == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=n_est, learning_rate=0.1, max_depth=-1, random_state=seed, n_jobs=-1, verbose=-1)
    elif model_id == "catboost":
        from catboost import CatBoostClassifier
        iterations = 10 if mode == "quick" else 100
        return CatBoostClassifier(iterations=iterations, learning_rate=0.1, depth=6, random_state=seed, thread_count=-1, verbose=0)
    else:
        raise ValueError(f"Unknown classical model ID: {model_id}")


def calculate_iou(y_true, y_pred):
    intersection = ((y_true == 1) & (y_pred == 1)).sum()
    union = ((y_true == 1) | (y_pred == 1)).sum()
    return intersection / union if union > 0 else 0.0

def run_reproducibility_benchmark(mode: str = "quick", model_filter: str = None, force: bool = False):
    logger.info(f"Starting Reproducibility Runner in Mode: {mode.upper()}")
    
    # 1. Load Dataset
    dataset_id = "ps10_sentinel_v1"
    dataset = DatasetRegistry.load_dataset(dataset_id, configs_dir=str(root_dir / "configs"))
    X, y = dataset.X.copy(), dataset.y.copy()
    
    # Pre-flight health checker gate
    from geoai.evaluation.health_checker import run_preflight_checks, generate_health_report
    coords, spatial_shape, valid_mask, feature_cube = get_pixel_coords(dataset_id, configs_dir=str(root_dir / "configs"))
    
    os.makedirs(root_dir / "outputs" / "reproducibility", exist_ok=True)
    os.makedirs(root_dir / "outputs" / "reproducibility" / "checkpoints", exist_ok=True)
    
    logger.info("Executing mandatory pre-flight health checker gate...")
    health_results = run_preflight_checks(
        X, y, coords, spatial_shape, patch_size=15, split_policy="spatial", dataset_id=dataset_id, force=force
    )
    generate_health_report(health_results, str(root_dir / "outputs" / "reproducibility" / "benchmark_health_report.md"))
    
    # Configure speedups
    if mode == "quick":
        os.environ["DL_MAX_SAMPLES"] = "100"
    else:
        if "DL_MAX_SAMPLES" in os.environ:
            del os.environ["DL_MAX_SAMPLES"]
            
    from geoai.experiments.experiment_registry import list_registered_models, UnimplementedBaseModel
    models_to_run = list_registered_models()
    if model_filter:
        models_to_run = [m for m in models_to_run if m == model_filter]
        
    results_list = []
    
    for model_id in models_to_run:
        model_wrapper = get_experiment_model(model_id)
        if isinstance(model_wrapper, UnimplementedBaseModel) or (hasattr(model_wrapper, "is_implemented") and not model_wrapper.is_implemented()):
            logger.warning(f"Skipping unimplemented model: {model_id}")
            continue
            
        logger.info(f"Evaluating Model: {model_id}")
        expected_channels = model_wrapper.get_capabilities().expected_input_channels
        
        from geoai.utils.constants import CANONICAL_FEATURE_NAMES
        col_indices = [
            list(CANONICAL_FEATURE_NAMES).index(channel)
            for channel in expected_channels
            if channel in CANONICAL_FEATURE_NAMES
        ]
        X_sub = X[:, col_indices] if col_indices else X
        
        # Centralized splitting
        split_result = split_dataset_unified(
            X=X_sub, y=y, dataset_id=dataset_id, split_policy="spatial", patch_size=15, test_size=0.20, random_state=42
        )
        
        X_train, y_train = split_result.X_train, split_result.y_train
        X_test, y_test = split_result.X_test, split_result.y_test
        X_val, y_val = split_result.X_val, split_result.y_val
        
        # Subsample test set in quick mode to speed up prediction loops
        if mode == "quick" and len(X_test) > 500:
            rng_test = np.random.default_rng(42)
            idx_test = rng_test.choice(len(X_test), size=500, replace=False)
            X_test_eval = X_test[idx_test]
            y_test_eval = y_test[idx_test]
        else:
            X_test_eval = X_test
            y_test_eval = y_test
            
        for seed in SEEDS:
            logger.info(f"  Running Seed: {seed}")
            
            checkpoint_name = f"{model_id}_seed_{seed}.pt" if isinstance(model_wrapper, DLBaseModelWrapper) else f"{model_id}_seed_{seed}.pkl"
            if mode == "quick":
                checkpoint_name = "quick_" + checkpoint_name
            checkpoint_path = root_dir / "outputs" / "reproducibility" / "checkpoints" / checkpoint_name
            
            # Reset/reinitialize wrapper
            model_wrapper = get_experiment_model(model_id)
            if hasattr(model_wrapper, "hyperparams"):
                model_wrapper.hyperparams["random_state"] = seed
                if mode == "quick":
                    model_wrapper.hyperparams["epochs"] = 1
            model_wrapper.dataset_id = dataset_id
            
            is_loaded = False
            t0 = time.time()
            
            # Checkpoint Reuse logic
            if checkpoint_path.exists():
                logger.info(f"    Reusing existing checkpoint: {checkpoint_path.name}")
                try:
                    if isinstance(model_wrapper, DLBaseModelWrapper):
                        model_wrapper.load(checkpoint_path)
                    else:
                        model_wrapper = joblib.load(checkpoint_path)
                    is_loaded = True
                except Exception as e:
                    logger.warning(f"    Failed to load checkpoint: {e}. Retraining...")
                    
            if not is_loaded:
                logger.info(f"    Training model (seed={seed})...")
                if isinstance(model_wrapper, DLBaseModelWrapper):
                    model_wrapper.fit(X_train, y_train, X_val, y_val)
                    model_wrapper.save(checkpoint_path)
                else:
                    # Classical model selection
                    model_wrapper = get_classical_classifier(model_id, seed, mode)
                    if mode == "quick" and len(X_train) > 1000:
                        rng_tr = np.random.default_rng(seed)
                        idx_tr = rng_tr.choice(len(X_train), size=1000, replace=False)
                        model_wrapper.fit(X_train[idx_tr], y_train[idx_tr])
                    else:
                        model_wrapper.fit(X_train, y_train)
                    joblib.dump(model_wrapper, checkpoint_path)
                    
            train_time = time.time() - t0
            
            # Run prediction
            t0_inf = time.time()
            y_pred = model_wrapper.predict(X_test_eval)
            inf_time = time.time() - t0_inf
            
            # Compute metrics
            acc = accuracy_score(y_test_eval, y_pred)
            prec = precision_score(y_test_eval, y_pred, zero_division=0)
            rec = recall_score(y_test_eval, y_pred, zero_division=0)
            f1 = f1_score(y_test_eval, y_pred, zero_division=0)
            iou = calculate_iou(y_test_eval, y_pred)
            
            results_list.append({
                "model_id": model_id,
                "seed": seed,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "iou": iou,
                "train_time_sec": train_time,
                "inference_time_sec": inf_time
            })
            
    # Save raw seed results
    df = pd.DataFrame(results_list)
    df.to_csv(root_dir / "outputs" / "reproducibility" / "reproducibility_results.csv", index=False)
    
    # Aggregate stats
    agg_rows = []
    for model_id, group in df.groupby("model_id"):
        for metric in ["accuracy", "precision", "recall", "f1", "iou"]:
            vals = group[metric].values
            mean = np.mean(vals)
            std = np.std(vals)
            var = np.var(vals)
            cov = std / mean if mean > 0 else 0.0
            sem = std / np.sqrt(len(vals))
            ci_half = 1.96 * sem
            
            agg_rows.append({
                "model_id": model_id,
                "metric": metric,
                "mean": round(float(mean), 6),
                "std": round(float(std), 6),
                "variance": round(float(var), 6),
                "cov": round(float(cov), 6),
                "ci_lower": round(float(mean - ci_half), 6),
                "ci_upper": round(float(mean + ci_half), 6)
            })
            
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(root_dir / "outputs" / "reproducibility" / "reproducibility_aggregates.csv", index=False)
    
    # Generate Markdown Report
    generate_reproducibility_report(df_agg, df, str(root_dir / "outputs" / "reproducibility" / "reproducibility_report.md"), mode)
    logger.info("Reproducibility Benchmark successfully executed.")

def generate_reproducibility_report(df_agg: pd.DataFrame, df_raw: pd.DataFrame, filepath: str, mode: str):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, "w") as f:
        f.write("# GeoAI Platform — Multi-Seed Reproducibility Report\n\n")
        f.write(f"- **Execution Timestamp**: {timestamp}\n")
        f.write(f"- **Execution Mode**: {mode.upper()}\n")
        f.write(f"- **Evaluated Seeds**: {', '.join(map(str, SEEDS))}\n\n")
        
        f.write("## 1. Statistical Aggregates Summary\n\n")
        f.write("| Model | Metric | Mean | Std Dev | Variance | Coeff of Variation | 95% Confidence Interval |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for idx, row in df_agg.iterrows():
            f.write(f"| {row['model_id']} | {row['metric'].upper()} | {row['mean']:.4f} | {row['std']:.4f} | {row['variance']:.6f} | {row['cov']:.4f} | [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] |\n")
            
        f.write("\n## 2. Raw Seed Iteration Logs\n\n")
        f.write("| Model | Seed | Accuracy | Precision | Recall | F1 | IoU | Train Time (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for idx, row in df_raw.iterrows():
            f.write(f"| {row['model_id']} | {row['seed']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['iou']:.4f} | {row['train_time_sec']:.2f} |\n")
            
        f.write("\n## Scientific Reproducibility Log\n\n")
        f.write("> [!NOTE]\n")
        f.write("> Coefficient of Variation (CoV) values <= 0.05 represent high configuration stability. Deep learning architectures can exhibits minor variance on CPU backends due to floating-point accumulators order.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed reproducibility evaluation runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"], help="Subsampled or full dataset evaluation")
    parser.add_argument("--model", type=str, default=None, help="Filter for a specific model ID")
    parser.add_argument("--force", action="store_true", help="Bypass critical pre-flight gates")
    args = parser.parse_args()
    
    run_reproducibility_benchmark(mode=args.mode, model_filter=args.model, force=args.force)
