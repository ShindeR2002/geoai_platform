import os
import sys
import argparse
import yaml
import json
import time
import datetime
import hashlib
import numpy as np
import pandas as pd
import torch
import joblib
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Ensure geoai package is in import path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from geoai.experiments.experiment_registry import get_experiment_model, list_registered_models, UnimplementedBaseModel
from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import split_dataset_unified, get_pixel_coords
from geoai.models.baselines.dl_wrapper import DLBaseModelWrapper
from geoai.evaluation.health_checker import run_preflight_checks, generate_health_report
from geoai.evaluation.consistency_validator import run_consistency_audit
from geoai.evaluation.profiling_manager import ProfilingManager
from geoai.evaluation.explainability_validator import run_explainability_audit, generate_explainability_report
from geoai.evaluation.failure_analysis import perform_failure_analysis, generate_failure_report
from scripts.run_reproducibility import get_classical_classifier

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="GeoAI Platform Scientific Benchmark Campaign Executor (Phases 1-3)")
    parser.add_argument("--config", type=str, default="configs/campaign.yaml", help="Path to YAML campaign configuration")
    parser.add_argument("--mode", type=str, choices=["quick", "full"], help="Override campaign execution mode")
    parser.add_argument("--models", type=str, nargs="+", help="Override campaign benchmark target model list")
    parser.add_argument("--force", action="store_true", help="Bypass validation checks failure and force execution")
    return parser.parse_args()

def calculate_iou(y_true, y_pred):
    intersection = ((y_true == 1) & (y_pred == 1)).sum()
    union = ((y_true == 1) | (y_pred == 1)).sum()
    return float(intersection / union if union > 0 else 0.0)

def main():
    args = parse_args()
    
    # Load Campaign Config
    config_path = root_dir / args.config
    if not config_path.exists():
        logger.error(f"Campaign config file '{config_path}' not found.")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Command overrides
    mode = args.mode or config.get("mode", "quick")
    force = args.force or config.get("force", False)
    dataset_id = config.get("dataset_id", "ps10_sentinel_v1")
    output_dir = Path(config.get("output_dir", "outputs/campaign"))
    
    # Resolve target models (discover dynamically from registry if 'all')
    raw_models = config.get("models", "all")
    if args.models:
        raw_models = args.models
        
    if raw_models == "all":
        discovered_models = list_registered_models()
    else:
        discovered_models = raw_models
        
    # Build clean targets list (excluding unimplemented models)
    models_to_run = []
    for model_id in discovered_models:
        try:
            wrapper = get_experiment_model(model_id)
            if isinstance(wrapper, UnimplementedBaseModel) or (hasattr(wrapper, "is_implemented") and not wrapper.is_implemented()):
                logger.info(f"Skipping unregistered/unimplemented model: {model_id}")
                continue
            models_to_run.append(model_id)
        except Exception as e:
            logger.warning(f"Failed to check capabilities for model '{model_id}': {e}. Skipping.")
            
    logger.info(f"Final Campaign Models List: {models_to_run}")
    
    # Configure speedups
    if mode == "quick":
        os.environ["DL_MAX_SAMPLES"] = "100"
    else:
        if "DL_MAX_SAMPLES" in os.environ:
            del os.environ["DL_MAX_SAMPLES"]
            
    # Resolve seeds
    seeds = config.get("seeds", [42, 123, 456, 789, 2025])
    
    # Create output directories
    runs_dir = output_dir / "runs"
    raw_results_dir = output_dir / "raw_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    raw_results_dir.mkdir(exist_ok=True)
    
    # Load and setup database
    db_path = output_dir / "experiment_db.json"
    if db_path.exists():
        with open(db_path, "r") as f:
            try:
                experiment_db = json.load(f)
            except Exception:
                experiment_db = {"entries": []}
    else:
        experiment_db = {"entries": []}
        
    # =========================================================================
    # PHASE 1: Dataset Validation
    # =========================================================================
    logger.info("--- PHASE 1: Dataset Validation ---")
    dataset = DatasetRegistry.load_dataset(dataset_id, configs_dir=str(root_dir / "configs"))
    X, y = dataset.X.copy(), dataset.y.copy()
    coords, spatial_shape, valid_mask, feature_cube = get_pixel_coords(dataset_id, configs_dir=str(root_dir / "configs"))
    
    # Run Health Checker
    health_results = run_preflight_checks(
        X, y, coords, spatial_shape, patch_size=15, split_policy="spatial", dataset_id=dataset_id, force=force
    )
    health_report_path = output_dir / "benchmark_health_report.md"
    generate_health_report(health_results, str(health_report_path))
    
    if health_results.get("preflight_gate") == "FAIL" and not force:
        logger.error("Dataset pre-flight health gate failed. Campaign aborted. Use --force to override.")
        sys.exit(1)
        
    # Save Dataset Fingerprint
    feature_hash = hashlib.sha256(X.tobytes()).hexdigest()
    label_hash = hashlib.sha256(y.tobytes()).hexdigest()
    
    fingerprint = {
        "dataset_id": dataset_id,
        "features_sha256": feature_hash,
        "labels_sha256": label_hash,
        "crs": "EPSG:32643",  # Enforce standardized projection mapping
        "raster_dimensions": [int(spatial_shape[0]), int(spatial_shape[1])],
        "spatial_resolution": 10.0,
        "feature_list": list(CANONICAL_FEATURE_NAMES) if 'CANONICAL_FEATURE_NAMES' in locals() else [f"f{i}" for i in range(X.shape[1])]
    }
    
    fingerprint_path = output_dir / "dataset_fingerprint.json"
    with open(fingerprint_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"Dataset fingerprint saved successfully to '{fingerprint_path.name}'.")
    
    # Save platform freeze release metadata
    freeze_meta_path = root_dir / "configs" / "platform_version.json"
    if freeze_meta_path.exists():
        shutil_dest = output_dir / "platform_version.json"
        import shutil
        shutil.copy(freeze_meta_path, shutil_dest)
        
    # Save campaign manifest
    import platform
    manifest = {
        "benchmark_version": "1.0",
        "platform_version": "1.0",
        "dataset_version": dataset_id,
        "mode": mode,
        "seeds": seeds,
        "execution_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu": platform.processor(),
        "packages": ["scikit-learn", "numpy", "pandas", "joblib"]
    }
    with open(output_dir / "campaign_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # =========================================================================
    # PHASE 2: Experiment Execution
    # =========================================================================
    logger.info("--- PHASE 2: Experiment Execution ---")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for model_id in models_to_run:
        # Load capability for channel splitting
        model_wrapper = get_experiment_model(model_id)
        expected_channels = model_wrapper.get_capabilities().expected_input_channels
        
        from geoai.utils.constants import CANONICAL_FEATURE_NAMES
        col_indices = [
            list(CANONICAL_FEATURE_NAMES).index(channel)
            for channel in expected_channels
            if channel in CANONICAL_FEATURE_NAMES
        ]
        X_sub = X[:, col_indices] if col_indices else X
        
        split_result = split_dataset_unified(
            X=X_sub, y=y, dataset_id=dataset_id, split_policy="spatial", patch_size=15, test_size=0.20, random_state=42
        )
        X_train, y_train = split_result.X_train, split_result.y_train
        X_test, y_test = split_result.X_test, split_result.y_test
        X_val, y_val = split_result.X_val, split_result.y_val
        
        if mode == "quick" and len(X_test) > 500:
            rng_test = np.random.default_rng(42)
            idx_test = rng_test.choice(len(X_test), size=500, replace=False)
            X_test_eval = X_test[idx_test]
            y_test_eval = y_test[idx_test]
            coords_eval = coords[:len(X_test)][idx_test]
        else:
            X_test_eval = X_test
            y_test_eval = y_test
            coords_eval = coords[:len(X_test)]
            
        for seed in seeds:
            experiment_id = f"{model_id}_{dataset_id}_seed{seed}"
            logger.info(f"Running Experiment ID: {experiment_id}")
            
            # Check Resume Logic
            existing_record = next((e for e in experiment_db["entries"] if e.get("experiment_id") == experiment_id), None)
            exp_dir = runs_dir / experiment_id
            status_file = exp_dir / "status.json"
            
            mode_mismatch = False
            snapshot_path = exp_dir / "config_snapshot.yaml"
            if snapshot_path.exists():
                try:
                    with open(snapshot_path, "r") as csf:
                        snap = yaml.safe_load(csf)
                    if snap.get("mode") != mode:
                        mode_mismatch = True
                        logger.info(f"Mode mismatch for {experiment_id} (found '{snap.get('mode')}', requested '{mode}'). Forcing rerun.")
                except Exception:
                    pass
            
            if not mode_mismatch and existing_record and existing_record.get("execution_status") == "SUCCESS" and status_file.exists():
                with open(status_file, "r") as sf:
                    s_data = json.load(sf)
                if s_data.get("status") == "SUCCESS" and (exp_dir / "predictions.npy").exists():
                    logger.info(f"Experiment {experiment_id} already completed successfully. Skipping.")
                    skipped_count += 1
                    continue
                    
            # Setup run directory
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration snapshot
            config_snapshot = {
                "experiment_id": experiment_id,
                "model_id": model_id,
                "seed": seed,
                "dataset_id": dataset_id,
                "mode": mode,
                "channels": expected_channels
            }
            with open(exp_dir / "config_snapshot.yaml", "w") as cf:
                yaml.dump(config_snapshot, cf)
                
            # Run trial
            t0 = time.time()
            pm = ProfilingManager()
            is_loaded = False
            
            checkpoint_name = f"{model_id}_seed_{seed}.pt" if isinstance(model_wrapper, DLBaseModelWrapper) else f"{model_id}_seed_{seed}.pkl"
            if mode == "quick":
                checkpoint_name = "quick_" + checkpoint_name
            checkpoint_path = exp_dir / checkpoint_name
            
            try:
                # Re-instantiate model wrapper to clear state
                model_wrapper = get_experiment_model(model_id)
                if hasattr(model_wrapper, "hyperparams"):
                    model_wrapper.hyperparams["random_state"] = seed
                    if mode == "quick":
                        model_wrapper.hyperparams["epochs"] = 1
                model_wrapper.dataset_id = dataset_id
                
                # Checkpoint Reuse
                if checkpoint_path.exists():
                    logger.info(f"  Reusing existing checkpoint: {checkpoint_path.name}")
                    try:
                        if isinstance(model_wrapper, DLBaseModelWrapper):
                            model_wrapper.load(checkpoint_path)
                        else:
                            model_wrapper = joblib.load(checkpoint_path)
                        is_loaded = True
                    except Exception as e:
                        logger.warning(f"  Failed to load checkpoint: {e}. Retraining...")
                        
                pm.start_training()
                if not is_loaded:
                    logger.info(f"  Training model...")
                    if isinstance(model_wrapper, DLBaseModelWrapper):
                        model_wrapper.fit(X_train, y_train, X_val, y_val)
                        model_wrapper.save(checkpoint_path)
                    else:
                        # Classical dynamic classifier selection
                        model_wrapper = get_classical_classifier(model_id, seed, mode)
                        if mode == "quick" and len(X_train) > 1000:
                            rng_tr = np.random.default_rng(seed)
                            idx_tr = rng_tr.choice(len(X_train), size=1000, replace=False)
                            model_wrapper.fit(X_train[idx_tr], y_train[idx_tr])
                        else:
                            model_wrapper.fit(X_train, y_train)
                        joblib.dump(model_wrapper, checkpoint_path)
                pm.end_training(len(X_train))
                
                # Inference
                pm.start_inference()
                y_pred = model_wrapper.predict(X_test_eval)
                y_prob = None
                if hasattr(model_wrapper, "predict_proba"):
                    try:
                        y_prob = model_wrapper.predict_proba(X_test_eval)
                    except Exception:
                        pass
                pm.end_inference(len(X_test_eval))
                
                # Generate Profile
                profile = pm.generate_profile(
                    model_wrapper.model if isinstance(model_wrapper, DLBaseModelWrapper) else model_wrapper,
                    checkpoint_path=str(checkpoint_path)
                )
                
                # Compute Metrics
                acc = accuracy_score(y_test_eval, y_pred)
                prec = precision_score(y_test_eval, y_pred, zero_division=0)
                rec = recall_score(y_test_eval, y_pred, zero_division=0)
                f1 = f1_score(y_test_eval, y_pred, zero_division=0)
                iou = calculate_iou(y_test_eval, y_pred)
                
                # Save Numpy arrays
                np.save(exp_dir / "predictions.npy", y_pred)
                np.save(exp_dir / "ground_truth.npy", y_test_eval)
                np.save(exp_dir / "test_coordinates.npy", coords_eval)
                if y_prob is not None:
                    np.save(exp_dir / "probabilities.npy", y_prob)
                    
                # Save Confusion Matrix
                cm = confusion_matrix(y_test_eval, y_pred)
                np.save(exp_dir / "confusion_matrix.npy", cm)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(x=j, y=i, s=str(cm[i, j]), va='center', ha='center', fontsize=12)
                plt.title("Confusion Matrix", fontsize=10)
                plt.xlabel("Predicted", fontsize=8)
                plt.ylabel("Actual", fontsize=8)
                plt.savefig(exp_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
                plt.close()
                
                # ROC / PR curves if probabilities exist
                if y_prob is not None:
                    # Select probability of change (class 1)
                    prob_change = y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob
                    
                    # ROC Curve
                    fpr, tpr, thresholds_roc = roc_curve(y_test_eval, prob_change)
                    roc_auc = auc(fpr, tpr)
                    df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds_roc})
                    df_roc.to_csv(exp_dir / "roc_curve.csv", index=False)
                    
                    plt.figure(figsize=(4, 4))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
                    plt.xlabel('False Positive Rate', fontsize=8)
                    plt.ylabel('True Positive Rate', fontsize=8)
                    plt.title('ROC Curve', fontsize=10)
                    plt.legend(loc="lower right", fontsize=8)
                    plt.savefig(exp_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
                    plt.close()
                    
                    # Precision-Recall Curve
                    precision, recall, thresholds_pr = precision_recall_curve(y_test_eval, prob_change)
                    ap = average_precision_score(y_test_eval, prob_change)
                    df_pr = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "thresholds": thresholds_pr})
                    df_pr.to_csv(exp_dir / "pr_curve.csv", index=False)
                    
                    plt.figure(figsize=(4, 4))
                    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.2f}')
                    plt.xlabel('Recall', fontsize=8)
                    plt.ylabel('Precision', fontsize=8)
                    plt.title('Precision-Recall Curve', fontsize=10)
                    plt.legend(loc="lower left", fontsize=8)
                    plt.savefig(exp_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
                    plt.close()
                else:
                    roc_auc = 0.0
                    ap = 0.0
                    
                # Save individual metrics
                metrics = {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "iou": float(iou),
                    "roc_auc": float(roc_auc),
                    "average_precision": float(ap)
                }
                with open(exp_dir / "metrics.json", "w") as mf:
                    json.dump(metrics, mf, indent=2)
                    
                # Record to db
                duration = time.time() - t0
                record = {
                    "experiment_id": experiment_id,
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "benchmark_version": "1.0",
                    "platform_version": "1.0",
                    "seed": seed,
                    "execution_status": "SUCCESS",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_sec": duration,
                    "metrics": metrics,
                    "profile": profile,
                    "paths": {
                        "predictions": str(exp_dir / "predictions.npy"),
                        "probabilities": str(exp_dir / "probabilities.npy") if y_prob is not None else None,
                        "ground_truth": str(exp_dir / "ground_truth.npy"),
                        "checkpoint": str(checkpoint_path),
                        "artifact_dir": str(exp_dir)
                    }
                }
                
                # Write to database (overwriting if exists, otherwise appending)
                experiment_db["entries"] = [e for e in experiment_db["entries"] if e.get("experiment_id") != experiment_id]
                experiment_db["entries"].append(record)
                
                with open(status_file, "w") as sf:
                    json.dump({"status": "SUCCESS"}, sf)
                with open(raw_results_dir / f"{experiment_id}.json", "w") as rf:
                    json.dump(record, rf, indent=2)
                    
                success_count += 1
                logger.info(f"Experiment {experiment_id} complete. F1: {f1:.4f} | IoU: {iou:.4f}")
                
            except Exception as e:
                logger.error(f"Experiment {experiment_id} failed with error: {e}", exc_info=True)
                failed_count += 1
                
                # Record failure status
                record = {
                    "experiment_id": experiment_id,
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "benchmark_version": "1.0",
                    "platform_version": "1.0",
                    "seed": seed,
                    "execution_status": "FAILED",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_sec": time.time() - t0,
                    "error": str(e)
                }
                experiment_db["entries"] = [e for e in experiment_db["entries"] if e.get("experiment_id") != experiment_id]
                experiment_db["entries"].append(record)
                
                with open(status_file, "w") as sf:
                    json.dump({"status": "FAILED", "error": str(e)}, sf)
                with open(raw_results_dir / f"{experiment_id}.json", "w") as rf:
                    json.dump(record, rf, indent=2)
                    
            # Save updated DB at each step for robustness
            with open(db_path, "w") as f:
                json.dump(experiment_db, f, indent=2)
                
    # =========================================================================
    # PHASE 3: Artifact Verification
    # =========================================================================
    logger.info("--- PHASE 3: Artifact Verification ---")
    verification_passed = True
    
    with open(db_path, "r") as f:
        db_audit = json.load(f)
        
    for entry in db_audit["entries"]:
        if entry.get("execution_status") != "SUCCESS":
            continue
        exp_id = entry.get("experiment_id")
        exp_dir = runs_dir / exp_id
        
        # Verify required arrays
        preds_path = exp_dir / "predictions.npy"
        gt_path = exp_dir / "ground_truth.npy"
        coords_path = exp_dir / "test_coordinates.npy"
        
        missing = []
        if not preds_path.exists(): missing.append("predictions.npy")
        if not gt_path.exists(): missing.append("ground_truth.npy")
        if not coords_path.exists(): missing.append("test_coordinates.npy")
        
        if missing:
            logger.warning(f"Experiment {exp_id} is missing required artifacts: {missing}. Marking as FAILED.")
            entry["execution_status"] = "FAILED"
            entry["error"] = f"Missing artifacts: {missing}"
            verification_passed = False
        else:
            # Verify arrays can be loaded
            try:
                np.load(preds_path)
                np.load(gt_path)
                np.load(coords_path)
            except Exception as e:
                logger.warning(f"Artifact corruption detected in {exp_id}: {e}. Marking as FAILED.")
                entry["execution_status"] = "FAILED"
                entry["error"] = f"Artifact corruption: {e}"
                verification_passed = False
                
    # Save audited database
    with open(db_path, "w") as f:
        json.dump(db_audit, f, indent=2)
        
    logger.info("Campaign execution complete.")
    logger.info(f"Summary: Success: {success_count} | Failed: {failed_count} | Skipped: {skipped_count} | Verification Status: {'PASS' if verification_passed else 'FAIL'}")

if __name__ == "__main__":
    main()
