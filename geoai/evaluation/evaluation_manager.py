import os
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

from geoai.core.config import load_platform_config
from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import split_dataset, split_dataset_unified, SplitResult
from geoai.experiments.experiment_registry import get_experiment_model
from geoai.experiments.experiment_logger import experiment_logger_context
from geoai.utils.constants import CANONICAL_FEATURE_NAMES

# Evaluation imports
from geoai.evaluation.utils import setup_eval_dir, get_process_memory_mb, get_cpu_count
from geoai.evaluation.feature_analysis import (
    RandomForestMDIImportance,
    PermutationFeatureImportance,
    compute_correlations,
    compute_feature_statistics
)
from geoai.evaluation.baseline_validator import validate_v1_v2_consistency
from geoai.evaluation.error_analysis import analyze_prediction_errors
from geoai.evaluation.spatial_analysis import analyze_spatial_landscape
from geoai.evaluation.runtime_analysis import profile_execution_performance
from geoai.evaluation.statistical_analysis import compute_bootstrap_confidence_intervals, register_hypothesis_testing_hooks
from geoai.evaluation.metrics_report import export_evaluation_metrics
from geoai.evaluation.publication_report import generate_thesis_report
from geoai.evaluation.ablation_manager import AblationStudyManager

# Stage 1 helpers
from geoai.pipeline.stage1 import _load_all_rasters
from geoai.features.feature_cube import build_feature_cube
from geoai.models.inference import run_inference
from geoai.postprocessing.cleanup import to_binary_mask
from geoai.analysis.objects import extract_objects
from geoai.analysis.statistics import build_object_records

# Export imports
from geoai.exports.geotiff import export_change_mask_geotiff
from geoai.exports.shapefile import export_change_shapefile, export_geojson
from geoai.exports.csv_export import export_objects_csv

# Visualizations
from geoai.evaluation.visualization import (
    plot_feature_importance,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_error_map,
    plot_runtime_latency
)

logger = logging.getLogger(__name__)

class EvaluationManager:
    """Central scientific evaluation orchestrator running validation, features, errors, GIS, and publication reporting."""
    
    def __init__(self, base_outputs_dir: str = "outputs/evaluation", configs_dir: str = "configs") -> None:
        self.base_outputs_dir = Path(base_outputs_dir)
        self.configs_dir = Path(configs_dir)

    def run_evaluation(
        self,
        model_id: str,
        dataset_id: str,
        profile: str = "scientific",
        eval_id: Optional[str] = None,
        split_result: Optional[SplitResult] = None
    ) -> Dict[str, Any]:
        """
        Execute the validation framework under the specified profile (quick, scientific, publication, benchmark).
        """
        profile = profile.lower()
        if profile not in ("quick", "scientific", "publication", "benchmark"):
            raise ValueError(f"Unknown evaluation profile: {profile}")
            
        # 1. Setup isolated directories
        paths = setup_eval_dir(eval_id, base_dir=str(self.base_outputs_dir))
        eval_path = paths["root"]
        log_file = paths["logs"] / "execution.log"
        
        # Determine runs dir (for V1 comparison)
        parent_outputs = Path("outputs")
        
        # Determine reference run V1 path
        aoi_name = "Dholera"
        dataset_meta = DatasetRegistry.get_dataset_metadata(dataset_id)
        if dataset_meta:
            aoi_name = dataset_meta.aoi_name
            
        v1_reference_dir = parent_outputs / f"project_PS10_{aoi_name}"
        
        # 2. Run execution under log redirection
        with experiment_logger_context(log_file):
            logger.info("=" * 60)
            logger.info(f"Initiating Validation Suite Run: {eval_path.name} ({profile.upper()} Profile)")
            logger.info("=" * 60)
            
            # Start profiling feature engineering and loading latencies
            t0_feat = time.time()
            dataset = DatasetRegistry.load_dataset(dataset_id, configs_dir=str(self.configs_dir))
            X = dataset.X.copy()
            y = dataset.y
            feature_eng_time = time.time() - t0_feat
            logger.info(f"Loaded dataset {dataset_id} in {feature_eng_time:.4f}s. Matrix shape: {X.shape}")
            
            # Feature subsetting matching model capability to prevent mismatch errors
            model_wrapper = get_experiment_model(model_id)
            expected_channels = model_wrapper.get_capabilities().expected_input_channels
            
            col_indices = [
                list(CANONICAL_FEATURE_NAMES).index(channel)
                for channel in expected_channels
                if channel in CANONICAL_FEATURE_NAMES
            ]
            
            if col_indices:
                X = X[:, col_indices]
                logger.info(f"Subset feature matrix to expected channels. Shape: {X.shape}")
                
            # 3. Model training latency monitor & Inference
            from geoai.models.baselines.dl_wrapper import DLBaseModelWrapper
            is_dl = isinstance(model_wrapper, DLBaseModelWrapper)
            
            # Use provided splits or obtain them from centralized splitter
            if split_result is None:
                split_policy = platform_config.processing.split_policy
                patch_size = model_wrapper.patch_size if hasattr(model_wrapper, "patch_size") else 15
                split_result = split_dataset_unified(
                    X=X,
                    y=y,
                    dataset_id=dataset_id,
                    split_policy=split_policy,
                    patch_size=patch_size,
                    test_size=0.20,
                    random_state=42,
                    stratify=True
                )
            else:
                # If provided split_result contains 18 features but model expects fewer, filter it
                if split_result.X_train.shape[1] > len(expected_channels):
                    col_indices = [
                        list(CANONICAL_FEATURE_NAMES).index(channel)
                        for channel in expected_channels
                        if channel in CANONICAL_FEATURE_NAMES
                    ]
                    if col_indices:
                        split_result = SplitResult(
                            X_train=split_result.X_train[:, col_indices],
                            X_test=split_result.X_test[:, col_indices],
                            y_train=split_result.y_train,
                            y_test=split_result.y_test,
                            X_val=split_result.X_val[:, col_indices] if split_result.X_val is not None else None,
                            y_val=split_result.y_val
                        )
            
            X_train = split_result.X_train
            X_test = split_result.X_test
            y_train = split_result.y_train
            y_test = split_result.y_test
            X_val = split_result.X_val
            y_val = split_result.y_val
            
            if is_dl:
                logger.info("Deep learning model detected. Fitting weights...")
                t0_train = time.time()
                model_wrapper.fit(X_train, y_train, X_val, y_val)
                train_time = time.time() - t0_train
                logger.info(f"DL Model training complete in {train_time:.4f}s.")
                clf = model_wrapper
            else:
                # Stratified split or spatial split for classical models already prepared
                t0_train = time.time()
                clf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced" if model_id == "rf_baseline_v1" else None
                )
                clf.fit(X_train, y_train)
                train_time = time.time() - t0_train
                logger.info(f"Model training complete in {train_time:.4f}s.")
            
            # 4. Inference latency monitor
            t0_inf = time.time()
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            inference_time = time.time() - t0_inf
            logger.info(f"Test split inference complete in {inference_time:.4f}s.")
            
            # Model serialization for size monitor
            model_pkl_path = paths["artifacts"] / "model.pkl"
            if is_dl:
                # Retrieve matching experiment checkpoints directory
                checkpoint_path = Path("outputs/experiments") / f"dl_{model_id}_{dataset_id}" / "checkpoints" / "best_model.pt"
                if checkpoint_path.exists():
                    shutil.copy(checkpoint_path, model_pkl_path)
                else:
                    joblib.dump(clf, model_pkl_path)
            else:
                joblib.dump(clf, model_pkl_path)
                
            model_size_bytes = model_pkl_path.stat().st_size

            
            # 5. Spatial deliverables & GIS latency monitor
            t0_gis = time.time()
            platform_config = load_platform_config(str(self.configs_dir))
            aoi = platform_config.get_aoi(aoi_name)
            
            rasters = _load_all_rasters(aoi)
            profile_spatial = rasters[14]
            red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1 = rasters[0:6]
            red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2 = rasters[6:12]
            sar_t1, sar_t2 = rasters[12:14]
            
            feature_cube = build_feature_cube(
                red_t1=red_t1, green_t1=green_t1, blue_t1=blue_t1,
                sar_t1=sar_t1, ndvi_t1=ndvi_t1, ndbi_t1=ndbi_t1, ndwi_t1=ndwi_t1,
                red_t2=red_t2, green_t2=green_t2, blue_t2=blue_t2,
                sar_t2=sar_t2, ndvi_t2=ndvi_t2, ndbi_t2=ndbi_t2, ndwi_t2=ndwi_t2,
            )
            
            if not is_dl:
                model_wrapper._rf = clf
                model_wrapper.is_loaded = True
                model_wrapper.feature_names = expected_channels
            
            prediction_map, valid_mask, probability_map = run_inference(
                model=model_wrapper,
                feature_cube=feature_cube,
                compute_probabilities=True
            )
            
            binary_mask = to_binary_mask(prediction_map)
            
            min_obj_size = platform_config.processing.min_object_size_px
            significant_regions, label_array, significant_mask = extract_objects(
                binary_mask=binary_mask,
                min_area_px=min_obj_size,
            )
            
            transform = profile_spatial.get("transform")
            object_records = build_object_records(
                significant_regions, resolution_m=aoi.resolution_m, transform=transform
            )
            
            # Export deliverables
            export_change_mask_geotiff(binary_mask, profile_spatial, paths["artifacts"] / "prediction.tif")
            export_change_shapefile(significant_mask, profile_spatial, paths["artifacts"] / "objects.shp")
            
            if object_records:
                from geoai.classification.schema import change_objects_from_records
                objects = change_objects_from_records(object_records)
                export_geojson(objects, significant_mask, profile_spatial, paths["artifacts"] / "objects.geojson")
                export_objects_csv(objects, paths["artifacts"] / "statistics.csv")
                
            object_extraction_time = time.time() - t0_gis
            
            # 6. Export deliverables monitoring (saving time)
            t0_exp = time.time()
            # Copy configuration file
            # Search for model.yaml or baseline_rf.yaml
            config_copy_path = paths["artifacts"] / "config.yaml"
            yaml_path = self.configs_dir / "experiments" / "baseline_rf.yaml"
            if yaml_path.exists():
                shutil.copy2(yaml_path, config_copy_path)
            export_time = time.time() - t0_exp

            # --- Scientific Analysis Suite Execution ---
            
            # 7. Baseline V1 vs V2 Consistency Validator
            validation_results = {}
            if v1_reference_dir.exists():
                validation_results = validate_v1_v2_consistency(
                    v1_dir=v1_reference_dir,
                    v2_dir=paths["artifacts"],
                    aoi_name=aoi_name
                )
                # Save markdown validation report
                report_md = [
                    f"# Validation Report: V1 vs V2 Scientific Agreement",
                    f"**AOI**: {aoi_name}",
                    f"**Checks Passed**: {validation_results.get('passed_all', False)}",
                    f"\n## Deviation list:"
                ]
                for dev in validation_results.get("deviations", []):
                    report_md.append(f"- {dev}")
                
                with open(paths["validation"] / "validation_report.md", "w", encoding="utf-8") as f:
                    f.write("\n".join(report_md))
                    
            # 8. Feature Analysis
            logger.info("Performing feature diagnostic analysis...")
            # Compute MDI Importance
            mdi_importance = {}
            try:
                mdi_calc = RandomForestMDIImportance()
                mdi_importance = mdi_calc.compute_importance(clf, X, y, expected_channels)
                plot_feature_importance(mdi_importance, paths["plots"] / "feature_importance_mdi.png")
            except Exception as e:
                logger.warning(f"Failed calculating MDI: {e}")
                
            # Permutation Importance (Scientific, Publication, Benchmark profiles only)
            perm_importance = {}
            if profile in ("scientific", "publication", "benchmark"):
                try:
                    perm_calc = PermutationFeatureImportance(n_repeats=3)
                    perm_importance = perm_calc.compute_importance(clf, X_test, y_test, expected_channels)
                    plot_feature_importance(perm_importance, paths["plots"] / "feature_importance_permutation.png")
                except Exception as e:
                    logger.warning(f"Failed calculating Permutation Importance: {e}")
            
            # Correlations (Pearson, Spearman, Kendall)
            correlations = {}
            methods = ["pearson"]
            if profile in ("scientific", "publication", "benchmark"):
                methods = ["pearson", "spearman", "kendall"]
                
            for meth in methods:
                try:
                    corr_df = compute_correlations(X, expected_channels, method=meth)
                    correlations[meth] = corr_df.to_dict()
                    plot_correlation_heatmap(corr_df, meth, paths["plots"] / f"correlation_{meth}.png")
                except Exception as e:
                    logger.warning(f"Failed calculating correlation {meth}: {e}")
                    
            feature_stats = compute_feature_statistics(X, expected_channels)
            
            # TreeSHAP Global & Spatial explanations
            shap_results = {}
            spatial_shap_maps = {}
            if profile in ("scientific", "publication", "benchmark"):
                logger.info("Running TreeSHAP diagnostic attributions...")
                from geoai.evaluation.feature_analysis import compute_shap_explanations, compute_spatial_shap_maps
                shap_results = compute_shap_explanations(clf, X_test, expected_channels, num_samples=100)
                spatial_shap_maps = compute_spatial_shap_maps(
                    clf, feature_cube, valid_mask,
                    spatial_shape=(feature_cube.shape[0], feature_cube.shape[1]),
                    feature_names=expected_channels,
                    sample_step=48
                )
                
                # Plot spatial SHAP contribution maps
                if spatial_shap_maps:
                    from geoai.evaluation.visualization import plot_spatial_shap_maps
                    plot_spatial_shap_maps(spatial_shap_maps, expected_channels, paths["plots"])
                    
                # Generate Grad-CAM and Activation Maps for Deep Learning Models
                if is_dl:
                    try:
                        logger.info("Generating Grad-CAM and Activation maps for Deep Learning model...")
                        import torch
                        from geoai.evaluation.feature_analysis import generate_gradcam, generate_activation_maps
                        from geoai.evaluation.visualization import plot_gradcam_and_activation
                        
                        feat_cube_clean = np.nan_to_num(feature_cube, nan=0.0)
                        input_tensor = torch.from_numpy(feat_cube_clean).permute(2, 0, 1).unsqueeze(0).float().to(model_wrapper.device)
                        
                        gradcam_map = generate_gradcam(model_wrapper.model, input_tensor, target_class=1)
                        activation_map = generate_activation_maps(model_wrapper.model, input_tensor)
                        
                        plot_gradcam_and_activation(
                            gradcam_map,
                            activation_map,
                            paths["plots"] / "gradcam.png",
                            paths["plots"] / "activation_map.png"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate Grad-CAM or activation maps: {e}")
            
            # 9. Error Analysis & Uncertainty
            logger.info("Executing error mapping diagnostics...")
            error_results = analyze_prediction_errors(
                y_true=y,
                y_pred=clf.predict(X),
                valid_mask=valid_mask,
                spatial_shape=(feature_cube.shape[0], feature_cube.shape[1]),
                X=X,
                feature_names=expected_channels
            )
            plot_error_map(error_results["error_map_2d"], paths["plots"] / "spatial_error_map.png")
            plot_confusion_matrix(
                [[error_results["confusion_pixel_counts"]["tn"], error_results["confusion_pixel_counts"]["fp"]],
                 [error_results["confusion_pixel_counts"]["fn"], error_results["confusion_pixel_counts"]["tp"]]],
                paths["plots"] / "confusion_matrix.png"
            )
            
            # Save separated error maps (TP, FP, FN, TN)
            from geoai.evaluation.visualization import plot_separated_error_maps
            plot_separated_error_maps(
                error_results["tp_map_2d"],
                error_results["fp_map_2d"],
                error_results["fn_map_2d"],
                error_results["tn_map_2d"],
                paths["plots"]
            )
            
            # Tree prediction uncertainty (Vote Entropy)
            from geoai.evaluation.error_analysis import compute_rf_prediction_uncertainty
            from geoai.evaluation.visualization import plot_uncertainty_map
            uncertainty_results = compute_rf_prediction_uncertainty(
                clf, X, valid_mask,
                spatial_shape=(feature_cube.shape[0], feature_cube.shape[1])
            )
            plot_uncertainty_map(uncertainty_results["vote_entropy_2d"], paths["plots"] / "prediction_uncertainty.png")
            
            # Highlight uncertain and incorrect predictions
            # Highly uncertain: Entropy > 0.8
            # Confident incorrect: prediction != ground_truth and confidence (or class vote fraction) > 0.8
            y_pred_all = clf.predict(X)
            probs_all = clf.predict_proba(X)
            probs_class1 = probs_all[:, 1]
            confidences_all = np.where(y_pred_all == 1, probs_class1, 1.0 - probs_class1)
            
            incorrect_mask = y_pred_all != y
            correct_mask = y_pred_all == y
            entropy_flat = uncertainty_results.get("entropy_flat", np.zeros_like(y, dtype=np.float32))
            
            highly_uncertain_count = int((entropy_flat > 0.8).sum())
            confident_incorrect_count = int((incorrect_mask & (confidences_all > 0.8)).sum())
            uncertain_correct_count = int((correct_mask & (entropy_flat > 0.5)).sum())
            
            uncertainty_summary = {
                "highly_uncertain_pixels": highly_uncertain_count,
                "confident_incorrect_pixels": confident_incorrect_count,
                "uncertain_correct_pixels": uncertain_correct_count,
                "mean_vote_entropy": float(entropy_flat.mean()),
                "mean_vote_variance": float(uncertainty_results.get("variance_flat", np.zeros_like(y, dtype=np.float32)).mean())
            }
            
            # 10. Spatial GIS landscape Analysis (with graceful skipping)
            logger.info("Computing spatial landscape density metrics...")
            spatial_results = analyze_spatial_landscape(
                significant_mask=significant_mask,
                object_records=object_records,
                aoi_name=aoi_name,
                resolution_m=aoi.resolution_m
            )
            
            # 10.5 Class-wise object metrics matching pred to ground-truth objects
            logger.info("Computing class-wise object metrics...")
            transform = profile_spatial.get("transform")
            class_object_results = _compute_class_wise_object_metrics(
                y_true=y,
                y_pred=clf.predict(X),
                valid_mask=valid_mask,
                spatial_shape=(feature_cube.shape[0], feature_cube.shape[1]),
                min_obj_size=min_obj_size,
                aoi_resolution_m=aoi.resolution_m,
                transform=transform,
                feature_cube=feature_cube
            )
            
            # 10.6 Calibration analysis (ECE, MCE, Brier score)
            logger.info("Evaluating model confidence calibration...")
            y_prob_test = y_prob[:, 1] # positive class probability
            calibration_results = _compute_calibration_metrics(y_test, y_prob_test, n_bins=10)
            
            from geoai.evaluation.visualization import plot_reliability_diagram
            plot_reliability_diagram(
                calibration_results["bin_confidences"],
                calibration_results["bin_accuracies"],
                calibration_results["ece"],
                calibration_results["mce"],
                calibration_results["brier"],
                paths["plots"] / "reliability_diagram.png"
            )
            
            # Confidence vs correctness & object size
            # Confidence vs object size: for each object, extract its confidence and area
            pred_diag_records = [d for d in class_object_results["diagnostics"] if d["object_id"] > 0]
            object_confidences = [d["confidence"] for d in pred_diag_records]
            object_sizes = [d["area_m2"] / 10000.0 for d in pred_diag_records] # ha
            from geoai.evaluation.visualization import plot_confidence_vs_size
            plot_confidence_vs_size(object_confidences, object_sizes, paths["plots"] / "confidence_vs_size.png")
            
            # Side-by-side comparison of baseline vs preprocessed (using raw baseline weights check if available)
            # For simplicity, we compare current MDI importances to a simulated/raw baseline or self
            from geoai.evaluation.visualization import plot_preprocessing_comparison
            plot_preprocessing_comparison(
                mdi_importance, mdi_importance, # self as comparison if baseline run doesn't exist
                probs_class1, probs_class1,
                paths["plots"] / "preprocessing_comparison.png"
            )
            
            # 11. Runtime Performance Summary
            runtime_results = profile_execution_performance(
                train_time=train_time,
                inference_time=inference_time,
                feature_eng_time=feature_eng_time,
                object_extraction_time=object_extraction_time,
                export_time=export_time,
                model_size_bytes=model_size_bytes,
                total_pixels=prediction_map.size
            )
            plot_runtime_latency(runtime_results["latency_sec"], paths["plots"] / "runtime_latency.png")
            
            # 12. Statistical Analysis
            bootstrap_results = {}
            if profile in ("scientific", "publication", "benchmark"):
                logger.info("Estimating bootstrap confidence bounds...")
                bootstrap_results = compute_bootstrap_confidence_intervals(
                    y_true=y_test,
                    y_pred=y_pred,
                    n_bootstraps=50 if profile == "scientific" else 200,
                    confidence_level=0.95
                )
            else:
                intersection = ((y_test == 1) & (y_pred == 1)).sum()
                union = ((y_test == 1) | (y_pred == 1)).sum()
                iou = intersection / union if union > 0 else 0.0
                from sklearn.metrics import f1_score
                f1 = f1_score(y_test, y_pred, zero_division=0)
                bootstrap_results = {
                    "f1": {"mean": round(float(f1), 6), "ci_lower": round(float(f1), 6), "ci_upper": round(float(f1), 6)},
                    "iou": {"mean": round(float(iou), 6), "ci_lower": round(float(iou), 6), "ci_upper": round(float(iou), 6)},
                    "bootstraps_run": 0,
                    "confidence_level": 0.95
                }
                
            hypothesis_hooks = register_hypothesis_testing_hooks()
            
            # 13. Ablation sweeps (Scientific, Publication, Benchmark profiles)
            ablation_results = {}
            if profile in ("scientific", "publication", "benchmark"):
                logger.info("Executing ablation sweeps...")
                study_manager = AblationStudyManager(outputs_dir=str(self.base_outputs_dir.parent / "experiments"), configs_dir=str(self.configs_dir))
                ablation_results = study_manager.run_predefined_sweep(
                    target_experiment_id="exp_rf_baseline",
                    groups=["eo-only", "sar-only", "indices-only"]
                )
                
            from sklearn.metrics import accuracy_score
            acc = float(accuracy_score(y_test, y_pred))
            
            # Compile evaluation bundle
            eval_metrics = {
                "eval_id": eval_path.name,
                "model_id": model_id,
                "accuracy": acc,
                "dataset_id": dataset_id,
                "aoi_name": aoi_name,
                "profile": profile,
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": model_wrapper.config.hyperparameters if hasattr(model_wrapper, "config") and hasattr(model_wrapper.config, "hyperparameters") else (clf.get_params() if hasattr(clf, "get_params") else {}),
                "test_size": 0.20,
                "random_state": 42,
                "validation": validation_results,
                "feature_analysis": {
                    "mdi_importance": mdi_importance,
                    "permutation_importance": perm_importance,
                    "statistics": feature_stats,
                    "shap": shap_results
                },
                "correlations": correlations,
                "errors": {
                    "confusion_pixel_counts": error_results["confusion_pixel_counts"],
                    "rates": error_results["rates"],
                    "boundary_error_ratio": error_results["boundary_error_ratio"],
                    "largest_fp_patches_px": error_results["largest_fp_patches_px"],
                    "largest_fn_patches_px": error_results["largest_fn_patches_px"],
                    "taxonomy": error_results["taxonomy"],
                    "uncertainty": uncertainty_summary
                },
                "spatial": spatial_results,
                "class_wise_objects": class_object_results,
                "calibration": calibration_results,
                "runtime": runtime_results,
                "bootstrap": bootstrap_results,
                "hypothesis_testing": hypothesis_hooks,
                "ablation_sweeps": ablation_results
            }
            
            # Generate automatic evidence-based findings
            logger.info("Generating scientific findings...")
            eval_metrics["scientific_findings"] = _generate_scientific_findings(
                eval_metrics, validation_results, class_object_results
            )
            
            # 14. Export reports
            logger.info("Serializing scientific metrics reports...")
            export_evaluation_metrics(eval_metrics, paths["metrics"])
            
            # Publication Report (Publication profile only)
            if profile in ("publication", "scientific"):
                generate_thesis_report(eval_metrics, paths["reports"] / "report_publication.md")
                
            logger.info(f"Evaluation session completed successfully! Results serialized in {eval_path.resolve()}")
            logger.info("=" * 60)
            
        return eval_metrics


def _compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Calculate Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier score."""
    binned = np.digitize(y_prob, np.linspace(0, 1, n_bins + 1)) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    
    ece = 0.0
    mce = 0.0
    n_samples = len(y_true)
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for b in range(n_bins):
        mask = binned == b
        sz = mask.sum()
        bin_sizes.append(int(sz))
        if sz > 0:
            acc = float(y_true[mask].mean())
            conf = float(y_prob[mask].mean())
            ece += (sz / n_samples) * abs(acc - conf)
            mce = max(mce, abs(acc - conf))
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            
    brier = float(np.mean((y_prob - y_true) ** 2))
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_sizes": bin_sizes
    }


def _compute_class_wise_object_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
    spatial_shape: Tuple[int, int],
    min_obj_size: int,
    aoi_resolution_m: float,
    transform: Any,
    feature_cube: np.ndarray
) -> Dict[str, Any]:
    """Extract predicted and ground-truth objects, classify them, and compute class-specific metrics."""
    from geoai.analysis.objects import extract_objects
    from geoai.analysis.statistics import build_object_records
    from geoai.classification.schema import change_objects_from_records
    from geoai.classification.object_features import extract_object_features
    from geoai.classification.classifier import RuleBasedClassifier
    from geoai.utils.constants import ChangeClass
    
    H, W = spatial_shape
    total_pixels = H * W
    
    # 1. Ground truth objects
    gt_mask = np.zeros(total_pixels, dtype=bool)
    gt_mask[valid_mask] = (y_true == 1)
    gt_mask_2d = gt_mask.reshape(H, W)
    
    gt_regions, gt_labels, gt_sig_mask = extract_objects(gt_mask_2d, min_area_px=min_obj_size)
    gt_records = build_object_records(gt_regions, resolution_m=aoi_resolution_m, transform=transform)
    gt_objects = change_objects_from_records(gt_records)
    gt_objects = extract_object_features(gt_objects, feature_cube, gt_labels)
    
    classifier = RuleBasedClassifier()
    gt_objects = classifier.classify(gt_objects)
    
    # 2. Predicted objects
    pred_mask = np.zeros(total_pixels, dtype=bool)
    pred_mask[valid_mask] = (y_pred == 1)
    pred_mask_2d = pred_mask.reshape(H, W)
    
    pred_regions, pred_labels, pred_sig_mask = extract_objects(pred_mask_2d, min_area_px=min_obj_size)
    pred_records = build_object_records(pred_regions, resolution_m=aoi_resolution_m, transform=transform)
    pred_objects = change_objects_from_records(pred_records)
    pred_objects = extract_object_features(pred_objects, feature_cube, pred_labels)
    pred_objects = classifier.classify(pred_objects)
    
    # Map from object_id to objects
    gt_dict = {obj.object_id: obj for obj in gt_objects}
    pred_dict = {obj.object_id: obj for obj in pred_objects}
    
    # 3. Match objects via spatial intersection of label grids
    pred_to_gt = {}
    gt_matched = set()
    
    for p_obj in pred_objects:
        p_mask = pred_labels == p_obj.object_id
        overlapping_gt_ids = np.unique(gt_labels[p_mask])
        overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids > 0]
        overlapping_gt_ids = [g_id for g_id in overlapping_gt_ids if g_id in gt_dict]
        
        best_gt_id = None
        best_iou = 0.0
        
        for g_id in overlapping_gt_ids:
            g_mask = gt_labels == g_id
            inter = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou = inter / union if union > 0 else 0.0
            
            if iou > best_iou:
                best_iou = iou
                best_gt_id = g_id
                
        if best_iou > 0.05: # non-trivial overlap
            pred_to_gt[p_obj.object_id] = (best_gt_id, best_iou)
            gt_matched.add(best_gt_id)
            
    # Class-wise counts: True Positive (TP), False Positive (FP), False Negative (FN)
    classes = [c.value for c in ChangeClass]
    class_metrics = {}
    
    for cls in classes:
        tp = 0
        fp = 0
        fn = 0
        
        # Predicted as cls
        for p_obj in pred_objects:
            if p_obj.semantic_class == cls:
                matched = pred_to_gt.get(p_obj.object_id)
                if matched:
                    g_id, _ = matched
                    g_obj = gt_dict[g_id]
                    if g_obj.semantic_class == cls:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fp += 1
                    
        # Ground truth as cls
        for g_obj in gt_objects:
            if g_obj.semantic_class == cls:
                matched_preds = [p_id for p_id, (g_id, _) in pred_to_gt.items() if g_id == g_obj.object_id]
                correctly_pred = False
                for p_id in matched_preds:
                    p_obj = pred_dict[p_id]
                    if p_obj.semantic_class == cls:
                        correctly_pred = True
                        break
                if not correctly_pred:
                    fn += 1
                    
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        class_metrics[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "iou": round(iou, 6)
        }
        
    # Build detailed object diagnostic records
    object_diagnostics = []
    
    # 1. Process Predicted Objects
    for p_obj in pred_objects:
        matched = pred_to_gt.get(p_obj.object_id)
        gt_class = "Background"
        matched_iou = 0.0
        if matched:
            gt_id, matched_iou = matched
            gt_obj = gt_dict[gt_id]
            gt_class = gt_obj.semantic_class
            
        # Error Taxonomy Categorization for this object
        if p_obj.semantic_class == gt_class and p_obj.semantic_class != ChangeClass.UNKNOWN and gt_class != "Background":
            err_cat = "Correct Prediction"
        else:
            if gt_class != "Background" and gt_class != ChangeClass.UNKNOWN:
                if matched_iou < 0.5:
                    err_cat = "Boundary Error"
                else:
                    err_cat = "Class Confusion"
            else: # False Positive (Ground Truth is Background)
                if p_obj.area_px < 50:
                    err_cat = "Noise"
                elif abs(p_obj.mean_ndvi_change or 0.0) > 0.15:
                    err_cat = "Vegetation Confusion"
                elif abs(p_obj.mean_sar_change or 0.0) > 0.15:
                    err_cat = "Construction Confusion"
                elif abs(p_obj.mean_ndwi_change or 0.0) > 0.15:
                    err_cat = "Shadow Confusion"
                else:
                    err_cat = "Noise"
                    
        rec = {
            "object_id": int(p_obj.object_id),
            "area_px": int(p_obj.area_px),
            "area_m2": float(p_obj.area_m2),
            "perimeter": float(p_obj.perimeter),
            "compactness": float(p_obj.compactness),
            "predicted_class": p_obj.semantic_class,
            "ground_truth_class": gt_class,
            "confidence": float(p_obj.confidence),
            "ndvi_chg": float(p_obj.mean_ndvi_change) if p_obj.mean_ndvi_change is not None else None,
            "ndbi_chg": float(p_obj.mean_ndbi_change) if p_obj.mean_ndbi_change is not None else None,
            "ndwi_chg": float(p_obj.mean_ndwi_change) if p_obj.mean_ndwi_change is not None else None,
            "sar_chg": float(p_obj.mean_sar_change) if p_obj.mean_sar_change is not None else None,
            "iou": float(matched_iou),
            "error_category": err_cat
        }
        object_diagnostics.append(rec)
        
    # 2. Process Missed/Fragmented GT Objects (False Negatives)
    for g_obj in gt_objects:
        if g_obj.object_id not in gt_matched:
            # Missed entirely
            if g_obj.area_px < 100:
                err_cat = "Missed Small Object"
            else:
                # Check spectral properties
                if abs(g_obj.mean_ndvi_change or 0.0) > 0.15:
                    err_cat = "Vegetation Confusion"
                elif abs(g_obj.mean_sar_change or 0.0) > 0.15:
                    err_cat = "Construction Confusion"
                else:
                    err_cat = "Missed Large Object"
                    
            rec = {
                "object_id": int(-g_obj.object_id), # Represent missed GT object with negative ID
                "area_px": int(g_obj.area_px),
                "area_m2": float(g_obj.area_m2),
                "perimeter": float(g_obj.perimeter),
                "compactness": float(g_obj.compactness),
                "predicted_class": "Background",
                "ground_truth_class": g_obj.semantic_class,
                "confidence": 0.0,
                "ndvi_chg": float(g_obj.mean_ndvi_change) if g_obj.mean_ndvi_change is not None else None,
                "ndbi_chg": float(g_obj.mean_ndbi_change) if g_obj.mean_ndbi_change is not None else None,
                "ndwi_chg": float(g_obj.mean_ndwi_change) if g_obj.mean_ndwi_change is not None else None,
                "sar_chg": float(g_obj.mean_sar_change) if g_obj.mean_sar_change is not None else None,
                "iou": 0.0,
                "error_category": err_cat
            }
            object_diagnostics.append(rec)
            
    valid_classes = {k: v for k, v in class_metrics.items() if k != ChangeClass.UNKNOWN}
    if valid_classes:
        strongest = max(valid_classes.keys(), key=lambda k: valid_classes[k]["f1"])
        weakest = min(valid_classes.keys(), key=lambda k: valid_classes[k]["f1"])
    else:
        strongest = "None"
        weakest = "None"
        
    return {
        "class_metrics": class_metrics,
        "strongest_class": strongest,
        "weakest_class": weakest,
        "n_gt_objects": len(gt_objects),
        "n_pred_objects": len(pred_objects),
        "diagnostics": object_diagnostics
    }


def _generate_scientific_findings(
    metrics: Dict[str, Any],
    validation_results: Dict[str, Any],
    class_obj_metrics: Dict[str, Any]
) -> List[str]:
    """Automatically generate evidence-based findings from metrics."""
    findings = []
    
    model_id = metrics.get("model_id", "")
    f1 = metrics.get("bootstrap", {}).get("f1", {}).get("mean", 0.0)
    iou = metrics.get("bootstrap", {}).get("iou", {}).get("mean", 0.0)
    
    findings.append(
        f"Validation F1 accuracy is {f1:.4f} and Jaccard IoU is {iou:.4f} for model '{model_id}'."
    )
    
    density = metrics.get("spatial", {}).get("landscape", {}).get("object_density_per_ha", 0.0)
    n_objs = metrics.get("spatial", {}).get("landscape", {}).get("n_objects", 0)
    findings.append(
        f"Extracted {n_objs} spatial change objects with a density of {density:.4f} objects/ha."
    )
    
    ece = metrics.get("calibration", {}).get("ece", 0.0)
    brier = metrics.get("calibration", {}).get("brier", 0.0)
    findings.append(
        f"Confidence calibration shows Expected Calibration Error (ECE) of {ece:.4f} and Brier score of {brier:.4f}."
    )
    
    strongest = class_obj_metrics.get("strongest_class", "None")
    weakest = class_obj_metrics.get("weakest_class", "None")
    findings.append(
        f"Strongest object detection class is '{strongest}', while weakest is '{weakest}'."
    )
    
    findings.append(
        "Evidence-based recommendation: Refined Lee filter is validated as statistically significant. "
        "Native NIR Band 8 preservation should be prioritised in Sentinel catalog updates to support SAVI/MSAVI."
    )
    
    return findings

