import os
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import joblib

# Matplotlib configuration for headless environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from geoai.core.config import load_platform_config
from geoai.core.exceptions import ModelError
from geoai.experiments.experiment import Experiment
from geoai.experiments.experiment_config import ExperimentConfig
from geoai.experiments.experiment_logger import experiment_logger_context
from geoai.experiments.experiment_registry import get_experiment_model, ModelImplementationUnavailableError
from geoai.models.baselines.dl_wrapper import DLBaseModelWrapper
from geoai.experiments.metrics import compute_metrics
from geoai.experiments.utils import get_git_commit, get_system_info
from geoai.experiments.experiment_report import generate_reports
from geoai.experiments.leaderboard import Leaderboard
from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import split_dataset, split_dataset_unified, SplitResult

# Stage 1 helpers
from geoai.pipeline.stage1 import _load_all_rasters
from geoai.features.feature_cube import build_feature_cube
from geoai.models.inference import run_inference
from geoai.postprocessing.cleanup import to_binary_mask
from geoai.analysis.objects import extract_objects
from geoai.analysis.statistics import build_object_records
from geoai.exports.geotiff import export_change_mask_geotiff
from geoai.exports.shapefile import export_change_shapefile, export_geojson
from geoai.exports.csv_export import export_objects_csv
from geoai.utils.constants import CANONICAL_FEATURE_NAMES

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Core orchestration engine to execute a research experiment run."""
    
    def __init__(self, outputs_dir: str = "outputs/experiments", configs_dir: str = "configs") -> None:
        self.outputs_dir = Path(outputs_dir)
        self.configs_dir = Path(configs_dir)

    def run(self, config_path: Path) -> Experiment:
        """
        Execute the experiment lifecycle steps.
        
        This loads config, validates baseline locks, subsets datasets, trains/evals,
        generates isolated artifact outputs, plots curves, and registers leaderboard entries.
        """
        config_path = Path(config_path)
        config = ExperimentConfig(config_path)
        
        experiment_id = config.experiment_id
        model_id = config.model_id
        dataset_id = config.dataset_id
        
        # 1. Output isolated workspace directory
        output_dir = self.outputs_dir / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = output_dir / "logs" / "execution.log"
        
        # 2. Run execution under the redirection logger context
        with experiment_logger_context(log_path):
            logger.info("=" * 60)
            logger.info(f"Starting Experiment: {experiment_id} (Model: {model_id})")
            logger.info("=" * 60)
            
            # Retrieve Dataset Metadata to identify AOI name
            dataset_meta = DatasetRegistry.get_dataset_metadata(dataset_id)
            if not dataset_meta:
                raise ValueError(f"Dataset '{dataset_id}' not found in registry catalog.")
            aoi_name = dataset_meta.aoi_name
            
            # Initialize Experiment object
            experiment = Experiment(
                experiment_id=experiment_id,
                model_id=model_id,
                dataset_id=dataset_id,
                aoi_name=aoi_name,
                config=config.to_dict(),
                random_seed=config.random_state,
                state="Running"
            )
            
            # Baseline lock validation & deviations collection
            lock_config = config.baseline_lock
            deviations: List[str] = []
            
            # Evaluate baseline deviations relative to contract locks
            if config.test_size != 0.20:
                deviations.append(f"Alternative train/test split size: {config.test_size * 100:.1f}% holdout")
            if config.random_state != 42:
                deviations.append(f"Alternative train/test split random seed: {config.random_state}")
                
            # Check explicit deviations configured in the yaml
            if lock_config:
                if not lock_config.get("inherit_preprocessing", True):
                    deviations.append("Custom preprocessing / no standard valid pixel mask")
                if not lock_config.get("inherit_feature_engineering", True):
                    deviations.append("Custom feature engineering / modified feature cube")
                custom_deviations = lock_config.get("deviations", [])
                for dev in custom_deviations:
                    if dev not in deviations:
                        deviations.append(dev)
                        
            experiment.deviations = deviations
            
            try:
                # 3. Model implementation validation check (prevent silent proxying)
                model_wrapper = get_experiment_model(model_id)
                if not model_wrapper.is_implemented():
                    logger.error(f"Execution halted: model {model_id} is not implemented.")
                    experiment.set_state("Failed")
                    raise ModelImplementationUnavailableError(model_id)
                
                # 4. Load dataset
                logger.info(f"Loading dataset: {dataset_id}")
                dataset = DatasetRegistry.load_dataset(
                    dataset_id,
                    configs_dir=str(self.configs_dir),
                    preprocessing_config=config.preprocessing,
                    features_config=config.features,
                    campaign_type=config.campaign_type,
                    track=config.track,
                )
                X = dataset.X
                y = dataset.y
                
                # 5. Extract expected features & apply mapping filter dynamically
                expected_channels = list(model_wrapper.get_capabilities().expected_input_channels)
                
                # Append features dynamically to expected_channels based on campaign type
                from geoai.preprocessing.pipeline import get_campaign_feature_names
                from geoai.utils.constants import CANONICAL_FEATURE_NAMES
                
                dataset_feature_names = get_campaign_feature_names(
                    campaign_type=config.campaign_type,
                    track=config.track,
                    config=config.to_dict()
                )
                
                extra_features = [f for f in dataset_feature_names if f not in CANONICAL_FEATURE_NAMES]
                for f in extra_features:
                    if f not in expected_channels:
                        expected_channels.append(f)
                            
                logger.info(f"Model expected features: {expected_channels}")
                
                col_indices = [
                    dataset_feature_names.index(channel)
                    for channel in expected_channels
                    if channel in dataset_feature_names
                ]
                
                if len(col_indices) < len(expected_channels):
                    missing = [c for c in expected_channels if c not in dataset_feature_names]
                    logger.warning(f"Some expected features were not found in dataset list: {missing}")
                
                if col_indices:
                    # Filter feature matrix columns to match expects
                    X = X[:, col_indices]
                    logger.info(f"Filtered feature matrix shape: {X.shape}")
                
                # 6. Stratified Split using centralized splitting logic
                split_result = split_dataset_unified(
                    X=X,
                    y=y,
                    dataset_id=dataset_id,
                    split_policy=config.split_policy,
                    patch_size=config.features.get("patch_size", 15),
                    test_size=config.test_size,
                    random_state=config.random_state,
                    stratify=True
                )
                X_train = split_result.X_train
                X_test = split_result.X_test
                y_train = split_result.y_train
                y_test = split_result.y_test
                X_val = split_result.X_val
                y_val = split_result.y_val
                
                # 7. Model training (fitting)
                logger.info("Fitting classifier weights...")
                hyperparams = config.hyperparameters
                t0_train = time.time()
                
                if model_id in ("rf_baseline_v1", "rf_enhanced_v1"):
                    clf = RandomForestClassifier(
                        n_estimators=hyperparams.get("n_estimators", 100),
                        random_state=hyperparams.get("random_state", 42),
                        n_jobs=hyperparams.get("n_jobs", -1),
                        class_weight="balanced" if model_id == "rf_baseline_v1" else None
                    )
                elif model_id == "extra_trees":
                    clf = ExtraTreesClassifier(
                        n_estimators=hyperparams.get("n_estimators", 100),
                        random_state=hyperparams.get("random_state", 42),
                        n_jobs=hyperparams.get("n_jobs", -1),
                        class_weight=hyperparams.get("class_weight", None)
                    )
                elif model_id == "xgboost":
                    clf = XGBClassifier(
                        n_estimators=hyperparams.get("n_estimators", 100),
                        learning_rate=hyperparams.get("learning_rate", 0.1),
                        max_depth=hyperparams.get("max_depth", 6),
                        random_state=hyperparams.get("random_state", 42),
                        n_jobs=hyperparams.get("n_jobs", -1),
                        eval_metric="logloss"
                    )
                elif model_id == "lightgbm":
                    clf = LGBMClassifier(
                        n_estimators=hyperparams.get("n_estimators", 100),
                        learning_rate=hyperparams.get("learning_rate", 0.1),
                        max_depth=hyperparams.get("max_depth", -1),
                        random_state=hyperparams.get("random_state", 42),
                        n_jobs=hyperparams.get("n_jobs", -1),
                        verbose=-1
                    )
                elif model_id == "catboost":
                    clf = CatBoostClassifier(
                        iterations=hyperparams.get("iterations", hyperparams.get("n_estimators", 100)),
                        learning_rate=hyperparams.get("learning_rate", 0.1),
                        depth=hyperparams.get("depth", hyperparams.get("max_depth", 6)),
                        random_state=hyperparams.get("random_state", 42),
                        thread_count=hyperparams.get("n_jobs", -1),
                        verbose=0
                    )
                elif isinstance(model_wrapper, DLBaseModelWrapper):
                    clf = model_wrapper
                    if hasattr(clf, "set_dataset_info"):
                        clf.set_dataset_info(dataset_id=dataset_id, config=config)
                else:
                    raise ValueError(f"Unknown classifier model_id: {model_id}")
                
                if isinstance(model_wrapper, DLBaseModelWrapper):
                    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                else:
                    clf.fit(X_train, y_train)
                train_time = time.time() - t0_train
                logger.info(f"Model training complete in {train_time:.4f}s.")
                
                # 8. Inference evaluation
                t0_inf = time.time()
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)
                inference_time = time.time() - t0_inf
                
                # Save serialized model file
                if isinstance(model_wrapper, DLBaseModelWrapper):
                    model_pkl_path = output_dir / "model.pt"
                    shutil.copy2(output_dir / "checkpoints" / "best_model.pt", model_pkl_path)
                else:
                    model_pkl_path = output_dir / "model.pkl"
                    joblib.dump(clf, model_pkl_path)
                model_size_bytes = model_pkl_path.stat().st_size
                
                # Measure process memory usage fallback
                mem_mb = 0.0
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    mem_mb = process.memory_info().rss / (1024 * 1024)
                except Exception:
                    pass
                
                # Compute performance metrics
                metrics = compute_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    train_time=train_time,
                    inference_time=inference_time,
                    model_size_bytes=model_size_bytes,
                    memory_usage_mb=mem_mb
                )
                experiment.metrics = metrics
                
                # 9. Plot ROC, PR, and Confusion Matrix heatmaps
                plot_dir = output_dir / "plots"
                self._generate_plots(y_test, y_pred, y_prob, plot_dir)
                
                # 10. Generate spatial change deliverables
                logger.info("Generating spatial prediction deliverables...")
                platform_config = load_platform_config(str(self.configs_dir))
                aoi = platform_config.get_aoi(aoi_name)
                
                # Load rasters and feature cube
                rasters_with_profile = _load_all_rasters(aoi)
                profile = rasters_with_profile[14]
                
                # Extract first 14 elements as inputs for building feature cube
                red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1 = rasters_with_profile[0:6]
                red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2 = rasters_with_profile[6:12]
                sar_t1, sar_t2 = rasters_with_profile[12:14]
                
                # Campaign A: Preprocessing
                if config.campaign_type == "preprocessing" and config.preprocessing:
                    s1_config = config.preprocessing.get("s1", {})
                    if s1_config.get("speckle_filter") == "refined_lee":
                        from geoai.preprocessing.pipeline import refined_lee_filter
                        window_size = s1_config.get("window_size", 7)
                        sar_t1 = refined_lee_filter(sar_t1, size=window_size)
                        sar_t2 = refined_lee_filter(sar_t2, size=window_size)

                feature_cube = build_feature_cube(
                    red_t1=red_t1, green_t1=green_t1, blue_t1=blue_t1,
                    sar_t1=sar_t1, ndvi_t1=ndvi_t1, ndbi_t1=ndbi_t1, ndwi_t1=ndwi_t1,
                    red_t2=red_t2, green_t2=green_t2, blue_t2=blue_t2,
                    sar_t2=sar_t2, ndvi_t2=ndvi_t2, ndbi_t2=ndbi_t2, ndwi_t2=ndwi_t2,
                )
                
                # Campaign B: Feature Engineering
                extra_bands = []
                if config.campaign_type == "feature_engineering" and config.features:
                    features_config = config.features
                    track = config.track
                    if track == "production" or track == "experimental":
                        if features_config.get("local_variance", {}).get("enabled", False):
                            from geoai.preprocessing.pipeline import local_variance_filter
                            size = features_config.get("local_variance", {}).get("window_size", 3)
                            lv_t1 = local_variance_filter(ndvi_t1, size=size)
                            lv_t2 = local_variance_filter(ndvi_t2, size=size)
                            extra_bands.extend([lv_t1, lv_t2])
                    if track == "experimental":
                        if features_config.get("savi", {}).get("enabled", False):
                            from geoai.preprocessing.pipeline import reconstruct_nir, compute_savi
                            nir_t1 = reconstruct_nir(red_t1, ndvi_t1)
                            nir_t2 = reconstruct_nir(red_t2, ndvi_t2)
                            savi_t1 = compute_savi(nir_t1, red_t1)
                            savi_t2 = compute_savi(nir_t2, red_t2)
                            extra_bands.extend([savi_t1, savi_t2])
                        if features_config.get("msavi", {}).get("enabled", False):
                            from geoai.preprocessing.pipeline import reconstruct_nir, compute_msavi
                            nir_t1 = reconstruct_nir(red_t1, ndvi_t1)
                            nir_t2 = reconstruct_nir(red_t2, ndvi_t2)
                            msavi_t1 = compute_msavi(nir_t1, red_t1)
                            msavi_t2 = compute_msavi(nir_t2, red_t2)
                            extra_bands.extend([msavi_t1, msavi_t2])

                # Campaign C: Boundary Engineering
                if config.campaign_type == "boundary_engineering" and config.features:
                    features_config = config.features
                    boundary_config = features_config.get("boundary", {})
                    method = boundary_config.get("method", "sobel").lower()
                    
                    from geoai.features.boundary import (
                        compute_continuous_gradient,
                        compute_distance_transform_from_edges,
                        compute_morphological_gradient,
                        compute_morphological_boundaries,
                        project_object_shapes
                    )
                    from geoai.features.temporal import compute_all_deltas
                    from geoai.features.pseudo_labels import generate_pseudo_labels
                    
                    t_delta_ndvi, _, _, t_delta_sar = compute_all_deltas(
                        ndvi_t1=ndvi_t1, ndvi_t2=ndvi_t2,
                        ndbi_t1=ndbi_t1, ndbi_t2=ndbi_t2,
                        ndwi_t1=ndwi_t1, ndwi_t2=ndwi_t2,
                        sar_t1=sar_t1, sar_t2=sar_t2
                    )
                    
                    if method in ("sobel", "scharr", "laplacian"):
                        f_ndvi = compute_continuous_gradient(t_delta_ndvi, method=method)
                        f_sar = compute_continuous_gradient(t_delta_sar, method=method)
                        extra_bands.extend([f_ndvi, f_sar])
                    elif method == "morphological_gradient":
                        f_ndvi = compute_morphological_gradient(t_delta_ndvi)
                        f_sar = compute_morphological_gradient(t_delta_sar)
                        extra_bands.extend([f_ndvi, f_sar])
                    elif method == "distance_transform":
                        f_ndvi = compute_distance_transform_from_edges(t_delta_ndvi)
                        f_sar = compute_distance_transform_from_edges(t_delta_sar)
                        extra_bands.extend([f_ndvi, f_sar])
                    elif method == "distance_to_boundary":
                        t_change_mask, _, _ = generate_pseudo_labels(
                            delta_sar=t_delta_sar,
                            delta_ndvi=t_delta_ndvi,
                            delta_ndbi=ndbi_t2 - ndbi_t1,
                            delta_ndwi=ndwi_t2 - ndwi_t1,
                            pseudo_label_config=platform_config.processing.pseudo_labels,
                        )
                        from scipy.ndimage import binary_dilation
                        struct = np.ones((3, 3))
                        boundary = binary_dilation(t_change_mask, structure=struct) & ~t_change_mask
                        from scipy.ndimage import distance_transform_edt
                        f_dist = distance_transform_edt(~boundary)
                        extra_bands.append(f_dist)
                    elif method == "morphological":
                        mask_ndvi = np.abs(t_delta_ndvi) > 0.15
                        mask_sar = np.abs(t_delta_sar) > 0.15
                        
                        i_ndvi, e_ndvi, t_ndvi = compute_morphological_boundaries(mask_ndvi)
                        i_sar, e_sar, t_sar = compute_morphological_boundaries(mask_sar)
                        
                        extra_bands.extend([i_ndvi, i_sar, e_ndvi, e_sar, t_ndvi, t_sar])
                    elif method == "boundary_refinement":
                        t_change_mask, _, _ = generate_pseudo_labels(
                            delta_sar=t_delta_sar,
                            delta_ndvi=t_delta_ndvi,
                            delta_ndbi=ndbi_t2 - ndbi_t1,
                            delta_ndwi=ndwi_t2 - ndwi_t1,
                            pseudo_label_config=platform_config.processing.pseudo_labels,
                        )
                        from scipy.ndimage import label
                        label_arr, num_feats = label(t_change_mask)
                        shapes = project_object_shapes(label_arr, ndvi_t1 != 0.0)
                        extra_bands.extend([
                            shapes["solidity"],
                            shapes["compactness"],
                            shapes["elongation"],
                            shapes["eccentricity"],
                            shapes["convexity"]
                         ])
                             
                if extra_bands:
                    extra_cube = np.stack(extra_bands, axis=-1)
                    feature_cube = np.concatenate([feature_cube, extra_cube], axis=-1)
                
                # Run inference with the trained estimator in the wrapper
                if model_id in ("rf_baseline_v1", "rf_enhanced_v1"):
                    model_wrapper._rf = clf
                elif model_id == "extra_trees":
                    model_wrapper._et = clf
                elif model_id == "xgboost":
                    model_wrapper._xgb = clf
                elif model_id == "lightgbm":
                    model_wrapper._lgb = clf
                elif model_id == "catboost":
                    model_wrapper._cat = clf
                elif isinstance(model_wrapper, DLBaseModelWrapper):
                    model_wrapper = clf
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
                
                transform = profile.get("transform")
                object_records = build_object_records(
                    significant_regions, resolution_m=aoi.resolution_m, transform=transform
                )
                
                # Export maps and shapefiles
                export_change_mask_geotiff(binary_mask, profile, output_dir / "prediction.tif")
                export_change_shapefile(significant_mask, profile, output_dir / "objects.shp")
                
                if object_records:
                    from geoai.classification.schema import change_objects_from_records
                    objects = change_objects_from_records(object_records)
                    export_geojson(objects, significant_mask, profile, output_dir / "objects.geojson")
                    export_objects_csv(objects, output_dir / "statistics.csv")
                
                # State: Completed
                experiment.set_state("Completed")
                logger.info(f"Experiment completed successfully! Metrics: {metrics}")
                
            except Exception as e:
                logger.exception(f"Exception raised during experiment execution: {e}")
                experiment.set_state("Failed")
                
            finally:
                # 11. Write Execution Provenance and environment reports
                experiment.execution_metadata = get_system_info()
                experiment.execution_metadata["git_commit"] = get_git_commit()
                experiment.execution_metadata["random_seed"] = config.random_state
                experiment.execution_metadata["test_size"] = config.test_size
                experiment.execution_metadata["model_pkl_path"] = str(output_dir / "model.pkl")
                experiment.execution_metadata["output_folder"] = str(output_dir)
                
                # Generate Reports
                generate_reports(experiment, output_dir)
                
                # Copies config.yaml to run dir
                shutil.copy2(config_path, output_dir / "config.yaml")
                
                # 12. Register in global leaderboard (if Completed)
                if experiment.get_state() == "Completed":
                    leaderboard_entry = {
                        "experiment_id": experiment.experiment_id,
                        "model_id": experiment.model_id,
                        "aoi_name": experiment.aoi_name,
                        "dataset_id": experiment.dataset_id,
                        "f1": experiment.metrics.get("f1", 0.0),
                        "iou": experiment.metrics.get("iou", 0.0),
                        "precision": experiment.metrics.get("precision", 0.0),
                        "recall": experiment.metrics.get("recall", 0.0),
                        "runtime_sec": experiment.metrics.get("train_time_sec", 0.0) + experiment.metrics.get("inference_time_sec", 0.0),
                        "memory_mb": experiment.metrics.get("memory_usage_mb", 0.0),
                        "date": experiment.timestamp
                    }
                    
                    # Write leaderboard_entry.json locally
                    with open(output_dir / "leaderboard_entry.json", "w", encoding="utf-8") as f:
                        import json
                        json.dump(leaderboard_entry, f, indent=2, default=str)
                        
                    # Add to global leaderboard database files
                    lb = Leaderboard(leaderboard_dir=str(self.outputs_dir))
                    lb.add_entry(leaderboard_entry)
                    
                    # Generate metrics.json locally
                    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(experiment.metrics, f, indent=2, default=str)
                        
                logger.info(f"Finished Experiment: {experiment_id} State: {experiment.get_state()}")
                logger.info("=" * 60)
                
        return experiment

    def _generate_plots(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, plot_dir: Path) -> None:
        """Create ROC, PR, and Confusion Matrix plots."""
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. ROC Curve
        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    prob_col = y_prob[:, 1]
                else:
                    prob_col = y_prob
                
                if len(np.unique(y_true)) > 1:
                    fpr, tpr, _ = roc_curve(y_true, prob_col)
                    plt.figure()
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic')
                    plt.legend(loc="lower right")
                    plt.savefig(plot_dir / 'roc_curve.png', bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.warning(f"Could not generate ROC curve: {e}")
                
        # 2. Precision-Recall Curve
        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    prob_col = y_prob[:, 1]
                else:
                    prob_col = y_prob
                    
                if len(np.unique(y_true)) > 1:
                    precision, recall, _ = precision_recall_curve(y_true, prob_col)
                    plt.figure()
                    plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Precision-Recall Curve')
                    plt.legend(loc="lower left")
                    plt.savefig(plot_dir / 'pr_curve.png', bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.warning(f"Could not generate PR curve: {e}")
                
        # 3. Confusion Matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=[0, 1], yticks=[0, 1],
                xticklabels=['No-Change', 'Change'], yticklabels=['No-Change', 'Change'],
                title='Confusion Matrix',
                ylabel='True label',
                xlabel='Predicted label'
            )
            
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            plt.savefig(plot_dir / 'confusion.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate Confusion Matrix plot: {e}")
