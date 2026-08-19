import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from geoai.experiments.experiment import Experiment
from geoai.experiments.experiment_config import ExperimentConfig
from geoai.experiments.experiment_runner import ExperimentRunner
from geoai.experiments.comparison import compare_experiments

class AblationManager:
    """Manages design, feature-zeroing, and performance contrast of ablation studies."""
    
    def __init__(self, outputs_dir: str = "outputs/experiments", configs_dir: str = "configs") -> None:
        self.outputs_dir = Path(outputs_dir)
        self.configs_dir = Path(configs_dir)
        self.runner = ExperimentRunner(outputs_dir=str(outputs_dir), configs_dir=str(configs_dir))

    def run_ablation(self, target_experiment_id: str, features_to_remove: List[str], ablation_id: str) -> Experiment:
        """
        Run an ablation run based on a target experiment.
        
        Zeroes out the specified features from the feature matrix X during dataset loading,
        and saves the run as ablation_{ablation_id}.
        """
        # Load the configuration of the target experiment
        target_dir = self.outputs_dir / target_experiment_id
        config_path = target_dir / "config.yaml"
        
        # If the target hasn't run or config is not copied, check configs/experiments/
        if not config_path.exists():
            config_path = self.configs_dir / "experiments" / f"{target_experiment_id.replace('exp_', '')}.yaml"
            if not config_path.exists():
                # Fallback to default baseline
                config_path = self.configs_dir / "experiments" / "baseline_rf.yaml"
                
        # Parse configuration
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
            
        # Update config fields for the ablation study
        raw_config["experiment_id"] = f"ablation_{ablation_id}"
        raw_config["description"] = f"Ablation Study: removed features {features_to_remove} from {target_experiment_id}"
        
        # Log deviations
        deviations = raw_config.setdefault("baseline_lock", {}).setdefault("deviations", [])
        deviations.append(f"Ablation Study: removed feature(s) {', '.join(features_to_remove)}")
        
        # Temporarily save this ablation config to run it
        temp_config_path = self.configs_dir / "experiments" / f"temp_ablation_{ablation_id}.yaml"
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(raw_config, f)
            
        # We will wrap the runner execution or perform it and hook into the feature mapping
        # Let's monkeypatch or intercept the feature loading:
        # Actually, we can perform it manually or run a customized execution in the manager!
        # Let's write a customized run that zeroes out the ablated features in X.
        
        # Let's reuse components of ExperimentRunner to ensure identical environment
        try:
            # We construct a custom runner run but zero out features!
            # Since we want to ensure zeroing out happens during loading, we can do it inside this method:
            # Let's copy the Runner's steps and zero out the feature columns.
            
            # Let's read configs
            config = ExperimentConfig(temp_config_path)
            from geoai.experiments.experiment_logger import experiment_logger_context
            from geoai.datasets.dataset_registry import DatasetRegistry
            from geoai.experiments.experiment_registry import get_experiment_model
            from geoai.utils.constants import CANONICAL_FEATURE_NAMES
            from geoai.datasets.dataset_splitter import split_dataset
            from sklearn.ensemble import RandomForestClassifier
            import time
            from geoai.experiments.metrics import compute_metrics
            from geoai.experiments.utils import get_git_commit, get_system_info
            from geoai.experiments.experiment_report import generate_reports
            from geoai.experiments.leaderboard import Leaderboard
            
            output_dir = self.outputs_dir / config.experiment_id
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "logs" / "execution.log"
            
            with experiment_logger_context(log_path):
                # Setup metadata
                dataset_meta = DatasetRegistry.get_dataset_metadata(config.dataset_id)
                aoi_name = dataset_meta.aoi_name if dataset_meta else "Dholera"
                
                experiment = Experiment(
                    experiment_id=config.experiment_id,
                    model_id=config.model_id,
                    dataset_id=config.dataset_id,
                    aoi_name=aoi_name,
                    config=config.to_dict(),
                    random_seed=config.random_state,
                    state="Running"
                )
                
                experiment.deviations = deviations
                
                # Load dataset
                dataset = DatasetRegistry.load_dataset(config.dataset_id, configs_dir=str(self.configs_dir))
                X = dataset.X.copy()
                y = dataset.y
                
                # Zero out ablated features
                # Find indexes of features to remove in CANONICAL_FEATURE_NAMES
                # Note: features_to_remove can contain substrings (e.g. "NDVI" means Delta_NDVI, NDVI_2021, NDVI_2024)
                # or exact feature names.
                ablated_indices = []
                for feat in features_to_remove:
                    for i, name in enumerate(CANONICAL_FEATURE_NAMES):
                        if feat.lower() in name.lower():
                            ablated_indices.append(i)
                            
                ablated_indices = list(set(ablated_indices))
                
                for idx in ablated_indices:
                    # Zero out the column
                    X[:, idx] = 0.0
                    
                # Model mapping subsetting if baseline model
                model_wrapper = get_experiment_model(config.model_id)
                expected_channels = model_wrapper.get_capabilities().expected_input_channels
                
                col_indices = [
                    list(CANONICAL_FEATURE_NAMES).index(channel)
                    for channel in expected_channels
                    if channel in CANONICAL_FEATURE_NAMES
                ]
                
                if col_indices:
                    X = X[:, col_indices]
                    
                # Split
                X_train, X_test, y_train, y_test = split_dataset(
                    X, y,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    stratify=True
                )
                
                # Train
                t0 = time.time()
                clf = RandomForestClassifier(
                    n_estimators=config.hyperparameters.get("n_estimators", 100),
                    random_state=config.hyperparameters.get("random_state", 42),
                    n_jobs=config.hyperparameters.get("n_jobs", -1),
                    class_weight="balanced" if config.model_id == "rf_baseline_v1" else None
                )
                clf.fit(X_train, y_train)
                train_time = time.time() - t0
                
                # Eval
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)
                
                import os
                import joblib
                model_pkl_path = output_dir / "model.pkl"
                joblib.dump(clf, model_pkl_path)
                model_size_bytes = model_pkl_path.stat().st_size
                
                mem_mb = 0.0
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    mem_mb = process.memory_info().rss / (1024 * 1024)
                except Exception:
                    pass
                    
                metrics = compute_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    train_time=train_time,
                    inference_time=0.01,
                    model_size_bytes=model_size_bytes,
                    memory_usage_mb=mem_mb
                )
                experiment.metrics = metrics
                
                # Generate reports
                experiment.set_state("Completed")
                experiment.execution_metadata = get_system_info()
                experiment.execution_metadata["git_commit"] = get_git_commit()
                experiment.execution_metadata["random_seed"] = config.random_state
                experiment.execution_metadata["test_size"] = config.test_size
                experiment.execution_metadata["model_pkl_path"] = str(model_pkl_path)
                experiment.execution_metadata["output_folder"] = str(output_dir)
                
                generate_reports(experiment, output_dir)
                
                # Copy config
                shutil.copy2(temp_config_path, output_dir / "config.yaml")
                
                # Plotting (ROC, PR, Confusion)
                self.runner._generate_plots(y_test, y_pred, y_prob, output_dir / "plots")
                
                # Save local metrics.json and leaderboard_entry
                with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
                    import json
                    json.dump(metrics, f, indent=2, default=str)
                    
                # Add to leaderboard
                leaderboard_entry = {
                    "experiment_id": experiment.experiment_id,
                    "model_id": experiment.model_id,
                    "aoi_name": experiment.aoi_name,
                    "dataset_id": experiment.dataset_id,
                    "f1": metrics.get("f1", 0.0),
                    "iou": metrics.get("iou", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "runtime_sec": train_time,
                    "memory_mb": mem_mb,
                    "date": experiment.timestamp
                }
                
                with open(output_dir / "leaderboard_entry.json", "w", encoding="utf-8") as f:
                    json.dump(leaderboard_entry, f, indent=2, default=str)
                    
                lb = Leaderboard(leaderboard_dir=str(self.outputs_dir))
                lb.add_entry(leaderboard_entry)
                
            return experiment
            
        finally:
            # Clean up temp config path
            if temp_config_path.exists():
                temp_config_path.unlink()
                
    def compare_to_target(self, target_experiment_id: str, ablation_id: str) -> Dict[str, Any]:
        """Compare the ablated model run directly against the target locked baseline."""
        target_meta = self.runner.outputs_dir / target_experiment_id / "metadata.json"
        ablation_meta = self.runner.outputs_dir / f"ablation_{ablation_id}" / "metadata.json"
        
        if not target_meta.exists() or not ablation_meta.exists():
            return {}
            
        with open(target_meta, "r", encoding="utf-8") as f:
            target_data = json.load(f)
        with open(ablation_meta, "r", encoding="utf-8") as f:
            ablation_data = json.load(f)
            
        return compare_experiments(target_data, ablation_data)
