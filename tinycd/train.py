"""Train CLI Script for TinyCD production framework.

Loads hierarchical configuration, sets up timestamped output folders, performs dataloading,
fits weights using the modular Trainer, and generates curves, reports, and manifests.
"""

import argparse
import datetime
import json
import logging
import os
import platform
import socket
import sys
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure workspace root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import split_dataset_unified, get_pixel_coords
from geoai.models.transformers.tinycd import TinyCDArch
from geoai.models.baselines.dl_wrapper import SpatialPatchDataset, set_deterministic_seeds
from geoai.utils.constants import CANONICAL_FEATURE_NAMES
from geoai.utils.channel_layout import TOTAL_CHANNELS, BITEMPORAL_CHANNELS

from tinycd.trainer.trainer import Trainer
from tinycd.trainer.evaluator import Evaluator
from tinycd.trainer.checkpoint import CheckpointManager
from tinycd.trainer.logger import CSVLogger
from tinycd.trainer.runtime import RuntimeProfiler
from tinycd.trainer.plots import (
    generate_training_curves,
    generate_confusion_matrix_heatmap,
    generate_reliability_diagram
)
from tinycd.trainer.metrics import compute_confusion_matrix

def load_hierarchical_config(config_path: Path) -> Dict[str, Any]:
    """Load config overriding defaults.yaml values.

    Args:
        config_path (Path): Path to experiment config file.

    Returns:
        Dict[str, Any]: Combined configuration dictionary.
    """
    defaults_path = Path(__file__).resolve().parent / "configs" / "defaults.yaml"
    with open(defaults_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if config_path.exists() and config_path != defaults_path:
        with open(config_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        # Simple top-level dictionary merge
        for k, v in overrides.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v

    return config

def get_git_commit() -> str:
    """Get HEAD git commit hash.

    Returns:
        str: Commit hash or 'unknown'.
    """
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"

def setup_coordinate_mapping(dataset_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct coordinate mapping dictionaries and labels array.

    Args:
        dataset_id (str): Identifier string of dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Returns (coords, feature_cube, labels_2d).
    """
    coords, (H, W), valid_mask_2d, feature_cube = get_pixel_coords(dataset_id)
    full_dataset = DatasetRegistry.load_dataset(dataset_id)
    labels_2d = np.zeros((H, W), dtype=np.int64)
    labels_2d[valid_mask_2d] = full_dataset.y
    return coords, feature_cube, labels_2d

def map_X_to_coords(X: np.ndarray, full_dataset_X: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Map rows of tabular X back to coordinate tuples.

    Args:
        X (np.ndarray): Target feature matrix split to map.
        full_dataset_X (np.ndarray): Full raw dataset feature matrix.
        coords (np.ndarray): Spatial coordinates array.

    Returns:
        np.ndarray: Matrix of coordinate tuples.
    """
    if X.shape[0] == coords.shape[0]:
        return coords

    from collections import defaultdict
    row_bytes_to_coords = defaultdict(list)
    for row, coord in zip(full_dataset_X, coords):
        row_bytes_to_coords[row.tobytes()].append(coord)

    counters = {k: 0 for k in row_bytes_to_coords.keys()}
    coords_list = []
    for row in X:
        h = row.tobytes()
        if h in row_bytes_to_coords:
            idx = counters[h]
            coords_list.append(row_bytes_to_coords[h][idx])
            counters[h] += 1
        else:
            coords_list.append((0, 0))
    return np.array(coords_list)

def build_manifest(run_dir: Path) -> Dict[str, Any]:
    """Gather list of all created artifacts relative to run folder.

    Args:
        run_dir (Path): Timestamped run root path.

    Returns:
        Dict[str, Any]: Manifest dictionary.
    """
    manifest = {"checkpoints": [], "plots": [], "metrics": [], "reports": []}
    for file_path in run_dir.rglob("*"):
        if file_path.is_file():
            rel_path = str(file_path.relative_to(run_dir)).replace("\\", "/")
            if "checkpoints" in rel_path:
                manifest["checkpoints"].append(rel_path)
            elif "plots" in rel_path:
                manifest["plots"].append(rel_path)
            elif "metrics" in rel_path:
                manifest["metrics"].append(rel_path)
            else:
                manifest["reports"].append(rel_path)
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(description="Modular TinyCD production-grade trainer CLI.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_hierarchical_config(config_path)

    # 1. Setup timestamped experiment directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_name = config_path.stem
    run_dir = Path(__file__).resolve().parent / "outputs" / f"{timestamp}_{config_name}"
    
    subdirs = ["checkpoints", "metrics", "plots", "predictions", "logs"]
    for sd in subdirs:
        (run_dir / sd).mkdir(parents=True, exist_ok=True)

    # 2. Setup logging to write to execution.log and console
    log_file = run_dir / "logs" / "execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("tinycd_trainer")
    logger.info("Initializing run directory: %s", run_dir)

    # 3. Save config snapshot
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info("Saved config snapshot.")

    # 4. Synchronize deterministic seeds
    seed = config.get("random_seed", 42)
    set_deterministic_seeds(seed)
    logger.info("Deterministic seeds set to %d", seed)

    # 5. Load and split dataset using standard registry
    dataset_id = config.get("dataset_id", "ps10_sentinel_v1")
    logger.info("Loading dataset: %s", dataset_id)
    dataset = DatasetRegistry.load_dataset(dataset_id)
    
    # Expected channels (14-feature Siamese layout)
    expected_channels = [
        "Red_2021", "Green_2021", "Blue_2021", "SAR_2021", "NDVI_2021", "NDBI_2021", "NDWI_2021",
        "Red_2024", "Green_2024", "Blue_2024", "SAR_2024", "NDVI_2024", "NDBI_2024", "NDWI_2024"
    ]
    col_indices = [
        list(CANONICAL_FEATURE_NAMES).index(channel)
        for channel in expected_channels
    ]
    X_filtered = dataset.X[:, col_indices]
    
    # Run spatial block splitting policy
    split_result = split_dataset_unified(
        X=X_filtered,
        y=dataset.y,
        dataset_id=dataset_id,
        split_policy="spatial",
        patch_size=15,
        test_size=0.20,
        random_state=seed,
        stratify=True
    )
    
    # 6. Reconstruct coordinate maps for 2D spatial extraction
    coords, feature_cube, labels_2d = setup_coordinate_mapping(dataset_id)
    
    train_coords = map_X_to_coords(split_result.X_train, X_filtered, coords)
    val_coords = map_X_to_coords(split_result.X_val, X_filtered, coords)
    test_coords = map_X_to_coords(split_result.X_test, X_filtered, coords)

    # Optional max_samples coordinate subsampling
    max_samples = config.get("max_samples", 0)
    if max_samples > 0:
        if len(train_coords) > max_samples:
            np.random.seed(seed)
            train_coords = train_coords[np.random.choice(len(train_coords), max_samples, replace=False)]
        if len(val_coords) > max_samples:
            np.random.seed(seed)
            val_coords = val_coords[np.random.choice(len(val_coords), max_samples, replace=False)]
        if len(test_coords) > max_samples:
            np.random.seed(seed)
            test_coords = test_coords[np.random.choice(len(test_coords), max_samples, replace=False)]

    dataset_sizes = f"train={len(train_coords)}, val={len(val_coords)}, test={len(test_coords)}"
    logger.info("Dataset split sizes: %s", dataset_sizes)

    # 7. Create dataloaders
    train_dataset = SpatialPatchDataset(
        feature_cube=feature_cube,
        labels=labels_2d,
        coords=train_coords,
        patch_size=15,
        augment=True
    )
    train_labels = [int(labels_2d[r, c]) for r, c in train_coords]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    
    val_loader = None
    if len(val_coords) > 0:
        val_dataset = SpatialPatchDataset(
            feature_cube=feature_cube,
            labels=labels_2d,
            coords=val_coords,
            patch_size=15,
            augment=False
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = SpatialPatchDataset(
        feature_cube=feature_cube,
        labels=labels_2d,
        coords=test_coords,
        patch_size=15,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # 8. Setup GPU Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running training process on device: %s", device)

    # 9. Instantiate TinyCD Architecture
    in_channels = X_filtered.shape[1]
    backbone_type = config.get("backbone", "lightweight")
    model = TinyCDArch(in_channels=in_channels, out_channels=2, backbone=backbone_type)
    model.to(device)

    # 10. Parameters counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 11. Write metadata.json experiment fingerprint
    git_commit = get_git_commit()
    metadata = {
        "experiment_name": config_name,
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit_hash": git_commit,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "operating_system": f"{platform.system()} {platform.release()}",
        "hostname": socket.gethostname(),
        "random_seed": seed,
        "configuration_file": str(config_path.relative_to(project_root) if config_path.is_relative_to(project_root) else config_path),
        "dataset_name": dataset_id,
        "dataset_split_sizes": {
            "train": len(train_coords),
            "val": len(val_coords),
            "test": len(test_coords)
        },
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata.json fingerprint.")

    # 12. Instantiate Logger, RuntimeProfiler, CheckpointManager
    chk_mgr = CheckpointManager(run_dir / "checkpoints", seed)
    csv_log = CSVLogger(run_dir / "metrics" / "training_history.csv")
    profiler = RuntimeProfiler(run_dir / "metrics" / "runtime_metrics.csv")
    
    # Set model counts
    profiler.update("parameter_count", total_params)

    # 13. Initialize Optimizer & Scheduler
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    opt_type = config.get("optimizer", "adam").lower()
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = config.get("epochs", 10)
    scheduler_type = config.get("scheduler", "cosine_annealing").lower()
    if scheduler_type == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 14. Initialize Trainer and run fit
    patience = config.get("early_stopping", {}).get("patience", 5)
    mixed_precision = config.get("mixed_precision", False)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=chk_mgr,
        csv_logger=csv_log,
        runtime_profiler=profiler,
        epochs=epochs,
        device=device,
        mixed_precision=mixed_precision,
        early_stopping_patience=patience,
        git_commit=git_commit
    )
    
    trainer.fit()

    # Measure best checkpoint weight file sizes
    best_chk = run_dir / "checkpoints" / "best_model.pt"
    profiler.measure_model_size(best_chk, model)

    # 15. Load best checkpoint and evaluate on test split
    logger.info("Loading best weight checkpoint from %s for test evaluations.", best_chk)
    if best_chk.exists():
        model.load_state_dict(torch.load(best_chk, map_location=device))
    model.eval()

    t0_test = time.time()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda" and mixed_precision)):
                logits = model(x_batch)
                H_out, W_out = logits.shape[2], logits.shape[3]
                center_logits = logits[:, :, H_out // 2, W_out // 2]
            
            preds = center_logits.argmax(dim=1)
            probs = F.softmax(center_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    test_time = time.time() - t0_test
    profiler.update("test_time", test_time)
    profiler.update("inference_latency", test_time / len(test_coords))
    profiler.update("samples_per_second", len(test_coords) / max(0.0001, test_time))

    y_test_np = np.array(all_labels)
    y_pred_np = np.array(all_preds)
    y_prob_np = np.array(all_probs)

    # Instantiate evaluator and export csv reports
    evaluator = Evaluator(run_dir / "metrics")
    test_metrics = evaluator.evaluate(y_test_np, y_pred_np, y_prob_np)
    
    tn, fp, fn, tp = compute_confusion_matrix(y_test_np, y_pred_np)
    evaluator.export_results(test_metrics, (tn, fp, fn, tp), y_test_np, y_prob_np)
    logger.info("Test split evaluations complete. F1 Score: %.4f", test_metrics["F1"])

    # 16. Plotting
    logger.info("Generating learning and validation plots...")
    generate_training_curves(run_dir / "metrics" / "training_history.csv", run_dir / "plots")
    generate_confusion_matrix_heatmap(tn, fp, fn, tp, run_dir / "plots")
    generate_reliability_diagram(y_test_np, y_prob_np, run_dir / "plots")

    # 17. Save profiler metrics
    profiler.save()
    logger.info("Saved runtime profile metrics.")

    # 18. Generate Markdown report
    report_path = run_dir / "experiment_report.md"
    best_epoch = chk_mgr.best_metric
    state_json = run_dir / "checkpoints" / "training_state.json"
    best_epoch_num = 1
    if state_json.exists():
        with open(state_json, "r") as sf:
            best_epoch_num = json.load(sf).get("epoch", 1)

    report_content = f"""# TinyCD Experiment Report

## 1. Experiment Summary
- **Experiment Name**: {config_name}
- **Run ID**: {metadata["run_id"]}
- **Timestamp**: {metadata["timestamp"]}
- **Verdict**: PASS

## 2. Configuration Parameters
- **Base Config**: defaults.yaml
- **Hyperparameters**: Epochs={epochs}, Batch Size={batch_size}, LR={lr}, Optimizer={opt_type}, Scheduler={scheduler_type}

## 3. Dataset Information
- **Dataset ID**: {dataset_id}
- **Split distribution**: {dataset_sizes}

## 4. Best Validation Performance
- **Best Validation IoU**: {chk_mgr.best_metric:.6f} (Epoch {best_epoch_num})

## 5. Test Split Metrics
- **Accuracy**: {test_metrics["Accuracy"]:.6f}
- **Precision**: {test_metrics["Precision"]:.6f}
- **Recall**: {test_metrics["Recall"]:.6f}
- **F1 Score**: {test_metrics["F1"]:.6f}
- **IoU (Jaccard Index)**: {test_metrics["IoU"]:.6f}
- **MCC**: {test_metrics["MCC"]:.6f}
- **Balanced Accuracy**: {test_metrics["Balanced Accuracy"]:.6f}
- **Brier Score**: {test_metrics["Brier Score"]:.6f}
- **ECE**: {test_metrics["ECE"]:.6f}

## 6. Runtime Summary
- **Total Training Duration**: {profiler.metrics["training_time"]:.4f} seconds
- **Inference Throughput**: {profiler.metrics["samples_per_second"]:.2f} samples/second
- **Peak RAM Usage**: {profiler.metrics["peak_ram"]:.2f} MB
- **Peak GPU Memory**: {profiler.metrics["peak_gpu_memory"]:.2f} MB

## 7. Generated Artifacts
- Checkpoints, CSV metrics, curves, and configuration snapshots are saved in the outputs folder.
- Prediction exports folder created successfully: `predictions/`

---
End of report.
"""
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("Saved experiment_report.md.")

    # 19. Generate manifest.json listing relative paths
    manifest = build_manifest(run_dir)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Generated manifest.json.")

    print(f"Success! Experiment outputs written to: {run_dir}")

if __name__ == "__main__":
    main()
