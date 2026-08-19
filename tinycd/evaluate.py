"""Evaluate CLI Script for TinyCD production framework.

Loads a saved checkpoint, executes inference on the test split, computes metrics
via Evaluator, and updates evaluation CSV manifests.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure workspace root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import split_dataset_unified
from geoai.models.transformers.tinycd import TinyCDArch
from geoai.models.baselines.dl_wrapper import SpatialPatchDataset, set_deterministic_seeds
from geoai.utils.constants import CANONICAL_FEATURE_NAMES

from tinycd.train import load_hierarchical_config, setup_coordinate_mapping, map_X_to_coords
from tinycd.trainer.evaluator import Evaluator
from tinycd.trainer.metrics import compute_confusion_matrix

def main() -> None:
    parser = argparse.ArgumentParser(description="Modular TinyCD evaluator CLI.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt checkpoint.")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    config = load_hierarchical_config(config_path)

    # Output directory defaults to checkpoint's parent folder
    # e.g., outputs/run_folder/checkpoints/best_model.pt -> outputs/run_folder
    run_dir = checkpoint_path.parent.parent
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("tinycd_evaluator")
    logger.info("Initializing post-hoc evaluation on checkpoint: %s", checkpoint_path)

    seed = config.get("random_seed", 42)
    set_deterministic_seeds(seed)

    # Load and split dataset
    dataset_id = config.get("dataset_id", "ps10_sentinel_v1")
    dataset = DatasetRegistry.load_dataset(dataset_id)
    
    expected_channels = [
        "Red_2021", "Green_2021", "Blue_2021", "SAR_2021", "NDVI_2021", "NDBI_2021", "NDWI_2021",
        "Red_2024", "Green_2024", "Blue_2024", "SAR_2024", "NDVI_2024", "NDBI_2024", "NDWI_2024"
    ]
    col_indices = [
        list(CANONICAL_FEATURE_NAMES).index(channel)
        for channel in expected_channels
    ]
    X_filtered = dataset.X[:, col_indices]
    
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
    
    coords, feature_cube, labels_2d = setup_coordinate_mapping(dataset_id)
    test_coords = map_X_to_coords(split_result.X_test, X_filtered, coords)

    max_samples = config.get("max_samples", 0)
    if max_samples > 0 and len(test_coords) > max_samples:
        np.random.seed(seed)
        test_coords = test_coords[np.random.choice(len(test_coords), max_samples, replace=False)]

    logger.info("Test split size: %d samples", len(test_coords))

    test_dataset = SpatialPatchDataset(
        feature_cube=feature_cube,
        labels=labels_2d,
        coords=test_coords,
        patch_size=15,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = X_filtered.shape[1]
    backbone_type = config.get("backbone", "lightweight")
    model = TinyCDArch(in_channels=in_channels, out_channels=2, backbone=backbone_type)
    model.to(device)

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logger.info("Running model predictions on test split...")
    all_preds, all_labels, all_probs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            H_out, W_out = logits.shape[2], logits.shape[3]
            center_logits = logits[:, :, H_out // 2, W_out // 2]
            
            preds = center_logits.argmax(dim=1)
            probs = F.softmax(center_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = time.time() - t0
    logger.info("Inference complete. Latency: %.4f seconds", inference_time)

    y_test_np = np.array(all_labels)
    y_pred_np = np.array(all_preds)
    y_prob_np = np.array(all_probs)

    evaluator = Evaluator(metrics_dir)
    test_metrics = evaluator.evaluate(y_test_np, y_pred_np, y_prob_np)
    
    tn, fp, fn, tp = compute_confusion_matrix(y_test_np, y_pred_np)
    evaluator.export_results(test_metrics, (tn, fp, fn, tp), y_test_np, y_prob_np)

    logger.info("Evaluation results written to: %s", metrics_dir)
    print("================ Evaluation Metrics ================")
    for k, v in test_metrics.items():
        print(f"{k:<20}: {v:.6f}")
    print("====================================================")

if __name__ == "__main__":
    main()
