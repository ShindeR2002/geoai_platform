import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from geoai.models.registry import get_model_class
from geoai.models.baselines.dl_wrapper import SpatialPatchDataset, set_deterministic_seeds
from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import get_pixel_coords

def run_overfitting_test(model_id, dataset_id, train_coords, feature_cube, labels_2d, patch_size, num_samples, max_epochs):
    set_deterministic_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the model class
    wrapper_cls = get_model_class(model_id)
    wrapper = wrapper_cls()
    
    in_ch = 18
    backbone_type = "lightweight"
    import inspect
    sig = inspect.signature(wrapper.architecture_class)
    if "backbone" in sig.parameters:
        model = wrapper.architecture_class(in_channels=in_ch, out_channels=2, backbone=backbone_type)
    else:
        model = wrapper.architecture_class(in_channels=in_ch, out_channels=2)
    
    model.to(device)
    model.train()
    
    # 2. Slice coordinates for the sub-sample
    np.random.seed(42)
    sample_idx = np.random.choice(len(train_coords), size=num_samples, replace=False)
    sub_coords = train_coords[sample_idx]
    
    # 3. Create Dataset and Loader
    dataset = SpatialPatchDataset(
        feature_cube=feature_cube,
        labels=labels_2d,
        coords=sub_coords,
        patch_size=patch_size,
        augment=False,  # Disable augmentation for deterministic memorization
        training_mode="center_pixel"
    )
    loader = DataLoader(dataset, batch_size=min(num_samples, 32), shuffle=False)
    
    # 4. Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    loss_history = []
    acc_history = []
    
    reached_zero_loss = False
    reached_hundred_acc = False
    target_epoch_loss = -1
    target_epoch_acc = -1
    
    start_time = time.time()
    
    for epoch in range(1, max_epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            logits = model(x_batch)
            H_out, W_out = logits.shape[2], logits.shape[3]
            center_logits = logits[:, :, H_out // 2, W_out // 2]
            
            loss = criterion(center_logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
            preds = center_logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
        epoch_loss /= total
        acc = correct / total
        
        loss_history.append(epoch_loss)
        acc_history.append(acc)
        
        # Check criteria
        if not reached_zero_loss and epoch_loss < 0.01:
            reached_zero_loss = True
            target_epoch_loss = epoch
        if not reached_hundred_acc and acc >= 0.999:
            reached_hundred_acc = True
            target_epoch_acc = epoch
            
        # Break early if perfect score is reached
        if reached_zero_loss and reached_hundred_acc:
            break
            
    elapsed = time.time() - start_time
    
    return {
        "model_id": model_id,
        "num_samples": num_samples,
        "final_loss": loss_history[-1],
        "final_accuracy": acc_history[-1],
        "epochs_run": len(loss_history),
        "reached_zero_loss": reached_zero_loss,
        "reached_hundred_acc": reached_hundred_acc,
        "epoch_reached_zero_loss": target_epoch_loss,
        "epoch_reached_hundred_acc": target_epoch_acc,
        "time_taken_sec": elapsed,
        "loss_curve": loss_history,
        "acc_curve": acc_history
    }

def main():
    print("Starting Sanity Overfitting Tests...")
    output_dir = root_dir / "outputs" / "root_cause_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_id = "ps10_sentinel_v1"
    
    print("Loading dataset coordinates and features...")
    coords, (H, W), valid_mask_2d, feature_cube = get_pixel_coords(dataset_id)
    full_dataset = DatasetRegistry.load_dataset(dataset_id)
    
    # Reconstruct labels_2d
    labels_2d = np.zeros((H, W), dtype=np.int64)
    labels_2d[valid_mask_2d] = full_dataset.y
    
    # Exclude coordinates whose patch neighborhood overflows the boundaries
    R = 7
    valid_coords = []
    for r, c in coords:
        if r - R >= 0 and r + R < H and c - R >= 0 and c + R < W:
            valid_coords.append([r, c])
    valid_coords = np.array(valid_coords)
    
    model_ids = ["fc_ef", "fc_siam_conc", "fc_siam_diff", "lightweight_siam_cnn", "changeformer", "bit", "tinycd"]
    
    results = {}
    
    for m_id in model_ids:
        print(f"\n--- Testing model: {m_id} ---")
        
        # Test A: 1 sample
        print("Running Test A (1 sample, max 100 epochs)...")
        res_a = run_overfitting_test(m_id, dataset_id, valid_coords, feature_cube, labels_2d, 15, 1, 100)
        print(f"Test A completed: Acc={res_a['final_accuracy']*100:.1f}%, Loss={res_a['final_loss']:.4f}, Epochs={res_a['epochs_run']}")
        
        # Test B: 10 samples
        print("Running Test B (10 samples, max 150 epochs)...")
        res_b = run_overfitting_test(m_id, dataset_id, valid_coords, feature_cube, labels_2d, 15, 10, 150)
        print(f"Test B completed: Acc={res_b['final_accuracy']*100:.1f}%, Loss={res_b['final_loss']:.4f}, Epochs={res_b['epochs_run']}")
        
        # Test C: 100 samples
        print("Running Test C (100 samples, max 200 epochs)...")
        res_c = run_overfitting_test(m_id, dataset_id, valid_coords, feature_cube, labels_2d, 15, 100, 200)
        print(f"Test C completed: Acc={res_c['final_accuracy']*100:.1f}%, Loss={res_c['final_loss']:.4f}, Epochs={res_c['epochs_run']}")
        
        results[m_id] = {
            "test_1_sample": res_a,
            "test_10_samples": res_b,
            "test_100_samples": res_c
        }
        
    with open(output_dir / "sanity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\nSanity Overfitting Tests completed successfully!")

if __name__ == "__main__":
    main()
