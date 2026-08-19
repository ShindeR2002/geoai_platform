import os
import random
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, jaccard_score

from geoai.models.base import BaseModel, ModelCapabilities
from geoai.core.exceptions import ModelError, ModelNotFoundError, InferenceError
from geoai.models.baselines.deep_learning import FC_EF, FC_Siam_Conc, FC_Siam_Diff, Lightweight_Siam_CNN
from geoai.utils.constants import CANONICAL_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Spatial block splitting code has been moved to geoai.datasets.dataset_splitter.
# The following imports and wrappers are maintained for backward compatibility.
import warnings
from geoai.datasets import dataset_splitter

BLOCK_SPLIT_MAP = dataset_splitter.BLOCK_SPLIT_MAP

def set_deterministic_seeds(seed: int):
    """Sync random seeds globally to ensure deterministic PyTorch initialization and execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Deterministic random seeds synchronized to: %d", seed)

def get_pixel_split(r: int, c: int, H: int, W: int, R: int) -> str:
    warnings.warn(
        "get_pixel_split is deprecated and moved to geoai.datasets.dataset_splitter",
        DeprecationWarning,
        stacklevel=2
    )
    return dataset_splitter.get_pixel_split(r, c, H, W, R)

def get_pixel_coords(dataset_id: str, configs_dir: str = "configs") -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray]:
    warnings.warn(
        "get_pixel_coords is deprecated and moved to geoai.datasets.dataset_splitter",
        DeprecationWarning,
        stacklevel=2
    )
    return dataset_splitter.get_pixel_coords(dataset_id, configs_dir)

def split_dataset_spatial(
    X: np.ndarray,
    y: np.ndarray,
    dataset_id: str,
    patch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    warnings.warn(
        "split_dataset_spatial is deprecated and moved to geoai.datasets.dataset_splitter",
        DeprecationWarning,
        stacklevel=2
    )
    return dataset_splitter.split_dataset_spatial(X, y, dataset_id, patch_size)

class SpatialPatchDataset(Dataset):
    """
    Yields patch neighborhoods of shape (C, P, P) around specified pixel coordinates.
    """
    def __init__(
        self,
        feature_cube: np.ndarray, # Shape (H, W, C)
        labels: np.ndarray,       # Shape (H, W)
        coords: np.ndarray,       # Shape (N, 2)
        patch_size: int = 15,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        training_mode: str = "center_pixel"
    ):
        self.patch_size = patch_size
        self.R = patch_size // 2
        self.coords = coords
        self.augment = augment
        self.augment_config = augment_config or {}
        self.training_mode = training_mode
        
        # Pad the feature cube
        self.padded_features = np.pad(
            feature_cube,
            ((self.R, self.R), (self.R, self.R), (0, 0)),
            mode='edge'
        )
        
        # Pad the labels. Padding with -100 ensures no fabricated labels are used in the loss.
        self.padded_labels = np.pad(
            labels,
            ((self.R, self.R), (self.R, self.R)),
            mode='constant',
            constant_values=-100
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        r, c = self.coords[idx]
        
        # Extract patch from padded features
        patch = self.padded_features[r : r + self.patch_size, c : c + self.patch_size, :]
        x_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        
        if self.training_mode == "dense_segmentation":
            label_patch = self.padded_labels[r : r + self.patch_size, c : c + self.patch_size]
            label = torch.from_numpy(label_patch).long()
        else:
            label = int(self.labels[r, c])
            
        if self.augment:
            if self.training_mode == "dense_segmentation":
                x_tensor, label = self._apply_augmentations_dense(x_tensor, label)
            else:
                x_tensor = self._apply_augmentations(x_tensor)
                
        return x_tensor, label

    def _apply_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        import random
        
        # Horizontal Flip
        if self.augment_config.get("horizontal_flip", True) and random.random() > 0.5:
            x = torch.flip(x, dims=[2])
            
        # Vertical Flip
        if self.augment_config.get("vertical_flip", True) and random.random() > 0.5:
            x = torch.flip(x, dims=[1])
            
        # Rotation (90, 180, 270 degrees)
        if self.augment_config.get("rotation", True) and random.random() > 0.5:
            k = random.choice([1, 2, 3])
            x = torch.rot90(x, k, dims=[1, 2])
            
        # Add Noise (Gaussian)
        if self.augment_config.get("gaussian_noise", False) and random.random() > 0.5:
            std = self.augment_config.get("gaussian_noise_std", 0.05)
            noise = torch.randn_like(x) * std
            x = x + noise
            
        return x

    def _apply_augmentations_dense(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        import random
        
        # Horizontal Flip
        if self.augment_config.get("horizontal_flip", True) and random.random() > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
            
        # Vertical Flip
        if self.augment_config.get("vertical_flip", True) and random.random() > 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
            
        # Rotation (90, 180, 270 degrees)
        if self.augment_config.get("rotation", True) and random.random() > 0.5:
            k = random.choice([1, 2, 3])
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[0, 1])
            
        # Add Noise (Gaussian)
        if self.augment_config.get("gaussian_noise", False) and random.random() > 0.5:
            std = self.augment_config.get("gaussian_noise_std", 0.05)
            noise = torch.randn_like(x) * std
            x = x + noise
            
        return x, y

import torchvision

class ResNet18SiameseAdapter(nn.Module):
    """
    Standard torchvision ResNet-18 adapter for 15x15 pixel patches.
    Projects 7 channels -> 64 channels, forwards through Layer 1,
    and performs spatial upsampling/interpolation back to 15x15 to prevent shape collapse.
    """
    def __init__(self, in_channels: int = 7):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        # Modify standard conv1 to accept 7 channels
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # (B, 64, 4, 4)
        x = self.resnet.layer1(x)  # (B, 64, 4, 4)
        # Upsample back to 15x15 using bilinear interpolation
        x = F.interpolate(x, size=(15, 15), mode='bilinear', align_corners=False)
        return x


class DLBaseModelWrapper(BaseModel):
    """Abstract/base model wrapper for PyTorch deep learning models."""

    def __init__(self, model_id: str, architecture_class):
        super().__init__()
        self.model_id = model_id
        self.architecture_class = architecture_class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dataset_id = None
        self.patch_size = 15
        self.config = None
        self.hyperparams = {}
        self.is_loaded = False
        self.feature_names = list(CANONICAL_FEATURE_NAMES)
        
        # Cache for matching tabular rows back to 2D spatial coordinates
        self.row_bytes_to_coords = None
        self.feature_cube = None
        self.labels_2d = None

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score on given features and labels for scikit-learn compatibility."""
        preds = self.predict(X)
        return float((preds == y).mean())

    def set_dataset_info(self, dataset_id: str, config: Any) -> None:
        """Configure wrapper dataset and experiment settings."""
        self.dataset_id = dataset_id
        self.config = config
        if config:
            self.hyperparams = config.hyperparameters
            self.patch_size = config.features.get("patch_size", 15)

    def load(self, model_path: Union[str, Path]) -> None:
        """Load a serialized model and its metadata from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(str(model_path))

        logger.info("Loading PyTorch model weights from '%s'.", model_path)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Recreate model architecture dynamically based on channels and backbone saved in checkpoint
            in_ch = checkpoint.get("in_channels", 18)
            backbone_type = checkpoint.get("backbone", "lightweight")
            import inspect
            sig = inspect.signature(self.architecture_class)
            if "backbone" in sig.parameters:
                self.model = self.architecture_class(in_channels=in_ch, out_channels=2, backbone=backbone_type)
            else:
                self.model = self.architecture_class(in_channels=in_ch, out_channels=2)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            # Restore metadata
            self.dataset_id = checkpoint.get("dataset_id", "ps10_sentinel_v1")
            self.patch_size = checkpoint.get("patch_size", 15)
            self.feature_names = checkpoint.get("feature_names", list(CANONICAL_FEATURE_NAMES))
            self.is_loaded = True
        except Exception as exc:
            raise ModelError(f"Failed to load PyTorch model from '{model_path}': {exc}.") from exc

    def save(self, model_path: Union[str, Path]) -> None:
        """Serialize model weights and metadata to disk."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint_data = {
                "model_state_dict": self.model.state_dict() if self.model is not None else {},
                "in_channels": len(self.feature_names) if self.feature_names else 18,
                "dataset_id": self.dataset_id,
                "patch_size": self.patch_size,
                "feature_names": self.feature_names,
                "backbone": self.hyperparams.get("backbone", "lightweight")
            }
            torch.save(checkpoint_data, model_path)
            logger.info("Saved PyTorch model weights and metadata to '%s'.", model_path)

        except Exception as exc:
            raise ModelError(f"Failed to save PyTorch model to '{model_path}': {exc}.") from exc

    def _ensure_coordinate_mapping(self, X: np.ndarray):
        """Build the cache mapping from tabular row byte values back to spatial coordinates."""
        if self.row_bytes_to_coords is not None:
            return
            
        logger.info("Building coordinate matching dictionary for dataset: %s", self.dataset_id)
        coords, (H, W), valid_mask_2d, feature_cube = get_pixel_coords(self.dataset_id or "ps10_sentinel_v1")
        
        # Load the full dataset Features (to find the matching mapping indices)
        from geoai.datasets.dataset_registry import DatasetRegistry
        full_dataset = DatasetRegistry.load_dataset(self.dataset_id or "ps10_sentinel_v1")
        full_X = full_dataset.X
        
        # Filter full_X columns to match expected_channels
        expected_channels = self.get_capabilities().expected_input_channels
        dataset_feature_names = list(CANONICAL_FEATURE_NAMES)
        col_indices = [
            dataset_feature_names.index(channel)
            for channel in expected_channels
            if channel in dataset_feature_names
        ]
        if col_indices:
            full_X = full_X[:, col_indices]
            
        # Build the byte hash mapping
        from collections import defaultdict
        self.row_bytes_to_coords = defaultdict(list)
        for row, coord in zip(full_X, coords):
            self.row_bytes_to_coords[row.tobytes()].append(coord)
            
        self.feature_cube = feature_cube
        self.labels_2d = np.zeros((H, W), dtype=np.int64)
        self.labels_2d[valid_mask_2d] = full_dataset.y
        self.coords = coords

    def _map_X_to_coords(self, X: np.ndarray) -> np.ndarray:
        """Map tabular X rows to their 2D spatial coordinate tuples."""
        self._ensure_coordinate_mapping(X)
        
        # Fast path: if X is the full dataset, coords match exactly
        if X.shape[0] == self.coords.shape[0]:
            return self.coords
            
        # O(N) mapping using integer index counters to avoid pop(0) list shifting
        counters = {k: 0 for k in self.row_bytes_to_coords.keys()}
        coords_list = []
        for row in X:
            h = row.tobytes()
            if h in self.row_bytes_to_coords:
                idx = counters[h]
                coords_list.append(self.row_bytes_to_coords[h][idx])
                counters[h] += 1
            else:
                coords_list.append((0, 0))
        return np.array(coords_list)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train the deep learning model wrapper on the reconstructed spatial dataset."""
        logger.info("Initializing fit on PyTorch Wrapper for model: %s", self.model_id)
        
        # Deterministic execution
        seed = self.hyperparams.get("random_state", 42)
        set_deterministic_seeds(seed)
        
        # 1. Fetch coords and reconstruct feature cube
        self._ensure_coordinate_mapping(X_train)
        H, W, C = self.feature_cube.shape
        
        # 2. Extract coords of training and validation splits
        train_coords = self._map_X_to_coords(X_train)
        val_coords = None
        if X_val is not None:
            val_coords = self._map_X_to_coords(X_val)
            
        # Optional: Subsample coordinates for fast development/campaign runs
        max_samples = int(os.environ.get("DL_MAX_SAMPLES", 0))
        if max_samples > 0:
            if len(train_coords) > max_samples:
                logger.info("Subsampling training coordinates to max_samples=%d for speedup", max_samples)
                np.random.seed(seed)
                idx = np.random.choice(len(train_coords), size=max_samples, replace=False)
                train_coords = train_coords[idx]
            if val_coords is not None and len(val_coords) > max_samples:
                logger.info("Subsampling validation coordinates to max_samples=%d for speedup", max_samples)
                np.random.seed(seed)
                idx = np.random.choice(len(val_coords), size=max_samples, replace=False)
                val_coords = val_coords[idx]
            
        # Supervision Safety Check
        training_mode = self.hyperparams.get("training_mode", "center_pixel")
        is_dense_available = (self.labels_2d is not None and self.labels_2d.ndim == 2)
        if training_mode == "dense_segmentation" and not is_dense_available:
            logger.warning("Dense labels unavailable. Automatically falling back to center_pixel training mode.")
            training_mode = "center_pixel"
            
        # 3. Create datasets and loader with WeightedRandomSampler (Class Imbalance)
        augment_config = self.config.preprocessing.get("augmentation", {}) if self.config else {}
        train_dataset = SpatialPatchDataset(
            feature_cube=self.feature_cube,
            labels=self.labels_2d,
            coords=train_coords,
            patch_size=self.patch_size,
            augment=True,
            augment_config=augment_config,
            training_mode=training_mode
        )
        
        train_labels = [int(self.labels_2d[r, c]) for r, c in train_coords]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[lbl] for lbl in train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        batch_size = self.hyperparams.get("batch_size", 64)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        
        val_loader = None
        if val_coords is not None:
            val_dataset = SpatialPatchDataset(
                feature_cube=self.feature_cube,
                labels=self.labels_2d,
                coords=val_coords,
                patch_size=self.patch_size,
                augment=False,
                training_mode=training_mode
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # 4. Instantiate PyTorch Network
        in_ch = X_train.shape[1]
        backbone_type = self.hyperparams.get("backbone", "lightweight")
        import inspect
        sig = inspect.signature(self.architecture_class)
        if "backbone" in sig.parameters:
            self.model = self.architecture_class(in_channels=in_ch, out_channels=2, backbone=backbone_type)
        else:
            self.model = self.architecture_class(in_channels=in_ch, out_channels=2)
        self.model.to(self.device)
        
        # 5. Define loss, optimizer, scheduler, and AMP Scaler
        lr = self.hyperparams.get("learning_rate", 1e-3)
        epochs = self.hyperparams.get("epochs", self.hyperparams.get("iterations", 20))
        patience = self.hyperparams.get("patience", 5)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.hyperparams.get("weight_decay", 1e-4))
        
        # Scheduler Selection
        scheduler_type = self.hyperparams.get("scheduler", "StepLR")
        if scheduler_type == "StepLR":
            step_size = self.hyperparams.get("scheduler_step_size", 10)
            gamma = self.hyperparams.get("scheduler_gamma", 0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "CosineAnnealingLR":
            T_max = self.hyperparams.get("scheduler_T_max", epochs)
            eta_min = self.hyperparams.get("scheduler_eta_min", 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            T_0 = self.hyperparams.get("scheduler_T_0", epochs // 2 or 1)
            T_mult = self.hyperparams.get("scheduler_T_mult", 1)
            eta_min = self.hyperparams.get("scheduler_eta_min", 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        else:
            logger.warning("Unknown scheduler type '%s'. Falling back to StepLR.", scheduler_type)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))
        
        # 6. Set up logging and directories
        experiment_id = self.config.experiment_id if self.config else f"dl_run_{int(time.time())}"
        checkpoint_dir = Path(f"outputs/experiments/{experiment_id}/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = Path(f"outputs/experiments/{experiment_id}/tb_logs")
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        
        # 7. Training loop
        best_val_iou = -1.0
        patience_counter = 0
        training_history = []
        resume_epoch = 1
        
        state_file = checkpoint_dir / "training_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)
                resume_epoch = state_data.get("epoch", 1) + 1
                best_val_iou = state_data.get("best_validation_metric", -1.0)
                if (checkpoint_dir / "last_model.pt").exists():
                    chk = torch.load(checkpoint_dir / "last_model.pt", map_location=self.device)
                    self.model.load_state_dict(chk["model_state_dict"])
                if (checkpoint_dir / "optimizer_state.pt").exists():
                    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer_state.pt", map_location=self.device))
                if (checkpoint_dir / "scheduler_state.pt").exists():
                    scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler_state.pt", map_location=self.device))
                if (checkpoint_dir / "training_history.json").exists():
                    with open(checkpoint_dir / "training_history.json", "r") as f:
                        training_history = json.load(f)
                logger.info("Resuming training from epoch %d with best IoU %.4f", resume_epoch, best_val_iou)
            except Exception as e:
                logger.warning("Failed to load resume state, starting from scratch: %s", e)

        logger.info("Beginning training loop on device %s for %d epochs.", self.device, epochs)
        
        for epoch in range(resume_epoch, epochs + 1):
            self.model.train()
            train_loss = 0.0
            all_preds, all_labels = [], []
            
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    logits = self.model(x_batch)
                    if training_mode == "dense_segmentation":
                        loss = F.cross_entropy(logits, y_batch)
                    else:
                        H_out, W_out = logits.shape[2], logits.shape[3]
                        center_logits = logits[:, :, H_out // 2, W_out // 2]
                        loss = F.cross_entropy(center_logits, y_batch)
                    
                scaler.scale(loss).backward()
                
                # Gradient Norm Clipping
                grad_clip_norm = self.hyperparams.get("gradient_clip_norm", None)
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_clip_norm))
                    
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * x_batch.size(0)
                
                H_out, W_out = logits.shape[2], logits.shape[3]
                center_logits = logits[:, :, H_out // 2, W_out // 2]
                preds = center_logits.argmax(dim=1)
                
                if training_mode == "dense_segmentation":
                    y_center = y_batch[:, H_out // 2, W_out // 2]
                else:
                    y_center = y_batch
                    
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_center.cpu().numpy())
                
            train_loss /= len(train_dataset)
            scheduler.step()
            
            # Compute train metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
            train_iou = jaccard_score(all_labels, all_preds, average='binary', zero_division=0)
            
            # Log train metrics to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("IoU/train", train_iou, epoch)
            writer.add_scalar("F1/train", train_f1, epoch)
            writer.add_scalar("Precision/train", train_prec, epoch)
            writer.add_scalar("Recall/train", train_rec, epoch)
            
            # Validation Step
            val_loss, val_iou, val_f1, val_prec, val_rec = 0.0, 0.0, 0.0, 0.0, 0.0
            if val_loader is not None:
                self.model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                            logits = self.model(x_batch)
                            if training_mode == "dense_segmentation":
                                loss = F.cross_entropy(logits, y_batch)
                            else:
                                H_out, W_out = logits.shape[2], logits.shape[3]
                                center_logits = logits[:, :, H_out // 2, W_out // 2]
                                loss = F.cross_entropy(center_logits, y_batch)
                            
                        val_loss += loss.item() * x_batch.size(0)
                        
                        H_out, W_out = logits.shape[2], logits.shape[3]
                        center_logits = logits[:, :, H_out // 2, W_out // 2]
                        preds = center_logits.argmax(dim=1)
                        
                        if training_mode == "dense_segmentation":
                            y_center = y_batch[:, H_out // 2, W_out // 2]
                        else:
                            y_center = y_batch
                            
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(y_center.cpu().numpy())
                        
                val_loss /= len(val_dataset)
                val_preds = np.array(val_preds)
                val_labels = np.array(val_labels)
                val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=0)
                val_iou = jaccard_score(val_labels, val_preds, average='binary', zero_division=0)
                
                # Log val metrics to TensorBoard
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("IoU/val", val_iou, epoch)
                writer.add_scalar("F1/val", val_f1, epoch)
                writer.add_scalar("Precision/val", val_prec, epoch)
                writer.add_scalar("Recall/val", val_rec, epoch)
                
            logger.info("Epoch %d/%d - Train Loss: %.4f, IoU: %.4f | Val Loss: %.4f, IoU: %.4f",
                        epoch, epochs, train_loss, train_iou, val_loss, val_iou)
            
            history_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_f1": val_f1
            }
            training_history.append(history_entry)
            
            # Checkpoint saving & Early Stopping
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                patience_counter = 0
                logger.info("Validation IoU improved to %.4f. Saving best checkpoint...", best_val_iou)
                checkpoint_data = {
                    "model_state_dict": self.model.state_dict(),
                    "in_channels": in_ch,
                    "dataset_id": self.dataset_id,
                    "patch_size": self.patch_size,
                    "feature_names": self.feature_names,
                    "backbone": self.hyperparams.get("backbone", "lightweight")
                }
                torch.save(checkpoint_data, checkpoint_dir / "best_model.pt")
            else:
                patience_counter += 1

            # Save state manifests at each epoch for failure recovery
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "in_channels": in_ch,
                "dataset_id": self.dataset_id,
                "patch_size": self.patch_size,
                "feature_names": self.feature_names,
                "backbone": self.hyperparams.get("backbone", "lightweight")
            }, checkpoint_dir / "last_model.pt")
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pt")
            torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler_state.pt")
            
            from geoai.experiments.provenance import get_git_commit
            with open(checkpoint_dir / "training_state.json", "w", encoding="utf-8") as f:
                json.dump({
                    "epoch": epoch,
                    "optimizer_state_path": str(checkpoint_dir / "optimizer_state.pt"),
                    "scheduler_state_path": str(checkpoint_dir / "scheduler_state.pt"),
                    "best_validation_metric": best_val_iou,
                    "best_checkpoint_path": str(checkpoint_dir / "best_model.pt"),
                    "random_seed": seed,
                    "dataset_id": self.dataset_id or "unknown",
                    "git_commit": get_git_commit(),
                    "experiment_id": experiment_id
                }, f, indent=2)

            from geoai.experiments.provenance import collect_provenance_metadata
            prov = collect_provenance_metadata(str(checkpoint_dir / "training_state.json"))
            
            # Extend provenance with comprehensive experiment parameters
            import hashlib
            prov["benchmark_protocol_version"] = "v1.0-freeze"
            prov["dataset_version"] = "v1"
            prov["dataset_hash"] = hashlib.sha256(self.feature_cube.tobytes()).hexdigest() if self.feature_cube is not None else "unknown"
            prov["preprocessing_version"] = "v1"
            prov["backbone"] = self.hyperparams.get("backbone", "lightweight")
            prov["optimizer"] = "Adam"
            prov["scheduler"] = scheduler_type
            prov["learning_rate"] = lr
            prov["epochs"] = epochs
            prov["batch_size"] = batch_size
            prov["patch_size"] = self.patch_size
            prov["random_seed"] = seed
            prov["training_mode"] = training_mode
            prov["gradient_clip_norm"] = grad_clip_norm
            prov["augmentation_pipeline"] = augment_config
            prov["loss_function"] = "CrossEntropyLoss"
            
            with open(checkpoint_dir / "run_manifest.json", "w", encoding="utf-8") as f:
                json.dump(prov, f, indent=2)
                
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without val IoU improvement.", patience)
                break


        # Save last checkpoint
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "in_channels": in_ch,
            "dataset_id": self.dataset_id,
            "patch_size": self.patch_size,
            "feature_names": self.feature_names
        }
        torch.save(checkpoint_data, checkpoint_dir / "last_model.pt")
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pt")
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler_state.pt")
        
        with open(checkpoint_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2)
            
        writer.close()
        logger.info("Model fitting complete.")
        
        # Load best weights into memory for inference
        self.load(checkpoint_dir / "best_model.pt")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict change / no change labels on valid pixels using fast single-pass FCN inference."""
        self._ensure_coordinate_mapping(X)
        coords = self._map_X_to_coords(X)
        
        self.model.eval()
        with torch.no_grad():
            feat_cube_clean = np.nan_to_num(self.feature_cube, nan=0.0)
            x_full = torch.from_numpy(feat_cube_clean).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                logits_full = self.model(x_full)
                
            rows = torch.from_numpy(coords[:, 0]).to(self.device)
            cols = torch.from_numpy(coords[:, 1]).to(self.device)
            logits_at_coords = logits_full[0, :, rows, cols].permute(1, 0)
            
            preds = logits_at_coords.argmax(dim=1).cpu().numpy()
            
        return preds.astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities on valid pixels using fast single-pass FCN inference."""
        self._ensure_coordinate_mapping(X)
        coords = self._map_X_to_coords(X)
        
        self.model.eval()
        with torch.no_grad():
            feat_cube_clean = np.nan_to_num(self.feature_cube, nan=0.0)
            x_full = torch.from_numpy(feat_cube_clean).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                logits_full = self.model(x_full)
                
            rows = torch.from_numpy(coords[:, 0]).to(self.device)
            cols = torch.from_numpy(coords[:, 1]).to(self.device)
            logits_at_coords = logits_full[0, :, rows, cols].permute(1, 0)
            
            probs = F.softmax(logits_at_coords, dim=1).cpu().numpy()
            
        return probs.astype(np.float32)

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_type=f"Deep Learning CNN ({self.model_id})",
            expected_input_channels=self.feature_names,
            training_status="Implemented / Milestone 10 Baseline",
            requirements=["torch", "torchvision", "tensorboard"]
        )

# Concrete implementations

class FCEFModel(DLBaseModelWrapper):
    def __init__(self):
        super().__init__("fc_ef", FC_EF)

class FCSiamConcModel(DLBaseModelWrapper):
    def __init__(self):
        super().__init__("fc_siam_conc", FC_Siam_Conc)

class FCSiamDiffModel(DLBaseModelWrapper):
    def __init__(self):
        super().__init__("fc_siam_diff", FC_Siam_Diff)

class LightweightSiamCNNModel(DLBaseModelWrapper):
    def __init__(self):
        super().__init__("lightweight_siam_cnn", Lightweight_Siam_CNN)

def get_dl_model_wrapper(model_id: str, hyperparams: Optional[dict] = None, config: Optional[Any] = None) -> DLBaseModelWrapper:
    """Instantiate the PyTorch model wrapper and set its hyperparameters."""
    if model_id == "fc_ef":
        wrapper = FCEFModel()
    elif model_id == "fc_siam_conc":
        wrapper = FCSiamConcModel()
    elif model_id == "fc_siam_diff":
        wrapper = FCSiamDiffModel()
    elif model_id == "lightweight_siam_cnn":
        wrapper = LightweightSiamCNNModel()
    else:
        raise ValueError(f"Unknown deep learning model_id: {model_id}")
        
    wrapper.set_dataset_info(
        dataset_id=config.dataset_id if config else "ps10_sentinel_v1",
        config=config
    )
    return wrapper
