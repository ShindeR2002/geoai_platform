from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geoai.models.baselines.dl_wrapper import DLBaseModelWrapper

class TransformerBaseWrapper(DLBaseModelWrapper, ABC):
    """Canonical abstract base class interface for all Transformer and Foundation models."""

    @abstractmethod
    def get_attention_maps(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Intercept Query-Key attention matrices and return a normalized 
        spatial attention heatmap tensor.
        """
        pass

    @abstractmethod
    def get_token_embeddings(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Extract token embeddings prior to the classification head 
        for PCA/UMAP explainability projections.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict change / no change labels on valid pixels using batch patch-based inference to prevent OOM."""
        self._ensure_coordinate_mapping(X)
        coords = self._map_X_to_coords(X)
        
        self.model.eval()
        
        from geoai.models.baselines.dl_wrapper import SpatialPatchDataset
        from torch.utils.data import DataLoader
        
        dataset = SpatialPatchDataset(
            feature_cube=self.feature_cube,
            labels=self.labels_2d,
            coords=coords,
            patch_size=self.patch_size,
            augment=False
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        
        preds_list = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    logits = self.model(x_batch)
                    H_out, W_out = logits.shape[2], logits.shape[3]
                    center_logits = logits[:, :, H_out // 2, W_out // 2]
                preds = center_logits.argmax(dim=1)
                preds_list.extend(preds.cpu().numpy())
                
        return np.array(preds_list, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities on valid pixels using batch patch-based inference to prevent OOM."""
        self._ensure_coordinate_mapping(X)
        coords = self._map_X_to_coords(X)
        
        self.model.eval()
        
        from geoai.models.baselines.dl_wrapper import SpatialPatchDataset
        from torch.utils.data import DataLoader
        
        dataset = SpatialPatchDataset(
            feature_cube=self.feature_cube,
            labels=self.labels_2d,
            coords=coords,
            patch_size=self.patch_size,
            augment=False
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        
        probs_list = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    logits = self.model(x_batch)
                    H_out, W_out = logits.shape[2], logits.shape[3]
                    center_logits = logits[:, :, H_out // 2, W_out // 2]
                probs = F.softmax(center_logits, dim=1)
                probs_list.extend(probs.cpu().numpy())
                
        return np.array(probs_list, dtype=np.float32)
