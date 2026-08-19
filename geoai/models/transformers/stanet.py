import torch
import torch.nn as nn
import torch.nn.functional as F
from geoai.utils.channel_layout import TOTAL_CHANNELS, BITEMPORAL_CHANNELS, T1_SLICE, T2_SLICE
import numpy as np
from geoai.models.transformers.base import TransformerBaseWrapper
from geoai.models.base import ModelCapabilities
from geoai.models.transformers.changeformer import SpatialAttention

class STANetArch(nn.Module):
    """STANet Siamese architecture with Spatial-Temporal Attention module."""
    def __init__(self, in_channels: int = 18, out_channels: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.attn = SpatialAttention(64)
        self.classifier = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centralized routing check
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        # Value assertions
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        f1 = F.relu(self.conv1(t1))
        f2 = F.relu(self.conv2(t2))
        concat = torch.cat([f1, f2], dim=1)
        feat = self.attn(concat)
        return self.classifier(feat)

class STANet(TransformerBaseWrapper):
    """Wrapper class executing the STANet baseline."""
    
    def __init__(self, in_channels: int = 18, out_channels: int = 2):
        super().__init__("stanet", STANetArch)
        
    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_type="Transformer",
            expected_input_channels=["Red_2021", "Green_2021", "Blue_2021", "SAR_2021", "NDVI_2021", "NDBI_2021", "NDWI_2021",
                                     "Red_2024", "Green_2024", "Blue_2024", "SAR_2024", "NDVI_2024", "NDBI_2024", "NDWI_2024"],
            training_status="Pluggable",
            requirements=["torch", "torchvision"]
        )

    def get_attention_maps(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x = torch.cat([x1, x2], dim=1).to(self.device)
            _ = self.model(x)
            attn = self.model.attn.last_attn
            if attn is not None:
                B, HW, HW_ = attn.size()
                H = W = int(np.sqrt(HW))
                return attn.mean(dim=1).view(B, 1, H, W)
        return torch.zeros((x1.size(0), 1, x1.size(2), x1.size(3))).to(self.device)

    def get_token_embeddings(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            f1 = F.relu(self.model.conv1(x1))
            f2 = F.relu(self.model.conv2(x2))
            return torch.cat([f1, f2], dim=1)
