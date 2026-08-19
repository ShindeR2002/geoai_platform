import torch
import torch.nn as nn
import torch.nn.functional as F
from geoai.utils.channel_layout import TOTAL_CHANNELS, BITEMPORAL_CHANNELS, T1_SLICE, T2_SLICE
import numpy as np
from geoai.models.transformers.base import TransformerBaseWrapper
from geoai.models.base import ModelCapabilities
from geoai.models.transformers.modules.attention import SpatialAttention


class TinyCDArch(nn.Module):
    """TinyCD lightweight change detection network utilizing multi-scale correlation maps."""
    def __init__(self, in_channels: int = 18, out_channels: int = 2, backbone: str = "lightweight"):
        super().__init__()
        self.backbone_type = backbone
        
        if self.backbone_type == "resnet18":
            import logging
            logging.getLogger(__name__).warning(
                "Scientific Receptive Field Warning: ResNet-18 is configured on 15x15 patches. "
                "The standard ResNet-18 receptive field size (>50x50) exceeds the patch bounds, "
                "which causes border truncation and aliasing. A minimum patch size of 64x64 is recommended."
            )
            from geoai.models.baselines.dl_wrapper import ResNet18SiameseAdapter
            self.resnet = ResNet18SiameseAdapter(in_channels=7)
            self.attn = SpatialAttention(128)
            self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)
        else:
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
        if self.backbone_type == "resnet18":
            f1 = self.resnet(t1)
            f2 = self.resnet(t2)
        else:
            f1 = F.relu(self.conv1(t1))
            f2 = F.relu(self.conv2(t2))
        concat = torch.cat([f1, f2], dim=1)
        feat = self.attn(concat)
        return self.classifier(feat)

class TinyCD(TransformerBaseWrapper):
    """Wrapper class executing the TinyCD baseline."""
    
    def __init__(self, in_channels: int = 18, out_channels: int = 2, backbone: str = "lightweight"):
        # Store architecture class directly without instantiation inside constructor
        super().__init__("tinycd", TinyCDArch)
        
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
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            if self.model.backbone_type == "resnet18":
                f1 = self.model.resnet(x1)
                f2 = self.model.resnet(x2)
            else:
                f1 = F.relu(self.model.conv1(x1))
                f2 = F.relu(self.model.conv2(x2))
            return torch.cat([f1, f2], dim=1)

