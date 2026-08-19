import torch
import torch.nn as nn
import torch.nn.functional as F
from geoai.utils.channel_layout import TOTAL_CHANNELS, BITEMPORAL_CHANNELS, T1_SLICE, T2_SLICE
import numpy as np
from geoai.models.transformers.base import TransformerBaseWrapper
from geoai.models.base import ModelCapabilities
from geoai.models.transformers.modules.attention import SpatialAttention
from geoai.models.transformers.modules.interaction import SpatialExchange, ChannelExchange

class ChangerArch(nn.Module):
    """
    Changer change detection network utilizing SpatialExchange and ChannelExchange.
    
    Architectural Adaptations & Deviations:
    - Official Changer: Employs deep hierarchical backbones (ResNet-18 or MiT) with multi-scale
      decoders designed for full 512x512 images.
    - Platform Adaptation: Standardized as a lightweight patch-based model operating over
      15x15 pixel neighborhoods, maintaining compatibility with the low-memory CPU constraints of
      the GeoAI training/evaluation pipeline.
    - Information Fusion: Utilizes a custom Siamese CNN encoder interspersed with SpatialExchange
      (spatial width swaps) and ChannelExchange (channel subset swaps), followed by a SpatialAttention
      neck to support explainability heatmaps, and a 1x1 Conv classification head.
    """
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
            self.spatial_exchange = SpatialExchange(p=2)
            self.channel_exchange1 = ChannelExchange(exchange_ratio=0.5)
            self.channel_exchange2 = ChannelExchange(exchange_ratio=0.5)
            self.attn = SpatialAttention(256)
            self.classifier = nn.Conv2d(256, out_channels, kernel_size=1)
        else:
            self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
            self.spatial_exchange = SpatialExchange(p=2)
            
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.channel_exchange1 = ChannelExchange(exchange_ratio=0.5)
            
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.channel_exchange2 = ChannelExchange(exchange_ratio=0.5)
            
            self.attn = SpatialAttention(128)
            self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centralized routing check
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        # Value assertions
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        
        if self.backbone_type == "resnet18":
            # Stage 1: resnet.conv1 + bn1 + relu + maxpool
            f1_1 = self.resnet.resnet.conv1(t1)
            f1_1 = self.resnet.resnet.bn1(f1_1)
            f1_1 = self.resnet.resnet.relu(f1_1)
            f1_1 = self.resnet.resnet.maxpool(f1_1)
            
            f2_1 = self.resnet.resnet.conv1(t2)
            f2_1 = self.resnet.resnet.bn1(f2_1)
            f2_1 = self.resnet.resnet.relu(f2_1)
            f2_1 = self.resnet.resnet.maxpool(f2_1)
            
            f1_1_ex, f2_1_ex = self.spatial_exchange(f1_1, f2_1)
            
            # Stage 2: resnet.layer1
            f1_2 = self.resnet.resnet.layer1(f1_1_ex)
            f2_2 = self.resnet.resnet.layer1(f2_1_ex)
            f1_2_ex, f2_2_ex = self.channel_exchange1(f1_2, f2_2)
            
            # Stage 3: resnet.layer2
            f1_3 = self.resnet.resnet.layer2(f1_2_ex)
            f2_3 = self.resnet.resnet.layer2(f2_2_ex)
            f1_3_ex, f2_3_ex = self.channel_exchange2(f1_3, f2_3)
            
            # Upsample layer2 features back to 15x15
            f1_3_ex = F.interpolate(f1_3_ex, size=(15, 15), mode='bilinear', align_corners=False)
            f2_3_ex = F.interpolate(f2_3_ex, size=(15, 15), mode='bilinear', align_corners=False)
        else:
            f1_1 = F.relu(self.conv1(t1))
            f2_1 = F.relu(self.conv2(t2))
            f1_1_ex, f2_1_ex = self.spatial_exchange(f1_1, f2_1)
            
            f1_2 = F.relu(self.conv3(f1_1_ex))
            f2_2 = F.relu(self.conv4(f2_1_ex))
            f1_2_ex, f2_2_ex = self.channel_exchange1(f1_2, f2_2)
            
            f1_3 = F.relu(self.conv5(f1_2_ex))
            f2_3 = F.relu(self.conv6(f2_2_ex))
            f1_3_ex, f2_3_ex = self.channel_exchange2(f1_3, f2_3)
            
        feat = torch.cat([f1_3_ex, f2_3_ex], dim=1)
        feat_attn = self.attn(feat)
        return self.classifier(feat_attn)

class Changer(TransformerBaseWrapper):
    """Wrapper class executing the Changer baseline."""
    
    def __init__(self, in_channels: int = 18, out_channels: int = 2, backbone: str = "lightweight"):
        super().__init__("changer", ChangerArch)
        
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
                f1_1 = self.model.resnet.resnet.conv1(x1)
                f1_1 = self.model.resnet.resnet.bn1(f1_1)
                f1_1 = self.model.resnet.resnet.relu(f1_1)
                f1_1 = self.model.resnet.resnet.maxpool(f1_1)
                
                f2_1 = self.model.resnet.resnet.conv1(x2)
                f2_1 = self.model.resnet.resnet.bn1(f2_1)
                f2_1 = self.model.resnet.resnet.relu(f2_1)
                f2_1 = self.model.resnet.resnet.maxpool(f2_1)
                
                f1_1_ex, f2_1_ex = self.model.spatial_exchange(f1_1, f2_1)
                
                f1_2 = self.model.resnet.resnet.layer1(f1_1_ex)
                f2_2 = self.model.resnet.resnet.layer1(f2_1_ex)
                f1_2_ex, f2_2_ex = self.model.channel_exchange1(f1_2, f2_2)
                
                f1_3 = self.model.resnet.resnet.layer2(f1_2_ex)
                f2_3 = self.model.resnet.resnet.layer2(f2_2_ex)
                f1_3_ex, f2_3_ex = self.model.channel_exchange2(f1_3, f2_3)
                
                f1_3_ex = F.interpolate(f1_3_ex, size=(15, 15), mode='bilinear', align_corners=False)
                f2_3_ex = F.interpolate(f2_3_ex, size=(15, 15), mode='bilinear', align_corners=False)
                return torch.cat([f1_3_ex, f2_3_ex], dim=1)
            else:
                f1_1 = F.relu(self.model.conv1(x1))
                f2_1 = F.relu(self.model.conv2(x2))
                f1_1_ex, f2_1_ex = self.model.spatial_exchange(f1_1, f2_1)
                
                f1_2 = F.relu(self.model.conv3(f1_1_ex))
                f2_2 = F.relu(self.model.conv4(f2_1_ex))
                f1_2_ex, f2_2_ex = self.model.channel_exchange1(f1_2, f2_2)
                
                f1_3 = F.relu(self.model.conv5(f1_2_ex))
                f2_3 = F.relu(self.model.conv6(f2_2_ex))
                f1_3_ex, f2_3_ex = self.model.channel_exchange2(f1_3, f2_3)
                return torch.cat([f1_3_ex, f2_3_ex], dim=1)

