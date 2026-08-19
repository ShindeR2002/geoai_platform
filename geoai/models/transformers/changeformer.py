import torch
import torch.nn as nn
import torch.nn.functional as F
from geoai.utils.channel_layout import TOTAL_CHANNELS, BITEMPORAL_CHANNELS, T1_SLICE, T2_SLICE
import numpy as np
from geoai.models.transformers.base import TransformerBaseWrapper
from geoai.models.base import ModelCapabilities
from geoai.models.transformers.modules.attention import SpatialAttention

class ChangeFormerArch(nn.Module):
    """
    ChangeFormer hierarchical change detection network.
    
    Architectural Adaptations & Deviations:
    - Official ChangeFormer: Uses pure Mix-Transformer (MiT) encoders downsampling to factor 32
      followed by an MLP decoder, typically designed for 256x256 or 512x512 bitemporal scenes.
    - Platform Adaptation: Standardized as a hierarchical multi-scale model operating over
      arbitrary spatial dimensions, implementing absolute difference skip connections at each
      downsampling scale.
    - Dynamic Backbone: Supports both a lightweight CNN stage-downsampler (3 levels: scale 1x, 0.5x, 0.25x)
      and a pluggable ResNet-18 stage adapter.
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
            
            # Projection heads for ResNet levels (Stage 1: 64, Stage 2: 64, Stage 3: 128)
            self.proj1 = nn.Conv2d(64, 64, kernel_size=1)
            self.proj2 = nn.Conv2d(64, 64, kernel_size=1)
            self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
            
            # Decoder fusion channels: 3 stages * 64 projected channels = 192 channels
            self.attn = SpatialAttention(192)
            self.classifier = nn.Conv2d(192, out_channels, kernel_size=1)
        else:
            # Stage 1 Siamese blocks
            self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
            
            # Stage 2 Siamese blocks (stride 2 downsampling)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            
            # Stage 3 Siamese blocks (stride 2 downsampling)
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
            self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
            
            # Projections to common embedding width
            self.proj1 = nn.Conv2d(32, 32, kernel_size=1)
            self.proj2 = nn.Conv2d(64, 32, kernel_size=1)
            self.proj3 = nn.Conv2d(64, 32, kernel_size=1)
            
            # Decoder fusion channels: 3 stages * 32 projected channels = 96 channels
            self.attn = SpatialAttention(96)
            self.classifier = nn.Conv2d(96, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centralized routing check
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        # Value assertions
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        H_in, W_in = t1.shape[2], t1.shape[3]
        
        if self.backbone_type == "resnet18":
            # Extract features via early ResNet blocks
            # Stage 1: resnet.conv1 + bn1 + relu + maxpool
            f1_1 = self.resnet.resnet.conv1(t1)
            f1_1 = self.resnet.resnet.bn1(f1_1)
            f1_1 = self.resnet.resnet.relu(f1_1)
            f1_1 = self.resnet.resnet.maxpool(f1_1)
            
            f2_1 = self.resnet.resnet.conv1(t2)
            f2_1 = self.resnet.resnet.bn1(f2_1)
            f2_1 = self.resnet.resnet.relu(f2_1)
            f2_1 = self.resnet.resnet.maxpool(f2_1)
            
            # Stage 2: resnet.layer1
            f1_2 = self.resnet.resnet.layer1(f1_1)
            f2_2 = self.resnet.resnet.layer1(f2_1)
            
            # Stage 3: resnet.layer2
            f1_3 = self.resnet.resnet.layer2(f1_2)
            f2_3 = self.resnet.resnet.layer2(f2_2)
            
            # Calculate absolute diffs at each stage
            d1 = torch.abs(f1_1 - f2_1)
            d2 = torch.abs(f1_2 - f2_2)
            d3 = torch.abs(f1_3 - f2_3)
            
            # Project to embed dim
            p1 = self.proj1(d1)
            p2 = self.proj2(d2)
            p3 = self.proj3(d3)
            
            # Spatial upsampling back to input spatial dimensions
            u1 = F.interpolate(p1, size=(H_in, W_in), mode='bilinear', align_corners=False)
            u2 = F.interpolate(p2, size=(H_in, W_in), mode='bilinear', align_corners=False)
            u3 = F.interpolate(p3, size=(H_in, W_in), mode='bilinear', align_corners=False)
        else:
            # Stage 1 CNN feature extraction
            f1_1 = F.relu(self.conv1(t1))
            f2_1 = F.relu(self.conv2(t2))
            
            # Stage 2 CNN feature extraction
            f1_2 = F.relu(self.conv3(f1_1))
            f2_2 = F.relu(self.conv4(f2_1))
            
            # Stage 3 CNN feature extraction
            f1_3 = F.relu(self.conv5(f1_2))
            f2_3 = F.relu(self.conv6(f2_2))
            
            # Calculate absolute diffs at each stage
            d1 = torch.abs(f1_1 - f2_1)
            d2 = torch.abs(f1_2 - f2_2)
            d3 = torch.abs(f1_3 - f2_3)
            
            # Project to common embed dim
            p1 = self.proj1(d1)
            p2 = self.proj2(d2)
            p3 = self.proj3(d3)
            
            u1 = p1
            u2 = F.interpolate(p2, size=(H_in, W_in), mode='bilinear', align_corners=False)
            u3 = F.interpolate(p3, size=(H_in, W_in), mode='bilinear', align_corners=False)
            
        feat = torch.cat([u1, u2, u3], dim=1)
        feat_attn = self.attn(feat)
        return self.classifier(feat_attn)

class ChangeFormer(TransformerBaseWrapper):
    """Wrapper class executing the ChangeFormer baseline."""
    
    def __init__(self, in_channels: int = 18, out_channels: int = 2, backbone: str = "lightweight"):
        super().__init__("changeformer", ChangeFormerArch)
        
    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_type="Transformer",
            expected_input_channels=["Red_2021", "Green_2021", "Blue_2021", "SAR_2021", "NDVI_2021", "NDBI_2021", "NDWI_2021",
                                     "Red_2024", "Green_2024", "Blue_2024", "SAR_2024", "NDVI_2024", "NDBI_2024", "NDWI_2024"],
            training_status="Pluggable",
            requirements=["torch", "torchvision"]
        )

    def get_attention_maps(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Extract spatial Query-Key attention coefficients."""
        self.model.eval()
        with torch.no_grad():
            x = torch.cat([x1, x2], dim=1).to(self.device)
            _ = self.model(x)
            attn = self.model.attn.last_attn
            if attn is not None:
                B, HW, HW_ = attn.size()
                H = W = int(np.sqrt(HW))
                # Average over queries to produce a single spatial attention map
                return attn.mean(dim=1).view(B, 1, H, W)
        return torch.zeros((x1.size(0), 1, x1.size(2), x1.size(3))).to(self.device)

    def get_token_embeddings(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features as token representations."""
        self.model.eval()
        with torch.no_grad():
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            H_in, W_in = x1.shape[2], x1.shape[3]
            if self.model.backbone_type == "resnet18":
                f1_1 = self.model.resnet.resnet.conv1(x1)
                f1_1 = self.model.resnet.resnet.bn1(f1_1)
                f1_1 = self.model.resnet.resnet.relu(f1_1)
                f1_1 = self.model.resnet.resnet.maxpool(f1_1)
                
                f2_1 = self.model.resnet.resnet.conv1(x2)
                f2_1 = self.model.resnet.resnet.bn1(f2_1)
                f2_1 = self.model.resnet.resnet.relu(f2_1)
                f2_1 = self.model.resnet.resnet.maxpool(f2_1)
                
                f1_2 = self.model.resnet.resnet.layer1(f1_1)
                f2_2 = self.model.resnet.resnet.layer1(f2_1)
                
                f1_3 = self.model.resnet.resnet.layer2(f1_2)
                f2_3 = self.model.resnet.resnet.layer2(f2_2)
                
                d1 = torch.abs(f1_1 - f2_1)
                d2 = torch.abs(f1_2 - f2_2)
                d3 = torch.abs(f1_3 - f2_3)
                
                p1 = self.model.proj1(d1)
                p2 = self.model.proj2(d2)
                p3 = self.model.proj3(d3)
                
                u1 = F.interpolate(p1, size=(H_in, W_in), mode='bilinear', align_corners=False)
                u2 = F.interpolate(p2, size=(H_in, W_in), mode='bilinear', align_corners=False)
                u3 = F.interpolate(p3, size=(H_in, W_in), mode='bilinear', align_corners=False)
            else:
                f1_1 = F.relu(self.model.conv1(x1))
                f2_1 = F.relu(self.model.conv2(x2))
                
                f1_2 = F.relu(self.model.conv3(f1_1))
                f2_2 = F.relu(self.model.conv4(f2_1))
                
                f1_3 = F.relu(self.model.conv5(f1_2))
                f2_3 = F.relu(self.model.conv6(f2_2))
                
                d1 = torch.abs(f1_1 - f2_1)
                d2 = torch.abs(f1_2 - f2_2)
                d3 = torch.abs(f1_3 - f2_3)
                
                p1 = self.model.proj1(d1)
                p2 = self.model.proj2(d2)
                p3 = self.model.proj3(d3)
                
                u1 = p1
                u2 = F.interpolate(p2, size=(H_in, W_in), mode='bilinear', align_corners=False)
                u3 = F.interpolate(p3, size=(H_in, W_in), mode='bilinear', align_corners=False)
                
            return torch.cat([u1, u2, u3], dim=1)
