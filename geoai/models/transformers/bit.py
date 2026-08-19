import torch
import torch.nn as nn
import torch.nn.functional as F
from geoai.utils.channel_layout import TOTAL_CHANNELS, BITEMPORAL_CHANNELS, T1_SLICE, T2_SLICE
import numpy as np
from geoai.models.transformers.base import TransformerBaseWrapper
from geoai.models.base import ModelCapabilities

class SpatialAttentionTokenizer(nn.Module):
    """
    Spatial Attention Tokenizer that extracts semantic visual tokens from feature maps.
    As proposed in "Remote Sensing Image Change Detection with Transformers" (BIT).
    """
    def __init__(self, in_ch: int, token_len: int = 4):
        super().__init__()
        self.token_len = token_len
        self.conv = nn.Conv2d(in_ch, token_len, kernel_size=1)
        self.last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        # Compute spatial attention map of shape (B, token_len, H*W)
        attn_map = self.conv(x).view(B, self.token_len, H * W)
        attn_map = F.softmax(attn_map, dim=-1) # (B, L, HW)
        self.last_attn = attn_map.detach()
        
        # Flatten input: (B, C, HW)
        x_flat = x.view(B, C, H * W)
        
        # Matrix multiplication: (B, L, HW) x (B, HW, C) -> (B, L, C)
        tokens = torch.bmm(attn_map, x_flat.transpose(1, 2))
        return tokens

class BITArch(nn.Module):
    """Bitemporal Image Transformer (BIT) change detection architecture."""
    def __init__(self, in_channels: int = 18, out_channels: int = 2, token_len: int = 4, embed_dim: int = 32, backbone: str = "lightweight"):
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
            # ResNet layer1 outputs 64 channels
            self.embed_dim = 64
        else:
            self.conv1 = nn.Conv2d(7, embed_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(7, embed_dim, kernel_size=3, padding=1)
            self.embed_dim = embed_dim
            
        # Tokenizers
        self.tokenizer1 = SpatialAttentionTokenizer(self.embed_dim, token_len=token_len)
        self.tokenizer2 = SpatialAttentionTokenizer(self.embed_dim, token_len=token_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=2, dim_feedforward=self.embed_dim * 2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim, nhead=2, dim_feedforward=self.embed_dim * 2, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # Classifier
        self.classifier = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)
        self.token_len = token_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centralized routing check
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        # Value assertions
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        
        # 1. CNN Feature extraction
        if self.backbone_type == "resnet18":
            f1 = self.resnet(t1)
            f2 = self.resnet(t2)
        else:
            f1 = F.relu(self.conv1(t1))
            f2 = F.relu(self.conv2(t2))
            
        B, C, H, W = f1.size()
        
        # 2. Tokenization
        tokens1 = self.tokenizer1(f1) # (B, L, C)
        tokens2 = self.tokenizer2(f2) # (B, L, C)
        
        # Concatenate tokens of t1 and t2 along sequence dimension -> shape (B, 2L, C)
        tokens = torch.cat([tokens1, tokens2], dim=1)
        
        # 3. Transformer Encoder for spatio-temporal modeling
        tokens_refined = self.transformer_encoder(tokens) # (B, 2L, C)
        
        tokens_refined_1 = tokens_refined[:, :self.token_len, :]
        tokens_refined_2 = tokens_refined[:, self.token_len:, :]
        
        # 4. Transformer Decoder
        f1_flat = f1.view(B, C, H * W).transpose(1, 2) # (B, HW, C)
        f2_flat = f2.view(B, C, H * W).transpose(1, 2) # (B, HW, C)
        
        y1_flat = self.transformer_decoder(tgt=f1_flat, memory=tokens_refined_1) # (B, HW, C)
        y2_flat = self.transformer_decoder(tgt=f2_flat, memory=tokens_refined_2) # (B, HW, C)
        
        y1 = y1_flat.transpose(1, 2).view(B, C, H, W)
        y2 = y2_flat.transpose(1, 2).view(B, C, H, W)
        
        # 5. Differencing
        diff = torch.abs(y1 - y2)
        
        # 6. Classification Head
        return self.classifier(diff)

class BIT(TransformerBaseWrapper):
    """Wrapper class executing the BIT baseline."""
    
    def __init__(self, in_channels: int = 18, out_channels: int = 2, backbone: str = "lightweight"):
        super().__init__("bit", BITArch)
        
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
            attn1 = self.model.tokenizer1.last_attn
            attn2 = self.model.tokenizer2.last_attn
            if attn1 is not None and attn2 is not None:
                B, L, HW = attn1.size()
                H = W = int(np.sqrt(HW))
                # Average attention across both time points and tokens to get a spatial map
                mean_attn = (attn1 + attn2) / 2.0
                return mean_attn.mean(dim=1).view(B, 1, H, W)
        return torch.zeros((x1.size(0), 1, x1.size(2), x1.size(3))).to(self.device)

    def get_token_embeddings(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features as token representations."""
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
