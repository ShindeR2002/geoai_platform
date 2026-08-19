import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """Functional Spatial Attention Module storing Query-Key weights maps."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.query = nn.Conv2d(in_ch, max(1, in_ch // 2), 1)
        self.key = nn.Conv2d(in_ch, max(1, in_ch // 2), 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        self.last_attn = attn.detach()
        
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x
