import torch
import torch.nn as nn

class SpatialExchange(nn.Module):
    """
    Spatial Exchange module.
    Exchanges a portion of features between two branches
    in the spatial dimension (columns/pixels).
    """
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w, device=x1.device) % self.p == 0
        out_x1 = x1.clone()
        out_x2 = x2.clone()
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]
        return out_x1, out_x2

class ChannelExchange(nn.Module):
    """
    Channel Exchange module.
    Exchanges a portion of features between two branches
    along the channel dimension.
    """
    def __init__(self, exchange_ratio: float = 0.5):
        super().__init__()
        self.exchange_ratio = exchange_ratio

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        batch, channels, h, w = x1.shape
        exchange_channels = int(channels * self.exchange_ratio)
        out1 = x1.clone()
        out2 = x2.clone()
        out1[:, :exchange_channels] = x2[:, :exchange_channels]
        out2[:, :exchange_channels] = x1[:, :exchange_channels]
        return out1, out2
