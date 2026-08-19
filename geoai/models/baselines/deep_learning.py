import torch
import torch.nn as nn
import torch.nn.functional as F
from geoai.utils.channel_layout import TOTAL_CHANNELS, BITEMPORAL_CHANNELS, T1_SLICE, T2_SLICE

class DoubleConv(nn.Module):
    """Helper module with double convolution, batch normalization, and ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class FC_EF(nn.Module):
    """Fully Convolutional Early Fusion Change Detection Network."""
    def __init__(self, in_channels: int = 18, out_channels: int = 2):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up1 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_up2 = DoubleConv(32, 16)
        self.outc = nn.Conv2d(16, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        u1 = self.up1(x3)
        diffY = x2.size()[2] - u1.size()[2]
        diffX = x2.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        u1 = torch.cat([x2, u1], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        diffY = x1.size()[2] - u2.size()[2]
        diffX = x1.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        u2 = torch.cat([x1, u2], dim=1)
        u2 = self.conv_up2(u2)
        
        return self.outc(u2)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return F.adaptive_avg_pool2d(x3, (1, 1)).view(x3.size(0), -1)

class FC_Siam_Conc(nn.Module):
    """Fully Convolutional Siamese Concatenation Change Detection Network."""
    def __init__(self, in_channels: int = 18, out_channels: int = 2):
        super().__init__()
        self.encoder = nn.ModuleDict({
            'inc': DoubleConv(7, 16),
            'down1': nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32)),
            'down2': nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        })
        
        self.up1 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        self.conv_up1 = DoubleConv(96, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_up2 = DoubleConv(48, 16)
        self.outc = nn.Conv2d(16, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centralized routing check
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        # Value assertions
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        
        x1_t1 = self.encoder['inc'](t1)
        x2_t1 = self.encoder['down1'](x1_t1)
        x3_t1 = self.encoder['down2'](x2_t1)
        
        x1_t2 = self.encoder['inc'](t2)
        x2_t2 = self.encoder['down1'](x1_t2)
        x3_t2 = self.encoder['down2'](x2_t2)
        
        bottleneck = torch.cat([x3_t1, x3_t2], dim=1)
        
        u1 = self.up1(bottleneck)
        x2_concat = torch.cat([x2_t1, x2_t2], dim=1)
        diffY = x2_concat.size()[2] - u1.size()[2]
        diffX = x2_concat.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        u1 = torch.cat([x2_concat, u1], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        x1_concat = torch.cat([x1_t1, x1_t2], dim=1)
        diffY = x1_concat.size()[2] - u2.size()[2]
        diffX = x1_concat.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        u2 = torch.cat([x1_concat, u2], dim=1)
        u2 = self.conv_up2(u2)
        
        return self.outc(u2)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        x3_t1 = self.encoder['down2'](self.encoder['down1'](self.encoder['inc'](t1)))
        x3_t2 = self.encoder['down2'](self.encoder['down1'](self.encoder['inc'](t2)))
        bottleneck = torch.cat([x3_t1, x3_t2], dim=1)
        return F.adaptive_avg_pool2d(bottleneck, (1, 1)).view(bottleneck.size(0), -1)

class FC_Siam_Diff(nn.Module):
    """Fully Convolutional Siamese Difference Change Detection Network."""
    def __init__(self, in_channels: int = 18, out_channels: int = 2):
        super().__init__()
        self.encoder = nn.ModuleDict({
            'inc': DoubleConv(7, 16),
            'down1': nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32)),
            'down2': nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        })
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up1 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_up2 = DoubleConv(32, 16)
        self.outc = nn.Conv2d(16, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centralized routing check
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        # Value assertions
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        
        x1_t1 = self.encoder['inc'](t1)
        x2_t1 = self.encoder['down1'](x1_t1)
        x3_t1 = self.encoder['down2'](x2_t1)
        
        x1_t2 = self.encoder['inc'](t2)
        x2_t2 = self.encoder['down1'](x1_t2)
        x3_t2 = self.encoder['down2'](x2_t2)
        
        bottleneck = torch.abs(x3_t1 - x3_t2)
        
        u1 = self.up1(bottleneck)
        x2_diff = torch.abs(x2_t1 - x2_t2)
        diffY = x2_diff.size()[2] - u1.size()[2]
        diffX = x2_diff.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        u1 = torch.cat([x2_diff, u1], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        x1_diff = torch.abs(x1_t1 - x1_t2)
        diffY = x1_diff.size()[2] - u2.size()[2]
        diffX = x1_diff.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        u2 = torch.cat([x1_diff, u2], dim=1)
        u2 = self.conv_up2(u2)
        
        return self.outc(u2)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        x3_t1 = self.encoder['down2'](self.encoder['down1'](self.encoder['inc'](t1)))
        x3_t2 = self.encoder['down2'](self.encoder['down1'](self.encoder['inc'](t2)))
        bottleneck = torch.abs(x3_t1 - x3_t2)
        return F.adaptive_avg_pool2d(bottleneck, (1, 1)).view(bottleneck.size(0), -1)

class Lightweight_Siam_CNN(nn.Module):
    """Lightweight Siamese CNN change detection network."""
    def __init__(self, in_channels: int = 18, out_channels: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centralized routing check
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        # Value assertions
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        
        feat_t1 = self.encoder(t1)
        feat_t2 = self.encoder(t2)
        
        diff = torch.abs(feat_t1 - feat_t2)
        return self.decoder(diff)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] in (TOTAL_CHANNELS, BITEMPORAL_CHANNELS), f"Expected {TOTAL_CHANNELS} or {BITEMPORAL_CHANNELS} channels, got {x.shape[1]}"
        t1 = x[:, T1_SLICE]
        t2 = x[:, T2_SLICE]
        assert torch.equal(t1, x[:, 0:7]), "T1 slice does not match raw indices 0:7 values"
        assert torch.equal(t2, x[:, 7:14]), "T2 slice does not match raw indices 7:14 values"
        feat_t1 = self.encoder(t1)
        feat_t2 = self.encoder(t2)
        diff = torch.abs(feat_t1 - feat_t2)
        return F.adaptive_avg_pool2d(diff, (1, 1)).view(diff.size(0), -1)
