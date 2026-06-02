"""Attention U-Net với encoder EfficientNet-B3 (Stage 1 — phân đoạn)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    from torchvision.models import EfficientNet_B3_Weights
except ImportError:
    EfficientNet_B3_Weights = None


def create_efficientnet_b3(pretrained: bool = True) -> nn.Module:
    if EfficientNet_B3_Weights is not None:
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        return models.efficientnet_b3(weights=weights)
    return models.efficientnet_b3(pretrained=pretrained)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate cho skip connection (g: decoder, x: encoder skip)."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        attn = self.psi(self.relu(g1 + x1))
        return x * attn


class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        inter_ch = max(skip_ch // 2, 16)
        self.attention = AttentionGate(in_ch, skip_ch, inter_ch)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip = self.attention(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class AttentionUNetEfficientNetB3(nn.Module):
    """Phân đoạn đa lớp (4 kênh logits) cho Severstal."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        base = create_efficientnet_b3(pretrained=pretrained)
        features = base.features

        self.stem = features[0]
        self.enc1 = features[1]
        self.enc2 = features[2]
        self.enc3 = features[3]
        self.enc4 = nn.Sequential(features[4], features[5])
        self.enc5 = nn.Sequential(features[6], features[7], features[8])

        self.center = ConvBlock(1536, 512, dropout=dropout)
        self.dec4 = AttentionDecoderBlock(512, 136, 256, dropout=dropout)
        self.dec3 = AttentionDecoderBlock(256, 48, 128, dropout=dropout)
        self.dec2 = AttentionDecoderBlock(128, 32, 64, dropout=dropout)
        self.dec1 = AttentionDecoderBlock(64, 24, 64, dropout=dropout)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        center = self.center(x5)
        d4 = self.dec4(center, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        out = self.final(d1)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
