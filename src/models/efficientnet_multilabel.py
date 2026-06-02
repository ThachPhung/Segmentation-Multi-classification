"""EfficientNet-B3 multi-label classifier (Stage 2 — phân loại RoI)."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models

try:
    from torchvision.models import EfficientNet_B3_Weights
except ImportError:
    EfficientNet_B3_Weights = None


class EfficientNetB3MultiLabel(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = False, dropout: float = 0.35):
        super().__init__()
        if EfficientNet_B3_Weights is not None and pretrained:
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1
            backbone = models.efficientnet_b3(weights=weights)
        else:
            backbone = models.efficientnet_b3(pretrained=pretrained)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
