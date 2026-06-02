"""Classifier multi-label Stage 2 — EfficientNet-B3 / ResNet50."""

from __future__ import annotations

import torch.nn as nn
import torchvision
import torchvision.models as models

try:
    from torchvision.models import EfficientNet_B3_Weights
except ImportError:
    EfficientNet_B3_Weights = None

from src.models.efficientnet_multilabel import EfficientNetB3MultiLabel


class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.35):
        super().__init__()
        try:
            weights = (
                torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            backbone = torchvision.models.resnet50(weights=weights)
        except Exception:
            backbone = models.resnet50(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def build_cls_model(
    backbone: str = "efficientnet_b3",
    num_classes: int = 4,
    pretrained: bool = False,
    dropout: float = 0.35,
) -> nn.Module:
    name = str(backbone).lower()
    if name == "efficientnet_b3":
        return EfficientNetB3MultiLabel(num_classes, pretrained, dropout)
    if name == "resnet50":
        return ResNet50MultiLabel(num_classes, pretrained, dropout)
    raise ValueError(f"Unknown backbone: {backbone}")


def configure_cls_finetune(
    model: nn.Module,
    backbone: str = "efficientnet_b3",
    unfreeze_last: int = 3,
) -> nn.Module:
    for p in model.parameters():
        p.requires_grad = False

    name = str(backbone).lower()
    if name == "efficientnet_b3":
        blocks = list(model.backbone.features.children())
        for block in blocks[-unfreeze_last:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.backbone.classifier.parameters():
            p.requires_grad = True
    elif name == "resnet50":
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True
        for p in model.backbone.fc.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model


def cls_param_groups(
    model: nn.Module,
    backbone: str = "efficientnet_b3",
    lr: float = 2e-4,
    lr_mult: float = 0.15,
) -> list[dict]:
    head_params, backbone_params = [], []
    name = str(backbone).lower()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "efficientnet_b3" and n.startswith("backbone.features"):
            backbone_params.append(p)
        elif name == "resnet50" and n.startswith("backbone.layer4"):
            backbone_params.append(p)
        else:
            head_params.append(p)

    groups = [{"params": head_params, "lr": lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * lr_mult})
    return groups
