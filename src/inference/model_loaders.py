"""Tải checkpoint segmentation & classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from inference_config import INFERENCE_CFG
from src.models.attention_unet_efficientnet_b3 import AttentionUNetEfficientNetB3
from src.models.classifier import build_cls_model


def load_thresholds(
    explicit: list[float] | None,
    npy_path: Path | str | None,
    default: list[float],
) -> list[float]:
    if explicit is not None:
        return list(explicit)
    if npy_path is not None and Path(npy_path).exists():
        return list(np.load(npy_path))
    return list(default)


def load_segmentation_model(
    checkpoint: Path | str | None = None,
    device: str | torch.device | None = None,
    cfg: dict | None = None,
) -> AttentionUNetEfficientNetB3:
    cfg = cfg or INFERENCE_CFG
    path = Path(checkpoint or cfg["seg_checkpoint"])
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy segmentation checkpoint: {path}")

    model = AttentionUNetEfficientNetB3(
        num_classes=cfg["num_classes"],
        pretrained=False,
        dropout=cfg["seg_dropout"],
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev).eval()


def load_classifier_model(
    checkpoint: Path | str | None = None,
    device: str | torch.device | None = None,
    cfg: dict | None = None,
    backbone: str = "efficientnet_b3",
) -> torch.nn.Module:
    cfg = cfg or INFERENCE_CFG
    path = Path(checkpoint or cfg["cls_checkpoint"])
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy classifier checkpoint: {path}")

    model = build_cls_model(
        backbone=backbone,
        num_classes=cfg["num_classes"],
        pretrained=False,
        dropout=cfg["cls_dropout"],
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev).eval()
