"""Suy luận segmentation — khớp `_predict_masks_batch` trong notebook."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_masks_batch(
    seg_model: torch.nn.Module,
    imgs: torch.Tensor,
    device: torch.device,
    thresholds: list[float] | None = None,
) -> np.ndarray:
    """
    Trả mask nhị phân uint8 [B, C, H, W].
    """
    seg_model.eval()
    if thresholds is None:
        thresholds = [0.5, 0.5, 0.5, 0.5]

    if not isinstance(imgs, torch.Tensor):
        imgs = torch.as_tensor(imgs)

    x = imgs.to(device)
    logits = seg_model(x)
    probs = torch.sigmoid(logits).cpu().numpy()
    thr = np.asarray(thresholds, dtype=np.float32).reshape(1, -1, 1, 1)
    return (probs > thr).astype(np.uint8)
