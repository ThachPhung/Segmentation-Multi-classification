"""Focal loss multi-label — Stage 2 classifier."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = (1.0 - pt).clamp(min=1e-6).pow(self.gamma) * bce
        return focal.mean()


@torch.no_grad()
def multilabel_f1(probs, targets, thr=0.5, eps=1e-9):
    if isinstance(thr, (float, int)):
        thr = [float(thr)] * probs.shape[1]
    thr_t = torch.tensor(thr, dtype=probs.dtype, device=probs.device).view(1, -1)
    preds = (probs > thr_t).float()
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return f1.mean().item(), f1.cpu().numpy()


def default_thr_grids(num_classes: int = 4) -> list[np.ndarray]:
    return [
        np.round(np.arange(0.15, 0.86, 0.03), 2),
        np.round(np.arange(0.15, 0.86, 0.03), 2),
        np.round(np.arange(0.15, 0.86, 0.03), 2),
        np.round(np.arange(0.35, 0.96, 0.03), 2),
    ][:num_classes]


@torch.no_grad()
def tune_cls_thresholds(probs, targets, grids=None):
    num_classes = probs.shape[1]
    if grids is None:
        grids = default_thr_grids(num_classes)

    best_thr = [0.5] * num_classes
    best_f1 = []
    for c in range(num_classes):
        y = targets[:, c]
        p = probs[:, c]
        grid = grids[c] if c < len(grids) else np.round(np.arange(0.15, 0.86, 0.03), 2)
        if y.sum() == 0:
            best_thr[c] = 0.5
            best_f1.append(0.0)
            continue
        best_fc, best_t = 0.0, 0.5
        for t in grid:
            pred = (p > t).float()
            tp = ((pred == 1) & (y == 1)).sum().item()
            fp = ((pred == 1) & (y == 0)).sum().item()
            fn = ((pred == 0) & (y == 1)).sum().item()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_fc:
                best_fc, best_t = f1, float(t)
        best_thr[c] = best_t
        best_f1.append(best_fc)

    macro, _ = multilabel_f1(probs, targets, thr=best_thr)
    return best_thr, best_f1, macro
