"""Dice metrics — Stage 1 validation."""

from __future__ import annotations

import numpy as np
import torch


def get_threshold_tensor(thresholds, device, num_classes: int = 4):
    if isinstance(thresholds, (float, int)):
        thresholds = [float(thresholds)] * num_classes
    return torch.tensor(thresholds, device=device, dtype=torch.float32).view(
        1, num_classes, 1, 1
    )


@torch.no_grad()
def logits_to_binary_masks(logits, thresholds=0.5):
    probs = torch.sigmoid(logits)
    thr = get_threshold_tensor(thresholds, probs.device, probs.shape[1])
    return (probs > thr).float()


@torch.no_grad()
def dice_matrix_from_binary(preds, targets):
    preds = preds.float()
    targets = targets.float()
    inter = (preds * targets).flatten(2).sum(dim=2)
    pred_sum = preds.flatten(2).sum(dim=2)
    target_sum = targets.flatten(2).sum(dim=2)
    den = pred_sum + target_sum
    return torch.where(
        den == 0,
        torch.ones_like(den),
        (2.0 * inter) / den.clamp_min(1e-7),
    )


@torch.no_grad()
def dice_from_logits(logits, targets, thresholds=0.5):
    preds = logits_to_binary_masks(logits, thresholds)
    dice_matrix = dice_matrix_from_binary(preds, targets)
    return dice_matrix.mean(), dice_matrix.mean(dim=0), dice_matrix


class DiceMeter:
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_sum = 0.0
        self.total_count = 0
        self.class_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.class_count = np.zeros(self.num_classes, dtype=np.float64)

    @torch.no_grad()
    def update(self, logits, targets, thresholds=0.5):
        _, _, dice_matrix = dice_from_logits(logits, targets, thresholds)
        dice_np = dice_matrix.detach().cpu().numpy()
        self.total_sum += float(dice_np.sum())
        self.total_count += int(dice_np.size)
        self.class_sum += dice_np.sum(axis=0)
        self.class_count += dice_np.shape[0]

    def compute(self):
        return {
            "dice": float(self.total_sum / max(self.total_count, 1)),
            "per_class_dice": (
                self.class_sum / np.maximum(self.class_count, 1)
            ).tolist(),
        }
