import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for segmentation.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (B, C, H, W) logits
            targets: (B, C, H, W) binary masks {0,1}
        """
        preds = torch.sigmoid(preds)
        eps = 1e-8

        # Binary cross entropy
        bce = -(targets * torch.log(preds + eps) + (1 - targets) * torch.log(1 - preds + eps))

        # Focal weight
        pt = torch.where(targets == 1, preds, 1 - preds)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss