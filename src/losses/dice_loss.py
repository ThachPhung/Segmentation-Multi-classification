import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (B, C, H, W) logits
            targets: (B, C, H, W) binary masks {0,1}
        """
        preds = torch.sigmoid(preds)  # sigmoid → [0,1]

        preds = preds.contiguous().view(preds.size(0), preds.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)

        intersection = (preds * targets).sum(dim=2)
        denominator = preds.sum(dim=2) + targets.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - dice.mean()
        return loss