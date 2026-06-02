import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss - concentrate on hard negative examples"""

    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - logits
            target: (B, H, W) - class indices
        """
        ce = F.cross_entropy(pred, target, weight=self.weight, reduction='none')

        p = torch.exp(-ce)
        focal_weight = (1 - p) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce

        return focal_loss.mean()