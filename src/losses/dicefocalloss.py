import torch
import torch.nn
from src.losses.focal_loss import FocalLoss

class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, class_weights=None):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights  # [w0, w1, w2, w3]


    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)

        dice_loss = 0.0
        num_classes = pred.shape[1]

        for c in range(num_classes):
            pred_c = pred[:, c, :, :]
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice_c = (2 * intersection + self.smooth) / (union + self.smooth)

            weight = self.class_weights[c] if self.class_weights is not None else 1.0
            dice_loss += weight * (1 - dice_c)

        return dice_loss

class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal Loss"""

    def __init__(self, alpha=0.5, dice_weight=None, focal_weight=None):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha  # blending factor
        self.dice = WeightedDiceLoss(class_weights=dice_weight)
        self.focal = FocalLoss(weight=focal_weight)

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)

        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss