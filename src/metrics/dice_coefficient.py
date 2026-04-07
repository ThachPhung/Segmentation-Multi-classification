import torch
from config import CONFIG

@torch.no_grad()
def dice_coefficient(logits, targets, threshold: float = 0.5, eps=1e-6):
    probs = torch.sigmoid(logits); preds = (probs>threshold).float()
    inter = (preds*targets).sum((2,3)); den = preds.sum((2,3))+targets.sum((2,3))
    dice = (2*inter+eps)/(den+eps)
    return dice.mean()

@torch.no_grad()
def dice_per_class(logits, targets, threshold: float = 0.5, eps=1e-6):
    probs = torch.sigmoid(logits); preds = (probs>threshold).float()
    inter = (preds*targets).sum((0,2,3)); den = preds.sum((0,2,3))+targets.sum((0,2,3))
    return ((2*inter+eps)/(den+eps)).cpu().tolist()