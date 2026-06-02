"""Suy luận classifier RoI — khớp `_cls_forward_probs` + TTA trong notebook."""

from __future__ import annotations

import torch


@torch.no_grad()
def forward_cls_probs(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    model.eval()
    x = imgs.to(device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=use_amp and device.type == "cuda"):
        return torch.sigmoid(model(x).float())


@torch.no_grad()
def classify_crop_probs(
    model: torch.nn.Module,
    crop_tensor: torch.Tensor,
    device: torch.device,
    use_tta: bool = True,
    use_amp: bool = False,
) -> torch.Tensor:
    """Xác suất [C] cho một crop [1, 3, H, W]."""
    probs = forward_cls_probs(model, crop_tensor, device, use_amp=use_amp)
    if use_tta:
        flipped = torch.flip(crop_tensor, dims=[-1])
        probs_flip = forward_cls_probs(model, flipped, device, use_amp=use_amp)
        probs = (probs + probs_flip) * 0.5
    return probs[0].cpu().numpy()
