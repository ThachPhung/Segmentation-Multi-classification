"""Chọn thiết bị suy luận (CUDA / MPS / CPU)."""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> str:
    """Trả về device hợp lệ: cuda, mps hoặc cpu."""
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("CUDA không khả dụng — dùng MPS (GPU Apple Silicon).")
            return "mps"
        print("CUDA/MPS không khả dụng — dùng CPU.")
        return "cpu"

    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        print("MPS không khả dụng — dùng CPU.")
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
