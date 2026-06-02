"""Mã hóa / giải mã RLE (định dạng Kaggle Severstal, Fortran order)."""

from __future__ import annotations

import numpy as np


def rle_decode(mask_rle: str, shape: tuple[int, int] = (256, 1600)) -> np.ndarray:
    """Giải mã chuỗi RLE thành mask nhị phân (H, W)."""
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    values = list(map(int, mask_rle.split()))
    starts = np.asarray(values[0::2]) - 1
    lengths = np.asarray(values[1::2])
    ends = starts + lengths

    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        flat[lo:hi] = 1
    return flat.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Mã hóa mask nhị phân (H, W) sang RLE."""
    pixels = mask.astype(np.uint8).flatten(order="F")
    padded = np.pad(pixels, (1, 1), constant_values=0)
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    starts, ends = changes[::2], changes[1::2]
    lengths = ends - starts
    if len(starts) == 0:
        return ""
    return " ".join(f"{int(s)} {int(l)}" for s, l in zip(starts, lengths))
