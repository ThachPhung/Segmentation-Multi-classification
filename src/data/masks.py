"""Xây mask từ CSV Severstal (RLE) — khớp notebook Stage 1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.inference.rle_utils import rle_decode


def is_valid_rle_series(s: pd.Series) -> pd.Series:
    return (
        s.notna()
        & (s.astype(str).str.strip() != "")
        & (s.astype(str).str.lower() != "nan")
    )


def build_masks(
    image_id: str,
    df: pd.DataFrame,
    shape: tuple[int, int] = (256, 1600),
    num_classes: int = 4,
) -> np.ndarray:
    """Mask [H, W, C] uint8 từ các RLE trong train.csv."""
    masks = np.zeros((shape[0], shape[1], num_classes), dtype=np.uint8)
    rows = df[df["ImageId"] == image_id]

    for c in range(1, num_classes + 1):
        rles = rows.loc[
            (rows["ClassId"] == c) & is_valid_rle_series(rows["EncodedPixels"]),
            "EncodedPixels",
        ]
        for rle in rles:
            masks[..., c - 1] |= rle_decode(rle, shape)

    return masks
