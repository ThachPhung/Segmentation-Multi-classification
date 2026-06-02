import pandas as pd
import numpy as np
from config import CONFIG

from src.data.masks import build_masks, is_valid_rle_series
from src.inference.rle_utils import rle_decode, rle_encode


class RLEprocessor:
    """RLE encoding/decoding — wrapper tương thích code cũ."""

    @staticmethod
    def rle_decoder(rle_str, shape=(CONFIG["Height"], CONFIG["Width"])) -> np.ndarray:
        return rle_decode(rle_str, shape)

    @staticmethod
    def build_masks(df: pd.DataFrame, image_id: str) -> np.ndarray:
        return build_masks(
            image_id,
            df,
            shape=(CONFIG["Height"], CONFIG["Width"]),
            num_classes=CONFIG["n_classes"],
        )

    @staticmethod
    def rle_encoder(mask):
        return rle_encode(mask)


__all__ = ["RLEprocessor", "build_masks", "is_valid_rle_series"]
