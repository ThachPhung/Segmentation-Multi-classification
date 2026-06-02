"""Dataset phân đoạn Severstal — khớp notebook."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.masks import build_masks


class SteelDefectDataset(Dataset):
    def __init__(
        self,
        image_ids,
        image_dir,
        df=None,
        transforms=None,
        has_mask: bool = True,
        image_size: tuple[int, int] = (256, 1600),
    ):
        self.image_ids = list(image_ids)
        self.image_dir = Path(image_dir)
        self.df = df
        self.transforms = transforms
        self.has_mask = has_mask
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def _read_image(self, image_id: str) -> np.ndarray:
        img = cv2.imread(str(self.image_dir / image_id), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_id)
        return np.repeat(img[..., None], 3, axis=2)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img = self._read_image(image_id)

        if self.has_mask:
            mask = build_masks(image_id, self.df, shape=self.image_size)
        else:
            h, w = self.image_size
            mask = np.zeros((h, w, 4), dtype=np.uint8)

        if self.transforms:
            out = self.transforms(image=img, mask=mask)
            img = out["image"]
            mask = out["mask"].permute(2, 0, 1).float()
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        return img, mask, {"image_id": image_id}


def collate_fn(batch):
    images, masks, metas = zip(*batch)
    return torch.stack(images), torch.stack(masks), metas


# Alias tương thích code cũ
SteverstalDataset = SteelDefectDataset
