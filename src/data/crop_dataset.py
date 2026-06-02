"""Dataset crop RoI cho classifier Stage 2."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SteelCropClsDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        self.label_cols = ["y_1", "y_2", "y_3", "y_4"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop_path = row["crop_path"]

        img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {crop_path}")

        img = np.repeat(img[..., None], 3, axis=2)
        labels = row[self.label_cols].values.astype(np.float32)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(labels, dtype=torch.float32), crop_path
