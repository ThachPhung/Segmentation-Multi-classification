# src/data/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple

class SteverstalDataset(Dataset):
    """PyTorch Dataset for Severstal"""

    def __init__(self, image_ids, df: pd.DataFrame, image_dir, transform=None, load_rgb=True):
        self.df = df
        self.image_dir = Path(image_dir)
        self.image_ids= list(image_ids)
        self.transform = transform
        self.load_rgb = load_rgb

    def __len__(self):
        return len(self.image_ids)

    def _read_image(self, image_id):
        img = cv2.imread(str(self.image_dir / image_id), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_id}")
        if self.load_rgb:
            img = np.repeat(img[..., None], 3, axis=2)  # (H,W,3)
        return img


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self._read_image(image_id)
        mask = RLEprocessor.build_masks(self.df, image_id)

        if self.transform:
            output = self.transform(image=image, mask=mask)
            image, mask = output['image'], output['mask'].permute(2, 0, 1) # (C, H, W)
        else:
            image = torch.from_numpy(image.transpose(2,0,1)).float()
            mask = torch.from_numpy(mask.transpose(2,0,1)).float()

        meta = {"image_id": image_id}
        return {
            'image': image,   # (3, H, W)
            'mask': mask,     # (4, H, W)
            'image_id': image_id
        }


# --- Collate function ---
def collate_fn(batch):
    images, masks, metas = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks, metas