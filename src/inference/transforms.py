"""Tiền xử lý ảnh inference — dùng chung transform với notebook."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from albumentations import Compose

from src.data.transform import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_cls_val_transform,
    get_valid_transforms,
)


def build_seg_transform(height: int, width: int) -> Compose:
    return get_valid_transforms(height, width)


def build_cls_transform(crop_height: int, crop_width: int) -> Compose:
    return get_cls_val_transform((crop_height, crop_width))


def read_image_as_rgb(image: np.ndarray | str) -> np.ndarray:
    if isinstance(image, str):
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image}")
        return np.repeat(gray[..., None], 3, axis=2)

    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[..., :3].copy()
    raise ValueError(f"Định dạng ảnh không hỗ trợ: shape={arr.shape}")


def preprocess_for_segmentation(
    image_rgb: np.ndarray,
    height: int,
    width: int,
    transform: Compose,
) -> tuple[torch.Tensor, np.ndarray]:
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = transform(image=resized)["image"].unsqueeze(0)
    return tensor, resized


def preprocess_crop_for_classifier(
    crop_rgb: np.ndarray,
    transform: Compose,
) -> torch.Tensor:
    return transform(image=crop_rgb)["image"].unsqueeze(0)
