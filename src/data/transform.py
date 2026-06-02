"""Albumentations — khớp notebook graduation-project-phase-2."""

from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(height: int = 256, width: int = 1600) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.02, 0.02),
                rotate=(-3, 3),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.35
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.25),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Resize(height, width, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )


def get_valid_transforms(height: int = 256, width: int = 1600) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height, width, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )


def get_cls_train_transform(crop_size: tuple[int, int] = (448, 448)) -> A.Compose:
    h, w = crop_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(
                translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                scale=(0.88, 1.12),
                rotate=(-12, 12),
                border_mode=0,
                p=0.45,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1.0
                    ),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                ],
                p=0.5,
            ),
            A.GaussNoise(std_range=(0.02, 0.10), p=0.25),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.CoarseDropout(
                num_holes_range=(4, 6),
                hole_height_range=(20, 28),
                hole_width_range=(20, 28),
                p=0.2,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
    )


def get_cls_val_transform(crop_size: tuple[int, int] = (448, 448)) -> A.Compose:
    h, w = crop_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
    )
