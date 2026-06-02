"""
Tạo crop Stage 2 từ DataLoader segmentation — khớp `build_stage2_crops` notebook.
Dùng khi tái tạo train_cls_labels.csv (không cần cho suy luận ảnh đơn).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.inference.roi import (
    clip_bbox,
    collect_bboxes_from_mask_chw,
    multilabel_active_in_roi,
)
from src.inference.seg_inference import predict_masks_batch
from src.inference.transforms import read_image_as_rgb


def _get_image_id_from_batch(image_ids, bi: int) -> str:
    if isinstance(image_ids, (list, tuple)) and len(image_ids) > bi:
        item = image_ids[bi]
        if isinstance(item, dict):
            for key in ("image_id", "ImageId", "id", "filename"):
                if key in item:
                    return str(item[key])
    if isinstance(image_ids, (list, tuple)):
        return str(image_ids[bi])
    return str(image_ids[bi])


def _unpack_seg_batch(batch):
    if isinstance(batch, dict):
        imgs = batch.get("image") or batch.get("images")
        masks = batch.get("mask") or batch.get("masks")
        image_ids = batch.get("image_id") or batch.get("metas")
        return imgs, masks, image_ids
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    raise TypeError("Batch cần (imgs, masks, metas)")


def _masks_to_numpy_uint8(masks, threshold=0.5):
    if isinstance(masks, torch.Tensor):
        m = masks.detach().cpu().numpy()
    else:
        m = np.asarray(masks)
    if isinstance(threshold, (list, tuple, np.ndarray)):
        thr = np.asarray(threshold, dtype=np.float32).reshape(1, -1, 1, 1)
        return (m > thr).astype(np.uint8)
    return (m > threshold).astype(np.uint8)


@torch.no_grad()
def build_stage2_crops(
    seg_loader,
    seg_model,
    image_dir,
    crop_dir,
    out_csv_path,
    device,
    *,
    bbox_mask_source: str = "pred",
    label_mask_source: str = "gt",
    seg_thresholds=None,
    min_component_area: int = 20,
    label_min_pixels: int = 8,
    bbox_margin: int = 12,
    max_crops_per_image: int = 16,
    crop_size: tuple[int, int] = (448, 448),
):
    if seg_thresholds is None:
        seg_thresholds = [0.5, 0.5, 0.5, 0.5]
    if bbox_mask_source == "pred" and seg_model is None:
        raise ValueError("Cần seg_model khi bbox_mask_source='pred'")

    rows = []
    image_dir = Path(image_dir)
    crop_dir = Path(crop_dir)
    out_csv_path = Path(out_csv_path)
    crop_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(seg_loader, desc=f"Crops → {out_csv_path.name}", leave=False):
        imgs, gt_masks, metas = _unpack_seg_batch(batch)
        gt_b = _masks_to_numpy_uint8(gt_masks, threshold=0.5)

        if bbox_mask_source == "pred":
            bbox_b = predict_masks_batch(seg_model, imgs, device, seg_thresholds)
        else:
            bbox_b = gt_b.copy()

        label_b = gt_b if label_mask_source == "gt" else bbox_b
        B, C, Hm, Wm = gt_b.shape

        for bi in range(B):
            image_id = Path(_get_image_id_from_batch(metas, bi)).name
            img_path = image_dir / image_id
            img0 = read_image_as_rgb(str(img_path))
            if img0.shape[0] != Hm or img0.shape[1] != Wm:
                img0 = cv2.resize(img0, (Wm, Hm), interpolation=cv2.INTER_LINEAR)

            H, W = img0.shape[:2]
            bbox_items = collect_bboxes_from_mask_chw(bbox_b[bi], min_component_area)
            if not bbox_items:
                continue
            bbox_items = bbox_items[:max_crops_per_image]

            for k, (main_class, (x1, y1, x2, y2)) in enumerate(bbox_items):
                x1 -= bbox_margin
                y1 -= bbox_margin
                x2 += bbox_margin
                y2 += bbox_margin
                x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, W, H)
                crop = img0[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                ys = multilabel_active_in_roi(label_b[bi], y1, x1, y2, x2, label_min_pixels)
                if ys.sum() == 0:
                    continue
                crop_rs = cv2.resize(
                    crop, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_LINEAR
                )
                crop_name = f"{Path(image_id).stem}_crop{k}_cls{main_class + 1}.jpg"
                crop_path = crop_dir / crop_name
                cv2.imwrite(str(crop_path), crop_rs)
                y_out = [int(ys[ci]) for ci in range(min(4, C))] + [0] * max(0, 4 - C)
                rows.append(
                    {
                        "crop_path": str(crop_path),
                        "image_id": image_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "y_1": y_out[0],
                        "y_2": y_out[1],
                        "y_3": y_out[2],
                        "y_4": y_out[3],
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    return df
