"""Trích xuất vùng quan tâm (RoI) từ mask phân đoạn."""

from __future__ import annotations

import cv2
import numpy as np


def find_bboxes_from_binary_mask(mask2d: np.ndarray, min_area: int = 20) -> list[tuple[int, int, int, int]]:
    """Tìm bounding box cho từng connected component trên mask nhị phân."""
    binary = (mask2d > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def clip_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(1, min(width, int(x2)))
    y2 = max(1, min(height, int(y2)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def collect_bboxes_from_mask_chw(
    mask_chw: np.ndarray,
    min_component_area: int = 20,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """
    Thu thập (class_index, bbox) từ mask [C, H, W].
    Sắp xếp theo diện tích bbox giảm dần.
    """
    items: list[tuple[int, tuple[int, int, int, int]]] = []
    for class_idx in range(mask_chw.shape[0]):
        for bbox in find_bboxes_from_binary_mask(mask_chw[class_idx], min_area=min_component_area):
            items.append((class_idx, bbox))
    items.sort(key=lambda it: (it[1][2] - it[1][0]) * (it[1][3] - it[1][1]), reverse=True)
    return items


def component_mask_in_bbox(
    class_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Lấy mask của component chứa tâm bbox trên kênh lớp đã cho.
    Trả về mask đầy đủ kích thước ảnh (H, W), uint8 {0,1}.
    """
    x1, y1, x2, y2 = bbox
    binary = (class_mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cy = min(max(cy, 0), binary.shape[0] - 1)
    cx = min(max(cx, 0), binary.shape[1] - 1)
    label_id = labels[cy, cx]
    if label_id == 0:
        # Tâm không nằm trên defect — chọn component có overlap lớn nhất với bbox
        roi_labels = labels[y1:y2, x1:x2]
        unique, counts = np.unique(roi_labels[roi_labels > 0], return_counts=True)
        if len(unique) == 0:
            return binary
        label_id = int(unique[np.argmax(counts)])

    return (labels == label_id).astype(np.uint8)


def multilabel_active_in_roi(
    mask_chw: np.ndarray,
    y1: int,
    x1: int,
    y2: int,
    x2: int,
    label_min_pixels: int = 8,
) -> np.ndarray:
    """Vector nhị phân [C] — lớp nào có đủ pixel trong RoI."""
    roi = mask_chw[:, y1:y2, x1:x2]
    return (roi.reshape(roi.shape[0], -1).sum(axis=1) >= label_min_pixels).astype(np.int32)
