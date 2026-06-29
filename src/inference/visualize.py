"""
Hiển thị kết quả inference: input, segmentation, output, classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.inference.roi import collect_bboxes_from_mask_chw, component_mask_in_bbox

if TYPE_CHECKING:
    from src.inference.pipeline import InferenceResult

CLASS_COLORS_RGB = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 128, 255),
    4: (255, 220, 0),
}

CLASS_COLORS_NORM = {k: tuple(c / 255.0 for c in v) for k, v in CLASS_COLORS_RGB.items()}


def _to_display_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.float32)
        if gray.max() > 1.5:
            gray = gray / 255.0
        return np.stack([gray, gray, gray], axis=-1)
    img = image[..., :3].astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0, 1)


def overlay_segmentation(image_rgb: np.ndarray, mask_chw: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = _to_display_rgb(image_rgb).copy()
    for class_idx in range(mask_chw.shape[0]):
        m = mask_chw[class_idx].astype(bool)
        if not m.any():
            continue
        color = np.array(CLASS_COLORS_NORM[class_idx + 1], dtype=np.float32)
        base[m] = (1 - alpha) * base[m] + alpha * color
    return np.clip(base, 0, 1)


def overlay_final_output(image_rgb: np.ndarray, detections: list, alpha: float = 0.5) -> np.ndarray:
    from src.inference.rle_utils import rle_decode

    base = _to_display_rgb(image_rgb).copy()
    h, w = base.shape[:2]

    for det in detections:
        class_id = int(det.class_id if hasattr(det, "class_id") else det["class_id"])
        rle = det.rle if hasattr(det, "rle") else det["rle"]
        if not rle:
            continue
        m = rle_decode(rle, (h, w)).astype(bool)
        color = np.array(CLASS_COLORS_NORM.get(class_id, (1, 1, 1)), dtype=np.float32)
        base[m] = (1 - alpha) * base[m] + alpha * color

    base_u8 = (base * 255).astype(np.uint8)
    for det in detections:
        x1, y1, x2, y2 = det.bbox if hasattr(det, "bbox") else det["bbox"]
        class_id = int(det.class_id if hasattr(det, "class_id") else det["class_id"])
        name = det.defect_name if hasattr(det, "defect_name") else det["defect_name"]
        conf = float(det.confidence if hasattr(det, "confidence") else det["confidence"])
        bgr = CLASS_COLORS_RGB.get(class_id, (255, 255, 255))[::-1]
        cv2.rectangle(base_u8, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 2)
        short = name.split("(")[0].strip()[:14]
        cv2.putText(
            base_u8,
            f"{short} {conf:.2f}",
            (int(x1), max(int(y1) - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            bgr,
            1,
            cv2.LINE_AA,
        )
    return base_u8.astype(np.float32) / 255.0


@dataclass
class CropMaskItem:
    """Mask component đã làm sạch — dùng cho RLE / crop classifier."""

    seg_class_id: int
    bbox: tuple[int, int, int, int]
    component_mask: np.ndarray


def collect_crop_mask_items(
    mask_chw: np.ndarray,
    *,
    min_component_area: int = 20,
    max_items: int | None = None,
) -> list[CropMaskItem]:
    """Thu thập mask component (giống bước chuẩn bị crop trong pipeline)."""
    items: list[CropMaskItem] = []
    for seg_class_idx, bbox in collect_bboxes_from_mask_chw(
        mask_chw, min_component_area=min_component_area
    ):
        component = component_mask_in_bbox(mask_chw[seg_class_idx], bbox)
        items.append(
            CropMaskItem(
                seg_class_id=int(seg_class_idx) + 1,
                bbox=bbox,
                component_mask=component,
            )
        )
        if max_items is not None and len(items) >= max_items:
            break
    return items


def overlay_component_masks(
    image_rgb: np.ndarray,
    crop_items: list[CropMaskItem],
    *,
    alpha: float = 0.55,
    draw_bboxes: bool = True,
) -> np.ndarray:
    """Overlay mask đã chuẩn bị để cắt (component) lên ảnh."""
    base = (_to_display_rgb(image_rgb) * 255).astype(np.uint8)
    for item in crop_items:
        m = item.component_mask.astype(bool)
        if not m.any():
            continue
        cid = item.seg_class_id
        bgr = CLASS_COLORS_RGB.get(cid, (255, 255, 255))[::-1]
        color = np.array(CLASS_COLORS_NORM.get(cid, (1, 1, 1)), dtype=np.float32)
        base_f = base.astype(np.float32) / 255.0
        base_f[m] = (1 - alpha) * base_f[m] + alpha * color
        base = (np.clip(base_f, 0, 1) * 255).astype(np.uint8)
        if draw_bboxes:
            x1, y1, x2, y2 = item.bbox
            cv2.rectangle(base, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 1)

    return base.astype(np.float32) / 255.0


def mask_for_crop_rgb(
    crop_items: list[CropMaskItem],
    shape_hw: tuple[int, int],
) -> np.ndarray:
    """Ảnh RGB chỉ gồm mask component (màu theo lớp), nền đen."""
    h, w = shape_hw
    out = np.zeros((h, w, 3), dtype=np.float32)
    for item in crop_items:
        m = item.component_mask.astype(bool)
        if not m.any():
            continue
        out[m] = np.array(CLASS_COLORS_NORM.get(item.seg_class_id, (1, 1, 1)), dtype=np.float32)
    return out


def _classification_text(probs: dict[int, float], thresholds: list[float], defect_names: dict) -> str:
    lines = []
    for cid in sorted(probs.keys()):
        p = probs[cid]
        thr = thresholds[cid - 1] if cid - 1 < len(thresholds) else 0.5
        mark = "OK" if p >= thr else "  "
        short = defect_names.get(cid, f"C{cid}").split("(")[0].strip()[:12]
        lines.append(f"{mark} {short}: {p:.2f}")
    return "\n".join(lines)


def plot_inference_result(
    result: "InferenceResult",
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 120,
) -> plt.Figure:
    roi_items = result.roi_items
    n_roi = min(len(roi_items), 6)
    n_cols = max(3, n_roi) if n_roi else 3

    fig = plt.figure(figsize=(max(14, n_cols * 2.8), 9))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1.2, 1], hspace=0.32, wspace=0.12)

    img_disp = _to_display_rgb(result.image_resized)
    seg_overlay = overlay_segmentation(result.image_resized, result.mask_seg)
    out_overlay = overlay_final_output(result.image_resized, result.detections)

    row1 = [
        ("1. Ảnh input", img_disp),
        ("2. Segmentation (U-Net predict)", seg_overlay),
        ("3. Output (mask + bbox + classification)", out_overlay),
    ]
    for col, (title, img) in enumerate(row1):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    if n_roi == 0:
        ax = fig.add_subplot(gs[1, :])
        ax.text(0.5, 0.5, "Không phát hiện RoI", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        fig.text(0.5, 0.46, "4. Classification từng RoI (EfficientNet-B3)", ha="center", fontsize=10)
        for j in range(n_roi):
            roi = roi_items[j]
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(_to_display_rgb(roi.crop_rgb))
            pred = ", ".join(
                result.defect_names.get(c, f"C{c}") for c in roi.predicted_class_ids
            ) or "(dưới ngưỡng)"
            ax.set_title(f"RoI #{j+1} | Seg C{roi.seg_class_id} → {pred}", fontsize=8)
            ax.text(
                0.02,
                0.02,
                _classification_text(roi.probabilities, result.cls_thresholds, result.defect_names),
                transform=ax.transAxes,
                fontsize=6.5,
                va="bottom",
                color="white",
                bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8),
            )
            ax.axis("off")
        for j in range(n_roi, n_cols):
            ax = fig.add_subplot(gs[1, j])
            ax.axis("off")

    name = Path(result.image_path or "image").name
    fig.suptitle(
        f"{name} — {len(result.detections)} khuyết tật phát hiện",
        fontsize=12,
        y=0.98,
    )

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def panels_from_result(
    result: "InferenceResult",
    *,
    min_component_area: int = 20,
) -> dict[str, np.ndarray]:
    """Trả về các ảnh RGB float [0,1] dùng cho báo cáo."""
    h, w = result.image_resized.shape[:2]
    crop_items = collect_crop_mask_items(
        result.mask_seg,
        min_component_area=min_component_area,
        max_items=None,
    )
    return {
        "crop_mask_pure": mask_for_crop_rgb(crop_items, (h, w)),
        "crop_mask_overlay": overlay_component_masks(result.image_resized, crop_items),
        "segmentation": overlay_segmentation(result.image_resized, result.mask_seg),
        "output": overlay_final_output(result.image_resized, result.detections),
    }


def plot_crop_mask(
    result: "InferenceResult",
    *,
    mode: str = "overlay",
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 120,
    min_component_area: int = 20,
) -> plt.Figure:
    """
    In mask chuẩn bị để cắt (component đã làm sạch).
    mode: \"overlay\" — mask + bbox trên ảnh; \"pure\" — chỉ mask màu trên nền đen.
    """
    panels = panels_from_result(result, min_component_area=min_component_area)
    img = panels["crop_mask_overlay"] if mode == "overlay" else panels["crop_mask_pure"]
    title = (
        "Mask component (chuẩn bị crop / RLE)"
        if mode == "overlay"
        else "Mask component (thuần)"
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    name = Path(result.image_path or "image").name
    fig.suptitle(name, fontsize=10, y=0.98)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_seg_output_pair(
    result: "InferenceResult",
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 150,
    figsize: tuple[float, float] = (12, 4.5),
) -> plt.Figure:
    """Hai cột giống báo cáo: Segmentation | Output (mask + bbox + classification)."""
    panels = panels_from_result(result)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (title, key) in zip(
        axes,
        [
            ("2. Segmentation (U-Net predict)", "segmentation"),
            ("3. Output (mask + bbox + classification)", "output"),
        ],
    ):
        ax.imshow(panels[key])
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    name = Path(result.image_path or "image").name
    fig.suptitle(name, fontsize=11, y=1.02)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_report_grid(
    results: list["InferenceResult"],
    *,
    columns: tuple[str, ...] = ("crop_mask_overlay", "segmentation", "output"),
    column_titles: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 150,
    min_component_area: int = 20,
    suptitle: str | None = None,
) -> plt.Figure:
    """
    Nhiều ảnh trong một khung (mỗi hàng = một mẫu).

    columns mặc định: mask crop | segmentation | output.
    Chỉ 2 cột như ảnh mẫu: columns=(\"segmentation\", \"output\").
    """
    default_titles = {
        "crop_mask_overlay": "1. Mask chuẩn bị crop",
        "crop_mask_pure": "1. Mask component (thuần)",
        "segmentation": "2. Segmentation (U-Net predict)",
        "output": "3. Output (mask + bbox + classification)",
    }
    titles = {**default_titles, **(column_titles or {})}

    n = len(results)
    if n == 0:
        raise ValueError("results rỗng")

    n_cols = len(columns)
    fig, axes = plt.subplots(
        n,
        n_cols,
        figsize=(4.2 * n_cols, 3.6 * n),
        squeeze=False,
    )

    for row, result in enumerate(results):
        panels = panels_from_result(result, min_component_area=min_component_area)
        label = Path(result.image_path or f"sample_{row}").stem
        for col, key in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(panels[key])
            if row == 0:
                ax.set_title(titles.get(key, key), fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=48, va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.0)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
