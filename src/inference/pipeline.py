"""
Pipeline suy luận hai giai đoạn (khớp notebook phase-2):
  Ảnh → Attention U-Net → RoI → EfficientNet-B3 (+ TTA) → RLE + tên khuyết tật.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference_config import DEFECT_NAMES_VI, INFERENCE_CFG
from src.inference.cls_inference import classify_crop_probs
from src.inference.model_loaders import load_classifier_model, load_segmentation_model, load_thresholds
from src.inference.roi import (
    clip_bbox,
    collect_bboxes_from_mask_chw,
    component_mask_in_bbox,
    multilabel_active_in_roi,
)
from src.inference.rle_utils import rle_encode
from src.inference.seg_inference import predict_masks_batch
from src.inference.transforms import (
    build_cls_transform,
    build_seg_transform,
    preprocess_crop_for_classifier,
    preprocess_for_segmentation,
    read_image_as_rgb,
)


@dataclass
class RoIInferenceItem:
    """Một vùng RoI: crop + xác suất classifier."""

    bbox: tuple[int, int, int, int]
    crop_rgb: np.ndarray
    seg_class_id: int
    probabilities: dict[int, float]
    predicted_class_ids: list[int]


@dataclass
class InferenceResult:
    """Kết quả đầy đủ để hiển thị / lưu."""

    image_path: str | None
    image_resized: np.ndarray
    mask_seg: np.ndarray
    detections: list["DefectDetection"]
    roi_items: list[RoIInferenceItem]
    defect_names: dict[int, str]
    seg_thresholds: list[float]
    cls_thresholds: list[float]


@dataclass
class DefectDetection:
    class_id: int
    defect_name: str
    rle: str
    bbox: tuple[int, int, int, int]
    confidence: float
    seg_class_id: int | None = None
    probabilities: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "defect_name": self.defect_name,
            "rle": self.rle,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "seg_class_id": self.seg_class_id,
            "probabilities": self.probabilities,
        }


class DefectInferencePipeline:
    def __init__(
        self,
        seg_checkpoint: str | Path | None = None,
        cls_checkpoint: str | Path | None = None,
        seg_thresholds: list[float] | None = None,
        cls_thresholds: list[float] | None = None,
        device: str | None = None,
        defect_names: dict[int, str] | None = None,
        cfg: dict | None = None,
        cls_backbone: str | None = None,
    ):
        self.cfg = dict(cfg or INFERENCE_CFG)
        cls_backbone = cls_backbone or self.cfg.get("cls_backbone", "efficientnet_b3")
        self.defect_names = defect_names or DEFECT_NAMES_VI
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        h, w = self.cfg["image_height"], self.cfg["image_width"]
        self.seg_transform = build_seg_transform(h, w)
        ch, cw = self.cfg["cls_crop_size"]
        self.cls_transform = build_cls_transform(ch, cw)

        self.seg_thresholds = load_thresholds(
            seg_thresholds,
            self.cfg.get("seg_thresholds_path"),
            self.cfg.get("seg_thresholds"),
        )
        self.cls_thresholds = load_thresholds(
            cls_thresholds,
            self.cfg.get("cls_thresholds_path"),
            self.cfg.get("cls_thresholds"),
        )

        self.seg_model = load_segmentation_model(
            seg_checkpoint, self.device, self.cfg
        )
        self.cls_model = load_classifier_model(
            cls_checkpoint, self.device, self.cfg, backbone=cls_backbone
        )

    def predict_detailed(self, image: np.ndarray | str) -> InferenceResult:
        image_path = str(image) if isinstance(image, str) else None
        image_rgb = read_image_as_rgb(image)
        h, w = self.cfg["image_height"], self.cfg["image_width"]

        seg_tensor, image_resized = preprocess_for_segmentation(
            image_rgb, h, w, self.seg_transform
        )
        mask_chw = predict_masks_batch(
            self.seg_model, seg_tensor, self.device, self.seg_thresholds
        )[0]

        detections = self._detect_from_masks(image_resized, mask_chw)
        roi_items = getattr(self, "_last_roi_items", [])

        return InferenceResult(
            image_path=image_path,
            image_resized=image_resized,
            mask_seg=mask_chw,
            detections=detections,
            roi_items=roi_items,
            defect_names=self.defect_names,
            seg_thresholds=[float(t) for t in self.seg_thresholds],
            cls_thresholds=[float(t) for t in self.cls_thresholds],
        )

    def predict(self, image: np.ndarray | str) -> list[DefectDetection]:
        return self.predict_detailed(image).detections

    def _detect_from_masks(
        self, image_resized: np.ndarray, mask_chw: np.ndarray
    ) -> list[DefectDetection]:
        bbox_items = collect_bboxes_from_mask_chw(
            mask_chw,
            min_component_area=self.cfg["min_component_area"],
        )
        self._last_roi_items = []
        if not bbox_items:
            return []

        margin = self.cfg["bbox_margin"]
        max_det = self.cfg["max_detections_per_image"]
        label_min_px = self.cfg["label_min_pixels"]
        cls_thr = np.asarray(self.cls_thresholds, dtype=np.float32)
        use_tta = bool(self.cfg.get("cls_use_tta", True))

        detections: list[DefectDetection] = []
        H, W = image_resized.shape[:2]

        for seg_class_idx, (x1, y1, x2, y2) in bbox_items[:max_det]:
            x1m, y1m, x2m, y2m = clip_bbox(
                x1 - margin, y1 - margin, x2 + margin, y2 + margin, W, H
            )
            crop = image_resized[y1m:y2m, x1m:x2m]
            if crop.size == 0:
                continue

            crop_tensor = preprocess_crop_for_classifier(crop, self.cls_transform)
            probs = classify_crop_probs(
                self.cls_model,
                crop_tensor,
                self.device,
                use_tta=use_tta,
                use_amp=False,
            )
            cls_active = np.where(probs >= cls_thr)[0].tolist()
            prob_dict = {i + 1: float(probs[i]) for i in range(len(probs))}

            seg_in_roi = multilabel_active_in_roi(
                mask_chw, y1m, x1m, y2m, x2m, label_min_pixels=label_min_px
            )

            # Giống logic notebook Stage 2 (bbox từ pred, nhãn giao seg∩classifier)
            candidate_classes = [c for c in cls_active if seg_in_roi[c] == 1]
            if not candidate_classes and cls_active:
                candidate_classes = cls_active
            if not candidate_classes:
                candidate_classes = [seg_class_idx]

            self._last_roi_items.append(
                RoIInferenceItem(
                    bbox=(x1m, y1m, x2m, y2m),
                    crop_rgb=crop.copy(),
                    seg_class_id=int(seg_class_idx) + 1,
                    probabilities=prob_dict,
                    predicted_class_ids=[int(c) + 1 for c in candidate_classes],
                )
            )

            component = component_mask_in_bbox(
                mask_chw[seg_class_idx], (x1, y1, x2, y2)
            )

            for class_idx in candidate_classes:
                class_id = int(class_idx) + 1
                if class_idx == seg_class_idx:
                    out_mask = component
                else:
                    out_mask = (
                        component.astype(bool) & mask_chw[class_idx].astype(bool)
                    ).astype(np.uint8)
                if out_mask.sum() < label_min_px:
                    continue

                rle = rle_encode(out_mask)
                if not rle:
                    continue

                detections.append(
                    DefectDetection(
                        class_id=class_id,
                        defect_name=self.defect_names.get(
                            class_id, f"Defect {class_id}"
                        ),
                        rle=rle,
                        bbox=(x1m, y1m, x2m, y2m),
                        confidence=float(probs[class_idx]),
                        seg_class_id=int(seg_class_idx) + 1,
                        probabilities={i + 1: float(probs[i]) for i in range(len(probs))},
                    )
                )

        return detections


_pipeline: DefectInferencePipeline | None = None


def get_pipeline(**kwargs) -> DefectInferencePipeline:
    global _pipeline
    if _pipeline is None or kwargs:
        _pipeline = DefectInferencePipeline(**kwargs)
    return _pipeline


def predict(image: np.ndarray | str, **pipeline_kwargs) -> list[dict[str, Any]]:
    pipe = get_pipeline(**pipeline_kwargs) if pipeline_kwargs else get_pipeline()
    return [d.to_dict() for d in pipe.predict(image)]


def predict_and_visualize(
    image: np.ndarray | str,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    **pipeline_kwargs,
) -> InferenceResult:
    """Chạy inference và hiển thị / lưu figure."""
    from src.inference.visualize import plot_inference_result

    pipe = get_pipeline(**pipeline_kwargs) if pipeline_kwargs else get_pipeline()
    result = pipe.predict_detailed(image)
    plot_inference_result(result, show=show, save_path=save_path)
    return result
