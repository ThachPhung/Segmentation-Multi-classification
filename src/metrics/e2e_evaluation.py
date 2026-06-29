"""
Đánh giá end-to-end thống nhất: bảng tổng hệ thống (Bảng 5.14) và bảng theo lớp
dùng cùng định nghĩa GT instance + cùng logic ghép cặp.

Mặc định (khớp bảng mới):
  - GT instance = một dòng RLE hợp lệ trong train.csv
  - Match end-to-end = IoU >= ngưỡng và cùng ClassId

Tùy chọn legacy (chỉ khi cần tái hiện báo cáo cũ):
  - GT instance = connected component trên mask GT
  - Spatial match = IoU >= ngưỡng (không bắt buộc cùng lớp)
  - Phát hiện đúng hoàn toàn = spatial match + cùng lớp
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.masks import build_masks, is_valid_rle_series
from src.inference.rle_utils import rle_decode
from src.inference.roi import collect_bboxes_from_mask_chw, component_mask_in_bbox


@dataclass
class DefectInstance:
    class_id: int
    mask: np.ndarray  # (H, W) uint8 {0, 1}


def _dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    den = a.sum() + b.sum()
    if den == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2.0 * inter / (den + eps))


def _iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / (union + eps))


def extract_gt_instances_rle(
    df: pd.DataFrame,
    image_id: str,
    shape: tuple[int, int] = (256, 1600),
) -> list[DefectInstance]:
    """Mỗi RLE hợp lệ trong CSV = một instance GT."""
    rows = df[
        (df["ImageId"] == image_id) & is_valid_rle_series(df["EncodedPixels"])
    ]
    instances: list[DefectInstance] = []
    for _, row in rows.iterrows():
        mask = rle_decode(str(row["EncodedPixels"]), shape)
        if mask.sum() == 0:
            continue
        instances.append(
            DefectInstance(class_id=int(row["ClassId"]), mask=mask)
        )
    return instances


def extract_gt_instances_component(
    df: pd.DataFrame,
    image_id: str,
    shape: tuple[int, int] = (256, 1600),
    *,
    min_component_area: int = 20,
) -> list[DefectInstance]:
    """Mỗi connected component trên mask GT = một instance."""
    mask_hwc = build_masks(image_id, df, shape=shape)
    mask_chw = np.transpose(mask_hwc, (2, 0, 1))
    instances: list[DefectInstance] = []
    for class_idx, bbox in collect_bboxes_from_mask_chw(
        mask_chw, min_component_area=min_component_area
    ):
        component = component_mask_in_bbox(mask_chw[class_idx], bbox)
        if component.sum() == 0:
            continue
        instances.append(
            DefectInstance(class_id=int(class_idx) + 1, mask=component)
        )
    return instances


def extract_gt_instances(
    df: pd.DataFrame,
    image_id: str,
    shape: tuple[int, int] = (256, 1600),
    *,
    gt_mode: str = "rle",
    min_component_area: int = 20,
) -> list[DefectInstance]:
    if gt_mode == "component":
        return extract_gt_instances_component(
            df, image_id, shape, min_component_area=min_component_area
        )
    if gt_mode != "rle":
        raise ValueError(f"gt_mode không hỗ trợ: {gt_mode}")
    return extract_gt_instances_rle(df, image_id, shape)


def extract_pred_instances(
    detections,
    shape: tuple[int, int] = (256, 1600),
) -> list[DefectInstance]:
    instances: list[DefectInstance] = []
    for det in detections:
        rle = det.rle if hasattr(det, "rle") else det["rle"]
        class_id = det.class_id if hasattr(det, "class_id") else det["class_id"]
        mask = rle_decode(rle, shape)
        if mask.sum() == 0:
            continue
        instances.append(DefectInstance(class_id=int(class_id), mask=mask))
    return instances


def match_instances_greedy(
    preds: list[DefectInstance],
    gts: list[DefectInstance],
    *,
    iou_threshold: float = 0.5,
    require_same_class: bool = True,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if require_same_class and pred.class_id != gt.class_id:
                continue
            score = _iou(pred.mask, gt.mask)
            if score >= iou_threshold:
                candidates.append((score, pi, gi))

    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for score, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, score))

    return matches


@dataclass
class E2EClassStats:
    class_id: int
    gt_instances: int = 0
    predicted_instances: int = 0
    spatial_matches: int = 0
    correct_matches: int = 0
    dice_sum: float = 0.0

    @property
    def precision(self) -> float:
        if self.predicted_instances == 0:
            return 0.0
        return self.spatial_matches / self.predicted_instances

    @property
    def recall(self) -> float:
        if self.gt_instances == 0:
            return 0.0
        return self.spatial_matches / self.gt_instances

    @property
    def dice(self) -> float:
        if self.correct_matches == 0:
            return 0.0
        return self.dice_sum / self.correct_matches


@dataclass
class E2EEvaluator:
    """
    Một evaluator duy nhất — bảng theo lớp và bảng tổng hệ thống luôn khớp nhau.
    """

    num_classes: int = 4
    iou_threshold: float = 0.5
    shape: tuple[int, int] = (256, 1600)
    gt_mode: str = "rle"
    match_mode: str = "end_to_end"
    min_component_area: int = 20
    num_images: int = 0
    images_with_correct_match: int = 0
    _stats: dict[int, E2EClassStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._stats:
            self._stats = {
                c: E2EClassStats(class_id=c) for c in range(1, self.num_classes + 1)
            }

    @property
    def require_same_class_for_match(self) -> bool:
        return self.match_mode == "end_to_end"

    def update(
        self,
        gt_instances: list[DefectInstance],
        pred_instances: list[DefectInstance],
    ) -> None:
        self.num_images += 1

        for inst in gt_instances:
            self._stats[inst.class_id].gt_instances += 1
        for inst in pred_instances:
            self._stats[inst.class_id].predicted_instances += 1

        matches = match_instances_greedy(
            pred_instances,
            gt_instances,
            iou_threshold=self.iou_threshold,
            require_same_class=self.require_same_class_for_match,
        )

        image_has_correct = False
        for pi, gi, _ in matches:
            pred = pred_instances[pi]
            gt = gt_instances[gi]
            cls_id = gt.class_id
            self._stats[cls_id].spatial_matches += 1

            same_class = pred.class_id == gt.class_id
            if same_class:
                self._stats[cls_id].correct_matches += 1
                self._stats[cls_id].dice_sum += _dice(pred.mask, gt.mask)
                image_has_correct = True

        if image_has_correct:
            self.images_with_correct_match += 1

    def totals(self) -> dict[str, float | int]:
        total_gt = sum(s.gt_instances for s in self._stats.values())
        total_pred = sum(s.predicted_instances for s in self._stats.values())
        total_spatial = sum(s.spatial_matches for s in self._stats.values())
        total_correct = sum(s.correct_matches for s in self._stats.values())
        total_dice = sum(s.dice_sum for s in self._stats.values())

        return {
            "gt_instances": total_gt,
            "predicted_instances": total_pred,
            "spatial_matches": total_spatial,
            "correct_matches": total_correct,
            "precision": total_spatial / max(total_pred, 1),
            "recall": total_spatial / max(total_gt, 1),
            "correct_precision": total_correct / max(total_pred, 1),
            "correct_recall": total_correct / max(total_gt, 1),
            "dice": total_dice / max(total_correct, 1),
            "image_recall": self.images_with_correct_match / max(self.num_images, 1),
        }

    def to_per_class_dataframe(
        self,
        class_names: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        rows = []
        use_correct_for_dice = self.match_mode == "end_to_end"

        for c in range(1, self.num_classes + 1):
            s = self._stats[c]
            label = (class_names or {}).get(c, f"Lớp {c}")
            dice_val = s.dice if s.correct_matches else 0.0
            if use_correct_for_dice and s.spatial_matches and not s.correct_matches:
                dice_val = 0.0

            row = {
                "Lớp": label,
                "GT instances": s.gt_instances,
                "Predicted instances": s.predicted_instances,
                "Spatial matches": s.spatial_matches,
                "Precision": round(s.precision, 4),
                "Recall": round(s.recall, 4),
                "Dice": round(dice_val, 4),
            }
            if self.match_mode == "legacy":
                row["Phát hiện đúng hoàn toàn"] = s.correct_matches
            rows.append(row)

        t = self.totals()
        total_row = {
            "Lớp": "Tổng / Macro",
            "GT instances": int(t["gt_instances"]),
            "Predicted instances": int(t["predicted_instances"]),
            "Spatial matches": int(t["spatial_matches"]),
            "Precision": round(float(t["precision"]), 4),
            "Recall": round(float(t["recall"]), 4),
            "Dice": round(float(t["dice"]), 4),
        }
        if self.match_mode == "legacy":
            total_row["Phát hiện đúng hoàn toàn"] = int(t["correct_matches"])
        rows.append(total_row)
        return pd.DataFrame(rows)

    def to_system_summary_dataframe(self) -> pd.DataFrame:
        """Bảng tổng hệ thống (định dạng Bảng 5.14) — cùng số liệu với dòng Tổng."""
        t = self.totals()
        spatial = int(t["spatial_matches"])
        correct = int(t["correct_matches"])
        gt = int(t["gt_instances"])
        pred = int(t["predicted_instances"])

        cls_acc = correct / max(spatial, 1) if self.match_mode == "legacy" else 1.0
        if self.match_mode == "end_to_end":
            cls_acc = 1.0

        system_rows = [
            ("Tổng số ảnh kiểm thử", self.num_images),
            (
                "Số ảnh dự đoán đúng ít nhất một khuyết tật",
                self.images_with_correct_match,
            ),
            (
                "Tỷ lệ Recall ở cấp độ ảnh (%)",
                round(float(t["image_recall"]) * 100, 2),
            ),
            ("Tổng số thực thể lỗi nhãn chuẩn (GT Instances)", gt),
            ("Tổng số vùng mô hình dự đoán (Predicted Instances)", pred),
            ("Số vùng trùng khớp không gian (Spatial Matches)", spatial),
            ("Số vùng phát hiện đúng hoàn toàn (End-to-end)", correct),
            (
                "Độ chính xác phân loại trên vùng RoI trùng khớp (%)",
                round(cls_acc * 100, 2),
            ),
            (
                "Tỷ lệ bỏ sót thực thể lỗi - Miss Rate (%)",
                round((1.0 - float(t["recall"])) * 100, 2),
            ),
            (
                "Tỷ lệ báo động giả - False Alarm Rate (%)",
                round((1.0 - float(t["precision"])) * 100, 2),
            ),
            ("Instance Recall (%)", round(float(t["recall"]) * 100, 2)),
            ("Instance Precision (%)", round(float(t["precision"]) * 100, 2)),
            ("Dice trung bình (cặp match đúng lớp)", round(float(t["dice"]), 4)),
        ]
        return pd.DataFrame(
            system_rows,
            columns=["Chỉ số đánh giá hệ thống", "Giá trị thực nghiệm"],
        )


def evaluate_e2e_on_images(
    pipeline,
    df: pd.DataFrame,
    image_ids: list[str],
    image_dir,
    *,
    iou_threshold: float = 0.5,
    shape: tuple[int, int] = (256, 1600),
    class_names: dict[int, str] | None = None,
    gt_mode: str = "rle",
    match_mode: str = "end_to_end",
    min_component_area: int = 20,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, E2EEvaluator]:
    """
    Trả về (bảng_theo_lớp, bảng_tổng_hệ_thống, evaluator).
    Hai bảng dùng chung một lần chạy pipeline — dòng Tổng luôn khớp Bảng 5.14.
    """
    from pathlib import Path
    from tqdm import tqdm

    image_dir = Path(image_dir)
    evaluator = E2EEvaluator(
        iou_threshold=iou_threshold,
        shape=shape,
        gt_mode=gt_mode,
        match_mode=match_mode,
        min_component_area=min_component_area,
    )

    iterator = image_ids
    if show_progress:
        iterator = tqdm(image_ids, desc="E2E eval")

    for image_id in iterator:
        img_path = image_dir / image_id
        if not img_path.exists():
            continue

        gt_instances = extract_gt_instances(
            df,
            image_id,
            shape=shape,
            gt_mode=gt_mode,
            min_component_area=min_component_area,
        )
        detections = pipeline.predict(str(img_path))
        pred_instances = extract_pred_instances(detections, shape=shape)
        evaluator.update(gt_instances, pred_instances)

    per_class = evaluator.to_per_class_dataframe(class_names=class_names)
    system = evaluator.to_system_summary_dataframe()
    return per_class, system, evaluator
