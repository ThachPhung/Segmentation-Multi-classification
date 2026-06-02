"""
Cấu hình pipeline suy luận (Inference).

Đặt checkpoint sau khi train từ notebook `graduation-project-phase-2.ipynb`:
  - weights/best_Attunet_efficientnet_b3.pth
  - weights/classifier_best.pth
  - weights/thresholds_seg.npy   (tùy chọn, 4 giá trị)
  - weights/thresholds_cls.npy   (tùy chọn, 4 giá trị)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

INFERENCE_CFG = {
  # Kích thước ảnh Severstal (H, W)
  "image_height": 256,
  "image_width": 1600,
  "num_classes": 4,
  # Segmentation
  "seg_dropout": 0.1,
  "seg_thresholds": [0.5, 0.5, 0.5, 0.5],
  "seg_checkpoint": WEIGHTS_DIR / "best_Attunet_efficientnet_b3.pth",
  "seg_thresholds_path": WEIGHTS_DIR / "thresholds_seg.npy",
  # Classifier RoI
  "cls_crop_size": (448, 448),
  "cls_dropout": 0.35,
  "cls_thresholds": [0.5, 0.5, 0.5, 0.5],
  "cls_checkpoint": WEIGHTS_DIR / "classifier_best.pth",
  "cls_thresholds_path": WEIGHTS_DIR / "thresholds_cls.npy",
  "cls_use_tta": True,
  "cls_backbone": "efficientnet_b3",
  # Hậu xử lý RoI
  "bbox_margin": 12,
  "min_component_area": 20,
  "max_detections_per_image": 16,
  "label_min_pixels": 8,
}

# Tên hiển thị (tiếng Việt) — Severstal 4 loại khuyết tật
DEFECT_NAMES_VI = {
    1: "Vết xước (Scratches)",
    2: "Tạp chất (Inclusions)",
    3: "Bề mặt lõm (Pitted surface)",
    4: "Vết bẩn (Stains)",
}

DEFECT_NAMES_EN = {
    1: "Scratches",
    2: "Inclusions",
    3: "Pitted surface",
    4: "Stains",
}
