#!/usr/bin/env python3
"""Kiểm tra nhanh môi trường trước khi chạy inference / đánh giá."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_WEIGHTS = [
    "best_Attunet_efficientnet_b3.pth",
    "classifier_best.pth",
]
OPTIONAL_WEIGHTS = [
    "thresholds_seg.npy",
    "thresholds_cls.npy",
]
DATASET_PATHS = [
    Path("Dataset/train.csv"),
    Path("Dataset/train_images"),
    Path("Dataset/test_images"),
]


def main() -> int:
    print("=== Kiểm tra môi trường ===\n")
    ok = True

    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  ✗ Cần Python 3.10 trở lên.")
        ok = False
    else:
        print("  ✓ Phiên bản Python phù hợp.")

    packages = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "cv2",
        "albumentations",
        "matplotlib",
        "sklearn",
    ]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ Thiếu package: {pkg}  →  pip install -r requirements.txt")
            ok = False

    weights_dir = ROOT / "weights"
    print(f"\nWeights ({weights_dir}):")
    for name in REQUIRED_WEIGHTS:
        path = weights_dir / name
        if path.exists():
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ Thiếu {name} — xem Readme.md mục Chuẩn bị weights")
            ok = False

    for name in OPTIONAL_WEIGHTS:
        path = weights_dir / name
        mark = "✓" if path.exists() else "○"
        note = "" if path.exists() else " (tùy chọn, mặc định ngưỡng 0.5)"
        print(f"  {mark} {name}{note}")

    print("\nDataset:")
    dataset_ok = True
    for path in DATASET_PATHS:
        full = ROOT / path
        if full.exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ Thiếu {path} — xem Readme.md mục Tải dataset")
            dataset_ok = False
    if not dataset_ok:
        ok = False

    try:
        from src.inference.device import resolve_device

        device = resolve_device()
        print(f"\nDevice suy luận mặc định: {device}")
    except Exception as exc:
        print(f"\n✗ Lỗi import pipeline: {exc}")
        ok = False

    print()
    if ok:
        print("✓ Sẵn sàng chạy. Ví dụ:")
        print('  python run_inference.py --image Dataset/train_images/0002cc93b.jpg --save-vis outputs/vis/demo.png')
        print("  python run_e2e_eval.py --limit 20 --device cpu")
        return 0

    print("✗ Còn thiếu bước cài đặt — xem Readme.md.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
