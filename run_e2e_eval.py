#!/usr/bin/env python3
"""
Đánh giá end-to-end thống nhất: bảng theo lớp + bảng tổng hệ thống (Bảng 5.14).

Ví dụ:
  python run_e2e_eval.py
  python run_e2e_eval.py --limit 50 --device cpu
  python run_e2e_eval.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_config import DEFECT_NAMES_VI  # noqa: E402
from src.inference.device import resolve_device  # noqa: E402
from src.inference.pipeline import DefectInferencePipeline  # noqa: E402
from src.metrics.e2e_evaluation import evaluate_e2e_on_images  # noqa: E402


def load_train_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ImageId_ClassId" in df.columns:
        split_cols = df["ImageId_ClassId"].str.split("_", expand=True)
        df["ImageId"] = split_cols[0]
        df["ClassId"] = split_cols[1].astype(int)
    df["ClassId"] = df["ClassId"].astype(int)
    return df


def build_val_ids(df: pd.DataFrame, val_split: float, seed: int) -> list[str]:
    all_ids = df["ImageId"].unique()
    _, val_ids = train_test_split(
        all_ids,
        test_size=val_split,
        random_state=seed,
        shuffle=True,
    )
    return list(val_ids)


def main():
    parser = argparse.ArgumentParser(
        description="E2E eval thống nhất: bảng theo lớp + bảng tổng hệ thống"
    )
    parser.add_argument("--csv", type=str, default="Dataset/train.csv")
    parser.add_argument("--image-dir", type=str, default="Dataset/train_images")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Thư mục lưu e2e_per_class.csv và e2e_system_summary.csv",
    )
    parser.add_argument(
        "--gt-mode",
        choices=["rle", "component"],
        default="rle",
        help="Cách đếm GT instance (mặc định rle — khớp bảng theo lớp mới)",
    )
    parser.add_argument(
        "--match-mode",
        choices=["end_to_end", "legacy"],
        default="end_to_end",
        help="end_to_end: IoU+cùng lớp | legacy: IoU trước, lớp sau (báo cáo cũ)",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=20,
        help="Chỉ dùng khi --gt-mode component",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    image_dir = Path(args.image_dir)
    if not csv_path.exists():
        parser.error(f"Không tìm thấy CSV: {csv_path}")
    if not image_dir.is_dir():
        parser.error(f"Không tìm thấy thư mục ảnh: {image_dir}")

    df = load_train_df(csv_path)
    val_ids = build_val_ids(df, args.val_split, args.seed)
    if args.limit > 0:
        val_ids = val_ids[: args.limit]

    print(f"Validation images: {len(val_ids)}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"GT mode: {args.gt_mode} | Match mode: {args.match_mode}")

    device = resolve_device(args.device)
    pipe = DefectInferencePipeline(device=device)
    print(f"Device: {pipe.device}")

    per_class, system, evaluator = evaluate_e2e_on_images(
        pipeline=pipe,
        df=df,
        image_ids=val_ids,
        image_dir=image_dir,
        iou_threshold=args.iou_threshold,
        class_names=DEFECT_NAMES_VI,
        gt_mode=args.gt_mode,
        match_mode=args.match_mode,
        min_component_area=args.min_component_area,
    )

    print("\n=== Bảng theo lớp ===")
    print(per_class.to_string(index=False))

    print("\n=== Bảng tổng hệ thống (Bảng 5.14) ===")
    print(system.to_string(index=False))

    t = evaluator.totals()
    total_row = per_class.iloc[-1]
    assert int(total_row["GT instances"]) == int(t["gt_instances"])
    assert int(total_row["Spatial matches"]) == int(t["spatial_matches"])
    print("\n✓ Dòng Tổng bảng theo lớp khớp bảng tổng hệ thống.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_path = out_dir / "e2e_per_class.csv"
    system_path = out_dir / "e2e_system_summary.csv"
    per_class.to_csv(per_class_path, index=False)
    system.to_csv(system_path, index=False)
    print(f"\nĐã lưu:\n  {per_class_path}\n  {system_path}")


if __name__ == "__main__":
    main()
