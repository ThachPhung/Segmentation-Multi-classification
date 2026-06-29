#!/usr/bin/env python3
"""
Chạy pipeline suy luận — chỉ cần weights/ đã có checkpoint.

Ví dụ:
  python run_inference.py --image Dataset/train_images/0002cc93b.jpg --show
  python run_inference.py --image path.jpg --save-vis outputs/vis/result.png
  python run_inference.py --image-dir Dataset/train_images --limit 3 --show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.pipeline import DefectInferencePipeline  # noqa: E402
from src.inference.submission import predict_folder_to_submission  # noqa: E402
from src.inference.visualize import plot_inference_result  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline Attention U-Net + EfficientNet-B3")
    parser.add_argument("--image", type=str, help="Một ảnh")
    parser.add_argument("--image-dir", type=str, help="Thư mục ảnh")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="JSON kết quả")
    parser.add_argument("--submission", type=str, default=None, help="CSV format Kaggle")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị figure: input | segmentation | output | classification",
    )
    parser.add_argument(
        "--save-vis",
        type=str,
        default=None,
        help="Lưu figure PNG (vd. outputs/vis/result.png). Tự lưu vào outputs/vis/<tên_ảnh>.png nếu chỉ --show",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="outputs/vis",
        help="Thư mục lưu ảnh khi dùng --show với nhiều ảnh",
    )
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Cần --image hoặc --image-dir")

    kwargs = {}
    if args.device:
        kwargs["device"] = args.device

    pipe = DefectInferencePipeline(**kwargs)
    print(f"Device: {pipe.device}")
    print(f"Seg thresholds: {[round(float(t), 2) for t in pipe.seg_thresholds]}")
    print(f"Cls thresholds: {[round(float(t), 2) for t in pipe.cls_thresholds]}")

    paths: list[Path] = []
    if args.image:
        paths.append(Path(args.image))
    if args.image_dir:
        d = Path(args.image_dir)
        paths.extend(sorted(d.glob("*.jpg")))
        if args.limit > 0:
            paths = paths[: args.limit]

    vis_dir = Path(args.vis_dir)
    do_vis = args.show or args.save_vis is not None

    all_results = {}
    for p in paths:
        if not p.exists():
            print(f"Bỏ qua: {p}")
            continue

        result = pipe.predict_detailed(str(p))
        dets = [d.to_dict() for d in result.detections]
        all_results[p.name] = dets

        print(f"\n{p.name}: {len(dets)} detection(s)")
        for i, det in enumerate(dets, 1):
            print(f"  [{i}] {det['defect_name']} | conf={det['confidence']:.3f}")

        if do_vis:
            if args.save_vis and len(paths) == 1:
                vis_path = Path(args.save_vis)
            else:
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f"{p.stem}_inference.png"

            # Chỉ hiện cửa sổ với 1 ảnh; batch thì lưu file
            show_window = args.show and len(paths) == 1
            plot_inference_result(
                result,
                save_path=vis_path,
                show=show_window,
            )
            print(f"  → Đã lưu visualization: {vis_path}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nJSON: {out}")

    if args.submission and args.image_dir:
        predict_folder_to_submission(args.image_dir, args.submission, pipeline=pipe)
        print(f"Submission: {args.submission}")


if __name__ == "__main__":
    main()
