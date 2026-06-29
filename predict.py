#!/usr/bin/env python3
"""
Script chạy pipeline suy luận trên ảnh hoặc thư mục.

Ví dụ:
  python predict.py --image Dataset/train_images/0a4ad45a5.jpg
  python predict.py --image-dir Dataset/train_images --limit 5
  python predict.py --image test.jpg --seg-weights weights/best_Attunet_efficientnet_b3.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Đảm bảo import từ root project
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.pipeline import DefectInferencePipeline  # noqa: E402
from src.inference.visualize import plot_inference_result  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suy luận khuyết tật thép: U-Net phân đoạn + EfficientNet-B3 phân loại RoI"
    )
    parser.add_argument("--image", type=str, help="Đường dẫn một ảnh")
    parser.add_argument("--image-dir", type=str, help="Thư mục ảnh (batch)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số ảnh (0 = tất cả)")
    parser.add_argument("--seg-weights", type=str, default=None, help="Checkpoint segmentation")
    parser.add_argument("--cls-weights", type=str, default=None, help="Checkpoint classifier")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ghi kết quả JSON (tùy chọn)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show", action="store_true", help="Hiển thị visualization")
    parser.add_argument("--save-vis", type=str, default=None, help="Lưu PNG visualization")
    return parser.parse_args()


def run_on_path(
    pipe: DefectInferencePipeline,
    image_path: Path,
    quiet: bool,
    show: bool = False,
    save_vis: str | None = None,
) -> list[dict]:
    if show or save_vis:
        result = pipe.predict_detailed(str(image_path))
        plot_inference_result(
            result,
            save_path=save_vis,
            show=show and save_vis is None,
        )
        results = [d.to_dict() for d in result.detections]
    else:
        results = [d.to_dict() for d in pipe.predict(str(image_path))]
    if not quiet:
        print(f"\n=== {image_path.name} ===")
        if not results:
            print("  (không phát hiện khuyết tật)")
        for i, det in enumerate(results, 1):
            rle_preview = det["rle"][:60] + ("..." if len(det["rle"]) > 60 else "")
            print(
                f"  [{i}] {det['defect_name']} | class_id={det['class_id']} | "
                f"conf={det['confidence']:.3f} | bbox={det['bbox']}"
            )
            print(f"       RLE: {rle_preview}")
    return results


def main() -> None:
    args = parse_args()
    if not args.image and not args.image_dir:
        print("Cần --image hoặc --image-dir", file=sys.stderr)
        sys.exit(1)

    kwargs = {}
    if args.seg_weights:
        kwargs["seg_checkpoint"] = args.seg_weights
    if args.cls_weights:
        kwargs["cls_checkpoint"] = args.cls_weights
    if args.device:
        kwargs["device"] = args.device

    pipe = DefectInferencePipeline(**kwargs)

    all_output: dict[str, list] = {}
    paths: list[Path] = []

    if args.image:
        paths.append(Path(args.image))
    if args.image_dir:
        d = Path(args.image_dir)
        paths.extend(sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")))
        if args.limit > 0:
            paths = paths[: args.limit]

    for p in paths:
        if not p.exists():
            print(f"Bỏ qua (không tồn tại): {p}", file=sys.stderr)
            continue
        vis_path = args.save_vis if args.save_vis and len(paths) == 1 else None
        if args.save_vis and len(paths) > 1:
            vis_path = str(Path("outputs/vis") / f"{p.stem}_inference.png")
        all_output[p.name] = run_on_path(
            pipe, p, args.quiet, show=args.show, save_vis=vis_path
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_output, f, ensure_ascii=False, indent=2)
        print(f"\nĐã ghi: {out_path}")


if __name__ == "__main__":
    main()
