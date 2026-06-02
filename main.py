"""
Điểm vào nhanh cho pipeline suy luận.

Chạy: python main.py [đường_dẫn_ảnh]
Hoặc:  python predict.py --image ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        print("Hoặc:  python predict.py --image <image_path>")
        sys.exit(1)

    from src.inference.pipeline import DefectInferencePipeline

    image_path = sys.argv[1]
    pipe = DefectInferencePipeline()
    results = [d.to_dict() for d in pipe.predict(image_path)]
    if not results:
        print("Không phát hiện khuyết tật.")
        return
    for i, det in enumerate(results, 1):
        print(f"[{i}] {det['defect_name']} (class {det['class_id']})")
        print(f"    confidence={det['confidence']:.4f}")
        print(f"    bbox={det['bbox']}")
        print(f"    rle={det['rle'][:100]}{'...' if len(det['rle']) > 100 else ''}")


if __name__ == "__main__":
    main()
