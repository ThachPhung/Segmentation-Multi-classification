"""Xuất submission Kaggle từ kết quả pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.inference.pipeline import DefectInferencePipeline


def detections_to_submission_rows(
    image_id: str,
    detections: list[dict],
    num_classes: int = 4,
) -> list[list]:
    """Mỗi detection → một dòng (ImageId, ClassId, EncodedPixels)."""
    rows = []
    for det in detections:
        rle = det.get("rle", "")
        if not rle:
            continue
        rows.append([image_id, int(det["class_id"]), rle])

    present = {r[1] for r in rows}
    for class_id in range(1, num_classes + 1):
        if class_id not in present:
            rows.append([image_id, class_id, np.nan])

    return rows


def predict_folder_to_submission(
    image_dir: str | Path,
    output_csv: str | Path,
    pipeline: DefectInferencePipeline | None = None,
    pattern: str = "*.jpg",
) -> pd.DataFrame:
    image_dir = Path(image_dir)
    pipe = pipeline or DefectInferencePipeline()
    all_rows = []

    for image_path in sorted(image_dir.glob(pattern)):
        dets = [d.to_dict() for d in pipe.predict(str(image_path))]
        all_rows.extend(detections_to_submission_rows(image_path.name, dets))

    df = pd.DataFrame(all_rows, columns=["ImageId", "ClassId", "EncodedPixels"])
    empty = df["EncodedPixels"].astype(str).str.strip().isin(("", "nan"))
    df.loc[empty, "EncodedPixels"] = np.nan
    df.to_csv(output_csv, index=False)
    return df
