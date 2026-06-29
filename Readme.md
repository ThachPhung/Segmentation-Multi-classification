# Phát hiện khuyết tật thép (Severstal)

Pipeline **hai giai đoạn**: Attention U-Net + EfficientNet-B3 (phân đoạn) → RoI + EfficientNet-B3 multi-label (phân loại). Nhận ảnh, trả về **mã RLE** và **tên loại khuyết tật**.

## Yêu cầu

- Python **3.10+**
- RAM ≥ 8 GB (khuyến nghị)
- GPU (CUDA / MPS) tùy chọn; có thể chạy CPU

## Cài đặt

```bash
cd "Graduation Project"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python verify_setup.py
```

## Chuẩn bị dữ liệu và weights

### Dataset

Nguồn: [Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection) (Kaggle).

```bash
pip install kaggle
# Đặt kaggle.json vào ~/.kaggle/
kaggle competitions download -c severstal-steel-defect-detection
unzip severstal-steel-defect-detection.zip -d Dataset
```

Cấu trúc cần có:

```
Dataset/
├── train.csv
├── train_images/*.jpg
└── test_images/*.jpg
```

### Weights

Đặt vào thư mục `weights/`:

| File | Bắt buộc |
|------|----------|
| `best_Attunet_efficientnet_b3.pth` | Có — Stage 1 segmentation |
| `classifier_best.pth` | Có — Stage 2 classifier |
| `thresholds_seg.npy` | Không — ngưỡng Stage 1 (mặc định 0.5) |
| `thresholds_cls.npy` | Không — ngưỡng Stage 2 (mặc định 0.5) |

## Script chính

| Script | Mục đích |
|--------|----------|
| `verify_setup.py` | Kiểm tra Python, packages, weights, dataset |
| `main.py` | Suy luận một ảnh, in ra terminal |
| `run_inference.py` | Suy luận: batch, JSON, visualization, submission Kaggle |
| `predict.py` | CLI suy luận thay thế |
| `run_e2e_eval.py` | Đánh giá end-to-end trên tập validation |

## Quy trình chạy

```bash
# 1. Kiểm tra môi trường
python verify_setup.py

# 2. Suy luận một ảnh
python main.py Dataset/train_images/0002cc93b.jpg

# 3. Suy luận + lưu hình minh họa
python run_inference.py \
  --image Dataset/train_images/0002cc93b.jpg \
  --save-vis outputs/vis/demo.png

# 4. Đánh giá nhanh (20 ảnh)
python run_e2e_eval.py --limit 20 --device cpu

# 5. Đánh giá đầy đủ trên tập validation
python run_e2e_eval.py
```

## Kiểm tra môi trường

```bash
python verify_setup.py
```

Kiểm tra Python ≥ 3.10, các package trong `requirements.txt`, file trong `weights/`, và `Dataset/`. In device mặc định (CUDA → MPS → CPU).

## Suy luận (inference)

### `main.py`

```bash
python main.py Dataset/train_images/0002cc93b.jpg
```

### `run_inference.py`

```bash
# Hiển thị: input | segmentation | output | classification
python run_inference.py --image Dataset/train_images/0002cc93b.jpg --show

# Lưu PNG
python run_inference.py --image path.jpg --save-vis outputs/vis/result.png

# Nhiều ảnh + JSON
python run_inference.py --image-dir Dataset/train_images --limit 5 --output outputs/predictions.json

# Submission Kaggle (cần --image-dir)
python run_inference.py --image-dir Dataset/test_images --submission outputs/submission.csv

# Chọn device
python run_inference.py --image path.jpg --device cpu
python run_inference.py --image path.jpg --device cuda
python run_inference.py --image path.jpg --device mps
```

| Tham số | Mô tả |
|---------|--------|
| `--image` | Một ảnh |
| `--image-dir` | Thư mục `*.jpg` |
| `--limit` | Giới hạn số ảnh (0 = tất cả) |
| `--output` | Lưu JSON kết quả |
| `--submission` | Lưu CSV Kaggle (cần `--image-dir`) |
| `--device` | `cpu`, `cuda`, `mps` |
| `--show` | Hiện figure matplotlib |
| `--save-vis` | Đường dẫn PNG |
| `--vis-dir` | Thư mục lưu ảnh khi batch (mặc định `outputs/vis`) |

### `predict.py`

```bash
python predict.py --image Dataset/train_images/0002cc93b.jpg
python predict.py --image-dir Dataset/train_images --limit 5 --output outputs/predictions.json
python predict.py --image path.jpg --seg-weights weights/best_Attunet_efficientnet_b3.pth --cls-weights weights/classifier_best.pth
python predict.py --image path.jpg --show
python predict.py --image path.jpg --save-vis outputs/vis/result.png
```

| Tham số | Mô tả |
|---------|--------|
| `--image` / `--image-dir` | Ảnh hoặc thư mục |
| `--limit` | Giới hạn số ảnh |
| `--seg-weights` / `--cls-weights` | Checkpoint tùy chỉnh |
| `--device` | `cpu` hoặc `cuda` |
| `--output` | File JSON |
| `--quiet` | Không in chi tiết |
| `--show` / `--save-vis` | Visualization |

## Đánh giá end-to-end

Chạy pipeline trên **20%** `train.csv` (validation, `seed=42`), so với ground truth, xuất CSV:

```bash
python run_e2e_eval.py
python run_e2e_eval.py --limit 50 --device cpu
python run_e2e_eval.py --output-dir outputs
```

| Tham số | Mặc định | Mô tả |
|---------|----------|--------|
| `--csv` | `Dataset/train.csv` | File nhãn |
| `--image-dir` | `Dataset/train_images` | Thư mục ảnh |
| `--val-split` | `0.2` | Tỷ lệ validation |
| `--seed` | `42` | Seed chia train/val |
| `--limit` | `0` | Giới hạn ảnh (0 = tất cả) |
| `--iou-threshold` | `0.5` | Ngưỡng IoU ghép cặp |
| `--device` | auto | `cpu`, `cuda`, `mps` |
| `--output-dir` | `outputs` | Thư mục lưu CSV |
| `--gt-mode` | `rle` | `rle` hoặc `component` |
| `--match-mode` | `end_to_end` | `end_to_end` hoặc `legacy` |

**Output:**

- `outputs/e2e_per_class.csv` — Precision, Recall, Dice theo lớp
- `outputs/e2e_system_summary.csv` — chỉ số tổng hệ thống

Mặc định: mỗi dòng RLE hợp lệ trong CSV = một GT instance; match khi IoU ≥ ngưỡng **và** cùng `ClassId`.

## Sử dụng trong Python

```python
from src.inference.pipeline import predict, DefectInferencePipeline

results = predict("Dataset/train_images/0a4ad45a5.jpg")
for det in results:
    print(det["defect_name"], det["confidence"])

pipe = DefectInferencePipeline(device="cuda")
detections = pipe.predict("path/to/image.jpg")

result = pipe.predict_detailed("path/to/image.jpg")
from src.inference.visualize import plot_inference_result
plot_inference_result(result, save_path="outputs/vis/out.png", show=False)

from src.inference.submission import predict_folder_to_submission
predict_folder_to_submission("Dataset/test_images", "outputs/submission.csv", pipeline=pipe)
```

## Định dạng kết quả

| Trường | Mô tả |
|--------|--------|
| `class_id` | 1–4 |
| `defect_name` | Tên tiếng Việt + tiếng Anh |
| `rle` | RLE mask ảnh 256×1600 |
| `bbox` | `(x1, y1, x2, y2)` |
| `confidence` | Xác suất classifier |
| `probabilities` | Xác suất 4 lớp |

| ClassId | Tên |
|--------|-----|
| 1 | Vết xước (Scratches) |
| 2 | Tạp chất (Inclusions) |
| 3 | Bề mặt lõm (Pitted surface) |
| 4 | Vết bẩn (Stains) |

## Cấu hình

Chỉnh trong `inference_config.py` → `INFERENCE_CFG`: kích thước ảnh (256×1600), ngưỡng seg/cls, `bbox_margin`, `max_detections_per_image`, `cls_use_tta`, v.v.

## Xử lý lỗi

| Lỗi | Cách xử lý |
|-----|------------|
| Không tìm thấy checkpoint | Đặt file `.pth` vào `weights/` |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Không tìm thấy CSV / ảnh | Tải dataset theo mục trên |
| `Cần --image hoặc --image-dir` | Thêm một trong hai tham số |
| Submission không tạo file | Cần cả `--image-dir` và `--submission` |
| Chạy chậm | Dùng `--limit` khi eval; hoặc `--device cuda` / `mps` |
