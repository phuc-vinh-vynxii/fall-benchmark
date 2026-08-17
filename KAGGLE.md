# Chạy benchmark trên Kaggle / Lambda

Thử nghiệm nhanh: train **cả 8 model** một chút trên **CS-train subset** (leak-safe), eval trên
**CS-val**, ra bảng so sánh để quyết định model nào đáng train tiếp.

---

## A. KAGGLE

### 1. Chuẩn bị 2 input
- **Data:** dataset đã upload `phucvinhvynxii/fall-dataset` (chứa `manifest.csv`, `videos/`, `frames/`).
- **Code:** nén folder `fall-benchmark/` (đã gồm `splits/cs/` build sẵn) → upload thành 1 Kaggle Dataset
  mới, ví dụ `fall-benchmark-code`. *(Cách khác: `git clone` nếu bạn đẩy code lên GitHub.)*

### 2. Tạo Notebook, bật **GPU (T4 x2 hoặc P100)** + **Internet ON**
> Internet bắt buộc: để `pip install` và tải trọng số pretrained (VideoMAE/TimeSformer/V-JEPA 2 từ HuggingFace).

### 3. Các cell

```python
# Cell 1 — copy code ra working dir (vì /kaggle/input chỉ đọc, code cần ghi runs/results)
!cp -r /kaggle/input/fall-benchmark-code/fall-benchmark /kaggle/working/
%cd /kaggle/working/fall-benchmark
```

```python
# Cell 2 — cài deps
!pip install -q pytorchvideo transformers timm einops av decord scikit-learn pyyaml
# (torch/torchvision đã có sẵn trên Kaggle)
# VideoMamba kernels chính chủ (tùy chọn, để chạy bản official thay vì lite):
# !pip install -q causal-conv1d mamba-ssm
```

```python
# Cell 3 — KIỂM TRA đường dẫn data (sửa nếu cần)
import os
DATA = "/kaggle/input/fall-dataset"          # phải chứa manifest.csv
print(os.path.exists(f"{DATA}/manifest.csv"), os.listdir(DATA)[:10])
```

```python
# Cell 4 — chạy thử cả 8 model (scratch + finetune) trên subset nhỏ
!python run_all.py --data-root /kaggle/input/fall-dataset \
                   --modes both --subset 80 --epochs 3 --num-workers 2
# Nhẹ & nhanh hơn: chỉ vài model, chỉ finetune:
# !python run_all.py --data-root /kaggle/input/fall-dataset \
#                    --models x3d tsm videomamba videomae --modes finetune --subset 80 --epochs 5
```

```python
# Cell 5 — xem bảng kết quả
import pandas as pd
pd.read_csv("results/experiment_table.csv")
```

### 4. Test trên video của bạn (gửi sau)
Upload vài video test (mp4) thành 1 dataset, rồi:
```python
!python predict.py --ckpt finetune/runs/x3d_best.pt --videos /kaggle/input/<your-test-videos>
# Có nhãn -> thêm metrics: CSV cột path,label (FALL/NoFALL)
# !python predict.py --ckpt finetune/runs/x3d_best.pt --list /kaggle/input/<...>/test_labeled.csv
```

---

## B. LAMBDA CLOUD (máy Linux thường)
```bash
cd fall-benchmark
pip install -r requirements.txt
pip install causal-conv1d mamba-ssm          # tùy chọn: VideoMamba official
python run_all.py --data-root /path/to/MergedFallDataset --modes both --subset 80 --epochs 3
```

---

## Đọc kết quả
Bảng có: `mode, model, balanced_accuracy, sensitivity, specificity, f1, auc_roc`.

| Tín hiệu | Ý nghĩa |
|----------|---------|
| **sensitivity cao** | Bắt được cú ngã (quan trọng nhất cho fall detection) |
| **balanced_accuracy >> 0.5** | Học thật, không đoán bừa |
| **auc_roc cao** | Phân biệt tốt bất kể ngưỡng |
| BAcc ~0.5 / sensitivity ~0 | Chưa học được → cần thêm data/epoch hoặc bỏ |

So **finetune vs scratch**: nếu finetune cao hẳn còn scratch ~0.5 trên subset → đúng như dự đoán
(model cần pretrain; đây chính là "câu chuyện" Case 1 vs Case 2 cho paper).

> ⚠️ Đây là kết quả **subset nhỏ** chỉ để sàng lọc model. Số liệu chính thức = chạy full split
> (`--subset 0`) ở Phase H.
