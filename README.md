# fall-benchmark

Benchmark 8 video action-recognition models cho fall detection trên `MergedFallDataset`, hai chế độ
(Case 1 **from-scratch** / Case 2 **fine-tune pretrained**), rồi đề xuất kiến trúc mới. Xem
[../PROJECT_PLAN.md](../PROJECT_PLAN.md).

## Cấu trúc
```
fall-benchmark/
├── common/                 # DÙNG CHUNG (1 nguồn sự thật)
│   ├── dataset.py          #   đọc manifest.csv: mp4 + chuỗi frame, subset cân bằng
│   ├── transforms.py       #   sampling T frame + aug + chuẩn hóa Kinetics
│   ├── metrics.py          #   bộ metric đã KHÓA (BAcc/Sens/Spec/F1/Macro-F1/AUC)
│   └── engine.py           #   vòng train/eval/checkpoint (giống nhau cho cả 2 case → công bằng)
├── train_from_scratch/     # CASE 1 — random init
│   ├── models/             #   i3d, slowfast, x3d, tsm, videomae, timesformer, videomamba
│   ├── train.py  config.yaml
├── fine_tune/              # CASE 2 — pretrained → fine-tune
│   ├── models/             #   7 model trên (pretrained=True) + vjepa2 (chỉ ở đây)
│   ├── finetune.py  config.yaml
└── requirements.txt
```

## Cài đặt
```bash
pip install -r requirements.txt
# VideoMamba kernels chính chủ (CHỈ Linux+CUDA, vd Kaggle/Lambda):
#   pip install causal-conv1d mamba-ssm   # + thêm repo OpenGVLab/VideoMamba vào PYTHONPATH
```
> ⚠️ Windows/Py3.13: `decord` và `mamba-ssm` hay lỗi build → code đã có fallback (torchvision/pyav,
> VideoMamba-lite). Train thật nên chạy trên Linux (Kaggle/Lambda).

## Chạy nhanh (smoke test, mẫu nhỏ)
```bash
cd train_from_scratch && python train.py --model x3d --subset 100 --epochs 3     # Case 1
cd fine_tune        && python finetune.py --model videomae --subset 100 --epochs 3 # Case 2
```
Model nhẹ hợp 6GB local: **x3d, tsm, videomamba**. Còn lại (i3d, slowfast, videomae, timesformer,
vjepa2) → Kaggle/Lambda.

## Chạy thật (sau khi build split CS/CV ở Phase C)
```bash
python train.py   --model x3d --train-split ../splits/cs/train.csv --val-split ../splits/cs/val.csv
python finetune.py --model x3d --train-split ../splits/cs/train.csv --val-split ../splits/cs/val.csv
```

## 8 model
| name | họ | hội nghị | rank | from-scratch | fine-tune |
|------|-----|----------|:----:|:---:|:---:|
| i3d | 3D-CNN | CVPR'17 | A* | ✅ | ✅ K400 |
| slowfast | 3D-CNN 2way | ICCV'19 | A* | ✅ | ✅ K400 |
| x3d | 3D-CNN eff | CVPR'20 | A* | ✅ | ✅ K400 |
| tsm | 2D+shift | ICCV'19 | A* | ✅ | ✅ ImageNet |
| videomae | Transformer | NeurIPS'22 | A* | ✅ | ✅ K400 |
| timesformer | Transformer | ICML'21 | A* | ✅ | ✅ K400 |
| videomamba | SSM/Mamba | ECCV'24 | A* | ✅ | ✅ official |
| **vjepa2** | Foundation | arXiv'25 | preprint | ❌ | ✅ chỉ đây |

## Trạng thái
✅ Kiến trúc 2 folder đã dựng. ⏳ Tiếp theo (Phase C): cài deps + build split CS/CV leak-safe rồi
chạy smoke test. Hiện `common/dataset.py` dùng **random subset** (chỉ để smoke); KHÔNG report số
trên split random.
