# tsm_bridge — nối MergedFallDataset vào falling-net (TSM)

`falling-net` (tyu0912/falling-net) vendor lại TSM của MIT HAN Lab. Nó không đọc video: `TSNDataSet`
mở sẵn `<root>/jpg/<LABEL>/<clip>/img_00001.jpg` theo một file list `"<đường_dẫn> <số_frame> <class_id>"`.
Thư mục này chỉ làm việc chuyển đổi đó. **Không sửa gì trong model.**

## Chạy

```bash
export W251_ROOT=$HOME/data/w251fall

bash tsm_bridge/setup_tsm.sh $W251_ROOT $HOME/falling-net

python tsm_bridge/prep_frames.py --job train --data-root <thư_mục_có_manifest.csv> --out $W251_ROOT
python tsm_bridge/prep_frames.py --job test  --test-root <dataset-test> \
       --test-manifest <fall-testset-clean/test_clean_manifest.csv> --out $W251_ROOT
python tsm_bridge/gen_lists.py --root $W251_ROOT --test-manifest <.../test_clean_manifest.csv>

cd $HOME/falling-net/train && python main.py w251fall RGB \
    --arch mobilenetv2 --num_segments 8 --shift --shift_div=8 --shift_place=blockres \
    --consensus_type=avg --epochs 25 --batch-size 8 -j 8 --npb

python tsm_bridge/infer_testset.py --ckpt <ckpt.best.pth.tar> --root $W251_ROOT --list extest
```

Hoặc chạy cả chuỗi: `bash tsm_bridge/pipeline.sh` (xem `--limit`, `EPOCHS`, `NO_TERMINATE`).

## Những chỗ khác với công cụ gốc của falling-net, và vì sao

| Công cụ gốc | Vấn đề | Ở đây |
|---|---|---|
| `tools/vid2img_w251Fall.py` | `if '.avi' not in file_name: return` — bỏ hết `.mp4` | `prep_frames.py` nhận mp4/avi/mov/mkv |
| — | 437 clip đã có sẵn frame, trích lại là phí và giảm chất lượng lần hai | symlink thẳng sang `img_%05d.jpg` |
| — | `extracted_falls` đặt tên `frame_1894.jpg` theo index video gốc → sort chữ cái làm loạn trình tự | sort theo số trong tên |
| `tools/gen_label_w251fall.py` | chia train/val bằng cách dò `.train`/`.val` trong tên thư mục → phá split cross-subject | `gen_lists.py` đọc `splits/cs/*.csv` |
| `ops/dataset_config.py` | `ROOT_DATASET = '/data/'` hardcode | symlink `/data/w251fall`, không sửa code gốc |
| `test_models.py` | in top-1/top-5, không có metric của paper | `infer_testset.py` dùng `common/metrics.py` |

## Vá cho PyTorch mới — `setup_tsm.sh` làm hết

| Cần | Vì sao |
|---|---|
| `pip install tensorboardX` | `main.py` import trực tiếp |
| symlink `/data/w251fall` | `ops/dataset_config.py` hardcode `ROOT_DATASET = '/data/'` |
| `ops/utils.py`: `correct[:k].view(-1)` → `.reshape(-1)` | sau `topk()/t()/eq()` tensor không còn liền mạch bộ nhớ; PyTorch mới báo *view size is not compatible with input tensor's size and stride* |

`main.py` thì sạch — repo đã dùng `clip_grad_norm_`, không còn `.data[0]`.

Phát sinh lỗi nào khác thì thêm bước vá vào `setup_tsm.sh` (kiểu idempotent: `grep` trước khi `sed`)
và ghi một dòng vào bảng trên.

## Quy ước nhãn

`NoFALL = 0`, `FALL = 1` — khớp `common/metrics.py` (`POS_INDEX = 1`), nên số liệu so sánh trực tiếp
được với 8 model trong `run_all.py`. `categories.txt` ghi mỗi dòng một tên lớp theo đúng thứ tự đó.

## File list sinh ra

| File | Nội dung |
|---|---|
| `w251fall_rgb_train_split_1.txt` | CS-train |
| `w251fall_rgb_val_split_1.txt` | CS-val (dùng để chọn checkpoint tốt nhất) |
| `w251fall_rgb_test_split_1.txt` | CS-test nội bộ — để đối chiếu với 8 model kia |
| `w251fall_rgb_extest_split_1.txt` | tập test ngoài (Kaggle `dataset-test` đã bỏ 314 video bị dùng để train) |

`gen_lists.py --min-frames 8` loại clip ngắn hơn `num_segments`. Bản thân TSM cũng tự loại clip
dưới 3 frame trong `_parse_list`, lọc trước để số liệu báo cáo khớp với số clip thực sự được chấm.
