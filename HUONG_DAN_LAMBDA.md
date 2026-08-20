# Windows → Kaggle → Lambda → kết quả → tự tắt máy

**Quy tắc tiền:** Lambda tính tiền từ lúc instance bật. Phần 1 và 2 miễn phí — xong hết mới bấm Launch ở Phần 3.

Giải thích kỹ thuật (tsm_bridge làm gì, vì sao) nằm ở [tsm_bridge/README.md](fall-benchmark/tsm_bridge/README.md).

---

# PHẦN 1 — Windows ($0)

### ✅ 1.1 Rebuild split — xong rồi
CS train 2519 / val 639 / test 589, `[OK] NO LEAK`. 314 video lấy từ `dataset-test` đã vào split (220/47/47) → bước 1.2 là bắt buộc.

### ☐ 1.2 Tạo dataset test sạch trên Kaggle
```powershell
cd "d:\NuverxAI - RD\data\fall-benchmark"
python lambda/make_test_manifest.py           # thử: phải ra 6512 video (FALL 2828 / NoFALL 3684)
python lambda/make_test_manifest.py --push    # ưng thì đẩy lên
```
Kiểm tra: kaggle.com/datasets/phucvinhvynxii/fall-testset-clean có `test_clean_manifest.csv`.

### ✅ 1.3 Code lên GitHub — xong rồi
`github.com/phuc-vinh-vynxii/fall-benchmark`, nhánh `main`.

Mỗi lần sửa code ở Windows: `git push`. Repo đang **private** → xem Cell 1 để clone được từ Kaggle.

**🚦 Cổng 1:** split 2519 dòng ✓ · dataset `fall-testset-clean` đã có ✓ · code đã push ✓

---

# PHẦN 2 — Kaggle Notebook ($0): bắt falling-net chạy

Debug ở đây $0, debug trên Lambda $0.75/giờ.

### ☐ 2.1 Tạo notebook
New Notebook → **GPU T4 x2** + **Internet ON** → Add Input 3 dataset: `phucvinhvynxii/fall-dataset`, `huuquoc0909/dataset-test`, `phucvinhvynxii/fall-testset-clean`.

### ☐ 2.2 Các cell

```python
# Cell 1 — lấy code (chạy lại bao nhiêu lần cũng đúng)
import os
D = "/kaggle/working/fb"
if os.path.isdir(D + "/.git"):
    !git -C {D} pull --ff-only
else:
    !git clone https://github.com/phuc-vinh-vynxii/fall-benchmark.git {D}
%cd {D}
!git log --oneline -1
```

> Repo private → clone lỗi xác thực. Cách 1: đổi repo sang **public** (Settings → Change visibility; đã kiểm tra không có key/token nào trong repo). Cách 2: tạo token ở github.com/settings/tokens (quyền `repo`) → notebook **Add-ons → Secrets** thêm `GH_TOKEN` → sửa dòng clone thành `https://{tok}@github.com/...` với `tok = UserSecretsClient().get_secret("GH_TOKEN")`.

```python
# Cell 2 — deps
!pip install -q tensorboardX
!apt-get -qq install -y ffmpeg > /dev/null
```

```python
# Cell 3 — falling-net + nối /data/w251fall
!bash tsm_bridge/setup_tsm.sh /kaggle/working/w251fall /kaggle/working/falling-net
```

```python
# Cell 4 — tìm manifest.csv
!ls /kaggle/input/
!find /kaggle/input/datasets/phucvinhvynxii/fall-dataset/manifest.csv -maxdepth 3 -name manifest.csv
```
`find` in ra đường dẫn đầy đủ → **thư mục cha của nó** chính là `DATA` cho Cell 5.

- `ls` rỗng → chưa Add Input. Panel phải → **Input → + Add Input** → thêm cả 3 dataset ở mục 2.1.
- Tên thư mục khác `fall-dataset` → Kaggle đặt tên theo tên dataset chứ không theo slug; dùng đúng tên hiện ra.

```python
# Cell 5 — smoke test: 4 clip mỗi lớp mỗi split
DATA = "/kaggle/input/fall-dataset/MergedFallDataset"      # sửa theo Cell 4
!python tsm_bridge/prep_frames.py --job train --data-root {DATA} \
    --out /kaggle/working/w251fall --limit 4
!python tsm_bridge/gen_lists.py --root /kaggle/working/w251fall
```
Kỳ vọng `24/24 ok=24 loi=0`, rồi `train: 8 dòng · val: 8 · test: 8`. Dòng `thieu frame: 25xx` là bình thường khi có `--limit`.

```python
# Cell 6 — train 2 epoch
!cd /kaggle/working/falling-net/train && python main.py w251fall RGB \
    --arch mobilenetv2 --num_segments 8 --shift --shift_div=8 --shift_place=blockres \
    --consensus_type=avg --epochs 2 --batch-size 8 -j 2 --npb
```
`setup_tsm.sh` đã vá sẵn `ops/utils.py` (`correct[:k].view(-1)` → `.reshape(-1)`, lỗi
*view size is not compatible with input tensor's size and stride*). Lỗi khác thì dán traceback cho agent —
sửa xong nhớ ghi vào `tsm_bridge/setup_tsm.sh` để lần sau khỏi mò lại.

```python
# Cell 7 — inference thử
!python tsm_bridge/infer_testset.py --root /kaggle/working/w251fall \
    --tsm-dir /kaggle/working/falling-net --list test --limit 16 \
    --ckpt /kaggle/working/falling-net/train/checkpoint/*/ckpt.best.pth.tar
```

**🚦 Cổng 2 — chưa đủ 4 ô này thì TUYỆT ĐỐI chưa bấm Launch:**
- [ ] Cell 5: train/val/test đều khác 0
- [ ] Cell 6: chạy hết 2 epoch, không traceback
- [ ] Có `ckpt.best.pth.tar`
- [ ] Cell 7: in ra bảng metric

---

# PHẦN 3 — Lambda (bắt đầu tính tiền)

### ☐ 3.1 Chuẩn bị TRƯỚC khi Launch
Chuẩn bị 3 thứ:

| | Ở đâu |
|---|---|
| **Lambda API key** | Dashboard → API keys → Generate (để tự terminate) |
| **SSH public key** | `~\.ssh\id_ed25519_lambda.pub` → dán vào Dashboard → **SSH Keys**; lúc Launch chọn đúng key này |
| **kaggle.json** | `d:\NuverxAI - RD\kaggle.json` — **không phải** bản ở `~\.kaggle\` (token cũ, trả 403) |

Kiểm tra token còn sống TRƯỚC khi Launch:
```powershell
$env:KAGGLE_CONFIG_DIR="d:\NuverxAI - RD"; kaggle datasets files phucvinhvynxii/fall-testset-clean
```

Chọn **A10 24 GB (~$0.75/h)** — TSM MobileNetV2 rất nhẹ, không cần H100. Cần ~150 GB đĩa. Launch → ghi IP.

### ☐ 3.2 Kết nối
```powershell
cd "d:\NuverxAI - RD\data\fall-benchmark\lambda"
.\win_ssh_setup.ps1 -Ip <IP>
ssh lambda
```

### ☐ 3.3 Đưa code lên
```powershell
scp "d:\NuverxAI - RD\kaggle.json" lambda:~/.kaggle/kaggle.json
ssh lambda
```
```bash
git clone https://github.com/phuc-vinh-vynxii/fall-benchmark.git ~/fall-benchmark
cd ~/fall-benchmark/lambda && sed -i 's/\r$//' *.sh && chmod +x *.sh
./setup.sh                                   # ~10 phút
```

### ☐ 3.4 Nhiều terminal = tmux
Không cần cài gì trên Windows — tmux chạy **trên máy Lambda**, Windows chỉ cần `ssh` (có sẵn).
OpenCode và Claude Code **không có panel terminal**. Terminal thật là tmux:
```bash
tmux new -s main
```
| Phím | |
|---|---|
| `Ctrl-B` `c` | terminal mới |
| `Ctrl-B` `0..9` | chuyển |
| `Ctrl-B` `d` | **thoát mà vẫn chạy** |
| `Ctrl-B` `[` | cuộn xem log (`q` thoát) |

Vào lại: `ssh -t lambda "tmux attach -t main || tmux new -s main"`

⚠️ Trong tmux gõ `exit` hay `Ctrl-C` là **giết job thật**. Thoát bằng `Ctrl-B` `d`.

Bố cục: cửa sổ 0 = `claude` · 1 = `job.sh log run1` · 2 = `nvidia-smi -l 5` · 3 = shell trống.

### ☐ 3.5 Claude Code + gói Max
```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH" && echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
cd ~/fall-benchmark && claude
```
Nó in ra 1 link → mở trên trình duyệt Windows → đăng nhập → copy mã dán ngược lại. Gói Max dùng bình thường.

> Không dùng OpenCode với gói Max: Anthropic cấm subscription qua tool bên thứ ba, chặn token từ 01/2026, OpenCode đã gỡ plugin ở 1.3.0 — dùng cố có thể **bị khoá tài khoản**.

### ☐ 3.6 Prompt giao cho agent
```
Đọc HUONG_DAN_LAMBDA.md và tsm_bridge/README.md trong repo này.

Đây là instance Lambda A10 đang tính tiền theo giờ. Train: Kaggle phucvinhvynxii/fall-dataset.
Test: huuquoc0909/dataset-test lọc theo phucvinhvynxii/fall-testset-clean.

Làm theo thứ tự, KHÔNG hỏi lại giữa chừng:
1. ./lambda/pull_kaggle.sh cho cả 3 dataset về ~/data
2. df -h xem còn đủ ~100 GB cho frame JPG; thiếu thì báo và dừng
3. tsm_bridge/setup_tsm.sh, xác nhận /data/w251fall trỏ đúng
4. tsm_bridge/prep_frames.py --job train trên TOÀN BỘ split; kiểm tra 20 thư mục đầu có img_00001.jpg
5. tsm_bridge/gen_lists.py, in số dòng và phân bố class của từng file list
6. Báo cáo rồi DỪNG. Không tự chạy train.
```

---

# PHẦN 4 — Chạy tự động rồi đi ngủ

### ☐ 4.1 Một lệnh
```bash
cd ~/fall-benchmark
export LAMBDA_API_KEY=<key>
./lambda/job.sh start run1 --retries 2 -- bash tsm_bridge/pipeline.sh
```
Đóng laptop được ngay. 9 bước: kiểm tra → setup → frame train → frame test → file list → train → inference test ngoài → inference CS-test → đẩy Kaggle → xác nhận OK → terminate.

Mỗi bước ghi cờ ở `~/.pipeline_state` nên retry tiếp tục từ chỗ dở. Upload lỗi thì **không** terminate.

Thử nghiệm:
```bash
bash tsm_bridge/terminate_self.sh --dry-run    # xem có tìm đúng instance không, không giết
NO_TERMINATE=1 ./lambda/job.sh start run1 -- bash tsm_bridge/pipeline.sh
EPOCHS=3      ./lambda/job.sh start run1 -- bash tsm_bridge/pipeline.sh
```

### ☐ 4.2 Theo dõi — đóng terminal rồi vào lại lúc nào cũng được
"ID session" chính là cái tên bạn đặt (`run1`). Job sống trong tmux session `job-run1` **trên server**;
terminal của bạn chỉ là cửa sổ nhìn vào, đóng lại không ảnh hưởng gì.

```bash
./lambda/job.sh ls            # run1  RUNNING  up
./lambda/job.sh log run1      # log trực tiếp — Ctrl-C thoát, job VẪN CHẠY
./lambda/job.sh attach run1   # vào thẳng màn hình đang chạy (Ctrl-B rồi D để ra)
./lambda/job.sh status run1
```
Hỏi nhanh không cần đăng nhập (dán được vào OpenCode):
```powershell
ssh lambda "cd fall-benchmark && ./lambda/job.sh status run1"
```
Log ghi liên tục ra `~/jobs/run1/live.log`, không attach vẫn xem lại được từ đầu.
tmux sống qua mất mạng và đóng terminal, **không** sống qua reboot instance — reboot thì
`job.sh start` lại, `pipeline.sh` chạy tiếp từ bước dở.

| Bước | Thời gian |
|---|---|
| tải 43 GB | 30-40 phút |
| trích frame ~10.000 video | 1-2 h |
| train 25 epoch | 2-3 h |
| inference 6.512 video | 1-2 h |
| đẩy Kaggle + terminate | 10 phút |

**Tổng ≈ 5-8 giờ ≈ $4-6.**

### ☐ 4.3 Lấy kết quả
kaggle.com/datasets/phucvinhvynxii/falling-net-results → `metrics_extest.json`, `predictions_extest.csv`, `ckpt.best.pth.tar`, log.

**🚦 Cổng cuối:**
- [ ] `predictions_extest.csv` ~6.512 dòng
- [ ] balanced accuracy rõ ràng > 0.5
- [ ] Dashboard Lambda **không còn instance nào** — nhìn bằng mắt, đừng tin script

> Sensitivity = 0 hoặc balanced accuracy đúng 0.5 → gần như chắc là sai `class_id`, kiểm `categories.txt` trước, đừng vội kết luận model kém.

---

# Ba lỗi tốn tiền nhất

1. **Bật instance rồi mới debug.** Debug ở Kaggle. Lambda chỉ để chạy.
2. **Quên terminate.** `shutdown -h now` không chắc ngừng tính tiền — phải Terminate qua dashboard/API.
3. **Terminate trước khi lấy kết quả.** Đĩa xoá sạch khi terminate — đó là lý do `pipeline.sh` chỉ terminate sau khi Kaggle xác nhận đã có file.
