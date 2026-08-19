# Hướng dẫn từng bước: từ máy Windows → Lambda Cloud → kết quả → tự tắt máy

> **Quy tắc tiền bạc:** Lambda tính tiền theo giờ **kể từ lúc instance bật**, kể cả khi GPU không làm gì.
> Phần 1 và Phần 2 dưới đây **miễn phí hoàn toàn** — làm xong hết mới được bấm Launch ở Phần 3.
> Đừng bao giờ vừa bật instance vừa ngồi debug code.

Tick vào ô khi xong. Mỗi phần có một **CỔNG KIỂM TRA** — không qua cổng thì không đi tiếp.

---

# PHẦN 1 — Làm trên Windows (miễn phí)

## ✅ Bước 1.1 — Rebuild split *(mình đã chạy giúp bạn rồi)*

Splits cũ thiếu 724 video. Đã chạy lại:

```powershell
cd "d:\NuverxAI - RD\data\fall-benchmark"
python common/build_splits.py
```

Kết quả: CS train 2519 / val 639 / test 589, `[OK] NO LEAK`. Trong đó 314 video lấy từ `dataset-test`
đã vào đúng chỗ (220 train / 47 val / 47 test) — nên bước 1.2 là **bắt buộc**, không phải làm cho vui.

## ☐ Bước 1.2 — Tạo dataset test sạch trên Kaggle

Chỉ upload danh sách (~1 MB), không upload 17 GB video.

```powershell
cd "d:\NuverxAI - RD\data\fall-benchmark"
python lambda/make_test_manifest.py            # chạy thử, chưa đẩy lên
```

Xem output: phải in ra khoảng **6.500 video giữ lại** (FALL ~2.828 / NoFALL ~3.684). Ưng thì đẩy lên:

```powershell
python lambda/make_test_manifest.py --push
```

Vào https://www.kaggle.com/datasets/phucvinhvynxii/fall-testset-clean kiểm tra đã có
`test_clean_manifest.csv` và `excluded_314.csv`.

> Vì sao không up lại 17 GB: dataset `dataset-test` thuộc tài khoản `huuquoc0909`, bạn không tạo version
> mới cho nó được. Cách này cho kết quả khoa học y hệt mà tốn 5 phút thay vì 3 tiếng.

## ✅ Bước 1.3 — Đẩy code lên GitHub *(đây là "đoạn đưa lên OpenCode")* — **bạn làm rồi**

Repo đã có: **`git@github.com:phuc-vinh-vynxii/fall-benchmark.git`**, nhánh `main`, đã push.

Còn 2 việc nhỏ:

**a) Commit các file mới** (mình vừa thêm `tsm_bridge/`, `lambda/make_test_manifest.py`, và bản
`HUONG_DAN_LAMBDA.md` trong repo — trước đó nó nằm ngoài repo nên `git clone` không kéo theo, mà Bước 3.6
lại bảo agent đọc file này):

```powershell
cd "d:\NuverxAI - RD\data\fall-benchmark"
git add -A
git commit -m "tsm_bridge: noi MergedFallDataset vao falling-net + huong dan"
git push
```

**b) Quyết định public hay private.** Repo đang private → Kaggle Notebook clone sẽ báo lỗi xác thực.
Xem 2 cách xử lý ở Cell 1 của Bước 2.2.

### Cách B — scp thẳng (không cần GitHub)

Bỏ qua bước này, Phần 3.3 có sẵn lệnh `scp`. Nhược điểm: mỗi lần sửa code phải scp lại thủ công, và
agent trên server không commit/push được.

---

### 🚦 CỔNG KIỂM TRA PHẦN 1

- [ ] `fall-benchmark/splits/cs/train.csv` có 2519 dòng (+1 dòng tiêu đề)
- [ ] Kaggle đã có dataset `phucvinhvynxii/fall-testset-clean`
- [ ] Code đã ở GitHub (hoặc bạn chấp nhận dùng scp)

---

# PHẦN 2 — Kaggle Notebook (miễn phí): bắt `falling-net` chạy được

**Đây là bước tiết kiệm tiền lớn nhất.** Sửa lỗi trên Kaggle: **$0**. Sửa lỗi trên Lambda: **$0.75/giờ**.

### `tsm_bridge/` là gì

`falling-net` (= TSM của MIT HAN Lab) **không mở file video bao giờ**. Nó đọc:

```
/data/w251fall/jpg/<LABEL>/<tên_clip>/img_00001.jpg, img_00002.jpg, ...
/data/w251fall/file_list/w251fall_rgb_train_split_1.txt   ← "<thư_mục> <số_frame> <class_id>"
```

Dữ liệu của bạn là `.mp4` + `manifest.csv`. `tsm_bridge/` chỉ làm nhiệm vụ xếp lại cho đúng khuôn —
**không đụng gì tới model**:

| File | Việc |
|---|---|
| `prep_frames.py` | mp4 → JPG bằng ffmpeg; **và** xử lý 437 clip đã có sẵn frame |
| `gen_lists.py` | sinh file list từ `splits/cs/` (giữ đúng split cross-subject) |
| `setup_tsm.sh` | clone falling-net, cài `tensorboardX`, symlink `/data/w251fall` |
| `infer_testset.py` | chấm điểm bằng `common/metrics.py` — cùng bộ metric với 8 model kia |
| `push_results.py` · `terminate_self.sh` · `pipeline.sh` | phần cloud, không liên quan model |

### Về 437 thư mục frame có sẵn trong data của bạn

`MergedFallDataset` có **3.312 video + 437 thư mục frame** (`extracted_falls` 367, `UR` 70). Chỗ này
đúng thứ TSM cần rồi nên `prep_frames.py` **không trích lại** — chỉ symlink sang tên `img_%05d.jpg`,
mất vài giây thay vì hàng giờ, và không giảm chất lượng ảnh lần hai.

Có một cái bẫy đã xử lý: `extracted_falls` đánh số theo frame gốc trong video (`frame_1894.jpg`), nên
sort theo chữ cái sẽ ra `frame_120 → frame_1894 → frame_23 → frame_9` — **sai hết trình tự thời gian**,
mà TSM thì sống bằng trình tự thời gian. Script sort theo số nằm trong tên. Đã test trên data thật của
bạn: `extracted_falls_Video_0_fall_01` (20 ảnh) và `UR_adl-01-cam0-rgb` (150 ảnh) đều ra đúng thứ tự.

## ☐ Bước 2.1 — Tạo notebook

1. https://www.kaggle.com/code → **New Notebook**
2. Bên phải: **Accelerator = GPU T4 x2**, **Internet = ON**
3. **Add Input** 3 dataset: `phucvinhvynxii/fall-dataset`, `huuquoc0909/dataset-test`,
   `phucvinhvynxii/fall-testset-clean`

## ☐ Bước 2.2 — Các cell

```python
# Cell 1 — lấy code của bạn
!git clone https://github.com/phuc-vinh-vynxii/fall-benchmark.git /kaggle/working/fb
%cd /kaggle/working/fb
```

Repo đang **private** nên lệnh trên sẽ báo lỗi xác thực. Chọn 1:
- **Để repo public** (Settings → General → Change visibility). Đã kiểm tra 42 file trong repo: không có
  `kaggle.json`, không có `.pem`, không có token — public an toàn. Rồi chạy nguyên văn Cell 1 ở trên.
- **Giữ private:** tạo token ở https://github.com/settings/tokens (quyền `repo`), vào notebook
  **Add-ons → Secrets** thêm secret tên `GH_TOKEN`, rồi Cell 1 thành:

```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GH_TOKEN")
!git clone https://{tok}@github.com/phuc-vinh-vynxii/fall-benchmark.git /kaggle/working/fb
%cd /kaggle/working/fb
```

```python
# Cell 2 — deps (torch đã có sẵn trên Kaggle)
!pip install -q tensorboardX
!apt-get -qq install -y ffmpeg > /dev/null
```

```python
# Cell 3 — lấy falling-net + nối /data/w251fall
%env W251_ROOT=/kaggle/working/w251fall
!bash tsm_bridge/setup_tsm.sh /kaggle/working/w251fall /kaggle/working/falling-net
```

`setup_tsm.sh` clone repo, cài `tensorboardX`, và tạo symlink `/data/w251fall` → thư mục của ta
(vì `ops/dataset_config.py` hardcode `ROOT_DATASET = '/data/'`). Làm bằng symlink để repo gốc
không bị sửa, sau này `git pull` vẫn sạch.

```python
# Cell 4 — smoke test: 4 clip mỗi lớp mỗi split (train/val/test đều có)
!python tsm_bridge/prep_frames.py --job train \
    --data-root /kaggle/input/fall-dataset/MergedFallDataset \
    --out /kaggle/working/w251fall --limit 4
!python tsm_bridge/gen_lists.py --root /kaggle/working/w251fall
!head -3 /kaggle/working/w251fall/file_list/w251fall_rgb_train_split_1.txt
```

Kỳ vọng in ra `train: 8 dòng`, `val: 8 dòng`, `test: 8 dòng`. Dòng "thieu frame: 25xx" là **bình
thường** khi có `--limit` — nó chỉ đang báo phần còn lại chưa trích.

> ⚠️ Kiểm tra đường dẫn `--data-root`: mở panel Input bên phải xem `manifest.csv` nằm ở
> `/kaggle/input/fall-dataset/` hay `/kaggle/input/fall-dataset/MergedFallDataset/`, sửa cho khớp.

```python
# Cell 5 — train 2 epoch
!cd /kaggle/working/falling-net/train && python main.py w251fall RGB \
    --arch mobilenetv2 --num_segments 8 --shift --shift_div=8 --shift_place=blockres \
    --consensus_type=avg --epochs 2 --batch-size 8 -j 2 --npb
```

Mình đã đọc code gốc: repo này **đã là bản TSM mới** (dùng `clip_grad_norm_`, không còn `.data[0]`),
nên **không cần vá gì cho PyTorch 2.x** — chỉ thiếu `tensorboardX` (Cell 2) và `/data/w251fall`
(Cell 3). Nếu vẫn có lỗi lạ, copy traceback dán vào agent kèm câu:

> "Lỗi khi chạy TSM trong repo falling-net trên PyTorch 2.x. Sửa tối thiểu, ghi lại thay đổi vào
> `tsm_bridge/README.md` để lần sau khỏi mò lại."

```python
# Cell 6 — inference thử
!python tsm_bridge/infer_testset.py \
    --ckpt /kaggle/working/falling-net/train/checkpoint/*/ckpt.best.pth.tar \
    --root /kaggle/working/w251fall --tsm-dir /kaggle/working/falling-net \
    --list test --limit 16
```

### 🚦 CỔNG KIỂM TRA PHẦN 2

- [ ] Cell 4 in ra cả 3 dòng train/val/test đều **khác 0**
- [ ] Cell 5 chạy hết 2 epoch, không traceback
- [ ] Có `ckpt.best.pth.tar`
- [ ] Cell 6 in ra bảng metric (balanced accuracy, sensitivity, ...)
- [ ] Nếu có sửa file nào thì đã `git commit && git push`

**Chưa qua đủ 5 ô này thì TUYỆT ĐỐI chưa bấm Launch trên Lambda.**

---

# PHẦN 3 — Lambda Cloud (đồng hồ bắt đầu chạy)

## ☐ Bước 3.1 — Chuẩn bị TRƯỚC khi bấm Launch

Có sẵn trong tay 3 thứ (mở sẵn notepad):

| Thứ | Lấy ở đâu |
|---|---|
| **Lambda API key** | Dashboard → API keys → Generate. Dùng để instance tự terminate |
| `kaggle.json` | đã có ở `d:\NuverxAI - RD\data\kaggle.json` |
| SSH key | đã có `~\.ssh\id_ed25519` |

Chọn máy: **A10 24 GB (~$0.75/giờ)**. TSM MobileNetV2 rất nhẹ, **không cần H100** — H100 đắt gấp ~4 lần
mà không nhanh hơn đáng kể cho model này. Cần ~150 GB đĩa.

Bấm **Launch**. Ghi lại **IP**.

## ☐ Bước 3.2 — Kết nối (2 phút)

```powershell
cd "d:\NuverxAI - RD\data\fall-benchmark\lambda"
.\win_ssh_setup.ps1 -Ip <IP_VỪA_GHI>
```

Script này copy key vào `~/.ssh`, chạy `icacls` khoá quyền (thiếu bước này `ssh` báo
`UNPROTECTED PRIVATE KEY FILE` và từ chối chạy — lỗi phổ biến nhất trên Windows), và tạo alias.
Từ giờ chỉ cần:

```powershell
ssh lambda
```

## ☐ Bước 3.3 — Đưa code + key lên server

**Nếu dùng GitHub (cách A):**
```powershell
scp "d:\NuverxAI - RD\data\kaggle.json" lambda:~/.kaggle/kaggle.json
ssh lambda
git clone https://github.com/phuc-vinh-vynxii/fall-benchmark.git ~/fall-benchmark
```

**Nếu dùng scp (cách B):**
```powershell
scp -r "d:\NuverxAI - RD\data\fall-benchmark" lambda:~/
scp "d:\NuverxAI - RD\data\kaggle.json" lambda:~/.kaggle/kaggle.json
ssh lambda
```

Rồi trên server:
```bash
cd ~/fall-benchmark/lambda
sed -i 's/\r$//' *.sh && chmod +x *.sh     # bỏ ký tự xuống dòng kiểu Windows
./setup.sh                                  # apt + venv + pip + kaggle CLI  (~10 phút)
```

## ☐ Bước 3.4 — Mở nhiều terminal (tmux)

**Trả lời thẳng câu hỏi của bạn: OpenCode và Claude Code KHÔNG có panel terminal, không có tab terminal.**
Chúng chỉ có (a) chạy lệnh bên trong hội thoại — OpenCode gõ `!lệnh`, Claude Code tự gọi Bash — và
(b) nhiều *phiên chat* (`/sessions`). Đó không phải terminal.

Terminal thật đến từ **tmux**, chạy trên server. Mở 1 kết nối SSH, bên trong tạo bao nhiêu terminal tuỳ ý,
và tất cả **sống sót khi bạn tắt laptop**:

```bash
tmux new -s main
```

| Phím | Tác dụng |
|---|---|
| `Ctrl-B` rồi `c` | mở thêm 1 terminal |
| `Ctrl-B` rồi `0`…`9` | nhảy qua lại giữa các terminal |
| `Ctrl-B` rồi `%` | chia đôi màn hình theo chiều dọc |
| `Ctrl-B` rồi `d` | **thoát ra mà mọi thứ vẫn chạy** |
| `Ctrl-B` rồi `[` | cuộn lên xem log cũ (`q` để thoát cuộn) |

Lần sau vào lại:
```powershell
ssh -t lambda "tmux attach -t main || tmux new -s main"
```

⚠️ **Đang ở trong tmux mà gõ `exit` hoặc `Ctrl-C` là giết job thật.** Thoát luôn bằng `Ctrl-B` rồi `d`.

Bố cục nên dùng:

| Cửa sổ | Chạy gì |
|---|---|
| 0 | `claude` — agent |
| 1 | `./lambda/job.sh log run1` — xem log train |
| 2 | `nvidia-smi -l 5` — canh GPU |
| 3 | shell trống để gõ lệnh vặt |

## ☐ Bước 3.5 — Cài agent và đăng nhập bằng gói Max

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
cd ~/fall-benchmark && claude
```

Lần đầu chạy, nó in ra một **đường link**. Copy link đó, mở trên trình duyệt **máy Windows của bạn**,
đăng nhập tài khoản Claude, rồi copy mã trả về dán ngược lại vào terminal. Xong — gói **Max** dùng bình
thường, không tính tiền theo token.

> **Vì sao không phải OpenCode:** Anthropic cấm dùng subscription Pro/Max qua tool bên thứ ba và đã chặn
> token từ 01/2026; OpenCode gỡ plugin Claude OAuth ở bản 1.3.0 sau yêu cầu pháp lý từ Anthropic. Cố dùng
> có thể **bị khoá tài khoản**. Nếu bạn vẫn muốn dùng OpenCode, nó chỉ còn hỗ trợ hợp lệ ChatGPT Plus /
> GitHub Copilot / GitLab Duo — gói Max sẽ không dùng được ở đó.

## ☐ Bước 3.6 — Giao việc cho agent

Trong cửa sổ tmux số 0, dán nguyên khối này vào `claude`:

```
Đọc HUONG_DAN_LAMBDA.md và tsm_bridge/README.md trong repo này.

Bối cảnh: đây là instance Lambda A10, đang tính tiền theo giờ. Dữ liệu train là Kaggle
phucvinhvynxii/fall-dataset, test là huuquoc0909/dataset-test lọc theo
phucvinhvynxii/fall-testset-clean. Repo model là tyu0912/falling-net (TSM MobileNetV2), đã port
sang PyTorch 2.x ở Phần 2.

Việc cần làm, theo thứ tự, KHÔNG hỏi lại giữa chừng:
1. Chạy ./lambda/pull_kaggle.sh cho cả 3 dataset về ~/data
2. Kiểm tra df -h còn đủ chỗ cho frame JPG (cần ~100 GB); nếu thiếu thì báo và dừng
3. Chạy tsm_bridge/setup_tsm.sh, xác nhận /data/w251fall trỏ đúng
4. Chạy tsm_bridge/prep_frames.py --job train trên TOÀN BỘ split, kiểm tra 20 thư mục
   đầu có img_00001.jpg
5. Chạy tsm_bridge/gen_lists.py, in ra số dòng và phân bố class của từng file list
6. Báo cáo lại cho tôi, rồi DỪNG. Không tự chạy train.
```

Cho agent làm bước chuẩn bị, còn **lệnh train dài thì bạn tự chạy qua `job.sh`** (bước 4.1) — vì job phải
sống độc lập với phiên chat của agent.

---

# PHẦN 4 — Chạy chuỗi tự động rồi đi ngủ

## ☐ Bước 4.1 — Một lệnh duy nhất

Ở cửa sổ tmux số 3:

```bash
cd ~/fall-benchmark
export LAMBDA_API_KEY=<api_key_lambda_của_bạn>
./lambda/job.sh start run1 --retries 2 -- bash tsm_bridge/pipeline.sh
```

Xong. **Đóng laptop được ngay.** `pipeline.sh` chạy 9 bước tuần tự: kiểm tra đầu vào → setup falling-net →
trích frame train → trích frame test → sinh file list → train → inference tập test ngoài → inference
CS-test nội bộ → đẩy kết quả lên Kaggle → xác nhận upload OK → **mới** gọi API terminate.

Hai chi tiết cố ý:
- Mỗi bước ghi một cờ trong `~/.pipeline_state`, nên `--retries` chạy lại sẽ **tiếp tục từ bước đang dở**,
  không trích lại 10.000 video từ đầu.
- Nếu upload lỗi, instance **không** bị terminate — thà tốn thêm ít tiền còn hơn mất kết quả.

Muốn thử mà chưa muốn nó tự tắt máy:

```bash
NO_TERMINATE=1 ./lambda/job.sh start run1 -- bash tsm_bridge/pipeline.sh
# hoặc chạy ít epoch cho nhanh:
EPOCHS=3 ./lambda/job.sh start run1 -- bash tsm_bridge/pipeline.sh
```

Kiểm tra lệnh terminate có hoạt động không **trước khi** cần tới nó:

```bash
bash tsm_bridge/terminate_self.sh --dry-run    # chỉ in ra instance id, không giết
```

## ☐ Bước 4.2 — Theo dõi (khi nào rảnh thì ngó)

```bash
./lambda/job.sh ls            # run1  RUNNING  up
./lambda/job.sh log run1      # xem log realtime — Ctrl-C thoát, job VẪN CHẠY
./lambda/job.sh status run1   # tóm tắt + 15 dòng cuối
```

Ước tính tổng: **5-8 giờ ≈ $4-6**.

| Bước | Thời gian |
|---|---|
| tải 43 GB dữ liệu | 30-40 phút |
| trích frame ~10.000 video | 1-2 giờ |
| train 25 epoch | 2-3 giờ |
| inference 6.500 video | 1-2 giờ |
| đẩy kết quả + terminate | 10 phút |

## ☐ Bước 4.3 — Lấy kết quả

Sau khi instance biến mất khỏi dashboard, vào Kaggle:
**https://www.kaggle.com/datasets/phucvinhvynxii/falling-net-results** — có `metrics.json`,
`predictions.csv`, `ckpt.best.pth.tar`, `train.log`.

### 🚦 KIỂM TRA CUỐI

- [ ] `predictions.csv` có ~6.500 dòng
- [ ] `metrics.json` có balanced accuracy **rõ ràng > 0.5**
- [ ] Dashboard Lambda: **không còn instance nào đang chạy** (kiểm tra bằng mắt, đừng tin script)

> Nếu sensitivity = 0 hoặc balanced accuracy đúng bằng 0.5: gần như chắc chắn là file list sai `class_id`,
> kiểm tra `categories.txt` trước, đừng vội kết luận model kém.

---

# Ba lỗi tốn tiền nhất, đọc lại trước khi bắt đầu

1. **Bật instance rồi mới ngồi debug code 2019.** Debug ở Kaggle. Lambda chỉ để chạy.
2. **Job xong nhưng quên terminate.** `shutdown -h now` trên Lambda **không** chắc chắn ngừng tính tiền —
   phải **Terminate** qua dashboard hoặc API. `pipeline.sh` gọi API, nhưng vẫn tự mắt kiểm tra dashboard.
3. **Terminate rồi mới nhớ chưa lấy kết quả.** Ổ đĩa instance xoá sạch khi terminate. Đó là lý do
   `pipeline.sh` chỉ terminate sau khi upload lên Kaggle đã xác nhận thành công.
