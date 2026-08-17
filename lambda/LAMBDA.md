# Chạy trên Lambda Cloud — session bền, không gián đoạn

Nguyên tắc: **mọi lệnh dài phải chạy trong tmux trên server**, không chạy trực tiếp trong
terminal SSH. SSH đứt → shell của bạn chết → lệnh chạy trực tiếp bị SIGHUP và chết theo.
tmux server nằm *trên máy Lambda*, nên nó sống tiếp; lần sau ssh vào `attach` lại là thấy
màn hình y nguyên.

---

## 0. Một lần duy nhất — đẩy code + key lên

Từ Windows (PowerShell):

```powershell
scp -r "d:\NuverxAI - RD\data\fall-benchmark" ubuntu@<LAMBDA_IP>:~/
scp "d:\NuverxAI - RD\data\kaggle.json" ubuntu@<LAMBDA_IP>:~/.kaggle/kaggle.json
ssh ubuntu@<LAMBDA_IP>
```

Trên Lambda:

```bash
cd ~/fall-benchmark/lambda
sed -i 's/\r$//' *.sh && chmod +x *.sh    # bỏ CRLF của Windows
./setup.sh                                # apt + venv + pip + kaggle CLI  (~10 phút)
./pull_kaggle.sh phucvinhvynxii/fall-dataset ~/data
```

`pull_kaggle.sh` in ra `DATA_ROOT` — thư mục chứa `manifest.csv`. Ghi lại, dùng cho mọi lệnh train.

---

## 1. Chạy job — cách duy nhất bạn cần nhớ

```bash
cd ~/fall-benchmark
./lambda/job.sh start bench --retries 3 -- \
    python run_all.py --data-root ~/data/fall-dataset/MergedFallDataset \
                      --modes both --subset 80 --epochs 3 --num-workers 8
```

Xong. Bạn có thể **đóng laptop ngay lúc này**. Job chạy trong tmux session `job-bench`.

`--retries 3` = nếu process chết (OOM, lỗi mạng khi tải weight HuggingFace) thì tự chạy lại,
cách nhau 60s, tối đa 3 lần. Đặt `--retries 1` nếu không muốn tự lặp.

### Quản lý

```bash
./lambda/job.sh ls              # bảng: tên | RUNNING/DONE/FAILED | tmux up/down
./lambda/job.sh log bench       # tail -f log realtime (Ctrl-C thoát, job KHÔNG chết)
./lambda/job.sh status bench    # tóm tắt + 15 dòng cuối
./lambda/job.sh attach bench    # vào thẳng màn hình tmux (Ctrl-B rồi D để thoát)
./lambda/job.sh stop bench      # giết job
./lambda/job.sh rm bench        # giết + xóa log
```

⚠️ Trong `attach`: thoát bằng **Ctrl-B rồi D** (detach). Gõ `exit` hay Ctrl-C là **giết job thật**.

Log lưu ở `~/jobs/<name>/live.log`; lần start sau log cũ được đổi tên `log-<timestamp>.log`, không mất.

### Chạy nhiều job song song

Mỗi job một tên → một session riêng, độc lập hoàn toàn:

```bash
./lambda/job.sh start ft   -- python run_all.py --data-root $D --modes finetune --models x3d tsm videomae
./lambda/job.sh start scr  -- python run_all.py --data-root $D --modes scratch  --models i3d slowfast
./lambda/job.sh start pred -- python predict.py --ckpt finetune/runs/x3d_best.pt --videos ~/data/test
```

Nhưng **1 GPU chỉ nên 1 job train**. Nhiều job cùng lúc → OOM. Máy nhiều GPU thì tách bằng
`CUDA_VISIBLE_DEVICES`:

```bash
./lambda/job.sh start ft -- env CUDA_VISIBLE_DEVICES=0 python run_all.py ...
./lambda/job.sh start scr -- env CUDA_VISIBLE_DEVICES=1 python run_all.py ...
```

---

## 2. Kéo kết quả về

```powershell
scp -r ubuntu@<LAMBDA_IP>:~/fall-benchmark/results "d:\NuverxAI - RD\data\"
scp -r ubuntu@<LAMBDA_IP>:~/jobs "d:\NuverxAI - RD\data\lambda_logs"
```

---

## 3. Ba điều dễ mất tiền / mất việc

1. **tmux sống qua SSH-disconnect, KHÔNG sống qua reboot/terminate instance.** Lambda tính tiền
   theo giờ instance bật, kể cả khi GPU idle. Job xong ≠ hết tiền — phải tự terminate.
   Muốn tự tắt máy khi job xong (tiết kiệm thật):
   ```bash
   ./lambda/job.sh start bench -- bash -c "python run_all.py ... && sudo shutdown -h now"
   ```
   (Lambda: shutdown thường vẫn tính tiền cho tới khi bạn Terminate trên dashboard — kiểm tra kỹ.)

2. **Ổ đĩa instance mất sạch khi terminate.** Checkpoint/log muốn giữ thì gắn *Persistent
   Filesystem* của Lambda rồi trỏ vào đó:
   ```bash
   ./lambda/job.sh start bench --cwd ~/fall-benchmark -- \
       python run_all.py --data-root /home/ubuntu/<fs-name>/data/... --out /home/ubuntu/<fs-name>/results/table.csv
   ```
   Hoặc đơn giản: clone repo + tải data thẳng vào thư mục filesystem đó.

3. **`run_all.py` hiện chưa có resume giữa chừng.** Retry = chạy lại model đó từ đầu epoch 1.
   Với run ngắn (`--subset 80 --epochs 3`) thì không sao. Khi scale lên full split nhiều epoch,
   nên chia nhỏ theo model — mỗi model một job — để một cú OOM không kéo lại cả 8 model:
   ```bash
   for m in i3d slowfast x3d tsm videomae timesformer videomamba vjepa2; do
     ./lambda/job.sh start ft-$m --retries 2 -- \
       python run_all.py --data-root $D --modes finetune --models $m --epochs 30
   done
   ```
   (chạy tuần tự trên 1 GPU thì gộp vào 1 job với `bash -c "... ; ... ; ..."` thay vì vòng lặp trên.)

---

## 4. Tự chạy lại sau khi instance reboot (tùy chọn)

```bash
crontab -e
# thêm:
@reboot /bin/bash -lc 'cd ~/fall-benchmark && ./lambda/job.sh start bench --retries 3 -- python run_all.py --data-root ~/data/fall-dataset/MergedFallDataset --modes both --subset 80 --epochs 3'
```

## 5. Theo dõi GPU

```bash
tmux new -d -s gpu 'nvidia-smi -l 5'   # session riêng, attach lúc nào cũng được
./lambda/job.sh ls                     # hoặc chỉ cần nhìn status
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5
```
