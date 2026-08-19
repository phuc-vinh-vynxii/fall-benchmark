#!/usr/bin/env bash
# pipeline.sh -- chuoi day du tren Lambda: frame -> list -> train -> inference -> Kaggle -> terminate.
#
# Moi buoc ghi mot co hoan thanh trong $STATE, nen khi job.sh --retries chay lai thi no bat dau
# tu buoc dang do chu khong lam lai tu dau.
#
# Terminate CHI chay khi push_results.py tra ve 0 (da xac nhan file co tren Kaggle).
#
#   export LAMBDA_API_KEY=...
#   ./lambda/job.sh start run1 --retries 2 -- bash tsm_bridge/pipeline.sh
#
# Bien moi truong dieu chinh duoc:
#   W251_ROOT DATA_ROOT TEST_ROOT TEST_MANIFEST TSM_DIR EPOCHS BATCH SEGMENTS NO_TERMINATE
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

W251_ROOT="${W251_ROOT:-$HOME/data/w251fall}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/fall-dataset/MergedFallDataset}"
TEST_ROOT="${TEST_ROOT:-$HOME/data/dataset-test}"
TEST_MANIFEST="${TEST_MANIFEST:-$HOME/data/fall-testset-clean/test_clean_manifest.csv}"
TSM_DIR="${TSM_DIR:-$HOME/falling-net}"
EPOCHS="${EPOCHS:-25}"
BATCH="${BATCH:-8}"
SEGMENTS="${SEGMENTS:-8}"
STATE="${STATE:-$HOME/.pipeline_state}"

mkdir -p "$STATE"
step() { [[ -f "$STATE/$1" ]]; }
mark() { touch "$STATE/$1"; }
banner() { echo; echo "############ $* ############"; date -Is; }

banner "0. kiem tra dau vao"
for p in "$DATA_ROOT/manifest.csv" "$TEST_ROOT" "$TEST_MANIFEST"; do
  [[ -e "$p" ]] || { echo "THIEU: $p"; exit 1; }
done
df -h "$HOME" | tail -1
echo "W251_ROOT=$W251_ROOT  EPOCHS=$EPOCHS  BATCH=$BATCH  SEGMENTS=$SEGMENTS"

banner "1. setup falling-net"
if ! step setup; then
  bash tsm_bridge/setup_tsm.sh "$W251_ROOT" "$TSM_DIR"
  mark setup
else echo "(da xong, bo qua)"; fi

banner "2. frame cho tap train"
if ! step frames_train; then
  python tsm_bridge/prep_frames.py --job train --data-root "$DATA_ROOT" --out "$W251_ROOT"
  mark frames_train
else echo "(da xong, bo qua)"; fi

banner "3. frame cho tap test ngoai"
if ! step frames_test; then
  python tsm_bridge/prep_frames.py --job test --test-root "$TEST_ROOT" \
         --test-manifest "$TEST_MANIFEST" --out "$W251_ROOT"
  mark frames_test
else echo "(da xong, bo qua)"; fi

banner "4. sinh file list"
python tsm_bridge/gen_lists.py --root "$W251_ROOT" --test-manifest "$TEST_MANIFEST"
du -sh "$W251_ROOT/jpg" || true

banner "5. train"
if ! step train; then
  cd "$TSM_DIR/train"
  python main.py w251fall RGB \
      --arch mobilenetv2 --num_segments "$SEGMENTS" \
      --shift --shift_div=8 --shift_place=blockres --consensus_type=avg \
      --epochs "$EPOCHS" --batch-size "$BATCH" -j 8 --dropout 0.5 \
      --lr 0.01 --wd 1e-4 --lr_steps 10 20 --gd 20 --eval-freq 1 --npb
  cd "$REPO"
  mark train
else echo "(da xong, bo qua)"; fi

CKPT="$(find "$TSM_DIR/train/checkpoint" -name 'ckpt.best.pth.tar' | head -1)"
[[ -n "$CKPT" ]] || { echo "KHONG tim thay ckpt.best.pth.tar -- dung"; exit 1; }
echo "checkpoint: $CKPT"

banner "6. inference tap test ngoai (da loc 314 video)"
python tsm_bridge/infer_testset.py --ckpt "$CKPT" --root "$W251_ROOT" \
       --tsm-dir "$TSM_DIR" --list extest --num-segments "$SEGMENTS"

banner "7. inference tap CS-test noi bo (de doi chieu voi 8 model kia)"
python tsm_bridge/infer_testset.py --ckpt "$CKPT" --root "$W251_ROOT" \
       --tsm-dir "$TSM_DIR" --list test --num-segments "$SEGMENTS" || \
  echo "[!] CS-test loi, bo qua -- ket qua chinh o buoc 6"

banner "8. day ket qua len Kaggle"
# set -e se thoat truoc khi kip gan $? -> phai boc trong if
if python tsm_bridge/push_results.py --ckpt-dir "$TSM_DIR/train/checkpoint"; then
  PUSH_RC=0
else
  PUSH_RC=$?
fi

banner "9. terminate"
if [[ $PUSH_RC -ne 0 ]]; then
  echo "push that bai -> GIU instance lai de con lay ket qua"
  exit 1
fi
if [[ -n "${NO_TERMINATE:-}" ]]; then
  echo "NO_TERMINATE dang bat -> khong tat may. Nho tu terminate!"
  exit 0
fi
if [[ -z "${LAMBDA_API_KEY:-}" ]]; then
  echo "[!] chua set LAMBDA_API_KEY -> khong tu terminate duoc."
  echo "    Vao https://cloud.lambda.ai/instances tat tay."
  exit 0
fi
bash tsm_bridge/terminate_self.sh
