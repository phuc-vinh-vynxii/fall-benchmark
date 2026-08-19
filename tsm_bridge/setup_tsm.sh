#!/usr/bin/env bash
# setup_tsm.sh -- lay falling-net va noi no vao du lieu cua ta.
#
# Hoa ra code cua repo da la ban TSM moi (dung clip_grad_norm_, khong con .data[0]) nen KHONG
# phai va gi ca. Chi thieu 2 thu:
#   1. tensorboardX  -- main.py import truc tiep
#   2. ROOT_DATASET hardcode '/data/' trong ops/dataset_config.py
#      -> tao symlink /data/w251fall thay vi sua code, de con git pull duoc ban goc
#
#   bash tsm_bridge/setup_tsm.sh [W251_ROOT] [thu_muc_clone]
set -euo pipefail

W251_ROOT="${1:-${W251_ROOT:-$HOME/data/w251fall}}"
TSM_DIR="${2:-$HOME/falling-net}"

echo "== 1. clone falling-net =="
if [[ -d "$TSM_DIR/.git" ]]; then
  echo "  da co $TSM_DIR"
else
  git clone --depth 1 https://github.com/tyu0912/falling-net.git "$TSM_DIR"
fi

echo "== 2. deps =="
pip install -q tensorboardX

echo "== 3. noi /data/w251fall -> $W251_ROOT =="
mkdir -p "$W251_ROOT"
if [[ -w /data ]] || sudo -n true 2>/dev/null; then
  sudo mkdir -p /data
  sudo ln -sfn "$W251_ROOT" /data/w251fall
  echo "  /data/w251fall -> $(readlink -f /data/w251fall)"
else
  echo "  [!] khong tao duoc /data -- sua tay ROOT_DATASET trong"
  echo "      $TSM_DIR/train/ops/dataset_config.py  thanh '$(dirname "$W251_ROOT")/'"
  exit 1
fi

echo "== 3b. va code cho PyTorch moi =="
# ops/utils.py: correct[:k].view(-1) -> .reshape(-1)
# Sau topk()/t()/eq() tensor khong con lien mach bo nho; PyTorch moi tu choi .view() trong
# truong hop do (RuntimeError: view size is not compatible with input tensor's size and stride).
UTILS="$TSM_DIR/train/ops/utils.py"
if grep -q 'correct\[:k\].view(-1)' "$UTILS" 2>/dev/null; then
  sed -i 's/correct\[:k\]\.view(-1)/correct[:k].reshape(-1)/' "$UTILS"
  echo "  da va ops/utils.py (view -> reshape)"
else
  echo "  ops/utils.py: khong can va"
fi

echo "== 4. categories.txt =="
# Phai tao ngay o day: dataset_config.return_dataset() mo file nay de biet n_class, nen neu
# doi den gen_lists.py moi ghi thi buoc kiem tra ben duoi (va ca main.py) se FileNotFoundError.
# gen_lists.py ghi de lai cung noi dung nay, khong sao.
mkdir -p "$W251_ROOT/labels"
printf 'NoFALL\nFALL\n' > "$W251_ROOT/labels/categories.txt"
echo "  $(tr '\n' ' ' < "$W251_ROOT/labels/categories.txt")"

echo "== 5. kiem tra =="
python - <<PY
import sys; sys.path.insert(0, "$TSM_DIR/train")
from ops import dataset_config
cats, tr, va, root, prefix = dataset_config.return_dataset("w251fall", "RGB")
print(f"  n_class={cats}  root_data={root}  prefix={prefix}")
print(f"  train list -> {tr}")
PY
echo
echo "XONG. Lenh train:"
echo "  cd $TSM_DIR/train && python main.py w251fall RGB \\"
echo "      --arch mobilenetv2 --num_segments 8 --shift --shift_div=8 --shift_place=blockres \\"
echo "      --consensus_type=avg --epochs 25 --batch-size 8 -j 8 --npb"
