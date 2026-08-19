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

echo "== 4. kiem tra =="
ls "$TSM_DIR/train/scripts/" 2>/dev/null || echo "  (khong co thu muc scripts/)"
python - <<PY
import sys; sys.path.insert(0, "$TSM_DIR/train")
from ops import dataset_config
print("  return_dataset ->", dataset_config.return_dataset("w251fall", "RGB"))
PY
echo
echo "XONG. Lenh train:"
echo "  cd $TSM_DIR/train && python main.py w251fall RGB \\"
echo "      --arch mobilenetv2 --num_segments 8 --shift --shift_div=8 --shift_place=blockres \\"
echo "      --consensus_type=avg --epochs 25 --batch-size 8 -j 8 --npb"
