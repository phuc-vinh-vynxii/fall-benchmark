#!/usr/bin/env bash
# setup.sh — bootstrap 1 lần trên instance Lambda Cloud mới.
# Dùng: bash lambda/setup.sh
set -euo pipefail

echo "== 1. gói hệ thống =="
sudo apt-get update -qq
sudo apt-get install -y -qq tmux unzip ffmpeg libgl1 libglib2.0-0 >/dev/null

echo "== 2. venv =="
[[ -d ~/venv ]] || python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip wheel

echo "== 3. deps =="
HERE="$(cd "$(dirname "$0")/.." && pwd)"
pip install -q -r "$HERE/requirements.txt"
pip install -q kaggle
# VideoMamba CUDA kernel chính chủ (bỏ qua nếu build lỗi -> code tự fallback lite)
pip install -q causal-conv1d>=1.2 mamba-ssm || echo "[warn] mamba-ssm build fail -> dùng VideoMamba-lite"

echo "== 4. kaggle credentials =="
if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo "  THIẾU ~/.kaggle/kaggle.json — từ máy Windows chạy:"
  echo '  scp "d:\NuverxAI - RD\data\kaggle.json" ubuntu@<LAMBDA_IP>:~/.kaggle/kaggle.json'
  mkdir -p ~/.kaggle
else
  chmod 600 ~/.kaggle/kaggle.json && echo "  OK"
fi

echo "== 5. auto-activate venv khi ssh vào =="
grep -q 'source ~/venv/bin/activate' ~/.bashrc || echo 'source ~/venv/bin/activate' >> ~/.bashrc

echo; echo "XONG. GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
