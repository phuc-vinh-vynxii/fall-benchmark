#!/usr/bin/env bash
# pull_kaggle.sh — kéo dataset từ Kaggle về Lambda (idempotent: có rồi thì bỏ qua).
# Dùng: bash lambda/pull_kaggle.sh [slug] [dest]
#   ./pull_kaggle.sh phucvinhvynxii/fall-dataset ~/data
set -euo pipefail

SLUG="${1:-phucvinhvynxii/fall-dataset}"
DEST="${2:-$HOME/data}"
NAME="$(basename "$SLUG")"
OUT="$DEST/$NAME"

mkdir -p "$OUT"
if [[ -f "$OUT/.complete" ]]; then
  echo "đã có $OUT (bỏ qua). Xóa $OUT/.complete để tải lại."; exit 0
fi

echo "== tải $SLUG -> $OUT =="
# -c cho competition, mặc định là dataset user-uploaded:
kaggle datasets download -d "$SLUG" -p "$OUT" --force
echo "== giải nén =="
shopt -s nullglob
for z in "$OUT"/*.zip; do unzip -q -o "$z" -d "$OUT"; rm -f "$z"; done
shopt -u nullglob

MAN="$(find "$OUT" -maxdepth 3 -name manifest.csv | head -1)"
[[ -n "$MAN" ]] || { echo "[warn] không thấy manifest.csv trong $OUT"; ls "$OUT"; exit 1; }
touch "$OUT/.complete"
echo "OK. DATA_ROOT = $(dirname "$MAN")"
du -sh "$OUT"
