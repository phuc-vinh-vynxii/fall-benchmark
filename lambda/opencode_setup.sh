#!/usr/bin/env bash
# opencode_setup.sh — cài OpenCode NGAY TRÊN instance Lambda.
# Chạy trên Lambda:  bash lambda/opencode_setup.sh
#
# Vì sao cài trên server thay vì chạy từ Windows: opencode khởi động trong phiên SSH sẽ
# kế thừa môi trường của server — nó đọc file của server, chạy lệnh trên server, thấy GPU
# và dataset trực tiếp. Không phải đồng bộ gì cả.
set -euo pipefail

curl -fsSL https://opencode.ai/install | bash -s -- --prefix "$HOME/.local"

if ! grep -q '.local/bin' ~/.bashrc 2>/dev/null; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
export PATH="$HOME/.local/bin:$PATH"

echo; opencode --version || true
cat <<'EOF'

TIẾP THEO:
  1. opencode auth login          # dán API key (Anthropic / OpenAI / ...)
  2. tmux new -s oc               # LUÔN chạy opencode trong tmux
  3. cd ~/fall-benchmark && opencode
     -> thoát tạm: Ctrl-B rồi D   (opencode vẫn sống)
     -> quay lại : tmux attach -t oc

Phiên làm việc cũ:  opencode --continue      (hoặc -s <session-id>)
Chạy 1 phát không mở TUI:  opencode run "tóm tắt log train mới nhất"
EOF
