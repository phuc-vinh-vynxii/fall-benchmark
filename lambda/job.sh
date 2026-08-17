#!/usr/bin/env bash
# job.sh — chạy lệnh dài (train/inference) trong tmux session BỀN trên Lambda Cloud.
#
# Mỗi job = 1 tmux session tên "job-<name>" + 1 thư mục $JOBS_DIR/<name>/ chứa
#   cmd.sh   lệnh gốc          run.sh   wrapper retry
#   live.log log đang chạy     status   RUNNING|DONE|FAILED  rc  exit code
# Đóng SSH / tắt máy local -> session vẫn chạy tiếp vì tmux server nằm trên Lambda.
#
#   ./job.sh start bench -- python run_all.py --data-root ~/data/MergedFallDataset
#   ./job.sh ls | attach bench | log bench | status bench | stop bench | rm bench
set -euo pipefail

JOBS_DIR="${JOBS_DIR:-$HOME/jobs}"
PREFIX="job-"

die() { echo "ERR: $*" >&2; exit 1; }
sess() { echo "${PREFIX}$1"; }
jdir() { echo "$JOBS_DIR/$1"; }
alive() { tmux has-session -t "$(sess "$1")" 2>/dev/null; }

cmd_start() {
  local name="" retries=1 cwd="$PWD" delay=60
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --retries) retries="$2"; shift 2 ;;
      --cwd)     cwd="$(cd "$2" && pwd)"; shift 2 ;;
      --delay)   delay="$2"; shift 2 ;;
      --)        shift; break ;;
      *)         [[ -z "$name" ]] && name="$1" || die "tham số lạ: $1"; shift ;;
    esac
  done
  [[ -n "$name" ]] || die "thiếu tên job"
  [[ $# -gt 0 ]] || die "thiếu lệnh sau --"
  alive "$name" && die "job '$name' đang chạy. Dùng: $0 attach $name"

  local d; d="$(jdir "$name")"; mkdir -p "$d"
  printf '%q ' "$@" > "$d/cmd.sh"; echo >> "$d/cmd.sh"

  cat > "$d/run.sh" <<EOF
#!/usr/bin/env bash
cd "$cwd" || exit 1
echo "RUNNING" > "$d/status"; date -Is > "$d/started_at"
for i in \$(seq 1 $retries); do
  echo "=== [\$(date -Is)] attempt \$i/$retries ==="
  bash "$d/cmd.sh"; rc=\$?
  echo "=== [\$(date -Is)] exit=\$rc ==="
  [ \$rc -eq 0 ] && break
  [ \$i -lt $retries ] && { echo "--- fail, retry sau ${delay}s ---"; sleep $delay; }
done
echo \$rc > "$d/rc"; date -Is > "$d/ended_at"
[ \$rc -eq 0 ] && echo "DONE" > "$d/status" || echo "FAILED" > "$d/status"
if [ -n "\${JOB_KEEP_ALIVE:-}" ]; then echo "(giữ pane, Ctrl-C để thoát)"; sleep infinity; fi
exit \$rc
EOF
  chmod +x "$d/run.sh"

  # Nối log: mv log cũ, tmux pipe-pane ghi toàn bộ output ra live.log
  if [[ -f "$d/live.log" ]]; then mv "$d/live.log" "$d/log-$(date +%Y%m%d-%H%M%S).log"; fi
  tmux new-session -d -s "$(sess "$name")" "bash '$d/run.sh'"
  tmux pipe-pane -t "$(sess "$name")" -o "cat >> '$d/live.log'"
  echo "started: $(sess "$name")  |  log: $d/live.log"
  echo "theo dõi: $0 log $name    (Ctrl-C thoát, job vẫn chạy)"
}

cmd_ls() {
  printf '%-18s %-9s %-6s %s\n' NAME STATUS TMUX STARTED
  for d in "$JOBS_DIR"/*/; do
    [[ -d "$d" ]] || continue
    local n; n="$(basename "$d")"
    printf '%-18s %-9s %-6s %s\n' "$n" \
      "$(cat "$d/status" 2>/dev/null || echo '-')" \
      "$(alive "$n" && echo up || echo down)" \
      "$(cat "$d/started_at" 2>/dev/null || echo '-')"
  done
}

cmd_status() {
  local d; d="$(jdir "$1")"; [[ -d "$d" ]] || die "không có job '$1'"
  echo "job     : $1"
  echo "status  : $(cat "$d/status" 2>/dev/null || echo '-')  (rc=$(cat "$d/rc" 2>/dev/null || echo '-'))"
  echo "tmux    : $(alive "$1" && echo up || echo down)"
  echo "cmd     : $(cat "$d/cmd.sh" 2>/dev/null)"
  echo "log     : $d/live.log"
  echo "--- 15 dòng cuối ---"; tail -n 15 "$d/live.log" 2>/dev/null
}

case "${1:-}" in
  start)  shift; cmd_start "$@" ;;
  ls|list) cmd_ls ;;
  attach) tmux attach -t "$(sess "${2:?tên job}")" ;;
  log)    tail -n 80 -f "$(jdir "${2:?tên job}")/live.log" ;;
  status) cmd_status "${2:?tên job}" ;;
  stop)   tmux kill-session -t "$(sess "${2:?tên job}")" && echo "đã dừng ${2}" ;;
  rm)     tmux kill-session -t "$(sess "${2:?}")" 2>/dev/null; rm -rf "$(jdir "${2}")"; echo "đã xóa ${2}" ;;
  *) sed -n '2,12p' "$0" ;;
esac
