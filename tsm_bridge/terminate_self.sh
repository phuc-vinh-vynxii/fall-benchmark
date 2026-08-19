#!/usr/bin/env bash
# terminate_self.sh -- instance tu terminate qua Lambda Cloud API.
#
# `shutdown -h now` KHONG chac ngung tinh tien -- phai terminate. Script tim instance id bang
# cach doi chieu IP public cua chinh may nay voi danh sach instance cua tai khoan.
#
#   export LAMBDA_API_KEY=...
#   bash tsm_bridge/terminate_self.sh            # terminate that
#   bash tsm_bridge/terminate_self.sh --dry-run  # chi in ra se giet cai gi
set -euo pipefail

API="https://cloud.lambdalabs.com/api/v1"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

: "${LAMBDA_API_KEY:?chua set LAMBDA_API_KEY -- khong terminate}"

MYIP="$(curl -fsS --max-time 10 https://checkip.amazonaws.com || curl -fsS --max-time 10 https://api.ipify.org)"
MYIP="$(echo "$MYIP" | tr -d '[:space:]')"
echo "IP public cua may nay: $MYIP"

LIST="$(curl -fsS -H "Authorization: Bearer $LAMBDA_API_KEY" "$API/instances")"

ID="$(printf '%s' "$LIST" | MYIP="$MYIP" python3 -c '
import json, os, sys
ip = os.environ["MYIP"]
for inst in json.load(sys.stdin).get("data", []):
    if inst.get("ip") == ip:
        print(inst["id"]); break
')" || true

if [[ -z "${ID:-}" ]]; then
  echo "[!] khong khop duoc instance nao voi IP $MYIP."
  echo "    Danh sach instance dang chay:"
  echo "$LIST" | python3 -c "import json,sys; [print(' ', i['id'], i.get('ip'), i.get('name','')) for i in json.load(sys.stdin).get('data',[])]"
  echo "    Vao dashboard terminate tay: https://cloud.lambda.ai/instances"
  exit 1
fi

echo "instance id: $ID"
if [[ $DRY -eq 1 ]]; then
  echo "(dry-run) se POST terminate cho $ID"
  exit 0
fi

echo "terminating..."
curl -fsS -X POST "$API/instance-operations/terminate" \
     -H "Authorization: Bearer $LAMBDA_API_KEY" \
     -H "Content-Type: application/json" \
     -d "{\"instance_ids\": [\"$ID\"]}"
echo
echo "da gui lenh terminate. Kiem tra lai o https://cloud.lambda.ai/instances"
