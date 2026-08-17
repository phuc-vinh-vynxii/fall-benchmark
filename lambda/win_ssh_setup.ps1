# win_ssh_setup.ps1 — cấu hình SSH tới Lambda Cloud trên Windows (chạy 1 lần).
#
#   .\win_ssh_setup.ps1 -Ip 129.x.x.x                          # dùng key sẵn có ~/.ssh/id_ed25519
#   .\win_ssh_setup.ps1 -Ip 129.x.x.x -Key "$HOME\Downloads\lambda.pem"
#
# Làm: copy key vào ~/.ssh, khoá quyền (ssh từ chối key "quá mở"), thêm alias "lambda"
# vào ~/.ssh/config, rồi test kết nối.
param(
  [Parameter(Mandatory=$true)][string]$Ip,
  [string]$Key  = "$env:USERPROFILE\.ssh\id_ed25519",
  [string]$User = "ubuntu",
  [string]$Alias = "lambda"
)

$ErrorActionPreference = "Stop"
$sshDir = "$env:USERPROFILE\.ssh"
if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory $sshDir | Out-Null }

if (-not (Test-Path $Key)) { throw "Không thấy key: $Key" }

# Nếu key nằm ngoài ~/.ssh (vd Downloads) -> copy vào
$dest = Join-Path $sshDir (Split-Path $Key -Leaf)
if ((Resolve-Path $Key).Path -ne (Join-Path $sshDir (Split-Path $Key -Leaf))) {
  Copy-Item $Key $dest -Force
  Write-Host "copied -> $dest"
} else { $dest = $Key }

# Khoá quyền: chỉ chính bạn đọc được, bỏ thừa kế. Thiếu bước này ssh báo
# "UNPROTECTED PRIVATE KEY FILE" và từ chối dùng key.
icacls $dest /inheritance:r  | Out-Null
icacls $dest /grant:r "$($env:USERNAME):(R)" | Out-Null
Write-Host "permissions locked"

$cfg = Join-Path $sshDir "config"
$block = @"

Host $Alias
    HostName $Ip
    User $User
    IdentityFile $dest
    ServerAliveInterval 30
    ServerAliveCountMax 6
    TCPKeepAlive yes
"@

if ((Test-Path $cfg) -and (Select-String -Path $cfg -Pattern "^Host\s+$Alias\s*$" -Quiet)) {
  Write-Host "[!] ~/.ssh/config đã có 'Host $Alias' — tự sửa IP thủ công nếu instance đổi."
} else {
  Add-Content -Path $cfg -Value $block -Encoding utf8
  Write-Host "added 'Host $Alias' -> $cfg"
}

Write-Host "`n== test =="
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new $Alias "hostname; nvidia-smi --query-gpu=name --format=csv,noheader"
Write-Host "`nOK. Từ giờ chỉ cần:  ssh $Alias"
