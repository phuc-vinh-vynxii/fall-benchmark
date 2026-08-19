"""Ship results to Kaggle and VERIFY the upload before anything terminates the machine.

The instance's disk is wiped on terminate, so this is the last line of defence: it exits 0 only
after re-listing the dataset on Kaggle and seeing the files there. pipeline.sh keys the
terminate step off that exit code.

    python tsm_bridge/push_results.py --ckpt-dir ~/falling-net/train/checkpoint --results results/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SLUG = "phucvinhvynxii/falling-net-results"
MAX_FILE_GB = 19.0          # Kaggle bo qua file qua lon; ckpt TSM mobilenetv2 ~ 30 MB


def sh(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--ckpt-dir", default=str(Path.home() / "falling-net" / "train" / "checkpoint"))
    ap.add_argument("--log-dir", default=str(Path.home() / "falling-net" / "train" / "log"))
    ap.add_argument("--jobs-dir", default=str(Path.home() / "jobs"))
    ap.add_argument("--slug", default=SLUG)
    ap.add_argument("--stage", default=str(Path.home() / "_results_upload"))
    args = ap.parse_args()

    stage = Path(args.stage)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    n = 0
    res = Path(args.results)
    if res.is_dir():
        for f in res.rglob("*"):
            if f.is_file():
                shutil.copy2(f, stage / f.name); n += 1

    for d, pattern in [(args.ckpt_dir, "ckpt.best.pth.tar"), (args.ckpt_dir, "log.csv"),
                       (args.log_dir, "*.csv"), (args.jobs_dir, "*/live.log")]:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.rglob(pattern) if "*" in pattern else p.rglob(pattern):
            if f.is_file() and f.stat().st_size < MAX_FILE_GB * 1e9:
                # giu ten thu muc cha de khoi de len nhau
                shutil.copy2(f, stage / f"{f.parent.name}__{f.name}"); n += 1

    if n == 0:
        sys.exit("KHONG co file nao de upload -- dung lai, KHONG terminate")
    print(f"gom {n} file -> {stage}")
    for f in sorted(stage.iterdir()):
        print(f"   {f.stat().st_size/1e6:8.2f} MB  {f.name}")

    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "falling-net (TSM) results on MergedFallDataset",
        "id": args.slug,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()

    exists = True
    try:
        api.dataset_list_files(args.slug)
    except Exception:
        exists = False

    cmd = (["kaggle", "datasets", "version", "-p", str(stage), "-m",
            f"run {time.strftime('%Y-%m-%d %H:%M')}", "--dir-mode", "zip"]
           if exists else
           ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"])
    rc = sh(cmd).returncode
    if rc != 0:
        sys.exit(f"upload that bai (rc={rc}) -- KHONG terminate")

    print("\ncho Kaggle xu ly roi kiem tra lai...")
    for attempt in range(12):
        time.sleep(20)
        try:
            names = {f.name for f in api.dataset_list_files(args.slug).files}
        except Exception as e:
            print(f"  chua doc duoc ({repr(e)[:60]}), thu lai")
            continue
        print(f"  lan {attempt+1}: {len(names)} file tren Kaggle")
        if names:
            print("\nXAC NHAN upload thanh cong:")
            for x in sorted(names)[:20]:
                print("   ", x)
            print(f"\nhttps://www.kaggle.com/datasets/{args.slug}")
            sys.exit(0)
    sys.exit("het gio cho ma chua thay file tren Kaggle -- KHONG terminate")


if __name__ == "__main__":
    main()
