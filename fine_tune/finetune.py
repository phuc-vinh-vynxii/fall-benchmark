"""Fine-tune trainer (Case 2). Loads pretrained weights for each model and fine-tunes on
MergedFallDataset. Same engine/data/metrics as from-scratch -> directly comparable.

Usage:
    python finetune.py --model videomae                 # small-subset smoke test
    python finetune.py --model x3d --subset 100 --epochs 5
    python finetune.py --model vjepa2 --batch-size 1     # fine-tune-only foundation model
    python finetune.py --config config.yaml
    python finetune.py --model x3d --train-split ../splits/cs/train.csv \\
                       --val-split ../splits/cs/val.csv  # full CS split

Run from inside the fine_tune/ folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # repo root -> `common`
sys.path.insert(0, str(HERE))                 # this folder -> `models`

from common.engine import run               # noqa: E402
from models import build_model, REGISTRY     # noqa: E402

DEFAULTS = {
    "data_root": str(HERE.parent.parent / "MergedFallDataset"),
    "manifest_csv": None,
    "model": "videomae",
    "num_frames": 16,
    "img_size": 224,
    "batch_size": 2,
    "lr": 1e-5,                # LOWER LR for fine-tuning pretrained weights
    "weight_decay": 0.05,
    "epochs": 5,
    "subset_per_class": 100,
    "train_split_csv": None,
    "val_split_csv": None,
    "num_workers": 0,
    "seed": 42,
    "out_dir": str(HERE / "runs"),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--model", choices=list(REGISTRY))
    p.add_argument("--subset", type=int, dest="subset_per_class")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--num-frames", type=int, dest="num_frames")
    p.add_argument("--lr", type=float)
    p.add_argument("--data-root", dest="data_root")
    p.add_argument("--train-split", dest="train_split_csv")
    p.add_argument("--val-split", dest="val_split_csv")
    return p.parse_args()


def main():
    args = parse()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update({k: v for k, v in yaml.safe_load(open(args.config)).items() if v is not None})
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    if cfg.get("subset_per_class") is not None and cfg["subset_per_class"] <= 0:
        cfg["subset_per_class"] = None       # --subset 0 -> full split (no subset)
    print(f"[config] {cfg}")
    run(cfg, build_model_fn=build_model, pretrained=True)


if __name__ == "__main__":
    main()
