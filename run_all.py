"""Orchestrator: train ALL models briefly on the CS-train SUBSET and evaluate on the CS-val set,
for both regimes (from-scratch + fine-tune), then emit one comparison table.

This is the "should I keep training this model?" experiment — short runs on a small leak-safe
subset of the cross-subject split. Designed for Kaggle/Lambda (Linux, big GPU).

Usage (Kaggle/Lambda):
    python run_all.py --data-root /kaggle/input/fall-dataset/MergedFallDataset \\
                      --modes both --subset 80 --epochs 3
    python run_all.py --models x3d tsm videomamba --modes finetune --subset 80 --epochs 5

Outputs: results/experiment_table.csv  (+ per-run json under <mode>/runs/)
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # -> common
sys.path.insert(0, str(HERE / "fine_tune"))   # -> superset registry (8 models incl. vjepa2)

from common.engine import run                # noqa: E402
from models import build_model, REGISTRY      # noqa: E402  (fine_tune registry: 8 models)

ALL_MODELS = ["i3d", "slowfast", "x3d", "tsm", "videomae", "timesformer", "videomamba", "vjepa2"]

# Per-model defaults tuned for a 16GB cloud GPU on the small subset.
MODEL_CFG = {
    "i3d":         {"num_frames": 16, "img_size": 224, "batch_size": 4},
    "slowfast":    {"num_frames": 32, "img_size": 224, "batch_size": 2},
    "x3d":         {"num_frames": 16, "img_size": 224, "batch_size": 4},
    "tsm":         {"num_frames": 8,  "img_size": 224, "batch_size": 8},
    "videomae":    {"num_frames": 16, "img_size": 224, "batch_size": 4},
    "timesformer": {"num_frames": 8,  "img_size": 224, "batch_size": 4},
    "videomamba":  {"num_frames": 16, "img_size": 224, "batch_size": 4},
    "vjepa2":      {"num_frames": 16, "img_size": 256, "batch_size": 1},   # finetune only
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(HERE.parent / "MergedFallDataset"))
    p.add_argument("--splits-dir", default=str(HERE / "splits" / "cs"))
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--modes", choices=["scratch", "finetune", "both"], default="both")
    p.add_argument("--subset", type=int, default=80, help="clips per class from CS-train")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default=str(HERE / "results" / "experiment_table.csv"))
    return p.parse_args()


def main():
    args = parse()
    models = ALL_MODELS if args.models == ["all"] else args.models
    modes = ["scratch", "finetune"] if args.modes == "both" else [args.modes]
    splits = Path(args.splits_dir)

    rows = []
    for mode in modes:
        pretrained = (mode == "finetune")
        for model in models:
            if model == "vjepa2" and not pretrained:
                print(f"\n[skip] vjepa2 has no from-scratch mode (foundation model).")
                continue
            mc = MODEL_CFG.get(model, {"num_frames": 16, "img_size": 224, "batch_size": 4})
            cfg = {
                "data_root": args.data_root, "manifest_csv": None,
                "model": model, "num_frames": mc["num_frames"], "img_size": mc["img_size"],
                "batch_size": mc["batch_size"],
                "lr": 1e-5 if pretrained else 1e-4, "weight_decay": 0.05,
                "epochs": args.epochs, "subset_per_class": args.subset,
                "train_split_csv": str(splits / "train.csv"),
                "val_split_csv": str(splits / "val.csv"),
                "num_workers": args.num_workers, "seed": 42,
                "out_dir": str(HERE / mode / "runs"),
            }
            tag = f"{mode}/{model}"
            print(f"\n{'#'*72}\n# {tag}  (frames={mc['num_frames']}, bs={mc['batch_size']}, "
                  f"subset={args.subset}/class, epochs={args.epochs})\n{'#'*72}")
            try:
                _, metrics = run(cfg, build_model_fn=build_model, pretrained=pretrained)
                rows.append({"mode": mode, "model": model,
                             **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}})
            except Exception as e:
                print(f"[ERROR] {tag} failed: {e}")
                traceback.print_exc()
                rows.append({"mode": mode, "model": model, "error": str(e)[:200]})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    print(f"\n{'='*72}\nEXPERIMENT TABLE  (saved {out})\n{'='*72}")
    cols = ["mode", "model", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc_roc"]
    show = [c for c in cols if c in table.columns]
    print(table[show].to_string(index=False) if show else table.to_string(index=False))
    print("\nJudge: high sensitivity (catches falls) + balanced_accuracy well above 0.5 on the "
          "CS-val subset => worth scaling to the full split. Near 0.5 / 0 sensitivity => not yet.")


if __name__ == "__main__":
    main()
