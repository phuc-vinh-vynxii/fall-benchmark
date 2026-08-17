"""Inference on held-out test videos (the ones you'll send later).

Loads a trained checkpoint and predicts FALL / NoFALL for each video in a folder (or a CSV list),
prints a table + saves predictions.csv. If you provide ground-truth labels it also computes the
locked metric set so you can judge whether a model is worth training further.

Usage:
    # predict every .mp4 in a folder
    python predict.py --ckpt train_from_scratch/runs/x3d_best.pt --videos path/to/test_videos

    # with labels for metrics: a CSV with columns  path,label   (label in {FALL,NoFALL})
    python predict.py --ckpt fine_tune/runs/videomae_best.pt --list test_labeled.csv

Frame-folder inputs work too (pass a dir whose subfolders are frame sequences via --list).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "fine_tune"))        # superset registry (8 models incl. vjepa2)

from common.transforms import build_clip, sample_indices   # noqa: E402
from common.metrics import compute_metrics, format_metrics  # noqa: E402
from common.dataset import _read_video_frames, _read_frame_dir, IMG_EXTS  # noqa: E402
from models import build_model                              # noqa: E402

LABELS = ["NoFALL", "FALL"]


def load_clip(path, num_frames, img_size):
    p = Path(path)
    if p.is_dir():
        n = len([f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS])
        frames = _read_frame_dir(str(p), sample_indices(n, num_frames))
    else:
        # probe length cheaply via reader (sample assuming >= num_frames, reader clips safely)
        frames = _read_video_frames(str(p), sample_indices(10**6, num_frames))
        if len(frames) < num_frames:
            frames = _read_video_frames(str(p), sample_indices(len(frames), num_frames))
    return build_clip(frames, num_frames, img_size, train=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--videos", help="folder of .mp4 files")
    ap.add_argument("--list", dest="list_csv", help="CSV with column 'path' (+ optional 'label')")
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    name = cfg.get("model", "x3d")
    num_frames = args.num_frames or cfg.get("num_frames", 16)
    img_size = args.img_size or cfg.get("img_size", 224)

    model = build_model(name, num_classes=2, num_frames=num_frames, img_size=img_size,
                        pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"[model] {name} | frames={num_frames} | img={img_size} | device={device}")

    # gather inputs
    if args.list_csv:
        df = pd.read_csv(args.list_csv)
    elif args.videos:
        vids = sorted(str(p) for p in Path(args.videos).rglob("*.mp4"))
        df = pd.DataFrame({"path": vids})
    else:
        raise SystemExit("provide --videos DIR or --list CSV")

    rows = []
    for _, r in df.iterrows():
        path = r["path"]
        try:
            clip = load_clip(path, num_frames, img_size).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(clip).float(), dim=1)[0].cpu().numpy()
            pred = int(prob[1] >= 0.5)
            rows.append({"path": path, "pred": LABELS[pred], "p_FALL": float(prob[1])})
            print(f"  {LABELS[pred]:<7} p_FALL={prob[1]:.3f}  {path}")
        except Exception as e:
            print(f"  [skip] {path}: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n[saved] {args.out}  ({len(out)} videos)")

    # metrics if labels available
    if "label" in df.columns:
        merged = out.merge(df[["path", "label"]], on="path")
        y_true = (merged["label"] == "FALL").astype(int).values
        y_score = merged["p_FALL"].values
        print("\n=== METRICS on labeled test set ===")
        print(format_metrics(compute_metrics(y_true, y_score)))


if __name__ == "__main__":
    main()
