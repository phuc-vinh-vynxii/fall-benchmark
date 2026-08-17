"""Notebook 2 driver: load trained checkpoints, run them on a few test videos, and draw a
Grad-CAM heatmap per (model x video) so you can see whether each model attends to the PERSON /
fall motion or cheats on the background.

Outputs (into --out, default viz_out/):
  <model>__<video>.png     one row = sampled frames with heatmap overlay + prediction
  SUMMARY_grid.png         rows = models, cols = videos (one overlay each) — quick comparison
  predictions.csv          model, video, pred, p_FALL (+ label/correct if labels given)
  metrics_<model>.txt      locked metrics per model (only if labels provided)

Usage (Kaggle Notebook 2):
    python heatmap_eval.py --ckpt-dir /kaggle/input/<trained-models> \\
                           --videos   /kaggle/input/<test-videos> --out viz_out
    # with labels: --list test_labeled.csv   (columns: path,label[FALL/NoFALL])
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "fine_tune"))          # 8-model registry (incl. vjepa2)

from common.transforms import build_clip, sample_indices          # noqa: E402
from common.dataset import _read_video_frames, _read_frame_dir, IMG_EXTS  # noqa: E402
from common.viz import (GradCAM, resolve_target, attention_rollout,          # noqa: E402
                        input_saliency, denormalize_frame, overlay)
CNN_MODELS = {"i3d", "slowfast", "x3d", "tsm"}
TRANSFORMER_MODELS = {"videomae", "timesformer", "vjepa2"}
from common.metrics import compute_metrics, format_metrics        # noqa: E402
from models import build_model                                    # noqa: E402

LABELS = ["NoFALL", "FALL"]


def load_clip(path, num_frames, img_size):
    p = Path(path)
    if p.is_dir():
        n = len([f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS])
        frames = _read_frame_dir(str(p), sample_indices(n, num_frames))
    else:
        frames = _read_video_frames(str(p), sample_indices(10**6, num_frames))
        if len(frames) < num_frames:
            frames = _read_video_frames(str(p), sample_indices(len(frames), num_frames))
    return build_clip(frames, num_frames, img_size, train=False)


def gather_videos(args):
    if args.list_csv:
        df = pd.read_csv(args.list_csv)
    else:
        exts = (".mp4", ".avi", ".mov", ".mkv")
        vids = sorted(str(p) for p in Path(args.videos).rglob("*") if p.suffix.lower() in exts)
        if not vids:  # maybe frame folders
            vids = sorted(str(p) for p in Path(args.videos).iterdir() if p.is_dir())
        df = pd.DataFrame({"path": vids})
    return df


def save_model_video_png(clip, cam, out_png, title, n_show=6):
    T = clip.shape[1]
    idxs = np.linspace(0, T - 1, n_show).round().astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(2.1 * n_show, 2.6))
    for ax, ti in zip(axes, idxs):
        frame = denormalize_frame(clip, int(ti))
        img = overlay(frame, cam) if cam is not None else frame
        ax.imshow(img); ax.axis("off"); ax.set_title(f"t={ti}", fontsize=7)
    fig.suptitle(title + ("" if cam is not None else "  [no heatmap]"), fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=90, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="folder with *_best.pt")
    ap.add_argument("--videos", help="folder of test .mp4 / frame folders")
    ap.add_argument("--list", dest="list_csv", help="CSV col 'path' (+ optional 'label')")
    ap.add_argument("--out", default="viz_out")
    ap.add_argument("--n-show", type=int, default=6)
    ap.add_argument("--models", nargs="+", help="subset of model names (default: all in ckpt-dir)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(Path(args.ckpt_dir).glob("*_best.pt")) or sorted(Path(args.ckpt_dir).glob("*.pt"))
    if args.models:
        ckpts = [c for c in ckpts if c.stem.replace("_best", "") in args.models]
    vids = gather_videos(args)
    print(f"[info] {len(ckpts)} models x {len(vids)} videos | device={device}")

    rows = []
    grid = {}                       # (model, vid_name) -> overlay image for summary
    for ck in ckpts:
        blob = torch.load(ck, map_location=device)
        cfg = blob.get("cfg", {})
        name = cfg.get("model") or ck.stem.replace("_best", "")
        nf, isz = cfg.get("num_frames", 16), cfg.get("img_size", 224)
        try:
            model = build_model(name, num_classes=2, num_frames=nf, img_size=isz, pretrained=False)
            model.load_state_dict(blob["model"]); model.to(device).eval()
        except Exception as e:
            print(f"[skip model] {name}: {e}"); continue

        cam_engine = None
        if name in CNN_MODELS:
            tgt, reshape = resolve_target(model, name)
            if tgt is not None:
                cam_engine = GradCAM(model, tgt, reshape)
        method = "grad-cam" if name in CNN_MODELS else \
                 ("attn-rollout" if name in TRANSFORMER_MODELS else "saliency")
        print(f"\n=== {name} (frames={nf}, img={isz}) | heatmap={method} ===")

        for _, r in vids.iterrows():
            vpath = r["path"]; vname = Path(vpath).stem
            try:
                clip = load_clip(vpath, nf, isz).to(device)
                with torch.no_grad():
                    prob = torch.softmax(model(clip.unsqueeze(0)).float(), 1)[0]
                p_fall = float(prob[1]); pred = int(p_fall >= 0.5)
                cam = None
                try:
                    xb = clip.unsqueeze(0)
                    if cam_engine is not None:                    # CNN -> Grad-CAM
                        cam = cam_engine(xb.requires_grad_(True), class_idx=1)
                    elif name in TRANSFORMER_MODELS:              # transformer -> attn rollout
                        cam = attention_rollout(model, xb)
                        if cam is None:                           # fallback if attn unavailable
                            cam = input_saliency(model, xb, 1)
                    else:                                         # videomamba / other -> saliency
                        cam = input_saliency(model, xb, 1)
                except Exception as e:
                    print(f"  [heatmap fail] {name}/{vname}: {e}")
                title = f"{name} | {vname} | pred={LABELS[pred]} p_FALL={p_fall:.2f} | {method}"
                save_model_video_png(clip.cpu(), cam, out / f"{name}__{vname}.png", title, args.n_show)
                mid = clip.shape[1] // 2
                base = denormalize_frame(clip.cpu(), mid)
                grid[(name, vname)] = overlay(base, cam) if cam is not None else base
                row = {"model": name, "video": vname, "pred": LABELS[pred], "p_FALL": round(p_fall, 4)}
                if "label" in vids.columns:
                    row["label"] = r["label"]; row["correct"] = (LABELS[pred] == r["label"])
                rows.append(row)
                print(f"  {vname:<28} pred={LABELS[pred]:<7} p_FALL={p_fall:.2f}"
                      f"{' heatmap' if cam is not None else ' (no cam)'}")
            except Exception as e:
                print(f"  [video fail] {name}/{vname}: {e}")
        if cam_engine is not None:
            cam_engine.remove()

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out / "predictions.csv", index=False)

    # summary grid: rows=models, cols=videos
    models = sorted({m for m, _ in grid})
    vnames = sorted({v for _, v in grid})
    if models and vnames:
        fig, axes = plt.subplots(len(models), len(vnames),
                                 figsize=(2.4 * len(vnames), 2.4 * len(models)), squeeze=False)
        for i, m in enumerate(models):
            for j, v in enumerate(vnames):
                ax = axes[i][j]; ax.axis("off")
                if (m, v) in grid:
                    ax.imshow(grid[(m, v)])
                if i == 0:
                    ax.set_title(v, fontsize=8)
                if j == 0:
                    ax.set_ylabel(m, fontsize=9, rotation=90, labelpad=12); ax.axis("on")
                    ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Grad-CAM: rows=models, cols=test videos", fontsize=12)
        fig.tight_layout(); fig.savefig(out / "SUMMARY_grid.png", dpi=95, bbox_inches="tight")
        plt.close(fig)

    # per-model metrics if labels available
    if "label" in vids.columns and rows:
        for m, g in pred_df.groupby("model"):
            y_true = (g["label"] == "FALL").astype(int).values
            y_score = g["p_FALL"].values
            (out / f"metrics_{m}.txt").write_text(format_metrics(compute_metrics(y_true, y_score)))

    print(f"\n[done] -> {out}  (predictions.csv, SUMMARY_grid.png, per-model PNGs)")


if __name__ == "__main__":
    main()
