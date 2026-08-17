"""Controlled experiment on the SAME 7 Le2i videos: compare 3 temporal-sampling protocols to
see how they affect generalization. Trimmed-trained models are tested under:

  1. uniform  — 16 frames spread over the WHOLE (untrimmed) video  [current baseline; fall diluted]
  2. trimmed  — for FALL videos, sample from the annotated fall window only [matches training]
  3. sliding  — split video into N windows, classify each, video = max p_FALL over windows [deploy]

Outputs a model x protocol table of balanced-accuracy / sensitivity / specificity.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "fine_tune"))
from common.transforms import build_clip                     # noqa: E402
from models import build_model                               # noqa: E402

BASE = str(HERE.parent / "test_raw" / "lei2fall" / "Le2i Fall")
CKPT_DIR = HERE.parent / "ket_qua_2"
MODELS = ["i3d", "slowfast", "x3d", "tsm", "videomae", "timesformer", "videomamba", "vjepa2"]
N_WIN, MARGIN = 4, 8


def read_all(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def sample(frames, n):
    if not frames:
        return np.zeros((n, 16, 16, 3), np.uint8)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.stack([frames[int(i)] for i in idx], 0)


def build_videos():
    vids = []
    for room, key in [("Coffee_room_01", "coffee"), ("Home_01", "home")]:
        found = sorted(glob.glob(f"{BASE}/{room}/**/*.avi", recursive=True))[:2]
        for i, v in enumerate(found, 1):
            ann = os.path.splitext(v.replace("Videos", "Annotation_files"))[0] + ".txt"
            L = open(ann).read().splitlines()
            vids.append((f"{key}_{i}", v, (int(L[0]), int(L[1])), "FALL"))
    adl = []
    for v in sorted(glob.glob(f"{BASE}/Home_02/**/Videos/*.avi", recursive=True)):
        ann = os.path.splitext(v.replace("Videos", "Annotation_files"))[0] + ".txt"
        try:
            L = open(ann).read().splitlines()
            if int(L[0]) == 0 and int(L[1]) == 0:
                adl.append(v)
        except Exception:
            pass
    for i, v in enumerate(adl[:3], 1):
        vids.append((f"adl_{i}", v, None, "NoFALL"))
    return vids


def clips_for(frames, interval, proto, nf, isz):
    """Return a list of [C,T,H,W] clips for the given protocol."""
    total = len(frames)
    if proto == "uniform":
        return [build_clip(sample(frames, nf), nf, isz, False)]
    if proto == "trimmed":
        if interval is not None:                     # FALL: window around the fall
            s, e = interval
            w = frames[max(0, s - MARGIN):min(total, e + MARGIN)] or frames
        else:                                        # NoFALL: whole ADL video
            w = frames
        return [build_clip(sample(w, nf), nf, isz, False)]
    # sliding: N non-overlapping windows over the whole video
    clips = []
    for k in range(N_WIN):
        w = frames[k * total // N_WIN:(k + 1) * total // N_WIN] or frames
        clips.append(build_clip(sample(w, nf), nf, isz, False))
    return clips


def metrics(df):
    tp = ((df.label == "FALL") & (df.pred == "FALL")).sum()
    fn = ((df.label == "FALL") & (df.pred == "NoFALL")).sum()
    tn = ((df.label == "NoFALL") & (df.pred == "NoFALL")).sum()
    fp = ((df.label == "NoFALL") & (df.pred == "FALL")).sum()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return round((sens + spec) / 2, 3), round(sens, 3), round(spec, 3)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    videos = build_videos()
    frames_cache = {n: read_all(v) for n, v, _, _ in videos}
    print(f"[info] {len(videos)} videos | device={device} | "
          f"frames: {[(n, len(f)) for n, f in frames_cache.items()]}")

    rows = []
    for name in MODELS:
        ck = CKPT_DIR / f"{name}_best.pt"
        if not ck.exists():
            continue
        blob = torch.load(ck, map_location=device)
        cfg = blob.get("cfg", {})
        nf, isz = cfg.get("num_frames", 16), cfg.get("img_size", 224)
        try:
            model = build_model(name, 2, num_frames=nf, img_size=isz, pretrained=False)
            model.load_state_dict(blob["model"]); model.to(device).eval()
        except Exception as e:
            print(f"[skip] {name}: {e}"); continue
        print(f"\n=== {name} (nf={nf}) ===")

        for proto in ["uniform", "trimmed", "sliding"]:
            recs = []
            for vname, _, interval, label in videos:
                clips = clips_for(frames_cache[vname], interval, proto, nf, isz)
                ps = []
                with torch.no_grad():
                    for c in clips:
                        p = torch.softmax(model(c.unsqueeze(0).to(device)).float(), 1)[0, 1].item()
                        ps.append(p)
                p_fall = max(ps)                     # aggregate: any window fall -> fall
                recs.append({"video": vname, "label": label,
                             "pred": "FALL" if p_fall >= 0.5 else "NoFALL", "p": round(p_fall, 2)})
            d = pd.DataFrame(recs)
            bacc, sens, spec = metrics(d)
            rows.append({"model": name, "protocol": proto, "bal_acc": bacc,
                         "sens": sens, "spec": spec})
            print(f"  {proto:<8} bal_acc={bacc}  sens={sens}  spec={spec}   "
                  f"{dict(zip(d.video, d.p))}")

    out = pd.DataFrame(rows)
    out.to_csv(HERE.parent / "protocol_compare.csv", index=False)
    print("\n===== BAL_ACC: model x protocol =====")
    print(out.pivot(index="model", columns="protocol", values="bal_acc").to_string())
    print("\n[saved] protocol_compare.csv")


if __name__ == "__main__":
    main()
