"""MergedFallDataset loader. Handles BOTH mp4 videos and frame-sequence folders.

Reads manifest.csv (columns: dataset, label, media_type, num_frames, rel_path, ...).
Binary task: NoFALL=0, FALL=1.

NOTE on splits: this loader supports a quick STRATIFIED-RANDOM split for smoke testing only.
The leak-safe cross-subject (CS) / cross-view (CV) split (Phase C of PROJECT_PLAN.md) is built
separately by joining subject/cam back from omnifall/labels/*.csv -> splits/{cs,cv}/*.csv, then
passed here via `split_csv`. DO NOT report final numbers on the random split.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import build_clip, sample_indices

LABEL_TO_IDX = {"NoFALL": 0, "FALL": 1}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- IO
def _read_video_frames(path: str, indices) -> np.ndarray:
    """Read len(indices) frames uniformly across the video (mp4/avi/...). Returns [n,H,W,C] uint8.

    The absolute values in `indices` are ignored; only their COUNT matters — frames are
    re-sampled from the video's true length (robust to unknown/guessed lengths).
    Reader priority: decord (if installed) -> OpenCV (cv2, always available here).
    """
    n_want = len(indices)
    # decord (fast) if available
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path))
        total = len(vr)
        sel = np.linspace(0, max(total - 1, 0), n_want).round().astype(int)
        return vr.get_batch([int(i) for i in sel]).asnumpy()
    except Exception:
        pass
    # OpenCV fallback (handles .avi/.mp4/.mov)
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        frames_all = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames_all.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames_all:
            return np.zeros((n_want, 16, 16, 3), dtype=np.uint8)
        sel = np.linspace(0, len(frames_all) - 1, n_want).round().astype(int)
        return np.stack([frames_all[int(i)] for i in sel], 0)
    full_sel = np.linspace(0, total - 1, n_want).round().astype(int)
    want = set(int(i) for i in full_sel)
    got, idx, maxi = {}, 0, int(max(want))
    while idx <= maxi:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            got[idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    if not got:
        return np.zeros((n_want, 16, 16, 3), dtype=np.uint8)
    last = got[max(got)]
    return np.stack([got.get(int(i), last) for i in full_sel], 0)


def _read_frame_dir(path: str, indices) -> np.ndarray:
    """Read given indices from a directory of image frames. Returns [len(indices),H,W,C] uint8."""
    from PIL import Image
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS))
    if not files:
        return np.zeros((len(indices), 16, 16, 3), dtype=np.uint8)
    idx = np.clip(np.asarray(indices, dtype=int), 0, len(files) - 1)
    imgs = [np.array(Image.open(os.path.join(path, files[i])).convert("RGB")) for i in idx]
    # pad to equal HxW if frames differ (rare): use first frame shape
    h, w, _ = imgs[0].shape
    imgs = [im if im.shape[:2] == (h, w) else np.array(Image.fromarray(im).resize((w, h))) for im in imgs]
    return np.stack(imgs, 0)


# ---------------------------------------------------------------------- Dataset
class FallClipDataset(Dataset):
    def __init__(self, data_root, manifest_csv=None, split_csv=None,
                 num_frames=16, img_size=224, train=True,
                 subset_per_class=None, seed=42):
        self.data_root = Path(data_root)
        manifest_csv = manifest_csv or (self.data_root / "manifest.csv")
        df = pd.read_csv(manifest_csv)
        df = df[df["label"].isin(LABEL_TO_IDX)].copy()

        # 1) restrict to a leak-safe CS/CV split if given
        if split_csv is not None:
            keep = set(pd.read_csv(split_csv)["rel_path"])
            df = df[df["rel_path"].isin(keep)].copy()
        # 2) THEN optionally take a small balanced subset *within* that split
        #    (so "subset of CS-train" stays leak-safe — exactly the quick-experiment case)
        if subset_per_class is not None:
            rng = np.random.default_rng(seed)
            parts = []
            for lab in LABEL_TO_IDX:
                sub = df[df["label"] == lab]
                n = min(subset_per_class, len(sub))
                parts.append(sub.iloc[rng.permutation(len(sub))[:n]])
            df = pd.concat(parts).reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.df)

    def label_counts(self):
        return self.df["label"].value_counts().to_dict()

    def class_weights(self):
        """Inverse-frequency weights [w_NoFALL, w_FALL] for weighted CE (imbalance 1:2.9)."""
        c = self.df["label"].value_counts()
        n0, n1 = c.get("NoFALL", 1), c.get("FALL", 1)
        total = n0 + n1
        return torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = str(self.data_root / row["rel_path"])
        label = LABEL_TO_IDX[row["label"]]
        total = int(row["num_frames"]) if not pd.isna(row.get("num_frames")) else 0

        is_video = str(row.get("media_type", "video")) == "video" and path.lower().endswith(".mp4")
        if total <= 0:
            total = self.num_frames * 4  # guess; readers clip safely
        indices = sample_indices(total, self.num_frames)

        try:
            frames = _read_video_frames(path, indices) if is_video else _read_frame_dir(path, indices)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")
            frames = np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)

        clip = build_clip(frames, self.num_frames, self.img_size, self.train)  # [C,T,H,W]
        return clip, label
