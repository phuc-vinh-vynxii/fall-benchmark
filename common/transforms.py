"""Temporal sampling + spatial transforms. Output contract: [C, T, H, W] float, Kinetics-normalized.

Every model wrapper in this repo expects input shaped [B, C, T, H, W]; HuggingFace models
(VideoMAE / TimeSformer / V-JEPA 2) permute to [B, T, C, H, W] internally.
"""
from __future__ import annotations

import numpy as np
import torch

# Kinetics-400 normalization (pytorchvideo convention)
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


def sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Uniformly sample `num_frames` indices from [0, total_frames). Pads by repeat if too short."""
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=int)
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames).round().astype(int)
    # too short -> uniform with repetition
    return np.clip(np.linspace(0, total_frames - 1, num_frames).round().astype(int), 0, total_frames - 1)


def _resize_shorter(frames: torch.Tensor, size: int) -> torch.Tensor:
    """frames: [T,C,H,W] float in [0,1]. Resize shorter side to `size`."""
    import torch.nn.functional as F
    _, _, h, w = frames.shape
    if h < w:
        nh, nw = size, int(round(w * size / h))
    else:
        nh, nw = int(round(h * size / w)), size
    return F.interpolate(frames, size=(nh, nw), mode="bilinear", align_corners=False)


def _crop(frames: torch.Tensor, size: int, train: bool) -> torch.Tensor:
    _, _, h, w = frames.shape
    if train:
        top = int(torch.randint(0, max(1, h - size + 1), (1,)))
        left = int(torch.randint(0, max(1, w - size + 1), (1,)))
    else:  # center crop
        top, left = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return frames[:, :, top:top + size, left:left + size]


def build_clip(frames_uint8: np.ndarray, num_frames: int, img_size: int, train: bool) -> torch.Tensor:
    """frames_uint8: [T,H,W,C] uint8 (already temporally sampled to ~num_frames). Returns [C,T,H,W] float."""
    x = torch.from_numpy(frames_uint8).float() / 255.0      # [T,H,W,C]
    x = x.permute(0, 3, 1, 2)                                # [T,C,H,W]
    x = _resize_shorter(x, int(img_size * 1.15))
    x = _crop(x, img_size, train)
    if train and torch.rand(1).item() < 0.5:                # horizontal flip
        x = torch.flip(x, dims=[3])
    x = x.permute(1, 0, 2, 3)                                # [C,T,H,W]
    x = (x - MEAN) / STD
    # enforce exact T
    if x.shape[1] != num_frames:
        idx = torch.linspace(0, x.shape[1] - 1, num_frames).round().long()
        x = x.index_select(1, idx)
    return x.contiguous()
