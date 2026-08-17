"""SlowFast (Feichtenhofer et al., ICCV 2019, CORE A*).

Two pathways: Slow (low frame-rate, posture/context) + Fast (high frame-rate, fall motion).
PyTorchVideo's SlowFast expects a list [slow_clip, fast_clip]; the wrapper builds both pathways
from a single [B,C,T,H,W] input via temporal subsampling (alpha=4).
pretrained=False -> random init (from-scratch).

Input contract: [B, C, T, H, W] -> logits [B, num_classes].
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _replace_head(model, num_classes):
    head = model.blocks[-1]
    if hasattr(head, "proj") and isinstance(head.proj, nn.Linear):
        head.proj = nn.Linear(head.proj.in_features, num_classes)
    return model


class SlowFastWrap(nn.Module):
    def __init__(self, model, alpha=4):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def _pack(self, x):                       # x: [B,C,T,H,W]
        fast = x
        t = x.shape[2]
        idx = torch.linspace(0, t - 1, t // self.alpha).long().to(x.device)
        slow = torch.index_select(x, 2, idx)
        return [slow, fast]

    def forward(self, x):
        return self.model(self._pack(x))


def build(num_classes=2, num_frames=32, img_size=224, pretrained=False, alpha=4, **kw):
    from pytorchvideo.models.hub import slowfast_r50
    model = slowfast_r50(pretrained=pretrained)
    model = _replace_head(model, num_classes)
    return SlowFastWrap(model, alpha=alpha)
