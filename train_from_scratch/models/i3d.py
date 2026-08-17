"""I3D — Inflated 3D ConvNet (Carreira & Zisserman, CVPR 2017, CORE A*).

Uses the PyTorchVideo I3D-ResNet50 builder. pretrained=False -> random init (from-scratch).
This is the exact architecture OmniFall benchmarks (their most domain-robust model).

Input contract: [B, C, T, H, W] -> logits [B, num_classes].
"""
from __future__ import annotations

import torch.nn as nn


def _replace_head(model, num_classes):
    head = model.blocks[-1]
    if hasattr(head, "proj") and isinstance(head.proj, nn.Linear):
        head.proj = nn.Linear(head.proj.in_features, num_classes)
    return model


class _Wrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):          # [B,C,T,H,W] -> [B,num_classes]
        return self.model(x)


def build(num_classes=2, num_frames=16, img_size=224, pretrained=False, **kw):
    from pytorchvideo.models.hub import i3d_r50
    model = i3d_r50(pretrained=pretrained)
    model = _replace_head(model, num_classes)
    return _Wrap(model)
