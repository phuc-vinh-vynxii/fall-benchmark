"""X3D — Efficient 3D CNN (Feichtenhofer, CVPR 2020, CORE A*).

Very lightweight -> fits the local RTX 3060 6GB and edge deployment. Uses PyTorchVideo x3d_m.
pretrained=False -> random init (from-scratch).

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

    def forward(self, x):
        return self.model(x)


def build(num_classes=2, num_frames=16, img_size=224, pretrained=False, **kw):
    from pytorchvideo.models.hub import x3d_m
    model = x3d_m(pretrained=pretrained)
    model = _replace_head(model, num_classes)
    return _Wrap(model)
