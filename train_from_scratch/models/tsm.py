"""TSM — Temporal Shift Module (Lin et al., ICCV 2019, CORE A*).

Hand-implemented from scratch: a 2D ResNet-50 where each residual block's first conv is
preceded by a zero-cost temporal shift (move 1/8 channels forward, 1/8 backward in time).
This is the canonical TSM construction. `pretrained` here = ImageNet-2D backbone (the standard
TSM init); set pretrained=False for the from-scratch ablation.

Input contract: [B, C, T, H, W] -> logits [B, num_classes].
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class TemporalShift(nn.Module):
    def __init__(self, net: nn.Module, n_segment: int, n_div: int = 8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    @staticmethod
    def _shift(x, n_segment, fold_div):
        nt, c, h, w = x.size()
        nb = nt // n_segment
        x = x.view(nb, n_segment, c, h, w)
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                 # shift to future
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # shift to past
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]             # stay
        return out.view(nt, c, h, w)

    def forward(self, x):
        return self.net(self._shift(x, self.n_segment, self.fold_div))


def _inject_shift(resnet: nn.Module, n_segment: int, n_div: int = 8):
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(resnet, layer_name)
        for block in layer:
            block.conv1 = TemporalShift(block.conv1, n_segment, n_div)
    return resnet


class TSM(nn.Module):
    def __init__(self, num_classes=2, num_frames=8, pretrained=False, n_div=8):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone = _inject_shift(backbone, n_segment=num_frames, n_div=n_div)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone
        self.num_frames = num_frames

    def forward(self, x):                     # x: [B,C,T,H,W]
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)   # [B*T,C,H,W]
        out = self.backbone(x)                                 # [B*T,num_classes]
        return out.view(b, t, -1).mean(dim=1)                  # temporal consensus


def build(num_classes=2, num_frames=8, img_size=224, pretrained=False, **kw):
    return TSM(num_classes=num_classes, num_frames=num_frames, pretrained=pretrained)
