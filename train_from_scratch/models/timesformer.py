"""TimeSformer (Bertasius et al., ICML 2021, CORE A*).

Divided space-time attention ViT. pretrained=False -> random-init from scratch.

Input contract: [B, C, T, H, W] -> logits [B, num_classes].
"""
from __future__ import annotations

import torch.nn as nn

HF_CKPT = "facebook/timesformer-base-finetuned-k400"


class _HFWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):                 # [B,C,T,H,W] -> [B,T,C,H,W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(pixel_values=x).logits


def build(num_classes=2, num_frames=8, img_size=224, pretrained=False, **kw):
    from transformers import TimesformerForVideoClassification, TimesformerConfig
    if pretrained:
        model = TimesformerForVideoClassification.from_pretrained(
            HF_CKPT, num_labels=num_classes, ignore_mismatched_sizes=True)
    else:
        cfg = TimesformerConfig(image_size=img_size, num_frames=num_frames, num_labels=num_classes)
        model = TimesformerForVideoClassification(cfg)
    return _HFWrap(model)
