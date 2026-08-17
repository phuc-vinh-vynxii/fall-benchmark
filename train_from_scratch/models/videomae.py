"""VideoMAE (Tong et al., NeurIPS 2022, CORE A*).

One of the two OmniFall anchor models. ViT backbone with tubelet embedding.
pretrained=False -> random-initialized ViT trained supervised from scratch (the from-scratch
ablation). NOTE: VideoMAE was designed for self-supervised pretraining; from-scratch supervised
on ~3k clips will underfit heavily — this is exactly the gap Case 1 vs Case 2 measures.

Input contract: [B, C, T, H, W] -> logits [B, num_classes].
"""
from __future__ import annotations

import torch.nn as nn

HF_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics"


class _HFWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):                 # [B,C,T,H,W] -> HF wants [B,T,C,H,W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(pixel_values=x).logits


def build(num_classes=2, num_frames=16, img_size=224, pretrained=False, **kw):
    from transformers import VideoMAEForVideoClassification, VideoMAEConfig
    if pretrained:
        model = VideoMAEForVideoClassification.from_pretrained(
            HF_CKPT, num_labels=num_classes, ignore_mismatched_sizes=True)
    else:
        cfg = VideoMAEConfig(image_size=img_size, num_frames=num_frames, num_labels=num_classes)
        model = VideoMAEForVideoClassification(cfg)
    return _HFWrap(model)
