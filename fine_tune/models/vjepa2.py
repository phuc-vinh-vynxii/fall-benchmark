"""V-JEPA 2 (Meta, 2025, arXiv:2506.09985) — FINE-TUNE ONLY.

Self-supervised video foundation model (pretrained on ~1M hours). SOTA motion understanding.
It CANNOT be trained from scratch on ~3k clips, so it exists only in fine_tune/.

⚠️ Scientific caveat: as of writing this is a Meta PREPRINT (not peer-reviewed). Report it as a
supplementary/reference baseline and flag the preprint status — do not treat its self-reported
SOTA as audited (see PROJECT_PLAN.md Phase A discussion).

Strategy: load the frozen/pretrained encoder, attach a linear classification head, fine-tune.
Needs a large GPU (Lambda) — too big for the 6GB local card.

Input contract: [B, C, T, H, W] -> logits [B, num_classes].
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Common HF checkpoints (pick per VRAM): vitl (large) / vitg (giant)
HF_CKPT = "facebook/vjepa2-vitl-fpc64-256"


class _VJEPA2Classifier(nn.Module):
    def __init__(self, encoder, hidden, num_classes, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):                      # [B,C,T,H,W] -> [B,T,C,H,W]
        x = x.permute(0, 2, 1, 3, 4)
        out = self.encoder(pixel_values_videos=x) if _accepts(self.encoder, "pixel_values_videos") \
            else self.encoder(x)
        feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        pooled = feats.mean(dim=1) if feats.dim() == 3 else feats   # mean over tokens
        return self.head(self.norm(pooled))


def _accepts(module, kw):
    import inspect
    try:
        return kw in inspect.signature(module.forward).parameters
    except (ValueError, TypeError):
        return False


def build(num_classes=2, num_frames=16, img_size=256, pretrained=True,
          freeze_encoder=False, **kw):
    if not pretrained:
        # Architecture-only (random init) so a TRAINED checkpoint can be loaded for
        # predict/heatmap (weights get overwritten by load_state_dict). Config fetch is tiny.
        from transformers import AutoModelForVideoClassification, AutoConfig
        cfg = AutoConfig.from_pretrained(HF_CKPT, num_labels=num_classes)
        return _HFHead(AutoModelForVideoClassification.from_config(cfg))
    # 1) Preferred: a ready classification head if the installed transformers exposes it.
    try:
        from transformers import AutoModelForVideoClassification
        return _HFHead(AutoModelForVideoClassification.from_pretrained(
            HF_CKPT, num_labels=num_classes, ignore_mismatched_sizes=True))
    except Exception as e1:
        print(f"[vjepa2] no AutoModelForVideoClassification ({e1}); falling back to encoder + linear head.")
    # 2) Fallback: load encoder via AutoModel and attach our own head.
    try:
        from transformers import AutoModel
        enc = AutoModel.from_pretrained(HF_CKPT)
        hidden = getattr(enc.config, "hidden_size", 1024)
        return _VJEPA2Classifier(enc, hidden, num_classes, freeze_encoder)
    except Exception as e2:
        raise RuntimeError(
            f"Could not load V-JEPA 2 from '{HF_CKPT}'. Update transformers, or load via "
            f"torch.hub('facebookresearch/vjepa2'). Underlying error: {e2}")


class _HFHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(pixel_values_videos=x).logits
