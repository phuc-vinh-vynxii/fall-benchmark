"""Grad-CAM heatmaps for the video models (Phase G).

Purpose: check whether a model attends to the PERSON / the fall motion, or cheats on the
background / data-source. Reliable for CNNs (x3d, i3d, slowfast, tsm); best-effort for the
transformer / mamba models (videomae, timesformer, videomamba, vjepa2) via a LayerNorm target
with a token->grid reshape (may need a per-model tweak — wrapped in try/except so it degrades
to "no heatmap, prediction still shown").

Hook-based, no heavy deps (numpy / torch / opencv / matplotlib).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .transforms import MEAN, STD


# --------------------------------------------------------------- target resolution
def _last_module(model, types):
    found = None
    for m in model.modules():
        if isinstance(m, types):
            found = m
    return found


def resolve_target(model, name):
    """Return (target_module, reshape_fn|None). reshape_fn maps activations -> [B,C,H,W]."""
    name = (name or "").lower()
    if name in {"x3d", "i3d", "slowfast"}:
        return _last_module(model, nn.Conv3d), None
    if name == "tsm":
        return _last_module(model, nn.Conv2d), None
    # transformers / mamba: target the last LayerNorm; reshape tokens to a square grid
    norm = _last_module(model, nn.LayerNorm)

    def reshape(act):                       # act: [B, N, C] (maybe with cls token(s))
        if act.dim() != 3:
            return act
        b, n, c = act.shape
        for drop in (0, 1):
            m = n - drop
            s = int(round(m ** 0.5))
            if s * s == m:
                return act[:, drop:, :].transpose(1, 2).reshape(b, c, s, s)
            for t in (8, 16, 2, 4, 32):     # token grid may be temporal*spatial
                if m % t == 0:
                    sp = m // t
                    s = int(round(sp ** 0.5))
                    if s * s == sp:
                        g = act[:, drop:, :].reshape(b, t, sp, c).mean(1)
                        return g.transpose(1, 2).reshape(b, c, s, s)
        return act
    return norm, reshape


# ----------------------------------------------------------------------- Grad-CAM
class GradCAM:
    def __init__(self, model, target_module, reshape_fn=None):
        self.model = model
        self.reshape_fn = reshape_fn
        self.acts = None
        self.grads = None
        self._h1 = target_module.register_forward_hook(self._fwd)
        self._h2 = target_module.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def remove(self):
        self._h1.remove(); self._h2.remove()

    def __call__(self, x, class_idx=1):
        """x: [1,C,T,H,W]. Returns cam [H,W] in [0,1] for the requested class, or None."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, class_idx].backward()
        acts, grads = self.acts, self.grads
        if acts is None or grads is None:
            return None
        if self.reshape_fn is not None:
            acts, grads = self.reshape_fn(acts), self.reshape_fn(grads)
        if acts.dim() == 5:                 # [B,C,T,H,W]
            w = grads.mean(dim=(2, 3, 4), keepdim=True)
            cam = torch.relu((w * acts).sum(1)).mean(1)     # -> [B,H,W]
        elif acts.dim() == 4:               # [B,C,H,W] or [B*T,C,H,W]
            w = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * acts).sum(1)).mean(0, keepdim=True)
        else:
            return None
        cam = cam[0].float().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


# ----------------------------------------------------------------- visualization
def denormalize_frame(clip, t_index):
    """clip: [C,T,H,W] normalized -> RGB uint8 [H,W,3] at time t_index."""
    f = clip[:, t_index] * STD[:, 0] + MEAN[:, 0]
    return (f.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def overlay(frame_uint8, cam2d, alpha=0.45):
    """Blend a [0,1] heatmap over an RGB frame. Returns uint8 [H,W,3]."""
    import cv2
    h, w = frame_uint8.shape[:2]
    cam = cv2.resize(cam2d.astype(np.float32), (w, h))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * frame_uint8).astype(np.uint8)


# ----------------------------------------------------- transformer: attention rollout
def _tokens_to_grid(v):
    """1-D token importance [N] -> square spatial grid [s,s] (drops cls, collapses time)."""
    n = int(v.shape[0])
    for drop in (0, 1):
        m = n - drop
        s = int(round(m ** 0.5))
        if s * s == m:
            return v[drop:].reshape(s, s)
        for t in (8, 16, 2, 4, 32):        # spatiotemporal tokens -> collapse time
            if m % t == 0:
                sp = m // t
                s = int(round(sp ** 0.5))
                if s * s == sp:
                    return v[drop:].reshape(t, sp).mean(0).reshape(s, s)
    return None


def attention_rollout(model, x):
    """Reliable heatmap for HF video transformers (VideoMAE/TimeSformer/V-JEPA2).

    Multiplies per-layer attention (with residual) to trace patch->output influence.
    model: wrapper whose `.model` is a *ForVideoClassification. x: [1,C,T,H,W]. Returns [H,W] or None.
    """
    hf = getattr(model, "model", None)
    if hf is None:
        return None
    pixel = x.permute(0, 2, 1, 3, 4)
    try:
        out = hf(pixel_values=pixel, output_attentions=True)
        atts = getattr(out, "attentions", None)
        if not atts:
            return None
        n = atts[0].shape[-1]
        eye = torch.eye(n)
        R = eye.clone()
        for a in atts:
            a = a.detach().float().mean(1)[0].cpu()   # avg heads -> [N,N]
            a = a + eye
            a = a / a.sum(-1, keepdim=True)
            R = a @ R
        mask = R.mean(0)                              # importance received per token
        grid = _tokens_to_grid(mask)
        if grid is None:
            return None
        g = grid.numpy().astype(np.float32)
        g -= g.min()
        return g / g.max() if g.max() > 0 else g
    except Exception:
        return None


# --------------------------------------------------- universal fallback: input saliency
def input_saliency(model, x, class_idx=1):
    """Model-agnostic |d logit / d input| heatmap (works for VideoMamba/SSM & anything). [H,W]."""
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, class_idx].backward()
    if x.grad is None:
        return None
    g = x.grad.detach()[0].abs().mean(dim=(0, 1)).cpu().numpy()   # [H,W]
    g -= g.min()
    return g / g.max() if g.max() > 0 else g
