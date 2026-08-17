"""VideoMamba (Li et al., ECCV 2024, CORE A*) — State Space Model for efficient video.

Linear-complexity temporal modeling; motion-sensitive; edge-friendly.

Two backends:
  1. OFFICIAL: if the `mamba_ssm` CUDA kernels are installed (Linux+CUDA, e.g. Kaggle/Lambda),
     use the real selective-scan blocks. Install: `pip install causal-conv1d mamba-ssm` and add
     the official VideoMamba repo to PYTHONPATH (OpenGVLab/VideoMamba).
  2. FALLBACK ("VideoMamba-lite"): a compact pure-PyTorch bidirectional gated-SSM approximation so
     the architecture instantiates and trains on Windows/CPU for smoke tests. NOT the official
     kernels — do not report fallback numbers as "VideoMamba" in the paper; rerun on Linux.

Input contract: [B, C, T, H, W] -> logits [B, num_classes].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------- pure-PyTorch fallback
class _BiSSMlite(nn.Module):
    """Bidirectional gated linear-recurrence block (Mamba-style, simplified, kernel-free)."""
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        inner = expand * dim
        self.in_proj = nn.Linear(dim, inner * 2)
        self.dwconv = nn.Conv1d(inner, inner, kernel_size=3, padding=1, groups=inner)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, d_state, inner).clamp(min=1e-3)))
        self.out_proj = nn.Linear(inner, dim)
        self.norm = nn.LayerNorm(dim)

    def _scan(self, x):                       # x: [B,L,inner]
        a = torch.exp(-F.softplus(self.A_log))            # decay in (0,1), [inner]
        out = torch.zeros_like(x)
        h = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        for t in range(x.shape[1]):           # linear recurrence h_t = a*h_{t-1} + x_t
            h = a * h + x[:, t]
            out[:, t] = h
        return out

    def forward(self, x):                     # x: [B,L,dim]
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)                  # [B,L,2*inner]
        xx, z = xz.chunk(2, dim=-1)
        xx = self.dwconv(xx.transpose(1, 2)).transpose(1, 2)
        xx = F.silu(xx)
        fwd = self._scan(xx)
        bwd = torch.flip(self._scan(torch.flip(xx, [1])), [1])
        y = (fwd + bwd) * F.silu(z)
        return res + self.out_proj(y)


class VideoMambaLite(nn.Module):
    def __init__(self, num_classes=2, num_frames=16, img_size=224,
                 patch=16, dim=384, depth=8):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n_patches = (img_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_patches, dim))
        self.tpos = nn.Parameter(torch.zeros(1, num_frames, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([_BiSSMlite(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.tpos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):                     # [B,C,T,H,W]
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)   # [B*T, N, dim]
        x = x + self.pos
        x = x.mean(dim=1).view(b, t, -1)                     # [B,T,dim] spatial pool
        x = x + self.tpos[:, :t]
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)                       # [B,T+1,dim]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))


def build(num_classes=2, num_frames=16, img_size=224, pretrained=False, **kw):
    # Try official kernels first.
    try:
        import mamba_ssm  # noqa: F401  (presence check)
        from videomamba import videomamba_tiny  # official repo on PYTHONPATH
        model = videomamba_tiny(num_classes=num_classes, num_frames=num_frames,
                                pretrained=pretrained)
        return model
    except Exception as e:
        if pretrained:
            print(f"[videomamba] official kernels unavailable ({e}); "
                  "pretrained weights need the official repo. Using random-init lite fallback.")
        else:
            print(f"[videomamba] using pure-PyTorch VideoMamba-lite fallback ({e}).")
        return VideoMambaLite(num_classes, num_frames, img_size)
