"""Model registry for the FROM-SCRATCH benchmark (random init, trainable on small data).

7 architectures spanning 5 families. V-JEPA 2 is intentionally ABSENT here — it is a
foundation model that cannot be meaningfully trained from scratch on ~3k clips; it lives
only in fine_tune/.

    from models import build_model
    net = build_model("x3d", num_classes=2, num_frames=16, img_size=224, pretrained=False)
"""
from __future__ import annotations

from . import i3d, slowfast, tsm, x3d, videomae, timesformer, videomamba

REGISTRY = {
    "i3d":         i3d.build,          # 3D-CNN      (CVPR'17, A*)  — OmniFall anchor
    "slowfast":    slowfast.build,     # 3D-CNN 2way (ICCV'19, A*)
    "x3d":         x3d.build,          # 3D-CNN eff. (CVPR'20, A*)  — fits 6GB
    "tsm":         tsm.build,          # 2D+shift    (ICCV'19, A*)  — fits 6GB, hand-implemented
    "videomae":    videomae.build,     # Transformer (NeurIPS'22, A*) — OmniFall anchor
    "timesformer": timesformer.build,  # Transformer (ICML'21, A*)
    "videomamba":  videomamba.build,   # SSM/Mamba   (ECCV'24, A*)
}

# Models that comfortably fit the local RTX 3060 6GB for smoke tests.
FITS_6GB = {"x3d", "tsm", "videomamba"}


def build_model(name, num_classes=2, num_frames=16, img_size=224, pretrained=False, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. available: {list(REGISTRY)}")
    return REGISTRY[name](num_classes=num_classes, num_frames=num_frames,
                          img_size=img_size, pretrained=pretrained, **kw)
