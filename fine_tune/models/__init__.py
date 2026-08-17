"""Model registry for the FINE-TUNE benchmark (Case 2): all pretrained weights fine-tuned on
MergedFallDataset.

The 7 architecture builders are reused verbatim from ../train_from_scratch/models (single source
of truth) with pretrained=True. V-JEPA 2 is added here only (fine-tune-only foundation model).

    from models import build_model
    net = build_model("videomae", num_classes=2, num_frames=16, pretrained=True)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the scratch models package under a UNIQUE name ("scratch_models") so the two same-named
# `models` packages never collide on sys.path / sys.modules.
_SCRATCH = Path(__file__).resolve().parents[2] / "train_from_scratch" / "models"
if "scratch_models" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "scratch_models", _SCRATCH / "__init__.py",
        submodule_search_locations=[str(_SCRATCH)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["scratch_models"] = _mod
    _spec.loader.exec_module(_mod)
_SCRATCH_REGISTRY = sys.modules["scratch_models"].REGISTRY   # 7 builders

from . import vjepa2                                          # noqa: E402  (fine-tune only)

# Pretrained sources per model:
#   i3d/slowfast/x3d  -> Kinetics-400 (pytorchvideo hub)
#   videomae          -> Kinetics-400 (MCG-NJU)
#   timesformer       -> Kinetics-400 (facebook)
#   tsm               -> ImageNet-2D backbone (standard TSM init; plug a TSM-K400 ckpt if available)
#   videomamba        -> official repo weights (Linux) else random-init lite
#   vjepa2            -> Meta self-supervised foundation weights (PREPRINT — reference baseline)
REGISTRY = dict(_SCRATCH_REGISTRY)
REGISTRY["vjepa2"] = vjepa2.build

FITS_6GB = {"x3d", "tsm"}            # everything else: use Kaggle/Lambda for fine-tune


def build_model(name, num_classes=2, num_frames=16, img_size=224, pretrained=True, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. available: {list(REGISTRY)}")
    return REGISTRY[name](num_classes=num_classes, num_frames=num_frames,
                          img_size=img_size, pretrained=pretrained, **kw)
