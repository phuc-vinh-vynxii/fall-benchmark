"""Shared utilities for both train_from_scratch/ and fine_tune/ folders.

Single source of truth for:
  - dataset.py     : reads MergedFallDataset/manifest.csv (mp4 + frame dirs)
  - transforms.py  : temporal sampling + spatial aug, Kinetics normalization
  - metrics.py     : the LOCKED metric set (Phase B of PROJECT_PLAN.md)
  - engine.py      : train/eval loop reused by train.py and finetune.py
"""
