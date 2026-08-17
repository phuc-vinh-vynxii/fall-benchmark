"""Shared train/eval engine, reused by train_from_scratch/train.py and fine_tune/finetune.py.

The ONLY difference between the two folders is:
  - which model registry is passed in (build_model_fn)
  - the `pretrained` flag and default LR
Everything else (data, metrics, loop, checkpoint) is identical -> fair comparison.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import FallClipDataset
from .metrics import compute_metrics, format_metrics


def _loaders(cfg):
    full = FallClipDataset(
        data_root=cfg["data_root"], manifest_csv=cfg.get("manifest_csv"),
        split_csv=cfg.get("train_split_csv"),
        num_frames=cfg["num_frames"], img_size=cfg["img_size"],
        train=True, subset_per_class=cfg.get("subset_per_class"), seed=cfg.get("seed", 42),
    )
    print(f"[data] total clips: {len(full)} | label counts: {full.label_counts()}")

    if cfg.get("val_split_csv"):                       # explicit CS/CV val set
        train_ds = full
        val_ds = FallClipDataset(
            data_root=cfg["data_root"], manifest_csv=cfg.get("manifest_csv"),
            split_csv=cfg["val_split_csv"], num_frames=cfg["num_frames"],
            img_size=cfg["img_size"], train=False,
        )
    else:                                              # smoke: random 80/20
        n_val = max(1, int(0.2 * len(full)))
        g = torch.Generator().manual_seed(cfg.get("seed", 42))
        train_ds, val_ds = random_split(full, [len(full) - n_val, n_val], generator=g)
        val_ds.dataset.train = False  # type: ignore[attr-defined]

    nw = cfg.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=nw, pin_memory=True)
    return train_loader, val_loader, full


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_true = [], []
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        logits = model(clips)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_scores.append(probs)
        all_true.append(labels.numpy())
    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_true)
    return compute_metrics(y_true, y_score)


def run(cfg, build_model_fn, pretrained: bool):
    """cfg: dict; build_model_fn(name, num_classes, num_frames, img_size, pretrained)->nn.Module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 42))
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, full = _loaders(cfg)

    model = build_model_fn(
        name=cfg["model"], num_classes=2,
        num_frames=cfg["num_frames"], img_size=cfg["img_size"], pretrained=pretrained,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg['model']} | pretrained={pretrained} | trainable params: {n_params/1e6:.1f}M | device={device}")

    # weighted CE for 1:2.9 imbalance
    base = full.dataset if hasattr(full, "dataset") else full
    weights = base.class_weights().to(device) if hasattr(base, "class_weights") else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.05))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_bacc, best_metrics, history = -1.0, {}, []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, running = time.time(), 0.0
        for clips, labels in train_loader:
            clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * clips.size(0)
        train_loss = running / len(train_loader.dataset)

        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{k: v for k, v in metrics.items()
                        if isinstance(v, float)}})
        print(f"\n[epoch {epoch+1}/{cfg['epochs']}] loss={train_loss:.4f} "
              f"({time.time()-t0:.0f}s)\n{format_metrics(metrics)}")

        if metrics["balanced_accuracy"] > best_bacc:
            best_bacc, best_metrics = metrics["balanced_accuracy"], metrics
            torch.save({"model": model.state_dict(), "cfg": cfg, "metrics": metrics},
                       out_dir / f"{cfg['model']}_best.pt")
            (out_dir / f"{cfg['model']}_best_metrics.json").write_text(json.dumps(metrics, indent=2))

    (out_dir / f"{cfg['model']}_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[done] best balanced_accuracy = {best_bacc:.4f} | saved to {out_dir}")
    return best_bacc, best_metrics
