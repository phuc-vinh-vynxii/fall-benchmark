"""Run the trained TSM over a list file and score it with the locked benchmark metrics.

Uses falling-net's own TSN model and TSNDataSet so the preprocessing matches training exactly,
but reports through common/metrics.py -- the same function every other model in this benchmark
is scored with, so the numbers stay comparable.

    python tsm_bridge/infer_testset.py --ckpt <ckpt.best.pth.tar> --root $W251_ROOT
    python tsm_bridge/infer_testset.py --ckpt ... --root ... --list test   # noi bo CS-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from common.metrics import compute_metrics, format_metrics, CLASS_NAMES  # noqa: E402


def build(args):
    sys.path.insert(0, str(Path(args.tsm_dir) / "train"))
    from ops.models import TSN
    from ops.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize

    net = TSN(len(CLASS_NAMES), args.num_segments, "RGB",
              base_model=args.arch, consensus_type="avg", dropout=0.5,
              img_feature_dim=256, partial_bn=False,
              is_shift=True, shift_div=args.shift_div, shift_place=args.shift_place)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[!] state_dict lech: thieu {len(missing)}, thua {len(unexpected)}")
        if len(missing) > 10:
            raise SystemExit("lech qua nhieu -- sai --arch hoac sai checkpoint?")
    print(f"checkpoint epoch={ck.get('epoch','?')} best_prec1={ck.get('best_prec1','?')}")

    tf = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.crop_size),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(net.input_mean, net.input_std),
    ])
    return net, tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="W251_ROOT")
    ap.add_argument("--tsm-dir", default=str(Path.home() / "falling-net"))
    ap.add_argument("--list", default="extest",
                    choices=["extest", "test", "val", "train"],
                    help="extest = tap test ngoai da loc; test = CS-test noi bo")
    ap.add_argument("--arch", default="mobilenetv2")
    ap.add_argument("--num-segments", type=int, default=8)
    ap.add_argument("--shift-div", type=int, default=8)
    ap.add_argument("--shift-place", default="blockres")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="chi chay N clip dau (smoke test)")
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args()

    root = Path(args.root)
    list_file = root / "file_list" / f"w251fall_rgb_{args.list}_split_1.txt"
    if not list_file.exists():
        raise SystemExit(f"khong thay {list_file} -- chay gen_lists.py truoc")

    net, tf = build(args)
    from ops.dataset import TSNDataSet

    if args.limit:
        trimmed = root / "file_list" / f"_tmp_{args.list}_limit.txt"
        lines = list_file.read_text(encoding="utf-8").splitlines(True)
        trimmed.write_text("".join(lines[:args.limit]), encoding="utf-8")
        list_file = trimmed

    ds = TSNDataSet(str(root / "jpg"), str(list_file), num_segments=args.num_segments,
                    new_length=1, modality="RGB", image_tmpl="img_{:05d}.jpg",
                    test_mode=True, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers, pin_memory=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev).eval()
    print(f"device={dev}  clip={len(ds)}  list={list_file.name}")

    probs, ys = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(dl, 1):
            logits = net(x.to(dev, non_blocking=True))
            probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
            ys.append(y.numpy())
            if i % 20 == 0 or i == len(dl):
                print(f"  {min(i*args.batch_size, len(ds))}/{len(ds)}", flush=True)

    probs = np.concatenate(probs); y_true = np.concatenate(ys)
    m = compute_metrics(y_true, probs)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    paths = [r.path for r in ds.video_list]
    pd.DataFrame({
        "clip": paths,
        "y_true": [CLASS_NAMES[int(v)] for v in y_true],
        "y_pred": [CLASS_NAMES[int(v)] for v in probs.argmax(1)],
        "prob_FALL": probs[:, 1],
    }).to_csv(out / f"predictions_{args.list}.csv", index=False)

    meta = {"list": args.list, "ckpt": str(args.ckpt), "n_clips": int(len(y_true)),
            "arch": args.arch, "num_segments": args.num_segments, **m}
    (out / f"metrics_{args.list}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + format_metrics(m))
    print(f"\n-> {out}/metrics_{args.list}.json")
    print(f"-> {out}/predictions_{args.list}.csv")


if __name__ == "__main__":
    main()
