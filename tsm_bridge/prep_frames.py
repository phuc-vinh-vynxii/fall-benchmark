"""Build the JPG frame tree that falling-net's TSNDataSet expects.

falling-net (= MIT TSM) never opens a video. It reads
    <root>/jpg/<LABEL>/<clip_key>/img_00001.jpg, img_00002.jpg, ...
so this converts MergedFallDataset into that layout. Two kinds of input:

  merged_media_type == 'video'   -> ffmpeg extracts frames
  merged_media_type == 'frames'  -> 437 clips ALREADY have frames; we only symlink them
                                    into img_%05d.jpg order (no re-encode, seconds not hours)

The existing frame folders use two different conventions and one of them numbers frames by
their index in the source video (frame_1894.jpg), so ordering is by the integer inside the
name -- a plain lexical sort would scramble the timeline.

    python tsm_bridge/prep_frames.py --job train --data-root ~/data/.../MergedFallDataset --out $W251_ROOT
    python tsm_bridge/prep_frames.py --job test  --test-root ~/data/dataset-test \
           --test-manifest ~/data/fall-testset-clean/test_clean_manifest.csv --out $W251_ROOT
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv")
CLASS_ID = {"NoFALL": 0, "FALL": 1}


def num_key(name: str):
    """Sort key: trailing integer in the filename, so frame_9 < frame_1894."""
    nums = re.findall(r"\d+", name)
    return (int(nums[-1]) if nums else 0, name)


def safe_key(rel_path: str) -> str:
    stem = Path(rel_path).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "clip"


def link_or_copy(src: Path, dst: Path):
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)          # Windows without developer mode


def build_one(task):
    """task = (kind, src, dst_dir, height, quality). Returns (dst_dir, n_frames, err)."""
    kind, src, dst_dir, height, quality = task
    src, dst_dir = Path(src), Path(dst_dir)
    done = dst_dir / ".done"
    if done.exists():
        return (str(dst_dir), int(done.read_text().strip() or 0), None)
    if dst_dir.exists():
        shutil.rmtree(dst_dir, ignore_errors=True)      # nua chung -> lam lai
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "frames":
            imgs = sorted((f for f in os.listdir(src)
                           if f.lower().endswith(IMG_EXT)), key=num_key)
            if not imgs:
                raise RuntimeError("thu muc frame rong")
            for i, f in enumerate(imgs, 1):
                link_or_copy(src / f, dst_dir / f"img_{i:05d}.jpg")
            n = len(imgs)
        else:
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(src),
                   "-threads", "1", "-vf", f"scale=-1:{height}", "-q:v", str(quality),
                   str(dst_dir / "img_%05d.jpg")]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            n = len([f for f in os.listdir(dst_dir) if f.startswith("img_")])
            if n == 0:
                raise RuntimeError("ffmpeg khong ra frame nao")
        done.write_text(str(n))
        return (str(dst_dir), n, None)
    except Exception as e:
        shutil.rmtree(dst_dir, ignore_errors=True)
        return (str(dst_dir), 0, f"{src.name}: {repr(e)[:160]}")


def rows_train(args):
    """Rows to build for the training corpus = everything in the CS train+val+test splits."""
    man = pd.read_csv(Path(args.data_root) / "manifest.csv")
    man = man[man["label"].isin(CLASS_ID)]
    media = dict(zip(man.rel_path, man.merged_media_type.astype(str)))

    frames = []
    for part in ("train", "val", "test"):
        f = Path(args.splits) / f"{part}.csv"
        if f.exists():
            d = pd.read_csv(f); d["part"] = part; frames.append(d)
    sp = pd.concat(frames, ignore_index=True)
    if args.limit:
        # lay dong deu cho TUNG split, khong thi train nuot het va val/test rong
        sp = sp.groupby(["part", "label"], group_keys=False).head(args.limit)
    sp = sp.drop_duplicates("rel_path")

    out = []
    for r in sp.itertuples():
        kind = "frames" if "frame" in media.get(r.rel_path, "video").lower() else "video"
        out.append((kind, Path(args.data_root) / r.rel_path, r.label, safe_key(r.rel_path)))
    return out


def rows_test(args):
    man = pd.read_csv(args.test_manifest)
    if args.limit:
        man = man.groupby("label", group_keys=False).head(args.limit)
    return [("video", Path(args.test_root) / r.kaggle_path, r.label,
             "test__" + safe_key(r.kaggle_path)) for r in man.itertuples()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", choices=["train", "test"], required=True)
    p.add_argument("--out", required=True, help="W251_ROOT, vd ~/data/w251fall")
    p.add_argument("--data-root", default="", help="job=train: thu muc chua manifest.csv")
    p.add_argument("--splits", default=str(REPO / "splits" / "cs"))
    p.add_argument("--test-root", default="", help="job=test: thu muc giai nen dataset-test")
    p.add_argument("--test-manifest", default="")
    p.add_argument("--limit", type=int, default=0, help="so clip MOI LOP (smoke test)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)))
    p.add_argument("--height", type=int, default=331, help="chieu cao khi resize (ffmpeg)")
    p.add_argument("--quality", type=int, default=2, help="ffmpeg -q:v, 2=tot, 0=to gap doi")
    args = p.parse_args()

    if args.job == "train" and not args.data_root:
        sys.exit("--job train can --data-root")
    if args.job == "test" and not (args.test_root and args.test_manifest):
        sys.exit("--job test can --test-root va --test-manifest")

    items = rows_train(args) if args.job == "train" else rows_test(args)

    # khoa trung ten
    seen, tasks, meta = {}, [], []
    jpg_root = Path(args.out) / "jpg"
    for kind, src, label, key in items:
        if key in seen:
            key = f"{key}_{seen[key]}"
        seen[key] = seen.get(key, 0) + 1
        dst = jpg_root / label / key
        tasks.append((kind, str(src), str(dst), args.height, args.quality))
        meta.append((label, key))

    missing = [t for t in tasks if not Path(t[1]).exists()]
    if missing:
        print(f"[!] {len(missing)} nguon khong ton tai, vd: {missing[0][1]}")
        keep = [i for i, t in enumerate(tasks) if Path(t[1]).exists()]
        tasks = [tasks[i] for i in keep]; meta = [meta[i] for i in keep]

    n_frames_src = sum(1 for t in tasks if t[0] == "frames")
    print(f"job={args.job}  clip={len(tasks)}  (video {len(tasks)-n_frames_src} | "
          f"frame co san {n_frames_src})  workers={args.workers}")
    print(f"out -> {jpg_root}")

    ok = err = 0
    errors = []
    with Pool(args.workers) as pool:
        for i, (dst, n, e) in enumerate(pool.imap_unordered(build_one, tasks, chunksize=4), 1):
            if e:
                err += 1; errors.append(e)
            else:
                ok += 1
            if i % 200 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  ok={ok} loi={err}", flush=True)

    print(f"\nXONG: {ok} clip co frame, {err} loi")
    for e in errors[:10]:
        print("  loi:", e)
    if err:
        print(f"  (con {max(0, err-10)} loi nua)" if err > 10 else "")


if __name__ == "__main__":
    main()
