"""Write the TSM label files from the leak-safe CS split.

Replaces falling-net's tools/gen_label_w251fall.py, which decides train-vs-val by looking for
the strings ".train"/".val" inside folder names. That would throw away the cross-subject split
in splits/cs/ -- the whole point of which is that the same person never appears in two splits.
Here train/val/test come straight from those CSVs instead.

Outputs under --root (= W251_ROOT):
    labels/categories.txt                          one class name per line -> n_class = 2
    file_list/w251fall_rgb_train_split_1.txt       "<LABEL>/<key> <n_frames> <class_id>"
    file_list/w251fall_rgb_val_split_1.txt
    file_list/w251fall_rgb_test_split_1.txt        internal CS test (from your own data)
    file_list/w251fall_rgb_extest_split_1.txt      external clean test set (Kaggle dataset-test)

Paths are relative to <root>/jpg, which is what TSNDataSet joins onto root_path.

    python tsm_bridge/gen_lists.py --root $W251_ROOT
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CLASS_NAMES = ["NoFALL", "FALL"]          # index = class_id, khop common/metrics.py
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def count_frames(d: Path) -> int:
    done = d / ".done"
    if done.exists():
        try:
            return int(done.read_text().strip())
        except ValueError:
            pass
    return len([f for f in os.listdir(d) if f.startswith("img_")]) if d.is_dir() else 0


def index_built(jpg_root: Path) -> dict[str, tuple[str, int]]:
    """key -> (LABEL, n_frames) cho moi clip da co frame."""
    out = {}
    for label_dir in sorted(p for p in jpg_root.iterdir() if p.is_dir()):
        for clip in sorted(p for p in label_dir.iterdir() if p.is_dir()):
            n = count_frames(clip)
            if n:
                out[clip.name] = (label_dir.name, n)
    return out


def safe_key(rel_path: str) -> str:
    import re
    stem = Path(rel_path).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "clip"


def write_list(path: Path, rows: list[tuple[str, int, int]], min_frames: int) -> tuple[int, dict]:
    """TSNDataSet._parse_list bo moi dong co num_frames < 3, nen loc truoc cho khoi lech so."""
    keep = [r for r in rows if r[1] >= min_frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p} {n} {c}\n" for p, n, c in keep), encoding="utf-8")
    dist = {CLASS_NAMES[i]: sum(1 for r in keep if r[2] == i) for i in range(len(CLASS_NAMES))}
    return len(keep), dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="W251_ROOT (chua jpg/)")
    ap.add_argument("--splits", default=str(REPO / "splits" / "cs"))
    ap.add_argument("--test-manifest", default="", help="test_clean_manifest.csv (tuy chon)")
    ap.add_argument("--min-frames", type=int, default=8,
                    help="bo clip qua ngan; nen >= num_segments cua TSM")
    args = ap.parse_args()

    root = Path(args.root)
    jpg_root = root / "jpg"
    if not jpg_root.is_dir():
        raise SystemExit(f"khong thay {jpg_root} -- chay prep_frames.py truoc")

    built = index_built(jpg_root)
    print(f"clip da co frame: {len(built)}")

    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "categories.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    fl = root / "file_list"
    total_missing = 0

    for part, out_name in [("train", "train"), ("val", "val"), ("test", "test")]:
        src = Path(args.splits) / f"{part}.csv"
        if not src.exists():
            print(f"[bo qua] khong co {src}")
            continue
        sp = pd.read_csv(src)
        rows, missing = [], 0
        for r in sp.itertuples():
            key = safe_key(r.rel_path)
            hit = built.get(key)
            if not hit:
                missing += 1
                continue
            label, n = hit
            rows.append((f"{label}/{key}", n, CLASS_ID[label]))
        n_kept, dist = write_list(fl / f"w251fall_rgb_{out_name}_split_1.txt", rows, args.min_frames)
        total_missing += missing
        print(f"  {out_name:6s}: {n_kept:5d} dong  {dist}"
              f"{f'   (thieu frame: {missing})' if missing else ''}")

    if args.test_manifest and Path(args.test_manifest).exists():
        man = pd.read_csv(args.test_manifest)
        rows, missing = [], 0
        for r in man.itertuples():
            key = "test__" + safe_key(r.kaggle_path)
            hit = built.get(key)
            if not hit:
                missing += 1
                continue
            label, n = hit
            rows.append((f"{label}/{key}", n, CLASS_ID[label]))
        n_kept, dist = write_list(fl / "w251fall_rgb_extest_split_1.txt", rows, args.min_frames)
        print(f"  extest: {n_kept:5d} dong  {dist}"
              f"{f'   (thieu frame: {missing})' if missing else ''}")

    print(f"\nghi vao {fl}")
    if total_missing:
        print(f"[!] {total_missing} clip trong split chua co frame -- chay lai prep_frames.py "
              f"neu day khong phai smoke test co --limit")


if __name__ == "__main__":
    main()
