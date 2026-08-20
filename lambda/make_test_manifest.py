"""Build the LEAK-FREE test manifest for huuquoc0909/dataset-test.

314 videos from that dataset were merged into MergedFallDataset and are now in the CS
train/val split (see taken_from_dataset_test.csv), so scoring on them would be scoring on
training data. Instead of re-uploading 17 GB of video, we publish a ~1 MB Kaggle dataset
holding the exclusion list + the label manifest; inference reads the ORIGINAL dataset and
skips the excluded paths.

Only lists filenames over the Kaggle API - downloads no video.

    python lambda/make_test_manifest.py            # build files only
    python lambda/make_test_manifest.py --push     # + create/version the Kaggle dataset
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent          # fall-benchmark/lambda
REPO = HERE.parent                              # fall-benchmark
DATA = REPO.parent                              # d:/NuverxAI - RD/data

SRC_SLUG = "huuquoc0909/dataset-test"
OUT_SLUG = "phucvinhvynxii/fall-testset-clean"
OUT_DIR = HERE / "testset_clean"
TAKEN_CSV = DATA / "taken_from_dataset_test.csv"

FALL_FOLDER = "abnormal"                        # test_dataset/Abnormal/* -> FALL
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def list_files(api, slug: str) -> list[str]:
    """Page through the whole file list. Kaggle 429s under load -> back off and retry."""
    files, tok, page = [], None, 0
    while True:
        for attempt in range(6):
            try:
                res = (api.dataset_list_files(slug, page_token=tok, page_size=200) if tok
                       else api.dataset_list_files(slug, page_size=200))
                break
            except Exception as e:
                if "429" not in repr(e) or attempt == 5:
                    raise
                wait = 5 * (attempt + 1)
                print(f"  429 rate-limited, retry in {wait}s", flush=True)
                time.sleep(wait)
        files += [f.name for f in res.files]
        page += 1
        tok = getattr(res, "nextPageToken", None) or getattr(res, "next_page_token", None)
        print(f"  page {page}: {len(files)} files", flush=True)
        if not tok or page > 300:
            break
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="create/version the Kaggle dataset")
    args = ap.parse_args()

    # kaggle.json co the nam o data/, o thu muc cha, hoac cho mac dinh ~/.kaggle.
    # Chi ep KAGGLE_CONFIG_DIR khi thuc su tim thay file, nguoc lai de kaggle tu tim.
    for cand in (DATA, DATA.parent):
        if (cand / "kaggle.json").is_file():
            os.environ["KAGGLE_CONFIG_DIR"] = str(cand)
            break
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()

    taken = pd.read_csv(TAKEN_CSV)
    excluded = set(taken["kaggle_path"])
    print(f"excluded (already used for training): {len(excluded)}")

    print(f"listing {SRC_SLUG} ...")
    allf = list_files(api, SRC_SLUG)
    vids = [f for f in allf if f.lower().endswith(VIDEO_EXT)]
    print(f"videos in source dataset: {len(vids)}  (all files: {len(allf)})")

    missing = excluded - set(vids)
    if missing:
        print(f"[!] {len(missing)} excluded paths not found in the dataset listing, e.g. "
              f"{sorted(missing)[:3]}")

    rows = []
    for p in vids:
        if p in excluded:
            continue
        parts = p.split("/")
        folder = parts[-2] if len(parts) >= 2 else ""
        name = parts[-1]
        rows.append({
            "kaggle_path": p,
            "label": "FALL" if folder.lower() == FALL_FOLDER else "NoFALL",
            "activity": folder,
            "subject": name.split("_")[0] if "_" in name else "",
        })
    man = pd.DataFrame(rows).sort_values("kaggle_path").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT_DIR / "test_clean_manifest.csv", index=False)
    taken.to_csv(OUT_DIR / "excluded_314.csv", index=False)

    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "Fall test set (clean) - exclusion list",
        "id": OUT_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")

    (OUT_DIR / "README.md").write_text(
        "# Fall test set (clean)\n\n"
        f"Label manifest for `{SRC_SLUG}` with the {len(excluded)} videos that were merged into\n"
        "the training set (`phucvinhvynxii/fall-dataset`) removed, so the test set stays disjoint\n"
        "from training data.\n\n"
        "- `test_clean_manifest.csv` - `kaggle_path,label,activity,subject` for every video to score.\n"
        "- `excluded_314.csv` - the removed videos, for the record.\n\n"
        "Labels: `test_dataset/Abnormal/*` -> FALL, every other activity folder -> NoFALL.\n"
        "Add BOTH this dataset and the original as notebook inputs; read videos from the original.\n",
        encoding="utf-8")

    print(f"\nkept {len(man)} videos  "
          f"(FALL {int((man.label=='FALL').sum())} / NoFALL {int((man.label=='NoFALL').sum())})")
    print(man.activity.value_counts().to_string())
    print(f"\nwrote -> {OUT_DIR}")

    if args.push:
        exists = False
        try:
            api.dataset_list_files(OUT_SLUG)
            exists = True
        except Exception:
            pass
        cmd = (["kaggle", "datasets", "version", "-p", str(OUT_DIR), "-m",
                f"rebuild: {len(man)} clean test videos"] if exists else
               ["kaggle", "datasets", "create", "-p", str(OUT_DIR)])
        print("\n$", " ".join(cmd))
        sys.exit(subprocess.call(cmd, env=os.environ.copy()))   # da set o tren neu tim thay


if __name__ == "__main__":
    main()
