"""Phase C — build LEAK-SAFE cross-subject (CS) and cross-view (CV) splits.

Recovers subject + camera from each source's filename convention, then splits SUBJECTS
(not clips) per dataset 70/15/15 so the same person/scene never appears in two splits.
This prevents the multiview leak described in PROJECT_PLAN.md.

Output: splits/cs/{train,val,test}.csv and splits/cv/{train,val,test}.csv
Each row: rel_path,label,dataset,subject,cam

Run (no torch needed):
    python common/build_splits.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]            # fall-benchmark/
DATA_ROOT = ROOT.parent / "MergedFallDataset"
OUT = ROOT / "splits"

VAL_FRAC, TEST_FRAC, SEED = 0.15, 0.15, 42


# ---------------------------------------------------- per-source subject/cam parsing
def parse_subject_cam(dataset: str, clip_id: str):
    """Return (subject_key, cam) parsed from the clip_id. cam=None if not applicable."""
    cid = clip_id
    if dataset == "ADL":
        m = re.search(r"_s(\d+)_e\d+", cid)
        return (f"ADL_s{m.group(1)}" if m else f"ADL_{cid}"), None
    if dataset == "CAUCAFall":
        m = re.search(r"Subject\.(\d+)", cid)
        return (f"CAUCA_s{m.group(1)}" if m else f"CAUCA_{cid}"), None
    if dataset == "MCFD":
        m = re.search(r"(chute\d+)/cam(\d+)", cid)
        return (f"MCFD_{m.group(1)}", f"cam{m.group(2)}") if m else (f"MCFD_{cid}", None)
    if dataset == "UR":
        m = re.search(r"(adl|fall)-(\d+)-cam(\d+)", cid)
        return (f"UR_{m.group(1)}{m.group(2)}", f"cam{m.group(3)}") if m else (f"UR_{cid}", None)
    if dataset == "extracted_falls":
        m = re.search(r"(Video_\d+)", cid)
        return (f"EF_{m.group(1)}" if m else f"EF_{cid}"), None
    if dataset == "fall_and_normal":
        m = re.search(r"S(\d+)C(\d+)P(\d+)R\d+A\d+", cid)   # NTU naming
        return (f"NTU_P{m.group(3)}", f"C{m.group(2)}") if m else (f"FN_{cid}", None)
    if dataset == "multiview":
        m = re.search(r"v(\d+)_s(\d+)_e\d+", cid)
        return (f"MV_s{m.group(2)}", f"v{m.group(1)}") if m else (f"MV_{cid}", None)
    return f"{dataset}_{cid}", None


def _split_subjects(subjects, seed):
    rng = np.random.default_rng(seed)
    subs = sorted(subjects)
    rng.shuffle(subs)
    n = len(subs)
    n_test = max(1, int(round(TEST_FRAC * n))) if n > 2 else 0
    n_val = max(1, int(round(VAL_FRAC * n))) if n > 3 else 0
    test = set(subs[:n_test])
    val = set(subs[n_test:n_test + n_val])
    train = set(subs[n_test + n_val:])
    return train, val, test


def build():
    df = pd.read_csv(DATA_ROOT / "manifest.csv")
    df = df[df["label"].isin(["FALL", "NoFALL"])].copy()
    # drop macOS resource-fork junk (clip_id basename starting with ._)
    df = df[~df["clip_id"].str.contains(r"/\._", regex=True)].copy()

    sc = df.apply(lambda r: parse_subject_cam(r["dataset"], r["clip_id"]), axis=1)
    df["subject"] = [s for s, _ in sc]
    df["cam"] = [c for _, c in sc]

    # ---- CS: split subjects per dataset ----
    cs = {"train": [], "val": [], "test": []}
    for ds, g in df.groupby("dataset"):
        tr, va, te = _split_subjects(g["subject"].unique(), SEED)
        for part, subs in [("train", tr), ("val", va), ("test", te)]:
            cs[part].append(g[g["subject"].isin(subs)])

    # ---- CV: split cameras per dataset (datasets w/o cam -> train only) ----
    cv = {"train": [], "val": [], "test": []}
    for ds, g in df.groupby("dataset"):
        cams = [c for c in g["cam"].dropna().unique()]
        if len(cams) >= 3:
            tr, va, te = _split_subjects(cams, SEED)   # reuse: split cam ids
            for part, cset in [("train", tr), ("val", va), ("test", te)]:
                cv[part].append(g[g["cam"].isin(cset)])
        else:                                          # no usable views -> train only
            cv["train"].append(g)

    OUT.mkdir(exist_ok=True)
    cols = ["rel_path", "label", "dataset", "subject", "cam"]
    for proto, dd in [("cs", cs), ("cv", cv)]:
        (OUT / proto).mkdir(parents=True, exist_ok=True)
        for part in ["train", "val", "test"]:
            out = pd.concat(dd[part]) if dd[part] else df.iloc[:0]
            out[cols].to_csv(OUT / proto / f"{part}.csv", index=False)

    _report(cs, cv)


def _report(cs, cv):
    def stats(frames):
        d = pd.concat(frames) if frames else pd.DataFrame(columns=["label", "subject"])
        n = len(d)
        f = int((d["label"] == "FALL").sum()) if n else 0
        nf = int((d["label"] == "NoFALL").sum()) if n else 0
        ns = d["subject"].nunique() if n else 0
        return n, f, nf, ns

    print(f"{'='*70}\nLEAK-SAFE SPLITS  (out: {OUT})\n{'='*70}")
    # CS must not share SUBJECTS across splits; CV must not share CAMERAS (same subject across
    # views is intentional for cross-view).
    for proto, dd, key in [("CS (cross-subject)", cs, "subject"), ("CV (cross-view)", cv, "cam")]:
        print(f"\n## {proto}  — leak check on '{key}'")
        print(f"{'split':<7}{'clips':>8}{'FALL':>8}{'NoFALL':>9}{'subjects':>10}")
        keysets = {}
        for part in ["train", "val", "test"]:
            n, f, nf, ns = stats(dd[part])
            print(f"{part:<7}{n:>8}{f:>8}{nf:>9}{ns:>10}")
            d = pd.concat(dd[part]) if dd[part] else pd.DataFrame(columns=[key])
            keysets[part] = set(d[key].dropna())
        leak = (keysets["train"] & keysets["test"]) | (keysets["train"] & keysets["val"])
        print(f"  {key} overlap train/val/test: {len(leak)}  "
              f"{'[OK] NO LEAK' if not leak else '[!] LEAK: '+str(list(leak)[:5])}")


if __name__ == "__main__":
    build()
