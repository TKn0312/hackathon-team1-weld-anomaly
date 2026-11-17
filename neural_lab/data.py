"""RIAWELC data for the lab: download + extract (if needed), and the team's split by weld.

The split is rebuilt with exactly the algorithm of the team notebook (SECTIONS 1, 2, 5a-5e: same file order,
duplicate removal, weld grouping, seed 42, 20,000 random assignments) and checked against the counts the
notebook printed. So 'test' tiles here are the same unseen welds the team evaluated on.

Command line (run from the repository root):
    python -m neural_lab.data                       # download/extract if missing, count, build + check split
    python -m neural_lab.data --rar-dir PATH        # reuse 19 RAR parts you already downloaded
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "01_autoencoder" / "data"                 # git-ignored (**/data/*)
RAR_DIR = DATA_DIR / "riawelc_rar"
EXTRACT_DIR = DATA_DIR / "riawelc_extracted"
RAW_DIR = EXTRACT_DIR / "DB - Copy"                               # same path the team notebook expects
CACHE_DIR = Path(__file__).resolve().parent / ".cache"            # git-ignored
SPLIT_CSV = CACHE_DIR / "split.csv"

N_PARTS = 19
URL = "https://github.com/stefyste/RIAWELC/raw/main/Dataset_partitioned/RIAWELC_dataset.part{:02d}.rar"
SEED = 42
LABELS = {"NoDifetto": "no_defect", "Difetto1": "crack", "Difetto2": "porosity", "Difetto4": "lack_of_penetration"}
ORDER = ["no_defect", "porosity", "crack", "lack_of_penetration"]
DEFECTS = ORDER[1:]
SPLITS = ["train", "val", "test"]

# What the original folders must contain, and what the team notebook printed after its split.
EXPECTED_RAW = {
    ("testing", "Difetto1"): 765, ("testing", "Difetto2"): 632, ("testing", "Difetto4"): 446, ("testing", "NoDifetto"): 600,
    ("training", "Difetto1"): 4962, ("training", "Difetto2"): 4108, ("training", "Difetto4"): 2893, ("training", "NoDifetto"): 3900,
    ("validation", "Difetto1"): 1908, ("validation", "Difetto2"): 1580, ("validation", "Difetto4"): 1113, ("validation", "NoDifetto"): 1500,
}
EXPECTED_SPLIT = {
    "train": {"no_defect": 3783, "porosity": 4207, "crack": 5175, "lack_of_penetration": 2921},
    "val": {"no_defect": 747, "porosity": 523, "crack": 1064, "lack_of_penetration": 454},
    "test": {"no_defect": 866, "porosity": 950, "crack": 577, "lack_of_penetration": 626},
}
EXPECTED_TEST_WELDS = ["RRT-23R", "RRT-28R", "RRT-31R", "RRT-40R"]
EXPECTED_VAL_WELDS = ["RRT-09R", "RRT-24R", "RRT-30R", "RRT-42R", "RRT-94R"]


# ---- 1. Download and extract ------------------------------------------------------------------------
def find_extractor():
    for c in [shutil.which("7z"), shutil.which("7za"), r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]:
        if c and Path(c).exists():
            return lambda rar, out: [c, "x", str(rar), f"-o{out}", "-y"]
    for c in [shutil.which("unrar"), r"C:\Program Files\WinRAR\UnRAR.exe"]:
        if c and Path(c).exists():
            return lambda rar, out: [c, "x", "-o+", str(rar), str(out) + "/"]
    if shutil.which("unar"):
        return lambda rar, out: ["unar", "-f", "-o", str(out), str(rar)]
    return None


def prepare(rar_source: Path | None = None):
    if RAW_DIR.is_dir():
        print(f"[data] already extracted: {RAW_DIR.relative_to(REPO_ROOT)}")
        return
    RAR_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, N_PARTS + 1):
        name = f"RIAWELC_dataset.part{i:02d}.rar"
        target = RAR_DIR / name
        if target.exists() and target.stat().st_size > 0:
            continue
        if rar_source and (rar_source / name).exists():
            shutil.copy2(rar_source / name, target)
            print(f"[data] copied {name}")
        else:
            print(f"[data] downloading {name}", flush=True)
            tmp = target.with_suffix(".download")
            urllib.request.urlretrieve(URL.format(i), tmp)
            tmp.replace(target)
    extractor = find_extractor()
    if extractor is None:
        sys.exit("[data] No RAR extractor found. Windows: winget install 7zip.7zip | macOS: brew install unar")
    print("[data] extracting all 19 parts (7-Zip reads them from part01) ...", flush=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(extractor(RAR_DIR / "RIAWELC_dataset.part01.rar", EXTRACT_DIR), check=True, stdout=subprocess.DEVNULL)
    if not RAW_DIR.is_dir():
        sys.exit(f"[data] extraction finished but {RAW_DIR} is missing")


def count_raw() -> dict:
    counts = {}
    for split_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            counts[(split_dir.name, cls_dir.name)] = len(list(cls_dir.glob("*.png")))
    return counts


# ---- 2. The team's split by weld (same steps and same order as the notebook) ----------------------------
def build_split() -> pd.DataFrame:
    rows = []
    for p in sorted(RAW_DIR.rglob("*.png")):
        img = np.asarray(Image.open(p))
        rows.append({"path": p, "orig_split": p.parent.parent.name, "label": LABELS[p.parent.name],
                     "source": p.stem.split("_[")[0], "pixel_hash": hashlib.md5(img.tobytes()).hexdigest()})
    df_raw = pd.DataFrame(rows)
    assert (df_raw.groupby("pixel_hash")["label"].nunique() == 1).all(), "duplicates with conflicting labels"
    df = df_raw[~df_raw.duplicated("pixel_hash", keep="first")].reset_index(drop=True)
    df["weld"] = df["source"].str.split("_").str[0]

    per_weld = pd.crosstab(df["weld"], df["label"])[ORDER]
    welds = per_weld.index.to_numpy()
    class_counts = per_weld[ORDER].to_numpy()
    class_totals = class_counts.sum(axis=0)
    target = np.array([0.70, 0.15, 0.15])

    def split_cost(assign):
        shares = np.stack([class_counts[assign == s].sum(axis=0) for s in range(3)]) / class_totals
        if (shares[1:] == 0).any():
            return np.inf
        return np.abs(shares - target[:, None]).sum()

    rng = np.random.default_rng(SEED)
    best_cost, best_assign = np.inf, None
    for _ in range(20_000):
        assign = rng.choice(3, size=len(welds), p=target)
        cost = split_cost(assign)
        if cost < best_cost:
            best_cost, best_assign = cost, assign
    weld_split = pd.Series(np.array(SPLITS)[best_assign], index=welds)
    df["split"] = df["weld"].map(weld_split)
    df["path"] = df["path"].map(lambda p: p.relative_to(RAW_DIR).as_posix())   # portable, no absolute paths
    return df[["path", "label", "weld", "source", "split"]]


def check_split(df: pd.DataFrame):
    got = pd.crosstab(df["split"], df["label"])
    for split, per_class in EXPECTED_SPLIT.items():
        for label, n in per_class.items():
            assert got.loc[split, label] == n, f"{split}/{label}: {got.loc[split, label]} != notebook {n}"
    assert sorted(df.loc[df["split"] == "test", "weld"].unique()) == EXPECTED_TEST_WELDS
    assert sorted(df.loc[df["split"] == "val", "weld"].unique()) == EXPECTED_VAL_WELDS


def load_split() -> pd.DataFrame:
    """The split table (built once, ~30 s, then read from neural_lab/.cache/split.csv)."""
    if SPLIT_CSV.exists():
        return pd.read_csv(SPLIT_CSV)
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Dataset not found at {RAW_DIR}. Run: python -m neural_lab.data")
    df = build_split()
    check_split(df)
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(SPLIT_CSV, index=False)
    return df


def tile_path(rel_path: str) -> Path:
    return RAW_DIR / rel_path


def evaluation_sets(df: pd.DataFrame):
    """train = sound tiles of training welds; val / test = ALL tiles of their welds (notebook SECTION 5e)."""
    train = df[(df["split"] == "train") & (df["label"] == "no_defect")].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rar-dir", type=Path, help="folder that already holds the 19 RAR parts")
    args = ap.parse_args()
    prepare(args.rar_dir)

    counts = count_raw()
    print("[data] images per original folder")
    for key in sorted(counts):
        status = "ok" if EXPECTED_RAW.get(key) == counts[key] else f"EXPECTED {EXPECTED_RAW.get(key)}"
        print(f"   {key[0]:<11}{key[1]:<10}{counts[key]:>6}  {status}")
    total = sum(counts.values())
    print(f"   TOTAL {total} ({'ok' if total == 24407 else 'EXPECTED 24407'})")
    assert counts == EXPECTED_RAW, "image counts differ from the official RIAWELC release"

    print("[data] building the team's split by weld (hashing 24,407 tiles, ~30 s) ...", flush=True)
    SPLIT_CSV.unlink(missing_ok=True)
    df = load_split()
    print(pd.crosstab(df["split"], df["label"])[ORDER].loc[SPLITS].to_string())
    print(f"[data] split identical to the notebook (21,893 unique tiles, test welds {', '.join(EXPECTED_TEST_WELDS)})")


if __name__ == "__main__":
    main()
