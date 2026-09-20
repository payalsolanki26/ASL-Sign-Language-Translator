"""
STAGE 1 - INGEST & VALIDATE  (Data Engineering)

Walks the raw dataset (one folder per class), validates every image, computes quality
signals and content hashes, and writes a *manifest* (one row per image). Everything
downstream reads the manifest, never the raw folder, so the pipeline is reproducible
and auditable.

Manifest columns
    rel_path      original relative path (kept only inside the trust boundary)
    label         class name (A..Z, space)
    label_id      integer id (sorted class order)
    capture_ts    epoch seconds parsed from the filename  (Image_<ts>.jpg)
    width/height/mode
    sha256        content hash (integrity + exact-duplicate detection)
    blur_var      variance of Laplacian  (sharpness)
    skeleton_px   number of red landmark pixels (sanity check that a hand was detected)
    brightness    mean grey level
    valid         bool - passed all hard checks
    issues        ';'-joined reasons when flagged
    take_id       continuous-recording id within a class
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

_TS_RE = re.compile(r"(\d{9,11}\.\d+)")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def parse_timestamp(name: str) -> float:
    m = _TS_RE.search(name)
    return float(m.group(1)) if m else float("nan")


def _analyse(path: Path, cfg_data: dict) -> dict:
    issues: list[str] = []
    row = dict(width=0, height=0, mode="", blur_var=np.nan, skeleton_px=0, brightness=np.nan)
    try:
        with Image.open(path) as im:
            im.verify()                       # cheap integrity check
        with Image.open(path) as im:          # verify() invalidates the handle -> reopen
            row["width"], row["height"], row["mode"] = im.width, im.height, im.mode
            rgb = np.asarray(im.convert("RGB"))
    except Exception as exc:                  # corrupt / truncated / not an image
        return {**row, "valid": False, "issues": f"unreadable:{type(exc).__name__}"}

    if (row["width"], row["height"]) != tuple(cfg_data["expected_size"]):
        issues.append("unexpected_size")
    if row["mode"] != cfg_data["expected_mode"]:
        issues.append("unexpected_mode")

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    row["blur_var"] = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    row["brightness"] = float(grey.mean())
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    row["skeleton_px"] = int(((r > 200) & (g < 80) & (b < 80)).sum())   # red landmark dots

    hard_fail = "unexpected_size" in issues or "unexpected_mode" in issues
    if row["skeleton_px"] < cfg_data["min_skeleton_pixels"]:
        issues.append("no_skeleton_overlay")
        hard_fail = True
    if row["blur_var"] < cfg_data["min_blur_var"]:
        issues.append("blurry")               # soft flag: kept, but reported
    return {**row, "valid": not hard_fail, "issues": ";".join(issues)}


def build_manifest(raw_dir: str | Path, cfg_data: dict) -> pd.DataFrame:
    raw = Path(raw_dir)
    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found in {raw}")
    label_id = {c: i for i, c in enumerate(classes)}

    rows = []
    for cls in classes:
        for p in sorted((raw / cls).iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            rows.append(dict(rel_path=f"{cls}/{p.name}", label=cls, label_id=label_id[cls],
                             capture_ts=parse_timestamp(p.name), sha256=sha256_file(p),
                             **_analyse(p, cfg_data)))
    df = pd.DataFrame(rows)

    # exact duplicates: keep first occurrence, flag the rest
    dup = df.duplicated("sha256", keep="first")
    df["issues"] = df["issues"].fillna("")
    for i in df.index[dup]:
        df.at[i, "issues"] = (df.at[i, "issues"] + ";exact_duplicate").strip(";")
    if cfg_data.get("drop_exact_duplicates", True):
        df.loc[dup, "valid"] = False

    # takes: a new take starts where the gap between consecutive frames exceeds the threshold
    df = df.sort_values(["label_id", "capture_ts"]).reset_index(drop=True)
    gap = df.groupby("label_id")["capture_ts"].diff()
    new_take = (gap.isna() | (gap > cfg_data["take_gap_seconds"])).astype(int)
    df["take_id"] = new_take.groupby(df["label_id"]).cumsum() - 1
    return df


def quality_report(df: pd.DataFrame) -> dict:
    v = df[df["valid"]]
    per_class = v.groupby("label").size()
    return {
        "n_files": int(len(df)),
        "n_valid": int(df["valid"].sum()),
        "n_invalid": int((~df["valid"]).sum()),
        "n_blurry_flagged": int(df["issues"].str.contains("blurry").sum()),
        "n_exact_duplicates": int(df["issues"].str.contains("exact_duplicate").sum()),
        "n_classes": int(df["label"].nunique()),
        "min_class_count": int(per_class.min()),
        "max_class_count": int(per_class.max()),
        "imbalance_ratio": round(float(per_class.max() / per_class.min()), 3),
        "median_frame_gap_s": round(float(df.groupby("label_id")["capture_ts"].diff().median()), 4),
        "n_takes_total": int(df.groupby("label_id")["take_id"].nunique().sum()),
        "mean_brightness": round(float(v["brightness"].mean()), 2),
        "median_blur_var": round(float(v["blur_var"].median()), 2),
    }
