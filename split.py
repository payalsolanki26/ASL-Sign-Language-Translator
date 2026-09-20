"""
STAGE 2 - LEAK-SAFE SPLITTING  (Data Engineering)

WHY THIS EXISTS
    The frames come from ~30 fps video: neighbouring frames are almost the same picture.
    A random 80/20 split puts a frame's twin in the training set for nearly every test
    frame, so the model is graded on images it has effectively memorised.

WHAT THIS DOES
    Each class recording (frames ordered by capture time) is cut into `n_blocks`
    contiguous blocks. Whole blocks go to val or test, the rest to train, and an
    *embargo* of `embargo_frames` train frames next to every held-out block is dropped
    so no train frame sits right beside a held-out frame.

HONEST LIMIT
    Blocks from the same take still share the signer, lighting and background. This split
    removes near-duplicate leakage; it does not measure generalisation to new people.
    Collect a fresh held-out set (new signers / rooms / cameras) for that.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def random_split(df: pd.DataFrame, seed: int = 42, val: float = 0.15, test: float = 0.15) -> pd.DataFrame:
    """Naive per-class random split. Kept ONLY to demonstrate leakage."""
    rng = np.random.default_rng(seed)
    out = df.sort_values(["label_id", "capture_ts"]).copy()
    out["frame_idx"] = out.groupby("label_id").cumcount()
    out["split"] = "train"
    for _, g in out.groupby("label_id"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val, n_test = int(len(idx) * val), int(len(idx) * test)
        out.loc[idx[:n_test], "split"] = "test"
        out.loc[idx[n_test:n_test + n_val], "split"] = "val"
    return out


def temporal_block_split(df: pd.DataFrame, n_blocks: int = 6, val_blocks=(2,), test_blocks=(4,),
                         embargo_frames: int = 10) -> pd.DataFrame:
    if set(val_blocks) & set(test_blocks):
        raise ValueError("val_blocks and test_blocks overlap")
    if max(list(val_blocks) + list(test_blocks)) >= n_blocks:
        raise ValueError("block index out of range")

    out = df.sort_values(["label_id", "capture_ts"]).copy()
    out["frame_idx"] = out.groupby("label_id").cumcount()
    out["split"] = "train"
    out["block"] = -1

    for _, g in out.groupby("label_id"):
        n = len(g)
        edges = np.linspace(0, n, n_blocks + 1).astype(int)
        blk = np.zeros(n, dtype=int)
        for b in range(n_blocks):
            blk[edges[b]:edges[b + 1]] = b
        split = np.array(["train"] * n, dtype=object)
        for b in val_blocks:
            split[blk == b] = "val"
        for b in test_blocks:
            split[blk == b] = "test"

        held = np.where(split != "train")[0]           # embargo: drop adjacent train frames
        drop = np.zeros(n, dtype=bool)
        for i in held:
            lo, hi = max(0, i - embargo_frames), min(n, i + embargo_frames + 1)
            drop[lo:hi] |= (split[lo:hi] == "train")
        split[drop] = "embargo"
        out.loc[g.index, "split"] = split
        out.loc[g.index, "block"] = blk
    return out


def audit_split(df: pd.DataFrame) -> dict:
    """Machine-checkable guarantees. Raises AssertionError if violated."""
    d = df[df["split"].isin(["train", "val", "test"])]
    assert not (set(d[d.split == "train"].sha256) & set(d[d.split != "train"].sha256)), "hash overlap"
    min_gap = None
    for _, g in d.groupby("label_id"):
        tr = g[g.split == "train"].frame_idx.to_numpy()
        ev = g[g.split != "train"].frame_idx.to_numpy()
        if len(tr) and len(ev):
            gap = int(np.abs(tr[:, None] - ev[None, :]).min())
            min_gap = gap if min_gap is None else min(min_gap, gap)
    return {"counts": {k: int(v) for k, v in d.groupby("split").size().items()},
            "n_embargoed": int((df["split"] == "embargo").sum()),
            "min_train_to_eval_gap_frames": min_gap}
