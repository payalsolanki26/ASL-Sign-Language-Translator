"""Exploratory data analysis figures  (Data Visualization) - all read the manifest."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .style import PALETTE, apply

plt = apply()


def plot_class_distribution(df: pd.DataFrame, out: Path):
    c = df[df.valid].groupby("label").size().reindex(sorted(df.label.unique()))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    bars = ax.bar(c.index, c.values, color=PALETTE["accent"], width=0.72)
    ax.axhline(c.mean(), color=PALETTE["muted"], ls="--", lw=1)
    ax.text(len(c) - 0.5, c.mean() + 6, f"mean {c.mean():.0f}", ha="right", color=PALETTE["muted"], fontsize=9)
    hi, lo = c.idxmax(), c.idxmin()
    for b, k in zip(bars, c.index):
        if k in (hi, lo):
            b.set_color(PALETTE["test"])
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, int(c[k]), ha="center", fontsize=9)
    ax.set_title(f"Images per class  (max/min = {c.max() / c.min():.2f}x - mild imbalance)")
    ax.set_ylabel("valid images")
    ax.grid(axis="x", visible=False)
    fig.savefig(out); plt.close(fig)


def plot_sample_grid(raw_dir: Path, df: pd.DataFrame, out: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = sorted(df.label.unique())
    cols = 9
    rows = int(np.ceil(len(labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.45))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, lab in zip(axes.ravel(), labels):
        g = df[(df.label == lab) & df.valid]
        p = raw_dir / g.iloc[int(rng.integers(len(g)))].rel_path
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(lab, fontsize=10, pad=2, loc="center")
    fig.suptitle("One random frame per class (skeleton overlay is part of the pixels)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.savefig(out); plt.close(fig)


def plot_capture_timeline(df: pd.DataFrame, out: Path):
    """Each class = one horizontal strip; every dot is a frame. Shows continuous 30 fps takes."""
    labels = sorted(df.label.unique())
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, lab in enumerate(labels):
        g = df[df.label == lab].sort_values("capture_ts")
        t = g.capture_ts.to_numpy() - g.capture_ts.min()
        ax.scatter(t, np.full_like(t, i), s=2.5, color=PALETTE["accent"], alpha=0.8, linewidths=0)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds since first frame of that class")
    ax.set_title("Capture timeline - every letter is a short continuous recording\n"
                 "(neighbouring frames are ~33 ms apart, so a random split would leak)")
    fig.savefig(out); plt.close(fig)


def plot_quality(df: pd.DataFrame, out: Path):
    v = df[df.valid]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].hist(v.brightness, bins=40, color=PALETTE["accent"]); axes[0].set_title("Brightness (mean grey)")
    axes[1].hist(np.log10(v.blur_var.clip(lower=1)), bins=40, color=PALETTE["val"])
    axes[1].set_title("Sharpness  log10(Laplacian var)")
    axes[2].hist(v.skeleton_px, bins=40, color=PALETTE["test"]); axes[2].set_title("Landmark (red) pixels")
    for a in axes:
        a.set_ylabel("images")
    fig.savefig(out); plt.close(fig)


def plot_split_strips(df: pd.DataFrame, out: Path, classes=("A", "M", "S", "space"), title="Temporal-block split"):
    """Show which frames of a recording went to train / val / test / embargo."""
    fig, axes = plt.subplots(len(classes), 1, figsize=(11, 1.0 * len(classes) + 1.2), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, classes):
        g = df[df.label == lab].sort_values("frame_idx")
        cols = g.split.map(PALETTE).to_numpy()
        ax.scatter(g.frame_idx, np.zeros(len(g)), c=cols, marker="|", s=260, linewidths=1.2)
        ax.set_yticks([]); ax.set_ylabel(lab, rotation=0, labelpad=18, va="center", fontweight="bold")
        ax.grid(False); ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("frame order within the recording")
    fig.subplots_adjust(hspace=1.0)
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=PALETTE[k], label=k) for k in ("train", "val", "test", "embargo")]
    axes[0].legend(handles=handles, ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False)
    axes[0].set_title(title, pad=22)
    fig.savefig(out); plt.close(fig)


def plot_split_counts(df: pd.DataFrame, out: Path):
    t = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    order = [c for c in ("train", "val", "test", "embargo") if c in t.columns]
    t = t[order].reindex(sorted(t.index))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    bottom = np.zeros(len(t))
    for c in order:
        ax.bar(t.index, t[c], bottom=bottom, color=PALETTE[c], label=c, width=0.75)
        bottom += t[c].to_numpy()
    ax.legend(ncol=4, frameon=False, loc="upper right")
    ax.set_title("Split composition per class"); ax.set_ylabel("images"); ax.grid(axis="x", visible=False)
    fig.savefig(out); plt.close(fig)


def plot_leakage_demo(res: dict, out: Path):
    """res = {'random': acc, 'temporal_block': acc, ...}"""
    names = list(res)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cols = [PALETTE["bad"] if n == "random" else PALETTE["good"] for n in names]
    bars = ax.bar([n.replace("_", "\n") for n in names], [res[n] * 100 for n in names], color=cols, width=0.55)
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, res[n] * 100 + 1, f"{res[n] * 100:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 108); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Same model, same data - only the split changes")
    ax.grid(axis="x", visible=False)
    fig.savefig(out); plt.close(fig)
