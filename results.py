"""Model-result figures  (Data Visualization) - read history.csv / preds.csv, no torch needed."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .style import PALETTE, apply

plt = apply()


def plot_training_curves(hist: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(hist.epoch, hist.train_loss, color=PALETTE["accent"], label="train")
    axes[0].plot(hist.epoch, hist.val_loss, color=PALETTE["test"], label="val"); axes[0].set_title("Loss")
    axes[1].plot(hist.epoch, hist.train_acc, color=PALETTE["accent"], label="train")
    axes[1].plot(hist.epoch, hist.val_acc, color=PALETTE["test"], label="val"); axes[1].set_title("Accuracy")
    axes[2].plot(hist.epoch, hist.val_macro_f1, color=PALETTE["val"]); axes[2].set_title("Validation macro-F1")
    best = hist.val_macro_f1.idxmax()
    axes[2].scatter([hist.epoch[best]], [hist.val_macro_f1[best]], color=PALETTE["ink"], zorder=3)
    for a in axes:
        a.set_xlabel("epoch")
    axes[0].legend(frameon=False); fig.savefig(out); plt.close(fig)


def plot_confusion(y_true, y_pred, classes, out: Path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9.2, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cmn[i, j] >= 0.08:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7, color=PALETTE["bad"])
    ax.set_title("Confusion matrix (row-normalised; off-diagonal >= 8% labelled)")
    fig.colorbar(im, ax=ax, fraction=0.04); fig.savefig(out); plt.close(fig)
    return cm


def top_confusions(cm: np.ndarray, classes, k: int = 10) -> pd.DataFrame:
    rows = [(classes[i], classes[j], int(cm[i, j]), cm[i, j] / max(cm[i].sum(), 1))
            for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    return pd.DataFrame(rows, columns=["true", "predicted", "count", "rate"]).sort_values("count", ascending=False).head(k)


def plot_per_class_f1(y_true, y_pred, classes, out: Path):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    order = np.argsort(f)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([classes[i] for i in order], f[order], color=[PALETTE["bad"] if v < 0.9 else PALETTE["accent"] for v in f[order]])
    ax.axvline(0.9, color=PALETTE["muted"], ls="--", lw=1)
    ax.set_xlim(0, 1.02); ax.set_xlabel("F1"); ax.set_title("Per-class F1 (red = below 0.90)")
    ax.grid(axis="y", visible=False); fig.savefig(out); plt.close(fig)
    return pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f})


def expected_calibration_error(conf, correct, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def plot_reliability(conf, correct, out: Path, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    ece = expected_calibration_error(conf, correct, bins)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], ls="--", lw=1)
    ax.plot(xs, ys, marker="o", color=PALETTE["accent"])
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title(f"Reliability  (ECE {ece:.3f})")
    fig.savefig(out); plt.close(fig)
    return ece


def summarize(y_true, y_pred, conf, classes) -> dict:
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    return {"accuracy": float(correct.mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "ece": expected_calibration_error(np.asarray(conf), correct),
            "n": int(len(y_true))}
