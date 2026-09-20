"""Shared plotting style so every figure in reports/ looks like one family."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "train": "#2F6F73",     # deep teal
    "val": "#E0A030",       # amber
    "test": "#C8553D",      # brick
    "embargo": "#B8B8B8",   # grey
    "ink": "#22262B",
    "muted": "#6B7280",
    "accent": "#2F6F73",
    "bad": "#C8553D",
    "good": "#2F6F73",
}


def apply():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": "#9AA0A6", "axes.labelcolor": PALETTE["ink"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"],
        "axes.grid": True, "grid.color": "#E6E8EB", "grid.linewidth": 0.8, "axes.axisbelow": True,
    })
    return plt
