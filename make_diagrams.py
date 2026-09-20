#!/usr/bin/env python
"""Regenerates design/*.png  (pipeline map, model architecture, deployment / trust boundaries)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "design"
OUT.mkdir(exist_ok=True)
INK, MUTED = "#22262B", "#6B7280"
LANES = {  # name: (edge, fill)
    "Data Engineering": ("#2F6F73", "#E3F0F0"),
    "Security & Privacy": ("#C8553D", "#F8E6E1"),
    "Deep Learning": ("#A8741A", "#FBF0DA"),
    "Visualization": ("#5B5F97", "#E9EAF5"),
}


def box(ax, x, y, w, h, text, ec, fc, fs=8, bold=False, dashed=False, tc=INK, r=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                lw=1.3, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, c=MUTED, lw=1.2, style="-|>", ls="-", rad=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, lw=lw, color=c, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h, title, sub=None):
    fig, ax = plt.subplots(figsize=(w / 10, h / 10))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.text(1, h - 2.5, title, fontsize=15, fontweight="bold", color=INK, va="top")
    if sub:
        ax.text(1, h - 7.2, sub, fontsize=9.5, color=MUTED, va="top")
    return fig, ax


# ------------------------------------------------------------------ 1. pipeline map
def pipeline_map():
    W, H = 170, 104
    fig, ax = canvas(W, H, "ASL classification - pipeline map",
                     "Rows are the four disciplines; columns are the stages. Every stage has a job in more than one row.")
    cols = ["1  Acquire", "2  Validate", "3  Protect + clean", "4  Split", "5  Train", "6  Evaluate", "7  Serve"]
    x0, cw, gap, lx = 22, 19.2, 1.4, 1
    for i, c in enumerate(cols):
        ax.text(x0 + i * (cw + gap) + cw / 2, 90, c, ha="center", fontsize=9.5, fontweight="bold", color=INK)
    cells = {
        "Data Engineering": [
            "Webcam video, 30 fps\n27 class folders\n(A-Z + space)", "Ingest + validate\nsize, mode, blur,\nskeleton pixels\nSHA-256 manifest",
            "Dedup + take\ndetection (gap > 2 s)\nmask magenta box", "Temporal-block split\nblocks + 10-frame\nembargo, audited",
            "Dataset + augment\nresize 128, rotate,\nscale, colour jitter", "Fresh new-signer\nhold-out set\n(recommended)", "Live frame -> training\nformat (landmark\noverlay, canvas)"],
        "Security & Privacy": [
            "Consent + retention\nkeys in secrets\nmanager, not repo", "Verify hashes vs\nmanifest\n(tamper check)",
            "Strip EXIF, blur\nfaces, pseudonymise\nIDs, AES-256-GCM", "Public manifest:\nno filenames, no\nabsolute time",
            "Decrypt in memory\nonly; hash-chained\naudit log", "HMAC-sign model;\nload weights_only", "TLS, API key, rate\nlimit, no image\npersistence"],
        "Deep Learning": [
            None, None, "Class weights from\ntrain counts", "Baseline: PCA +\nk-NN sanity check",
            "AslNet residual CNN\nAdamW, OneCycle,\nlabel smoothing", "Macro-F1, confusion,\ncalibration (ECE),\nGrad-CAM", "Top-3 + 'not sure'\nconfidence threshold"],
        "Visualization": [
            "Sample grid", "Quality histograms\nbrightness, sharpness", "Capture timeline\n(leakage evidence)",
            "Split strips +\ncomposition", "Training curves", "Confusion, per-class F1\nreliability, Grad-CAM", "Live UI: letter,\nconfidence, privacy\nbadge"],
    }
    dashed_cells = {("Data Engineering", 5)}
    lane_h, top = 19.5, 86
    for li, (name, (ec, fc)) in enumerate(LANES.items()):
        y = top - (li + 1) * lane_h - li * 1.2
        ax.add_patch(Rectangle((lx, y), x0 - 3, lane_h, fc=ec, ec="none", alpha=0.95))
        ax.text(lx + (x0 - 3) / 2, y + lane_h / 2, name.replace(" & ", "\n& ").replace(" ", "\n", 1) if " " in name and "&" not in name else name.replace(" & ", "\n& "),
                ha="center", va="center", color="white", fontsize=9, fontweight="bold", linespacing=1.3)
        ax.add_patch(Rectangle((x0 - 1.2, y), W - x0 + 0.2, lane_h, fc=fc, ec="none", alpha=0.55))
        prev = None
        for ci, t in enumerate(cells[name]):
            if t is None:
                prev = None; continue
            bx = x0 + ci * (cw + gap)
            box(ax, bx, y + 2, cw, lane_h - 4, t, ec, "white", fs=7.6, dashed=(name, ci) in dashed_cells)
            if prev is not None:
                arrow(ax, prev + cw, y + lane_h / 2, bx, y + lane_h / 2, c=ec)
            prev = bx
    ax.text(W - 1, 2, "solid = code included in this project   dashed = recommended next step", ha="right", fontsize=8, color=MUTED)
    fig.savefig(OUT / "01_pipeline_map.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ 2. model architecture
def model_arch():
    W, H = 170, 108
    fig, ax = canvas(W, H, "AslNet - model architecture",
                     "Residual CNN with squeeze-excite blocks, about 2.8 M parameters, 128x128 input, 27 classes")
    ec, fc = LANES["Deep Learning"]
    tec, tfc = LANES["Data Engineering"]
    y, h, w = 60, 22, 17
    items = [
        ("Input\n3 x 128 x 128", "ImageNet norm.\nmagenta box masked", tec, tfc),
        ("Stem\nConv 3x3 /2\nBN, ReLU", "32 x 64 x 64", ec, "white"),
        ("Stage 1\nResBlock 32>64 /2\nResBlock 64", "64 x 32 x 32", ec, "white"),
        ("Stage 2\nResBlock 64>128 /2\nResBlock 128", "128 x 16 x 16", ec, "white"),
        ("Stage 3\nResBlock 128>256 /2\nResBlock 256", "256 x 8 x 8", ec, "white"),
        ("Global\navg pool", "256", ec, "white"),
        ("Dropout 0.3\nLinear 256>27", "27 logits", ec, "white"),
        ("Softmax", "class + confidence", ec, "white"),
    ]
    gap = 3.8
    for i, (t, s, e, f) in enumerate(items):
        x = 2 + i * (w + gap)
        box(ax, x, y, w, h, t, e, f, fs=8, bold=False)
        ax.text(x + w / 2, y - 2.6, s, ha="center", fontsize=7.6, color=MUTED, va="top", linespacing=1.3)
        if i:
            arrow(ax, x - gap + 0.3, y + h / 2, x - 0.3, y + h / 2, c=INK)
    # Grad-CAM tap
    sx = 2 + 4 * (w + gap)
    arrow(ax, sx + w / 2, y + h, sx + w / 2, y + h + 7, c=LANES["Visualization"][0], ls="--")
    box(ax, sx - 8, y + h + 7, w + 16, 6.5, "Grad-CAM taps the last feature map\n(does the heat sit on the hand?)", LANES["Visualization"][0], LANES["Visualization"][1], fs=7.6)

    # ResBlock detail
    ax.text(2, 47, "ResBlock detail", fontsize=11, fontweight="bold", color=INK)
    by, bh, bw = 30, 11, 15
    chain = ["Conv 3x3\n(stride s)", "BN + ReLU", "Conv 3x3", "BN", "Squeeze-Excite\npool > FC/4 > FC\n> sigmoid > scale"]
    xs = []
    for i, t in enumerate(chain):
        x = 2 + i * (bw + 4.5); xs.append(x)
        box(ax, x, by, bw, bh, t, ec, "white", fs=7.6)
        if i:
            arrow(ax, x - 4.2, by + bh / 2, x - 0.3, by + bh / 2, c=INK)
    addx = xs[-1] + bw + 6
    ax.add_patch(plt.Circle((addx + 3, by + bh / 2), 3.0, fc="white", ec=ec, lw=1.4))
    ax.text(addx + 3, by + bh / 2, "+", ha="center", va="center", fontsize=13, color=ec, fontweight="bold")
    arrow(ax, xs[-1] + bw + 0.3, by + bh / 2, addx - 0.1, by + bh / 2, c=INK)
    box(ax, addx + 9, by, 10, bh, "ReLU", ec, "white", fs=8)
    arrow(ax, addx + 6.2, by + bh / 2, addx + 8.7, by + bh / 2, c=INK)
    # skip path
    ax.plot([xs[0] - 1.5, xs[0] - 1.5], [by + bh / 2, by - 6.5], color=MUTED, lw=1.2)
    ax.plot([xs[0] - 1.5, addx + 3], [by - 6.5, by - 6.5], color=MUTED, lw=1.2)
    arrow(ax, addx + 3, by - 6.5, addx + 3, by + bh / 2 - 3.1, c=MUTED)
    ax.text((xs[0] + addx) / 2, by - 9.6, "skip path: identity, or 1x1 conv + BN when channels or stride change", ha="center", fontsize=8, color=MUTED)

    # training recipe
    ax.text(2, 15.5, "Training recipe", fontsize=11, fontweight="bold", color=INK)
    recipe = ["Loss: cross-entropy, label smoothing 0.1,\nclass-weighted (inverse frequency)",
              "Optimiser: AdamW, OneCycle LR (max 2e-3),\nweight decay 1e-4, mixed precision on CUDA",
              "Augment: rotate +-12 deg, scale 0.9-1.1,\nshift 6%, colour jitter, random erasing.\nNo horizontal flip (mirrored hand = other sign)",
              "Model selection: best validation macro-F1,\nearly stop after 8 epochs without gain.\nTest block touched once, at the end"]
    for i, t in enumerate(recipe):
        box(ax, 2 + i * 41, 1.5, 38.5, 11.5, t, ec, fc, fs=7.6)
    fig.savefig(OUT / "02_model_architecture.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ 3. deployment / trust boundaries
def deployment():
    W, H = 170, 100
    fig, ax = canvas(W, H, "Deployment design - where the data goes and what is protected",
                     "Two options: on-device (nothing leaves the phone or laptop) or a hardened server")
    sec_e, sec_f = LANES["Security & Privacy"]
    de_e, de_f = LANES["Data Engineering"]
    dl_e, dl_f = LANES["Deep Learning"]

    # option A
    ax.text(2, 84, "Option A - on-device (most private, recommended default)", fontsize=10.5, fontweight="bold", color=INK)
    ax.add_patch(FancyBboxPatch((2, 62), 166, 18, boxstyle="round,pad=0,rounding_size=1.2", fc="#F3F7F7", ec=de_e, lw=1.4, ls="--"))
    ax.text(4, 78, "Trust boundary: the user's device", fontsize=8, color=de_e, va="top")
    a = [("Camera frame", de_e, "white"), ("MediaPipe hand\nlandmarks + overlay\ncanvas", de_e, "white"), ("AslNet\n(ONNX / exported)", dl_e, "white"),
         ("Letter + confidence\nUI", LANES["Visualization"][0], "white"), ("Frame discarded\nnothing stored or sent", sec_e, sec_f)]
    for i, (t, e, f) in enumerate(a):
        x = 6 + i * 32.4
        box(ax, x, 65, 26, 10, t, e, f, fs=7.8)
        if i:
            arrow(ax, x - 6, 70, x - 0.3, 70, c=INK)

    # option B
    ax.text(2, 55, "Option B - server API (for shared kiosks, classrooms, web apps)", fontsize=10.5, fontweight="bold", color=INK)
    ax.add_patch(FancyBboxPatch((2, 1.5), 30, 49.5, boxstyle="round,pad=0,rounding_size=1.2", fc="#F7F7F7", ec=MUTED, lw=1.2, ls="--"))
    ax.text(4, 49.5, "User's browser", fontsize=8, color=MUTED, va="top")
    box(ax, 5, 33, 24, 10, "Webcam frame\n(JPEG/PNG, <= 2 MB)", MUTED, "white", fs=7.8)
    box(ax, 5, 12, 24, 10, "Result: top-3\nor 'not sure'", LANES["Visualization"][0], "white", fs=7.8)
    ax.add_patch(FancyBboxPatch((44, 1.5), 124, 49.5, boxstyle="round,pad=0,rounding_size=1.2", fc="#FDF6F4", ec=sec_e, lw=1.4, ls="--"))
    ax.text(46, 49.5, "Trust boundary: the server (TLS terminated at the reverse proxy)", fontsize=8, color=sec_e, va="top")
    steps = [("Gateway\nAPI key, rate limit,\nCORS allow-list", sec_e, sec_f), ("Validate\ntype, size cap,\npixel cap", sec_e, sec_f),
             ("Hand landmarks\n+ training-format\ncanvas", de_e, de_f), ("AslNet\nsignature checked\nat startup", dl_e, dl_f),
             ("Response\ntop-3 / uncertain", LANES["Visualization"][0], LANES["Visualization"][1])]
    for i, (t, e, f) in enumerate(steps):
        x = 47 + i * 24.2
        box(ax, x, 27, 21, 14, t, e, f, fs=7.4)
        if i:
            arrow(ax, x - 3.1, 34, x - 0.2, 34, c=INK)
    arrow(ax, 29.3, 38, 46.8, 34, c=INK)
    ax.text(38, 40.5, "HTTPS", fontsize=7.5, color=INK, ha="center")
    rx = 47 + 4 * 24.2 + 10.5
    ax.plot([rx, rx, 17], [27, 5.2, 5.2], color=INK, lw=1.2)
    arrow(ax, 17, 5.2, 17, 11.7, c=INK)
    ax.text(90, 3.2, "HTTPS response", fontsize=7.5, color=INK, ha="center")
    box(ax, 47, 9.5, 55, 11.5, "In memory only: the image is decoded, used, then dropped.\nNever written to disk, never logged.", sec_e, "white", fs=7.6)
    box(ax, 105, 9.5, 48, 11.5, "Logs keep: request id, latency, label,\nconfidence bucket, keyed-hash client id.\nNo images, no IPs, no keys.", sec_e, "white", fs=7.6)
    fig.savefig(OUT / "03_deployment_design.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    pipeline_map(); model_arch(); deployment(); print("diagrams ->", OUT)
