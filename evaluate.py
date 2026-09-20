"""
STAGE 8 - EVALUATION  (Deep Learning + Data Visualization)

Loads the *signed* checkpoint (weights_only=True), scores the untouched test block, and writes
metrics.json, per-class metrics, preds.csv and the result figures (confusion matrix, per-class
F1, reliability diagram, training curves, Grad-CAM samples).
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd

from asl.config import resolve
from asl.data.dataset import AslDataset, build_transforms
from asl.models.asl_net import build_model
from asl.security import integrity
from asl.security.audit import AuditLog
from asl.training.train import make_loader_fn
from asl.viz import explain, results


def load_verified(ckpt_path, device):
    import torch
    if os.environ.get(integrity.SIGN_ENV):
        if not integrity.verify_file(ckpt_path):
            raise RuntimeError("Model signature check FAILED - refusing to load this checkpoint.")
    else:
        print("[warn] ASL_SIGNING_KEY not set: checkpoint signature NOT verified")
    return torch.load(ckpt_path, map_location=device, weights_only=True)


def run(cfg: dict):
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    p = {k: resolve(cfg, k) for k in cfg["paths"]}
    p["figures_dir"].mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_verified(p["artifacts_dir"] / "asl_model.pt", device)
    classes, sz = ck["meta"]["classes"], ck["meta"]["image_size"]
    model = build_model(cfg).to(device); model.load_state_dict(ck["state_dict"]); model.eval()

    df = pd.read_csv(p["artifacts_dir"] / "manifest_split.csv")
    te = df[df.split == "test"].reset_index(drop=True)
    ds = AslDataset(te, make_loader_fn(cfg, p), build_transforms(sz, False, cfg["train"]["augment"]), ck["meta"]["mask_magenta_box"])
    dl = DataLoader(ds, 64, shuffle=False, num_workers=cfg["train"]["num_workers"])

    ys, ps, conf = [], [], []
    with torch.no_grad():
        for x, y in dl:
            pr = torch.softmax(model(x.to(device)), 1)
            c, k = pr.max(1)
            ys += y.tolist(); ps += k.cpu().tolist(); conf += c.cpu().tolist()
    ys, ps, conf = np.array(ys), np.array(ps), np.array(conf)

    f = p["figures_dir"]
    summary = results.summarize(ys, ps, conf, classes)
    cm = results.plot_confusion(ys, ps, classes, f / "08_confusion_matrix.png")
    pc = results.plot_per_class_f1(ys, ps, classes, f / "09_per_class_f1.png")
    results.plot_reliability(conf, (ys == ps).astype(float), f / "10_reliability.png")
    hist = p["artifacts_dir"] / "history.csv"
    if hist.exists():
        results.plot_training_curves(pd.read_csv(hist), f / "11_training_curves.png")
    pc.to_csv(p["reports_dir"] / "per_class_metrics.csv", index=False)
    results.top_confusions(cm, classes).to_csv(p["reports_dir"] / "top_confusions.csv", index=False)
    pd.DataFrame({"y_true": ys, "y_pred": ps, "conf": conf}).to_csv(p["artifacts_dir"] / "preds.csv", index=False)
    (p["reports_dir"] / "metrics.json").write_text(json.dumps(summary, indent=2))

    # Grad-CAM: 6 correct + 6 wrong (or fewer) test samples
    wrong, right = np.where(ys != ps)[0][:6], np.where(ys == ps)[0][:6]
    picks = list(right) + list(wrong)
    if picks:
        fig, axes = plt.subplots(2, 6, figsize=(13, 5.2)); axes = axes.ravel()
        for ax in axes:
            ax.axis("off")
        for ax, i in zip(axes, picks):
            x, _ = ds[i]
            cam, pred = explain.grad_cam(model, x.unsqueeze(0).to(device), model.cam_layer())
            img = (x * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None])
            img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            ax.imshow(explain.overlay(img, cam))
            ax.set_title(f"true {classes[ys[i]]} / pred {classes[pred]}", fontsize=9,
                         color="#C8553D" if pred != ys[i] else "#22262B")
        fig.suptitle("Grad-CAM - top row correct, bottom row errors. Heat should sit on the hand/skeleton.", x=0.01, ha="left", fontweight="bold")
        fig.savefig(f / "12_gradcam.png"); plt.close(fig)

    AuditLog(p["audit_log"]).log("evaluate", **summary)
    print(json.dumps(summary, indent=2))
