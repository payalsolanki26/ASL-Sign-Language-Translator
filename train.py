"""
STAGE 7 - TRAINING  (Deep Learning)

    AdamW + OneCycle LR, label smoothing, class-weighted loss, mixed precision (CUDA),
    early stopping on validation *macro-F1*, best-checkpoint saving, then HMAC-signing.
Only the train/val rows of the split manifest are used; the test block is never touched here.
"""
from __future__ import annotations
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from asl.config import resolve
from asl.data.dataset import AslDataset, build_transforms, encrypted_loader, raw_loader
from asl.models.asl_net import build_model
from asl.security import integrity, privacy
from asl.security.audit import AuditLog


def seed_everything(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_loader_fn(cfg, p):
    if cfg["data"].get("source", "raw") == "encrypted":
        from asl.security.crypto import EncryptedImageStore
        store, salt = EncryptedImageStore(p["encrypted_store"]), os.environ[privacy.PSEUDO_ENV]
        return encrypted_loader(store, lambda rel: privacy.pseudonymize_id(rel, salt))
    return raw_loader(p["raw_dir"])


def evaluate_loader(model, loader, criterion, device):
    import torch
    model.eval(); losses, ys, ps = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            losses += criterion(out, y).item() * len(y)
            ys += y.cpu().tolist(); ps += out.argmax(1).cpu().tolist()
    ys, ps = np.array(ys), np.array(ps)
    return losses / len(ys), float((ys == ps).mean()), float(f1_score(ys, ps, average="macro", zero_division=0))


def run(cfg: dict):
    import torch
    from torch.utils.data import DataLoader

    p = {k: resolve(cfg, k) for k in cfg["paths"]}
    p["artifacts_dir"].mkdir(parents=True, exist_ok=True)
    log = AuditLog(p["audit_log"]); t = cfg["train"]
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(p["artifacts_dir"] / "manifest_split.csv")
    classes = df.sort_values("label_id").drop_duplicates("label_id")["label"].tolist()
    tr, va = df[df.split == "train"], df[df.split == "val"]
    loader_fn, sz = make_loader_fn(cfg, p), cfg["model"]["image_size"]
    mask = cfg["data"]["mask_magenta_box"]
    ds_tr = AslDataset(tr, loader_fn, build_transforms(sz, True, t["augment"]), mask)
    ds_va = AslDataset(va, loader_fn, build_transforms(sz, False, t["augment"]), mask)
    dl_tr = DataLoader(ds_tr, t["batch_size"], shuffle=True, num_workers=t["num_workers"], drop_last=True)
    dl_va = DataLoader(ds_va, t["batch_size"], shuffle=False, num_workers=t["num_workers"])

    model = build_model(cfg).to(device)
    w = None
    if t["class_weighted_loss"]:
        cnt = tr.groupby("label_id").size().reindex(range(len(classes))).to_numpy()
        w = torch.tensor(cnt.sum() / (len(cnt) * cnt), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=t["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], epochs=t["epochs"], steps_per_epoch=len(dl_tr))
    use_amp = bool(t["amp"] and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist, best, bad = [], -1.0, 0
    ckpt_path = p["artifacts_dir"] / "asl_model.pt"
    meta = {"classes": classes, "image_size": sz, "model": cfg["model"], "mask_magenta_box": mask}
    for epoch in range(1, t["epochs"] + 1):
        model.train(); t0 = time.time(); tl, tc, n = 0.0, 0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
        vl, va_acc, vf1 = evaluate_loader(model, dl_va, criterion, device)
        hist.append(dict(epoch=epoch, train_loss=tl / n, train_acc=tc / n, val_loss=vl, val_acc=va_acc,
                         val_macro_f1=vf1, seconds=round(time.time() - t0, 1)))
        print(f"epoch {epoch:02d}  loss {tl / n:.3f}  acc {tc / n:.3f} | val loss {vl:.3f}  acc {va_acc:.3f}  F1 {vf1:.3f}")
        if vf1 > best:
            best, bad = vf1, 0
            torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_path)
        else:
            bad += 1
            if bad >= t["early_stop_patience"]:
                print("early stopping"); break

    pd.DataFrame(hist).to_csv(p["artifacts_dir"] / "history.csv", index=False)
    signed = os.environ.get(integrity.SIGN_ENV) is not None
    if signed:
        integrity.sign_file(ckpt_path)
    log.log("train", best_val_macro_f1=round(best, 4), epochs_run=len(hist), device=device,
            model_sha256=integrity.sha256_file(ckpt_path), signed=signed)
    print(f"best val macro-F1 {best:.4f}  ->  {ckpt_path}  (signed={signed})")
