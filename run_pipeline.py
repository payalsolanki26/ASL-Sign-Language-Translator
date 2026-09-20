#!/usr/bin/env python
"""
ASL classification pipeline - one entry point, one command per stage.

    python scripts/run_pipeline.py keygen            # print fresh secrets (store them safely!)
    python scripts/run_pipeline.py ingest            # 1. validate + manifest
    python scripts/run_pipeline.py split             # 2. leak-safe split (+ pseudonymised public manifest)
    python scripts/run_pipeline.py privacy-scan      # 3. faces / EXIF audit of the raw images
    python scripts/run_pipeline.py eda               # 4. data figures
    python scripts/run_pipeline.py leakage-demo      # 5. random vs temporal split on a classical baseline
    python scripts/run_pipeline.py encrypt           # 6. AES-256-GCM encrypted copy of the dataset
    python scripts/run_pipeline.py train             # 7. deep learning  (needs PyTorch)
    python scripts/run_pipeline.py evaluate          # 8. test metrics + figures (needs PyTorch)
    python scripts/run_pipeline.py dashboard         # 9. one-file HTML report
    python scripts/run_pipeline.py all               # 1,2,3,4,5 (+ train/evaluate/dashboard if torch is present)

Secrets are read from the environment:  ASL_PSEUDO_SALT, ASL_DATA_KEY, ASL_SIGNING_KEY
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from asl.config import load_config, resolve
from asl.data import ingest as ing
from asl.data import split as spl
from asl.security import crypto, integrity
from asl.security.audit import AuditLog
from asl.security import privacy


def _paths(cfg):
    p = {k: resolve(cfg, k) for k in cfg["paths"]}
    for k in ("manifest_dir", "artifacts_dir", "reports_dir", "figures_dir"):
        p[k].mkdir(parents=True, exist_ok=True)
    return p


def cmd_keygen(cfg, a):
    print("# Store these in a secrets manager / .env (git-ignored). Losing ASL_DATA_KEY = losing the encrypted data.")
    print(f"export ASL_DATA_KEY={crypto.generate_key()}")
    print(f"export ASL_SIGNING_KEY={crypto.generate_key()}")
    print(f"export ASL_PSEUDO_SALT={crypto.generate_key()}")


def cmd_ingest(cfg, a):
    p = _paths(cfg); log = AuditLog(p["audit_log"])
    df = ing.build_manifest(p["raw_dir"], cfg["data"])
    df.to_csv(p["artifacts_dir"] / "manifest.csv", index=False)
    rep = ing.quality_report(df)
    (p["reports_dir"] / "data_quality.json").write_text(json.dumps(rep, indent=2))
    log.log("ingest", n_files=rep["n_files"], n_valid=rep["n_valid"])
    print(json.dumps(rep, indent=2))


def cmd_split(cfg, a):
    p = _paths(cfg); log = AuditLog(p["audit_log"])
    df = pd.read_csv(p["artifacts_dir"] / "manifest.csv")
    df = df[df.valid].copy()
    s = cfg["split"]
    df = (spl.temporal_block_split(df, s["n_blocks"], s["val_blocks"], s["test_blocks"], s["embargo_frames"])
          if s["strategy"] == "temporal_block" else spl.random_split(df, cfg["seed"]))
    audit = spl.audit_split(df)
    df.to_csv(p["artifacts_dir"] / "manifest_split.csv", index=False)
    (p["reports_dir"] / "split_audit.json").write_text(json.dumps(audit, indent=2))

    # shareable manifest: no filenames, no absolute time, keyed-hash ids
    salt = os.environ.get(privacy.PSEUDO_ENV)
    if cfg["privacy"]["pseudonymize_ids"] and salt:
        pub = df[["label", "label_id", "split", "frame_idx", "take_id"]].copy()
        pub.insert(0, "image_id", [privacy.pseudonymize_id(r, salt) for r in df.rel_path])
        pub.to_csv(p["manifest_dir"] / "manifest_public.csv", index=False)
    elif cfg["privacy"]["pseudonymize_ids"]:
        print(f"[warn] {privacy.PSEUDO_ENV} not set - skipped writing the pseudonymised public manifest")
    log.log("split", **audit)
    print(json.dumps(audit, indent=2))


def cmd_privacy_scan(cfg, a):
    p = _paths(cfg); log = AuditLog(p["audit_log"])
    df = pd.read_csv(p["artifacts_dir"] / "manifest.csv"); df = df[df.valid]
    pv = cfg["privacy"]; n_face, n_exif, flagged = 0, 0, []
    for r in df.itertuples():
        path = p["raw_dir"] / r.rel_path
        with Image.open(path) as im:
            if len(im.getexif()) > 0:
                n_exif += 1
            rgb = np.asarray(im.convert("RGB"))
        f = privacy.detect_faces(rgb, pv["face_scale_factor"], pv["face_min_neighbors"], pv["face_min_size"])
        if len(f):
            n_face += 1; flagged.append(r.rel_path)
    rep = {"images_scanned": int(len(df)), "images_with_exif": n_exif, "images_with_face_detected": n_face,
           "face_hit_rate_pct": round(100 * n_face / len(df), 3),
           "note": "Haar cascade misses partial/side faces: manually review flagged files and a random sample."}
    (p["reports_dir"] / "privacy_scan.json").write_text(json.dumps(rep, indent=2))
    pd.Series(flagged, name="rel_path").to_csv(p["reports_dir"] / "privacy_flagged_faces.csv", index=False)
    log.log("privacy_scan", **{k: v for k, v in rep.items() if k != "note"})
    print(json.dumps(rep, indent=2))


def cmd_eda(cfg, a):
    from asl.viz import eda
    p = _paths(cfg); f = p["figures_dir"]
    df = pd.read_csv(p["artifacts_dir"] / "manifest.csv")
    eda.plot_class_distribution(df, f / "01_class_distribution.png")
    eda.plot_sample_grid(p["raw_dir"], df, f / "02_sample_grid.png")
    eda.plot_capture_timeline(df, f / "03_capture_timeline.png")
    eda.plot_quality(df, f / "04_quality_distributions.png")
    sp = p["artifacts_dir"] / "manifest_split.csv"
    if sp.exists():
        d = pd.read_csv(sp)
        eda.plot_split_counts(d, f / "05_split_composition.png")
        eda.plot_split_strips(d, f / "06_split_strips.png")
    print("figures ->", f)


def _features(cfg, p, df, size=32):
    cache = p["artifacts_dir"] / f"feat{size}.npy"
    if cache.exists():
        X = np.load(cache)
        if len(X) == len(df):
            return X
    X = np.zeros((len(df), size * size * 3), dtype=np.float32)
    for i, rel in enumerate(df.rel_path):
        a = np.asarray(Image.open(p["raw_dir"] / rel).convert("RGB")).copy()
        if cfg["data"]["mask_magenta_box"]:
            m = (a[..., 0] > 200) & (a[..., 1] < 80) & (a[..., 2] > 200)
            a[m] = 255
        X[i] = np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
    np.save(cache, X)
    return X


def cmd_leakage_demo(cfg, a):
    """Classical baseline (PCA + k-NN / logistic regression) - NOT the deep model.
    Point: the same model gets very different scores depending only on how you split."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from asl.viz import eda

    p = _paths(cfg)
    df = pd.read_csv(p["artifacts_dir"] / "manifest.csv"); df = df[df.valid].reset_index(drop=True)
    X = _features(cfg, p, df)
    s = cfg["split"]
    variants = {"random": spl.random_split(df, cfg["seed"]),
                "temporal_block": spl.temporal_block_split(df, s["n_blocks"], s["val_blocks"], s["test_blocks"], s["embargo_frames"])}
    models = {"knn1": lambda: make_pipeline(StandardScaler(), PCA(80, random_state=0), KNeighborsClassifier(1)),
              "logreg": lambda: make_pipeline(StandardScaler(), PCA(80, random_state=0),
                                              LogisticRegression(max_iter=300, C=0.5))}
    out = {}
    for mname, mk in models.items():
        out[mname] = {}
        for vname, d in variants.items():
            tr, te = d[d.split == "train"], d[d.split == "test"]
            clf = mk().fit(X[tr.index], tr.label_id)
            out[mname][vname] = float((clf.predict(X[te.index]) == te.label_id).mean())
    (p["reports_dir"] / "leakage_demo.json").write_text(json.dumps(out, indent=2))
    eda.plot_leakage_demo(out["knn1"], p["figures_dir"] / "07_leakage_demo.png")
    print(json.dumps(out, indent=2))


def cmd_encrypt(cfg, a):
    p = _paths(cfg); log = AuditLog(p["audit_log"])
    df = pd.read_csv(p["artifacts_dir"] / "manifest.csv"); df = df[df.valid]
    store = crypto.EncryptedImageStore(p["encrypted_store"])
    salt = os.environ.get(privacy.PSEUDO_ENV)
    for r in df.itertuples():
        raw = (p["raw_dir"] / r.rel_path).read_bytes()
        clean = privacy.strip_metadata(raw) if cfg["privacy"]["strip_metadata"] else raw
        store.put(privacy.pseudonymize_id(r.rel_path, salt), clean)
    log.log("encrypt_dataset", n_files=int(len(df)))
    print(f"encrypted {len(df)} images -> {p['encrypted_store']}")


def cmd_train(cfg, a):
    from asl.training.train import run
    run(cfg)


def cmd_evaluate(cfg, a):
    from asl.training.evaluate import run
    run(cfg)


def cmd_dashboard(cfg, a):
    from asl.viz.dashboard import build
    print("dashboard ->", build(cfg))


def cmd_all(cfg, a):
    for c in (cmd_ingest, cmd_split, cmd_privacy_scan, cmd_eda, cmd_leakage_demo):
        c(cfg, a)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[info] PyTorch not installed - skipping train/evaluate. `pip install torch torchvision`, then run them.")
    else:
        cmd_train(cfg, a); cmd_evaluate(cfg, a)
    cmd_dashboard(cfg, a)


COMMANDS = {"keygen": cmd_keygen, "ingest": cmd_ingest, "split": cmd_split, "privacy-scan": cmd_privacy_scan,
            "eda": cmd_eda, "leakage-demo": cmd_leakage_demo, "encrypt": cmd_encrypt, "train": cmd_train,
            "evaluate": cmd_evaluate, "dashboard": cmd_dashboard, "all": cmd_all}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    COMMANDS[args.command](load_config(args.config), args)
