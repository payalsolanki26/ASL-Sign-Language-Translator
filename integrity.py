"""
Integrity & supply-chain protection  (Data Security)

  * SHA-256 manifest check: detects a modified / swapped / missing dataset file
  * HMAC-SHA256 signature over model files: refuse to load a checkpoint that was not
    produced by your own pipeline. (Pickle-based checkpoints can execute code on load,
    so we also load with torch.load(weights_only=True).)
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path

SIGN_ENV = "ASL_SIGNING_KEY"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_dataset(raw_dir: str | Path, manifest_rows: list[dict]) -> list[str]:
    """Return a list of problems (empty list == dataset matches the manifest)."""
    problems = []
    for r in manifest_rows:
        p = Path(raw_dir) / r["rel_path"]
        if not p.exists():
            problems.append(f"missing: {r['rel_path']}")
        elif sha256_file(p) != r["sha256"]:
            problems.append(f"modified: {r['rel_path']}")
    return problems


def _signing_key(key: str | None) -> bytes:
    k = key or os.environ.get(SIGN_ENV)
    if not k:
        raise RuntimeError(f"Set {SIGN_ENV} to sign / verify model files.")
    return k.encode()


def sign_file(path: str | Path, key: str | None = None) -> Path:
    sig = hmac.new(_signing_key(key), sha256_file(path).encode(), hashlib.sha256).hexdigest()
    sig_path = Path(str(path) + ".sig")
    sig_path.write_text(json.dumps({"file": Path(path).name, "sha256": sha256_file(path), "hmac": sig}))
    return sig_path


def verify_file(path: str | Path, key: str | None = None) -> bool:
    sig_path = Path(str(path) + ".sig")
    if not sig_path.exists():
        return False
    meta = json.loads(sig_path.read_text())
    digest = sha256_file(path)
    expected = hmac.new(_signing_key(key), digest.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, meta["hmac"]) and digest == meta["sha256"]
