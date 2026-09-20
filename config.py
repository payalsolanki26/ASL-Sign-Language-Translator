"""Configuration loading. Paths in the YAML are resolved relative to the project root."""
from __future__ import annotations
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve cfg['paths'][key] to an absolute path (relative to project root)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
