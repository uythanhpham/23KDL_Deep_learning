from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override vào base."""
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_config(config_path: str | Path, base_path: str | Path | None = None) -> Dict[str, Any]:
    """Load base.yaml rồi merge config riêng."""
    config_path = Path(config_path)
    if base_path is None:
        base_path = config_path.parent / "base.yaml"
    base_path = Path(base_path)

    if base_path.exists() and base_path.resolve() != config_path.resolve():
        cfg = deep_update(load_yaml(base_path), load_yaml(config_path))
    else:
        cfg = load_yaml(config_path)

    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
