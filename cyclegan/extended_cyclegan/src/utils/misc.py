from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[Warning] Bạn chọn cuda nhưng máy không thấy CUDA. Chuyển sang CPU.")
        return torch.device("cpu")
    return torch.device(name)


def append_csv(path: str | Path, row: Dict[str, object], fieldnames: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def format_losses(losses: Dict[str, float]) -> str:
    return " | ".join(f"{k}={v:.4f}" for k, v in losses.items())
