from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""
    return (x.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def save_sample_grid(
    real_A: torch.Tensor,
    fake_B: torch.Tensor,
    rec_A: torch.Tensor,
    real_B: torch.Tensor,
    fake_A: torch.Tensor,
    rec_B: torch.Tensor,
    out_path: str | Path,
    nrow: int = 3,
) -> None:
    """
    Lưu grid 2 hàng:
    real_A -> fake_B -> rec_A
    real_B -> fake_A -> rec_B
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = [
        denorm(real_A[0]),
        denorm(fake_B[0]),
        denorm(rec_A[0]),
        denorm(real_B[0]),
        denorm(fake_A[0]),
        denorm(rec_B[0]),
    ]
    grid = make_grid(images, nrow=nrow)
    save_image(grid, out_path)
