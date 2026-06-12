from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(root: str | Path, exts: Iterable[str] = DEFAULT_EXTS) -> list[Path]:
    root = Path(root)
    exts = tuple(e.lower() for e in exts)
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    if not files:
        raise RuntimeError(f"Không tìm thấy ảnh hợp lệ trong: {root}")
    return files


def build_transform(image_size: int, crop_size: int, phase: str = "train") -> Callable:
    ops = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if phase == "train":
        ops += [
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        ops += [
            transforms.CenterCrop(crop_size),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(ops)


def safe_open_rgb(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Lỗi đọc ảnh: {path}") from exc


class UnpairedImageDataset(Dataset):
    """
    Dataset CycleGAN không paired:
    - A: photo/content domain
    - B: art/style domain
    Mỗi __getitem__ lấy A theo index và B random để đúng tinh thần unpaired.
    """

    def __init__(
        self,
        root: str | Path,
        style: str,
        phase: str = "train",
        dir_A: str = "trainA",
        dir_B: str = "trainB",
        image_size: int = 286,
        crop_size: int = 256,
        exts: Iterable[str] = DEFAULT_EXTS,
        serial_batches: bool = False,
    ):
        self.root = Path(root)
        self.style = style
        self.phase = phase
        self.dir_A_name = dir_A
        self.dir_B_name = dir_B
        self.serial_batches = serial_batches

        self.dir_A = self.root / style / dir_A
        self.dir_B = self.root / style / dir_B

        self.paths_A = list_images(self.dir_A, exts)
        self.paths_B = list_images(self.dir_B, exts)
        self.transform = build_transform(image_size, crop_size, phase=phase)

    def __len__(self) -> int:
        return max(len(self.paths_A), len(self.paths_B))

    def __getitem__(self, index: int) -> dict:
        path_A = self.paths_A[index % len(self.paths_A)]
        if self.serial_batches:
            path_B = self.paths_B[index % len(self.paths_B)]
        else:
            path_B = self.paths_B[random.randint(0, len(self.paths_B) - 1)]

        image_A = self.transform(safe_open_rgb(path_A))
        image_B = self.transform(safe_open_rgb(path_B))

        return {
            "A": image_A,
            "B": image_B,
            "A_path": str(path_A),
            "B_path": str(path_B),
        }


class SingleImageDataset(Dataset):
    """Dataset dùng cho infer một chiều."""

    def __init__(
        self,
        image_dir: str | Path,
        image_size: int = 286,
        crop_size: int = 256,
        exts: Iterable[str] = DEFAULT_EXTS,
    ):
        self.image_dir = Path(image_dir)
        self.paths = list_images(self.image_dir, exts)
        self.transform = build_transform(image_size, crop_size, phase="test")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        return {
            "image": self.transform(safe_open_rgb(path)),
            "path": str(path),
        }
