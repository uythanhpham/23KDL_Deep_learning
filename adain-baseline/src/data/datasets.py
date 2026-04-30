"Dataset/DataLoader cho AdaIN theo thiết lập paper."
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS

def scan_image_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if is_image_file(p)])

def get_transform(resize_size: int = 512, crop_size: int = 256, split: str = "train") -> transforms.Compose:
    crop = transforms.RandomCrop(crop_size) if split == "train" else transforms.CenterCrop(crop_size)
    return transforms.Compose([
        transforms.Resize(resize_size),
        crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class AdaINPairDataset(Dataset):
    def __init__(
        self,
        content_dir: str | Path,
        style_dir: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
        pair_mode: str = "random",
        seed: int = 42,
    ) -> None:
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.transform = transform
        self.pair_mode = pair_mode
        self.rng = random.Random(seed)
        self.content_files = scan_image_files(self.content_dir)
        self.style_files = scan_image_files(self.style_dir)
        if not self.content_files:
            raise FileNotFoundError(f"Không tìm thấy ảnh content trong: {self.content_dir}")
        if not self.style_files:
            raise FileNotFoundError(f"Không tìm thấy ảnh style trong: {self.style_dir}")

    def __len__(self) -> int:
        return len(self.content_files)

    def _pick_style(self, index: int) -> Path:
        if self.pair_mode == "cycle":
            return self.style_files[index % len(self.style_files)]
        if self.pair_mode == "random":
            return self.rng.choice(self.style_files)
        raise ValueError(f"Unsupported pair_mode: {self.pair_mode}")

    def _load_image(self, path: Path) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Không đọc được ảnh: {path}") from e

    def __getitem__(self, index: int) -> Dict[str, object]:
        content_path = self.content_files[index % len(self.content_files)]
        style_path = self._pick_style(index)
        with self._load_image(content_path) as img:
            content_tensor = self.transform(img)
        with self._load_image(style_path) as img:
            style_tensor = self.transform(img)
        return {
            "content": content_tensor,
            "style": style_tensor,
            "content_path": str(content_path),
            "style_path": str(style_path),
        }

def build_dataloaders(
    train_content_dir: str,
    train_style_dir: str,
    val_content_dir: str,
    val_style_dir: str,
    resize_size: int = 512,
    crop_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    pair_mode: str = "random",
) -> tuple[DataLoader, DataLoader]:
    train_set = AdaINPairDataset(
        train_content_dir,
        train_style_dir,
        get_transform(resize_size, crop_size, split="train"),
        pair_mode=pair_mode,
        seed=seed,
    )
    val_set = AdaINPairDataset(
        val_content_dir,
        val_style_dir,
        get_transform(resize_size, crop_size, split="val"),
        pair_mode="cycle",
        seed=seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader

def summarize_batch(batch: Dict[str, object]) -> None:
    content = batch["content"]; style = batch["style"]
    print("=" * 80)
    print(f"content shape : {tuple(content.shape)}")
    print(f"style shape   : {tuple(style.shape)}")
    print(f"content range : {content.min().item():.4f} / {content.max().item():.4f}")
    print(f"style range   : {style.min().item():.4f} / {style.max().item():.4f}")
    print("=" * 80)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--style_dir", type=str, required=True)
    parser.add_argument("--resize_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_mode", type=str, default="random", choices=["cycle", "random"])
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ds = AdaINPairDataset(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        transform=get_transform(args.resize_size, args.crop_size, split="train"),
        pair_mode=args.pair_mode,
        seed=args.seed,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Content images: {len(ds.content_files)}")
    print(f"Style images  : {len(ds.style_files)}")
    if args.smoke_test:
        summarize_batch(next(iter(dl)))
    print("PASS: Dataset/DataLoader OK.")

if __name__ == "__main__":
    main()
