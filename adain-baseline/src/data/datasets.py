"""Biến ảnh trong processed thành tensor."""
import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # Đã thêm thư viện này để chuẩn hóa

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS

def scan_image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if is_image_file(p)])

# Hàm transform chuẩn cho VGG19
def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), # Tự động đưa về [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]) # Chuẩn hóa theo ImageNet
    ])

class AdaINDebugDataset(Dataset):
    def __init__(
        self,
        content_dir: str,
        style_dir: str,
        transform: Callable[[Image.Image], torch.Tensor],
        seed: int = 42,
        pair_mode: str = "cycle",
    ) -> None:
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.transform = transform
        self.seed = seed
        self.pair_mode = pair_mode

        self.content_files = scan_image_files(self.content_dir)
        self.style_files = scan_image_files(self.style_dir)

        if len(self.content_files) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh content: {self.content_dir}")
        if len(self.style_files) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh style: {self.style_dir}")

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.content_files)

    def _get_style_path(self, index: int) -> Path:
        if self.pair_mode == "cycle":
            return self.style_files[index % len(self.style_files)]
        if self.pair_mode == "random":
            return self._rng.choice(self.style_files)
        raise ValueError(f"Unsupported pair_mode: {self.pair_mode}")

    def __getitem__(self, index: int) -> Dict[str, object]:
        content_path = self.content_files[index]
        style_path = self._get_style_path(index)

        # Đọc và apply transform (đã có Resize + ToTensor + Normalize)
        with Image.open(content_path).convert("RGB") as img:
            content_tensor = self.transform(img)
            
        with Image.open(style_path).convert("RGB") as img:
            style_tensor = self.transform(img)

        return {
            "content": content_tensor,
            "style": style_tensor,
            "content_path": str(content_path),
            "style_path": str(style_path),
        }

def build_debug_dataset(
    root_dir: str = "debug_data",
    image_size: int = 256,
    seed: int = 42,
    pair_mode: str = "cycle",
) -> AdaINDebugDataset:
    root = Path(root_dir)
    return AdaINDebugDataset(
        content_dir=str(root / "content"),
        style_dir=str(root / "style"),
        transform=get_transform(image_size), # Dùng transform chuẩn ở đây
        seed=seed,
        pair_mode=pair_mode,
    )

def build_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def summarize_batch(batch: Dict[str, object]) -> None:
    # (Giữ nguyên logic in thông tin như cũ của bạn)
    content = batch["content"]
    print("=" * 80)
    print(f"Batch content shape: {tuple(content.shape)}")
    print(f"Batch content min/max: {content.min().item():.4f} / {content.max().item():.4f}")
    print("=" * 80)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="debug_data")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_mode", type=str, default="cycle", choices=["cycle", "random"])
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = build_debug_dataset(args.root_dir, args.image_size, args.seed, args.pair_mode)
    dataloader = build_dataloader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)

    first_batch = next(iter(dataloader))
    if args.smoke_test:
        summarize_batch(first_batch)
    
    print("PASS: Dataset đã được Normalize chuẩn để tránh ám màu.")

if __name__ == "__main__":
    main()
# python -m src.data.datasets --split train --root_dir debug_data --image_size 256 --smoke_test
