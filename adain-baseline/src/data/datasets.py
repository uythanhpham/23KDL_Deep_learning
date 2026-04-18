"""Biến ảnh trong processed thành tensor."""
# E:\Nam3_ki2\TH DL\PROJECT\23KDL_Deep_learning\adain-baseline\src\data\datasets.py
import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader


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


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL RGB image -> torch.FloatTensor [3, H, W], range [0, 1]
    """
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def resize_pil(image: Image.Image, image_size: int) -> Image.Image:
    return image.resize((image_size, image_size), Image.BILINEAR)


def default_image_loader(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = resize_pil(img, image_size)
        return pil_to_tensor(img)


class AdaINDebugDataset(Dataset):
    """
    Giai đoạn 0:
    - Đọc debug_data/content và debug_data/style
    - Trả ra contract chung cho cả nhóm:
        {
            "content": Tensor [3, H, W],
            "style": Tensor [3, H, W],
            "content_path": str,
            "style_path": str,
        }

    Khi DataLoader batch lại:
    - content: [B, 3, H, W]
    - style:   [B, 3, H, W]
    - content_path: list[str]
    - style_path:   list[str]
    """

    def __init__(
        self,
        content_dir: str,
        style_dir: str,
        image_size: int = 256,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        seed: int = 42,
        pair_mode: str = "cycle",
    ) -> None:
        """
        pair_mode:
            - cycle: style được lấy theo index % len(style_files)
            - random: style được random theo seed
        """
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.image_size = image_size
        self.transform = transform
        self.seed = seed
        self.pair_mode = pair_mode

        self.content_files = scan_image_files(self.content_dir)
        self.style_files = scan_image_files(self.style_dir)

        if len(self.content_files) == 0:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh content trong thư mục: {self.content_dir}"
            )
        if len(self.style_files) == 0:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh style trong thư mục: {self.style_dir}"
            )

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.content_files)

    def _get_style_path(self, index: int) -> Path:
        if self.pair_mode == "cycle":
            return self.style_files[index % len(self.style_files)]
        if self.pair_mode == "random":
            return self._rng.choice(self.style_files)
        raise ValueError(f"Unsupported pair_mode: {self.pair_mode}")

    def _load_tensor(self, path: Path) -> torch.Tensor:
        if self.transform is not None:
            with Image.open(path) as img:
                img = img.convert("RGB")
                return self.transform(img)
        return default_image_loader(path, self.image_size)

    def __getitem__(self, index: int) -> Dict[str, object]:
        content_path = self.content_files[index]
        style_path = self._get_style_path(index)

        try:
            content_tensor = self._load_tensor(content_path)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise RuntimeError(f"Lỗi đọc content image: {content_path}") from e

        try:
            style_tensor = self._load_tensor(style_path)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise RuntimeError(f"Lỗi đọc style image: {style_path}") from e

        sample = {
            "content": content_tensor,          # [3, H, W]
            "style": style_tensor,              # [3, H, W]
            "content_path": str(content_path),
            "style_path": str(style_path),
        }
        return sample


def build_debug_dataset(
    root_dir: str = "debug_data",
    image_size: int = 256,
    seed: int = 42,
    pair_mode: str = "cycle",
) -> AdaINDebugDataset:
    root = Path(root_dir)
    content_dir = root / "content"
    style_dir = root / "style"

    return AdaINDebugDataset(
        content_dir=str(content_dir),
        style_dir=str(style_dir),
        image_size=image_size,
        transform=None,
        seed=seed,
        pair_mode=pair_mode,
    )


def load_manifest_if_exists(root_dir: str) -> Optional[Dict]:
    manifest_path = Path(root_dir) / "manifest.json"
    if manifest_path.exists():
        try:
            return load_json(manifest_path)
        except Exception:
            return None
    return None


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def summarize_batch(batch: Dict[str, object]) -> None:
    content = batch["content"]
    style = batch["style"]
    content_path = batch["content_path"]
    style_path = batch["style_path"]

    print("=" * 80)
    print("SMOKE TEST: AdaIN debug dataset contract")
    print(f"type(content)     : {type(content)}")
    print(f"type(style)       : {type(style)}")
    print(f"content.shape     : {tuple(content.shape)}")
    print(f"style.shape       : {tuple(style.shape)}")
    print(f"content.dtype     : {content.dtype}")
    print(f"style.dtype       : {style.dtype}")

    if isinstance(content_path, list) and len(content_path) > 0:
        print(f"content_path[0]   : {content_path[0]}")
    else:
        print(f"content_path      : {content_path}")

    if isinstance(style_path, list) and len(style_path) > 0:
        print(f"style_path[0]     : {style_path[0]}")
    else:
        print(f"style_path        : {style_path}")

    if torch.is_tensor(content):
        print(f"content min/max   : {content.min().item():.4f} / {content.max().item():.4f}")
    if torch.is_tensor(style):
        print(f"style min/max     : {style.min().item():.4f} / {style.max().item():.4f}")

    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Giai đoạn 0 - AdaIN debug dataset contract smoke test."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Tùy chọn. Chưa bắt buộc dùng ở GĐ0, giữ để tương thích CLI kế hoạch.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="GĐ0 chủ yếu dùng train/debug, nhưng giữ tham số để khớp CLI kế hoạch.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="debug_data",
        help="Thư mục chứa content/, style/, manifest.json",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Resize ảnh về kích thước vuông image_size x image_size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size dùng cho smoke test",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Num workers cho DataLoader",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed tái lập",
    )
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="cycle",
        choices=["cycle", "random"],
        help="Cách ghép style cho từng content",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="In batch đầu tiên để xác nhận contract",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manifest = load_manifest_if_exists(args.root_dir)
    if manifest is not None:
        print(f"Found manifest: {Path(args.root_dir) / 'manifest.json'}")
        print(
            f"Manifest summary -> "
            f"num_content={manifest.get('num_content')}, "
            f"num_style={manifest.get('num_style')}, "
            f"image_size={manifest.get('image_size')}"
        )

    dataset = build_debug_dataset(
        root_dir=args.root_dir,
        image_size=args.image_size,
        seed=args.seed,
        pair_mode=args.pair_mode,
    )

    print(f"Dataset length    : {len(dataset)}")
    print(f"Content dir       : {Path(args.root_dir) / 'content'}")
    print(f"Style dir         : {Path(args.root_dir) / 'style'}")
    print(f"Split             : {args.split}")

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True if args.split == "train" else False,
        num_workers=args.num_workers,
    )

    first_batch = next(iter(dataloader))

    if args.smoke_test:
        summarize_batch(first_batch)

    # Kiểm tra contract cứng cho GĐ0
    assert "content" in first_batch, "Thiếu key 'content'"
    assert "style" in first_batch, "Thiếu key 'style'"
    assert "content_path" in first_batch, "Thiếu key 'content_path'"
    assert "style_path" in first_batch, "Thiếu key 'style_path'"

    assert torch.is_tensor(first_batch["content"]), "'content' phải là tensor"
    assert torch.is_tensor(first_batch["style"]), "'style' phải là tensor"

    assert first_batch["content"].ndim == 4, "'content' batch phải có shape [B, C, H, W]"
    assert first_batch["style"].ndim == 4, "'style' batch phải có shape [B, C, H, W]"

    assert first_batch["content"].shape[1] == 3, "content phải có 3 channels"
    assert first_batch["style"].shape[1] == 3, "style phải có 3 channels"

    assert first_batch["content"].shape[2] == args.image_size, "content H không đúng"
    assert first_batch["content"].shape[3] == args.image_size, "content W không đúng"
    assert first_batch["style"].shape[2] == args.image_size, "style H không đúng"
    assert first_batch["style"].shape[3] == args.image_size, "style W không đúng"

    print("PASS: Dataset contract hợp lệ cho Giai đoạn 0.")
    print("Batch keys: content, style, content_path, style_path")
    print(f"Batch content shape: {tuple(first_batch['content'].shape)}")
    print(f"Batch style shape  : {tuple(first_batch['style'].shape)}")


if __name__ == "__main__":
    main()

# python -m src.data.datasets --split train --root_dir debug_data --image_size 256 --smoke_test