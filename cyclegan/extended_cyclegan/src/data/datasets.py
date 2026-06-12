from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

# Đảm bảo đọc được các ảnh bị lỗi ghi đĩa hoặc thiếu dữ liệu nhẹ
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_jsonl_palettes(jsonl_path: str | Path) -> dict[str, torch.Tensor]:
    """Đọc file JSONL chứa thông tin palette, gộp colors_lab (18) và weights (6)

    thành một Tensor 24 chiều. Kèm key là tên file ảnh để tra cứu.
    """
    palettes = {}
    path = Path(jsonl_path)
    if not path.exists():
        print(f"[Warning] Không tìm thấy file palette: {path}")
        return palettes

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)

            # Lấy 6 màu LAB duỗi thẳng thành mảng 18 phần tử
            colors_flat = [val for color in data["colors_lab"] for val in color]
            # Lấy 6 giá trị trọng số (weights)
            weights = data["weights"]

            # Gộp lại thành vector 24 chiều và đưa về Float Tensor
            vector_24 = colors_flat + weights
            tensor_24 = torch.tensor(vector_24, dtype=torch.float32)

            # Lấy tên file ảnh làm khóa định danh (vd: 'vangogh_trainB_0001.jpg')
            img_filename = Path(data["source_image"]).name
            palettes[img_filename] = tensor_24

    return palettes


def list_images(
    root: str | Path, exts: Iterable[str] = DEFAULT_EXTS
) -> list[Path]:
    root = Path(root)
    exts = tuple(e.lower() for e in exts)
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {root}")
    files = [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
    ]
    files.sort()
    if not files:
        raise RuntimeError(f"Không tìm thấy ảnh hợp lệ trong: {root}")
    return files


def build_transform(
    image_size: int, crop_size: int, phase: str = "train"
) -> Callable:
    ops = [
        transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
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
    """Dataset CycleGAN kết hợp Palette Loss:

    - A: photo/content domain
    - B: art/style domain
    - Trả ra thêm thông tin palette gốc của A, B và một palette làm target ngẫu nhiên.
    """

    def __init__(
        self,
        root: str | Path,
        style: str,
        cfg_data: dict,  # Truyền dictionary 'data' từ file config (.yaml) vào đây
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

        # Tải dữ liệu bảng màu (Palettes) từ file JSONL vào RAM để truy xuất nhanh
        self.palettes_photo = load_jsonl_palettes(
            cfg_data.get("palette_photo", "")
        )
        self.palettes_art = load_jsonl_palettes(cfg_data.get("palette_art", ""))

        # Tạo danh sách các palette tranh nghệ thuật có sẵn để bốc random làm target
        self.art_palette_list = (
            list(self.palettes_art.values())
            if self.palettes_art
            else [torch.zeros(24)]
        )

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

        # --- TRUY XUẤT TENSOR PALETTE (24 CHIỀU) ---
        # Nếu ảnh không có palette tương ứng, gán mặc định tensor 0
        p_A = self.palettes_photo.get(path_A.name, torch.zeros(24))
        p_B = self.palettes_art.get(path_B.name, torch.zeros(24))

        # Bốc ngẫu nhiên một mẫu palette từ domain nghệ thuật làm mục tiêu định hướng style transfer
        p_target = random.choice(self.art_palette_list)

        return {
            "A": image_A,
            "B": image_B,
            "p_A": p_A,  # Mảng đặc trưng màu của ảnh thực tế A
            "p_B": p_B,  # Mảng đặc trưng màu của tranh nghệ thuật B
            "p_target": p_target,  # Palette mục tiêu bốc ngẫu nhiên từ kho nghệ thuật
            "A_path": str(path_A),
            "B_path": str(path_B),
        }


class SingleImageDataset(Dataset):
    """Dataset dùng cho infer một chiều (Không đổi so với phiên bản gốc)."""

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