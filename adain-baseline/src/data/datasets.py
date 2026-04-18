"""Biến ảnh trong debug_data / data đã chuẩn hóa thành tensor.

Giai đoạn 4 - Người 1:
- Giữ lại phần smoke test ổn định của GĐ3
- Bổ sung real-mode
- Bổ sung augmentation train cơ bản:
    + random horizontal flip
    + random crop nhẹ (thực hiện bằng resize lớn hơn rồi crop)
- Dataset hỗ trợ cả debug-mode và real-mode
"""

# E:\Nam3_ki2\TH DL\PROJECT\23KDL_Deep_learning\adain-baseline\src\data\datasets.py

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import DataLoader, Dataset


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


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
    return image.resize((image_size, image_size), RESAMPLE_BILINEAR)


def center_crop_pil(image: Image.Image, crop_size: int) -> Image.Image:
    w, h = image.size
    left = max(0, (w - crop_size) // 2)
    top = max(0, (h - crop_size) // 2)
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def random_crop_pil(image: Image.Image, crop_size: int) -> Image.Image:
    w, h = image.size
    if w == crop_size and h == crop_size:
        return image

    max_left = max(0, w - crop_size)
    max_top = max(0, h - crop_size)

    left = random.randint(0, max_left) if max_left > 0 else 0
    top = random.randint(0, max_top) if max_top > 0 else 0
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def default_image_loader(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = resize_pil(img, image_size)
        return pil_to_tensor(img)


def validate_image_file(path: Path) -> Tuple[bool, str]:
    """
    Kiểm tra nhanh file ảnh có mở/verify được không.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True, ""
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, str(e)


def filter_valid_images(paths: List[Path]) -> Tuple[List[Path], List[Dict[str, str]]]:
    valid_paths: List[Path] = []
    invalid_records: List[Dict[str, str]] = []

    for path in paths:
        ok, error_msg = validate_image_file(path)
        if ok:
            valid_paths.append(path)
        else:
            invalid_records.append(
                {
                    "path": str(path),
                    "error": error_msg,
                }
            )
    return valid_paths, invalid_records


def safe_collate_fn(batch: List[Optional[Dict[str, object]]]) -> Dict[str, object]:
    """
    Loại bỏ sample None để DataLoader không văng nếu 1 vài ảnh lỗi lúc runtime.
    """
    valid_samples = [sample for sample in batch if sample is not None]

    if len(valid_samples) == 0:
        raise RuntimeError(
            "Toàn bộ sample trong batch đều lỗi/None. "
            "Hãy kiểm tra lại dữ liệu hoặc log invalid images."
        )

    content = torch.stack([sample["content"] for sample in valid_samples], dim=0)
    style = torch.stack([sample["style"] for sample in valid_samples], dim=0)
    content_path = [sample["content_path"] for sample in valid_samples]
    style_path = [sample["style_path"] for sample in valid_samples]

    return {
        "content": content,
        "style": style,
        "content_path": content_path,
        "style_path": style_path,
    }


class BasicImageTransform:
    """
    Transform train/eval cơ bản, không phụ thuộc torchvision.

    Train augmentation:
    - random horizontal flip
    - random crop nhẹ: resize lên một chút rồi crop về image_size

    Eval:
    - resize thẳng về image_size
    """

    def __init__(
        self,
        image_size: int = 256,
        is_train: bool = False,
        enable_hflip: bool = False,
        enable_random_crop: bool = False,
        flip_prob: float = 0.5,
        crop_scale: float = 1.1,
    ) -> None:
        self.image_size = image_size
        self.is_train = is_train
        self.enable_hflip = enable_hflip
        self.enable_random_crop = enable_random_crop
        self.flip_prob = flip_prob
        self.crop_scale = max(1.0, crop_scale)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")

        if self.is_train and self.enable_random_crop:
            aug_size = max(self.image_size, int(round(self.image_size * self.crop_scale)))
            image = resize_pil(image, aug_size)
            image = random_crop_pil(image, self.image_size)
        else:
            image = resize_pil(image, self.image_size)

        if self.is_train and self.enable_hflip:
            if random.random() < self.flip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return pil_to_tensor(image)


class AdaINUnpairedDataset(Dataset):
    """
    Dataset unpaired dùng chung cho debug-mode và real-mode.

    Contract mỗi sample:
        {
            "content": Tensor [3, H, W],
            "style": Tensor [3, H, W],
            "content_path": str,
            "style_path": str,
        }
    """

    def __init__(
        self,
        content_files: List[Path],
        style_files: List[Path],
        image_size: int = 256,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        seed: int = 42,
        pair_mode: str = "cycle",
        validate_on_init: bool = True,
        max_retry: int = 8,
        dataset_name: str = "adain_dataset",
    ) -> None:
        self.image_size = image_size
        self.transform = transform
        self.seed = seed
        self.pair_mode = pair_mode
        self.validate_on_init = validate_on_init
        self.max_retry = max_retry
        self.dataset_name = dataset_name

        self.invalid_content_files: List[Dict[str, str]] = []
        self.invalid_style_files: List[Dict[str, str]] = []

        if validate_on_init:
            self.content_files, self.invalid_content_files = filter_valid_images(content_files)
            self.style_files, self.invalid_style_files = filter_valid_images(style_files)
        else:
            self.content_files = content_files
            self.style_files = style_files

        if len(self.content_files) == 0:
            raise FileNotFoundError(
                f"[{self.dataset_name}] Không tìm thấy ảnh content hợp lệ."
            )
        if len(self.style_files) == 0:
            raise FileNotFoundError(
                f"[{self.dataset_name}] Không tìm thấy ảnh style hợp lệ."
            )

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.content_files)

    def get_invalid_summary(self) -> Dict[str, int]:
        return {
            "invalid_content_count": len(self.invalid_content_files),
            "invalid_style_count": len(self.invalid_style_files),
        }

    def _get_style_index(self, index: int) -> int:
        if self.pair_mode == "cycle":
            return index % len(self.style_files)
        if self.pair_mode == "random":
            return self._rng.randrange(len(self.style_files))
        raise ValueError(f"Unsupported pair_mode: {self.pair_mode}")

    def _load_tensor(self, path: Path) -> torch.Tensor:
        if self.transform is not None:
            with Image.open(path) as img:
                return self.transform(img)
        return default_image_loader(path, self.image_size)

    def _build_content_candidates(self, index: int) -> List[Path]:
        candidates: List[Path] = []
        total = len(self.content_files)
        retry_count = min(self.max_retry, total - 1)

        for offset in range(retry_count + 1):
            candidate = self.content_files[(index + offset) % total]
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _build_style_candidates(self, index: int) -> List[Path]:
        candidates: List[Path] = []
        total = len(self.style_files)

        first_idx = self._get_style_index(index)
        candidates.append(self.style_files[first_idx])

        retry_count = min(self.max_retry, total - 1)

        if self.pair_mode == "cycle":
            for offset in range(1, retry_count + 1):
                candidate = self.style_files[(first_idx + offset) % total]
                if candidate not in candidates:
                    candidates.append(candidate)
        elif self.pair_mode == "random":
            all_indices = list(range(total))
            self._rng.shuffle(all_indices)
            for idx_ in all_indices:
                candidate = self.style_files[idx_]
                if candidate not in candidates:
                    candidates.append(candidate)
                if len(candidates) >= retry_count + 1:
                    break

        return candidates

    def _try_load_first_valid(
        self,
        candidates: List[Path],
        kind: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[Path], List[str]]:
        errors: List[str] = []

        for path in candidates:
            try:
                tensor = self._load_tensor(path)
                return tensor, path, errors
            except (UnidentifiedImageError, OSError, ValueError) as e:
                errors.append(f"{kind}: {path} -> {str(e)}")

        return None, None, errors

    def __getitem__(self, index: int) -> Optional[Dict[str, object]]:
        content_candidates = self._build_content_candidates(index)
        style_candidates = self._build_style_candidates(index)

        content_tensor, content_path, _ = self._try_load_first_valid(
            content_candidates,
            kind="content",
        )
        if content_tensor is None or content_path is None:
            return None

        style_tensor, style_path, _ = self._try_load_first_valid(
            style_candidates,
            kind="style",
        )
        if style_tensor is None or style_path is None:
            return None

        return {
            "content": content_tensor,
            "style": style_tensor,
            "content_path": str(content_path),
            "style_path": str(style_path),
        }


def normalize_style_domain(style_domain: str) -> str:
    style_domain = style_domain.strip().lower()
    mapping = {
        "anime": "anime",
        "style_anime": "anime",
        "watercolor": "watercolor",
        "style_watercolor": "watercolor",
        "sketch": "sketch",
        "style_sketch": "sketch",
    }
    if style_domain not in mapping:
        raise ValueError(
            f"Unsupported style_domain: {style_domain}. "
            f"Supported: anime, watercolor, sketch"
        )
    return mapping[style_domain]


def build_train_transform(
    image_size: int,
    split: str,
    enable_hflip: bool = True,
    enable_random_crop: bool = True,
    flip_prob: float = 0.5,
    crop_scale: float = 1.1,
) -> BasicImageTransform:
    is_train = split == "train"
    return BasicImageTransform(
        image_size=image_size,
        is_train=is_train,
        enable_hflip=enable_hflip if is_train else False,
        enable_random_crop=enable_random_crop if is_train else False,
        flip_prob=flip_prob,
        crop_scale=crop_scale,
    )


def build_debug_dataset(
    root_dir: str = "debug_data",
    image_size: int = 256,
    seed: int = 42,
    pair_mode: str = "cycle",
    validate_on_init: bool = True,
    max_retry: int = 8,
) -> AdaINUnpairedDataset:
    root = Path(root_dir)
    content_files = scan_image_files(root / "content")
    style_files = scan_image_files(root / "style")

    transform = BasicImageTransform(
        image_size=image_size,
        is_train=False,
        enable_hflip=False,
        enable_random_crop=False,
    )

    return AdaINUnpairedDataset(
        content_files=content_files,
        style_files=style_files,
        image_size=image_size,
        transform=transform,
        seed=seed,
        pair_mode=pair_mode,
        validate_on_init=validate_on_init,
        max_retry=max_retry,
        dataset_name="debug_dataset",
    )


def resolve_real_dirs(
    real_root_dir: Path,
    split: str,
    style_domain: str,
) -> Tuple[Path, Path]:
    """
    Hỗ trợ 2 layout:

    1) Chưa split (GĐ2/GĐ4):
       data/processed/content
       data/processed/style_anime
       data/processed/style_watercolor
       data/processed/style_sketch

    2) Đã split (hữu ích cho GĐ6 về sau):
       data/processed/train/content
       data/processed/train/style_anime
       ...
    """
    style_domain = normalize_style_domain(style_domain)
    style_folder = f"style_{style_domain}"

    split_content_dir = real_root_dir / split / "content"
    split_style_dir = real_root_dir / split / style_folder

    if split_content_dir.exists() and split_style_dir.exists():
        return split_content_dir, split_style_dir

    base_content_dir = real_root_dir / "content"
    base_style_dir = real_root_dir / style_folder
    return base_content_dir, base_style_dir


def build_real_dataset(
    real_root_dir: str = "data/processed",
    split: str = "train",
    style_domain: str = "anime",
    image_size: int = 256,
    seed: int = 42,
    pair_mode: str = "random",
    validate_on_init: bool = True,
    max_retry: int = 8,
    enable_hflip: bool = True,
    enable_random_crop: bool = True,
    flip_prob: float = 0.5,
    crop_scale: float = 1.1,
    content_dir: Optional[str] = None,
    style_dir: Optional[str] = None,
) -> AdaINUnpairedDataset:
    if content_dir is not None and style_dir is not None:
        content_folder = Path(content_dir)
        style_folder = Path(style_dir)
    else:
        content_folder, style_folder = resolve_real_dirs(
            real_root_dir=Path(real_root_dir),
            split=split,
            style_domain=style_domain,
        )

    content_files = scan_image_files(content_folder)
    style_files = scan_image_files(style_folder)

    transform = build_train_transform(
        image_size=image_size,
        split=split,
        enable_hflip=enable_hflip,
        enable_random_crop=enable_random_crop,
        flip_prob=flip_prob,
        crop_scale=crop_scale,
    )

    return AdaINUnpairedDataset(
        content_files=content_files,
        style_files=style_files,
        image_size=image_size,
        transform=transform,
        seed=seed,
        pair_mode=pair_mode,
        validate_on_init=validate_on_init,
        max_retry=max_retry,
        dataset_name=f"real_dataset_{style_domain}_{split}",
    )


def build_dataset(
    mode: str,
    root_dir: str = "debug_data",
    real_root_dir: str = "data/processed",
    split: str = "train",
    style_domain: str = "anime",
    image_size: int = 256,
    seed: int = 42,
    pair_mode: str = "cycle",
    validate_on_init: bool = True,
    max_retry: int = 8,
    enable_hflip: bool = True,
    enable_random_crop: bool = True,
    flip_prob: float = 0.5,
    crop_scale: float = 1.1,
    content_dir: Optional[str] = None,
    style_dir: Optional[str] = None,
) -> AdaINUnpairedDataset:
    if mode == "debug":
        return build_debug_dataset(
            root_dir=root_dir,
            image_size=image_size,
            seed=seed,
            pair_mode=pair_mode,
            validate_on_init=validate_on_init,
            max_retry=max_retry,
        )

    if mode == "real":
        return build_real_dataset(
            real_root_dir=real_root_dir,
            split=split,
            style_domain=style_domain,
            image_size=image_size,
            seed=seed,
            pair_mode=pair_mode,
            validate_on_init=validate_on_init,
            max_retry=max_retry,
            enable_hflip=enable_hflip,
            enable_random_crop=enable_random_crop,
            flip_prob=flip_prob,
            crop_scale=crop_scale,
            content_dir=content_dir,
            style_dir=style_dir,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def load_manifest_if_exists(root_dir: str, mode: str) -> Optional[Dict]:
    root = Path(root_dir)

    if mode == "debug":
        manifest_path = root / "manifest.json"
    else:
        manifest_path = root / "manifest_real.json"

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
        collate_fn=safe_collate_fn,
    )


def summarize_batch(batch: Dict[str, object], batch_idx: int = 0) -> None:
    content = batch["content"]
    style = batch["style"]
    content_path = batch["content_path"]
    style_path = batch["style_path"]

    print("=" * 80)
    print(f"SMOKE TEST: AdaIN dataset contract | batch_idx={batch_idx}")
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


def assert_batch_contract(batch: Dict[str, object]) -> None:
    assert "content" in batch, "Thiếu key 'content'"
    assert "style" in batch, "Thiếu key 'style'"
    assert "content_path" in batch, "Thiếu key 'content_path'"
    assert "style_path" in batch, "Thiếu key 'style_path'"

    assert torch.is_tensor(batch["content"]), "'content' phải là tensor"
    assert torch.is_tensor(batch["style"]), "'style' phải là tensor"

    assert batch["content"].ndim == 4, "'content' batch phải có shape [B, C, H, W]"
    assert batch["style"].ndim == 4, "'style' batch phải có shape [B, C, H, W]"

    assert batch["content"].shape[1] == 3, "content phải có 3 channels RGB"
    assert batch["style"].shape[1] == 3, "style phải có 3 channels RGB"

    assert batch["content"].shape[-1] == batch["content"].shape[-2], "content phải là ảnh vuông"
    assert batch["style"].shape[-1] == batch["style"].shape[-2], "style phải là ảnh vuông"

    assert isinstance(batch["content_path"], list), "'content_path' phải là list[str] sau collate"
    assert isinstance(batch["style_path"], list), "'style_path' phải là list[str] sau collate"

    assert len(batch["content_path"]) == batch["content"].shape[0], "Số content_path không khớp batch size"
    assert len(batch["style_path"]) == batch["style"].shape[0], "Số style_path không khớp batch size"

    assert batch["content"].dtype == torch.float32, "content phải là float32"
    assert batch["style"].dtype == torch.float32, "style phải là float32"

    assert batch["content"].min().item() >= 0.0 and batch["content"].max().item() <= 1.0, \
        "content tensor phải nằm trong [0,1]"
    assert batch["style"].min().item() >= 0.0 and batch["style"].max().item() <= 1.0, \
        "style tensor phải nằm trong [0,1]"


def run_smoke_test(
    dataloader: DataLoader,
    smoke_steps: int = 3,
    verbose: bool = True,
) -> None:
    seen_batches = 0
    seen_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        assert_batch_contract(batch)

        if verbose:
            summarize_batch(batch, batch_idx=batch_idx)

        seen_batches += 1
        seen_samples += int(batch["content"].shape[0])

        if seen_batches >= smoke_steps:
            break

    if seen_batches == 0:
        raise RuntimeError("Smoke test thất bại: không đọc được batch nào từ DataLoader.")

    print(f"Smoke test passed: seen_batches={seen_batches}, seen_samples={seen_samples}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Giai đoạn 4 - AdaIN dataset hỗ trợ debug-mode + real-mode + augmentation train."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Giữ để tương thích CLI kế hoạch. Hiện tại chưa parse config.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        choices=["debug", "real"],
        help="debug: dùng debug_data | real: dùng dữ liệu đã chuẩn hóa",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split logic cho augmentation và resolve thư mục",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="debug_data",
        help="Thư mục root cho debug mode",
    )
    parser.add_argument(
        "--real_root_dir",
        type=str,
        default="data/processed",
        help="Thư mục root cho real mode",
    )
    parser.add_argument(
        "--content_dir",
        type=str,
        default=None,
        help="Tùy chọn override trực tiếp content_dir cho real mode",
    )
    parser.add_argument(
        "--style_dir",
        type=str,
        default=None,
        help="Tùy chọn override trực tiếp style_dir cho real mode",
    )
    parser.add_argument(
        "--style_domain",
        type=str,
        default="anime",
        choices=["anime", "watercolor", "sketch", "style_anime", "style_watercolor", "style_sketch"],
        help="Chọn style domain cho real mode",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Kích thước ảnh vuông image_size x image_size",
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
        "--max_retry",
        type=int,
        default=8,
        help="Số lần fallback tối đa khi gặp ảnh lỗi lúc runtime",
    )
    parser.add_argument(
        "--skip_validate_on_init",
        action="store_true",
        help="Bỏ qua bước lọc ảnh lỗi khi khởi tạo dataset",
    )
    parser.add_argument(
        "--disable_hflip",
        action="store_true",
        help="Tắt random horizontal flip ở train",
    )
    parser.add_argument(
        "--disable_random_crop",
        action="store_true",
        help="Tắt random crop ở train",
    )
    parser.add_argument(
        "--flip_prob",
        type=float,
        default=0.5,
        help="Xác suất horizontal flip",
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        default=1.1,
        help="Tỉ lệ resize trước khi crop, ví dụ 1.1",
    )
    parser.add_argument(
        "--smoke_steps",
        type=int,
        default=3,
        help="Số batch dùng để smoke test",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Chạy smoke test nhiều batch để xác nhận DataLoader ổn định",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manifest_root = args.root_dir if args.mode == "debug" else args.real_root_dir
    manifest = load_manifest_if_exists(manifest_root, args.mode)

    if manifest is not None:
        if args.mode == "debug":
            print(f"Found manifest: {Path(args.root_dir) / 'manifest.json'}")
            print(
                f"Manifest summary -> "
                f"num_content={manifest.get('num_content')}, "
                f"num_style={manifest.get('num_style')}, "
                f"image_size={manifest.get('image_size')}"
            )
        else:
            print(f"Found manifest: {Path(args.real_root_dir) / 'manifest_real.json'}")
            print(
                f"Manifest summary -> "
                f"mode={manifest.get('mode')}, "
                f"image_size={manifest.get('image_size')}, "
                f"total_processed={manifest.get('total_processed')}"
            )

    dataset = build_dataset(
        mode=args.mode,
        root_dir=args.root_dir,
        real_root_dir=args.real_root_dir,
        split=args.split,
        style_domain=args.style_domain,
        image_size=args.image_size,
        seed=args.seed,
        pair_mode=args.pair_mode,
        validate_on_init=not args.skip_validate_on_init,
        max_retry=args.max_retry,
        enable_hflip=not args.disable_hflip,
        enable_random_crop=not args.disable_random_crop,
        flip_prob=args.flip_prob,
        crop_scale=args.crop_scale,
        content_dir=args.content_dir,
        style_dir=args.style_dir,
    )

    invalid_summary = dataset.get_invalid_summary()

    print(f"Mode                   : {args.mode}")
    print(f"Dataset length         : {len(dataset)}")
    print(f"Split                  : {args.split}")
    print(f"Pair mode              : {args.pair_mode}")
    print(f"Validate on init       : {not args.skip_validate_on_init}")
    print(f"Invalid content files  : {invalid_summary['invalid_content_count']}")
    print(f"Invalid style files    : {invalid_summary['invalid_style_count']}")

    if args.mode == "debug":
        print(f"Debug root             : {args.root_dir}")
    else:
        print(f"Real root              : {args.real_root_dir}")
        print(f"Style domain           : {normalize_style_domain(args.style_domain)}")
        print(f"Aug hflip(train only)  : {not args.disable_hflip}")
        print(f"Aug crop(train only)   : {not args.disable_random_crop}")
        print(f"Flip prob              : {args.flip_prob}")
        print(f"Crop scale             : {args.crop_scale}")

    if len(dataset.invalid_content_files) > 0:
        print("Ví dụ invalid content đầu tiên:")
        print(dataset.invalid_content_files[0])

    if len(dataset.invalid_style_files) > 0:
        print("Ví dụ invalid style đầu tiên:")
        print(dataset.invalid_style_files[0])

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True if args.split == "train" else False,
        num_workers=args.num_workers,
    )

    first_batch = next(iter(dataloader))
    assert_batch_contract(first_batch)
    summarize_batch(first_batch, batch_idx=0)

    if args.smoke_test:
        run_smoke_test(
            dataloader=dataloader,
            smoke_steps=args.smoke_steps,
            verbose=False,
        )

    print("DONE: Dataset/DataLoader hợp lệ cho GĐ4 (debug-mode + real-mode + augmentation).")


if __name__ == "__main__":
    main()


# --------------------------------------------------
# Ví dụ chạy giữ nguyên debug-mode cũ:
# python -m src.data.datasets --config configs/config.yaml --split train --smoke_test
#
# Ví dụ chạy debug-mode rõ ràng:
# python -m src.data.datasets --mode debug --root_dir debug_data --split train --smoke_test
#
# Ví dụ chạy real-mode với data/processed/content + style_anime:
# python -m src.data.datasets --mode real --real_root_dir data/processed --style_domain anime --split train --smoke_test
#
# Ví dụ chạy real-mode với style_sketch:
# python -m src.data.datasets --mode real --real_root_dir data/processed --style_domain sketch --split train --smoke_test
#
# Ví dụ tắt augmentation:
# python -m src.data.datasets --mode real --real_root_dir data/processed --style_domain watercolor --split train --disable_hflip --disable_random_crop --smoke_test