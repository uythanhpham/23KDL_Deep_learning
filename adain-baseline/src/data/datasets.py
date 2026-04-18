"""Biến ảnh trong processed/debug_data thành tensor.

Giai đoạn 3 - Người 1:
- Bổ sung kiểm tra lỗi đọc ảnh
- Smoke test DataLoader
- Đảm bảo batch chạy ổn định trên mock/debug data
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
            "Hãy kiểm tra lại debug_data hoặc log invalid images."
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


class AdaINDebugDataset(Dataset):
    """
    Giai đoạn 3:
    - Đọc debug_data/content và debug_data/style
    - Lọc trước ảnh lỗi nếu validate_on_init=True
    - Nếu runtime có file bị hỏng/xóa đột xuất, thử fallback sang ảnh khác
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
        validate_on_init: bool = True,
        max_retry: int = 8,
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
        self.validate_on_init = validate_on_init
        self.max_retry = max_retry

        raw_content_files = scan_image_files(self.content_dir)
        raw_style_files = scan_image_files(self.style_dir)

        self.invalid_content_files: List[Dict[str, str]] = []
        self.invalid_style_files: List[Dict[str, str]] = []

        if validate_on_init:
            self.content_files, self.invalid_content_files = filter_valid_images(raw_content_files)
            self.style_files, self.invalid_style_files = filter_valid_images(raw_style_files)
        else:
            self.content_files = raw_content_files
            self.style_files = raw_style_files

        if len(self.content_files) == 0:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh content hợp lệ trong thư mục: {self.content_dir}"
            )
        if len(self.style_files) == 0:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh style hợp lệ trong thư mục: {self.style_dir}"
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
                img = img.convert("RGB")
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

        content_tensor, content_path, content_errors = self._try_load_first_valid(
            content_candidates,
            kind="content",
        )
        if content_tensor is None or content_path is None:
            return None

        style_tensor, style_path, style_errors = self._try_load_first_valid(
            style_candidates,
            kind="style",
        )
        if style_tensor is None or style_path is None:
            return None

        sample = {
            "content": content_tensor,   # [3, H, W]
            "style": style_tensor,       # [3, H, W]
            "content_path": str(content_path),
            "style_path": str(style_path),
        }
        return sample


def build_debug_dataset(
    root_dir: str = "debug_data",
    image_size: int = 256,
    seed: int = 42,
    pair_mode: str = "cycle",
    validate_on_init: bool = True,
    max_retry: int = 8,
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
        validate_on_init=validate_on_init,
        max_retry=max_retry,
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
        collate_fn=safe_collate_fn,
    )


def summarize_batch(batch: Dict[str, object], batch_idx: int = 0) -> None:
    content = batch["content"]
    style = batch["style"]
    content_path = batch["content_path"]
    style_path = batch["style_path"]

    print("=" * 80)
    print(f"SMOKE TEST: AdaIN debug dataset contract | batch_idx={batch_idx}")
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
        description="Giai đoạn 3 - AdaIN debug dataset smoke test ổn định."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Giữ để tương thích CLI kế hoạch. GĐ3 chưa bắt buộc dùng config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="GĐ3 chủ yếu dùng train/debug, nhưng giữ tham số để khớp CLI kế hoạch.",
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
        "--max_retry",
        type=int,
        default=8,
        help="Số lần fallback tối đa khi gặp ảnh lỗi lúc runtime",
    )
    parser.add_argument(
        "--skip_validate_on_init",
        action="store_true",
        help="Bỏ qua bước lọc ảnh lỗi ở thời điểm khởi tạo dataset",
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
        validate_on_init=not args.skip_validate_on_init,
        max_retry=args.max_retry,
    )

    invalid_summary = dataset.get_invalid_summary()

    print(f"Dataset length         : {len(dataset)}")
    print(f"Content dir            : {Path(args.root_dir) / 'content'}")
    print(f"Style dir              : {Path(args.root_dir) / 'style'}")
    print(f"Split                  : {args.split}")
    print(f"Pair mode              : {args.pair_mode}")
    print(f"Validate on init       : {not args.skip_validate_on_init}")
    print(f"Invalid content files  : {invalid_summary['invalid_content_count']}")
    print(f"Invalid style files    : {invalid_summary['invalid_style_count']}")

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

    print("DONE: Dataset/DataLoader contract hợp lệ và ổn định cho GĐ3.")


if __name__ == "__main__":
    main()


# Ví dụ chạy:
# python -m src.data.datasets --config configs/config.yaml --split train --smoke_test
# python -m src.data.datasets --root_dir debug_data --image_size 256 --batch_size 4 --smoke_test --smoke_steps 3
# python -m src.data.datasets --root_dir debug_data --pair_mode random --smoke_test