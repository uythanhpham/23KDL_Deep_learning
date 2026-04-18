"""Dọn dẹp, chuẩn hóa ảnh từ raw -> processed."""

# E:\Nam3_ki2\TH DL\PROJECT\23KDL_Deep_learning\adain-baseline\src\data\prepare_data.py

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_path(path: Path) -> str:
    return path.expanduser().absolute().as_posix()


def normalize_ext(image_format: str) -> str:
    ext = image_format.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    return ext


# =========================================================
# DEBUG MODE (GĐ 0 - GĐ 1)
# =========================================================
def make_random_image_array(
    image_size: int,
    pattern: str = "noise",
) -> np.ndarray:
    """
    Tạo ảnh RGB dummy kích thước (H, W, 3), dtype=uint8.

    pattern:
        - noise: nhiễu ngẫu nhiên hoàn toàn
        - gradient: ảnh có chuyển màu
        - blocks: các khối màu
        - stripes: sọc ngang/dọc
    """
    h = w = image_size

    if pattern == "noise":
        arr = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        return arr

    if pattern == "gradient":
        x = np.linspace(0, 255, w, dtype=np.uint8)
        y = np.linspace(0, 255, h, dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)

        r = xv
        g = yv
        b = ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)
        arr = np.stack([r, g, b], axis=-1)
        return arr

    if pattern == "blocks":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        block_size = max(8, image_size // 8)

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
                arr[y:y + block_size, x:x + block_size] = color
        return arr

    if pattern == "stripes":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        stripe_width = max(4, image_size // 16)
        vertical = random.choice([True, False])

        if vertical:
            for x in range(0, w, stripe_width):
                color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
                arr[:, x:x + stripe_width] = color
        else:
            for y in range(0, h, stripe_width):
                color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
                arr[y:y + stripe_width, :] = color
        return arr

    raise ValueError(f"Unsupported pattern: {pattern}")


def save_image(array: np.ndarray, output_path: Path) -> None:
    image = Image.fromarray(array, mode="RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def generate_debug_images(
    target_dir: Path,
    prefix: str,
    count: int,
    image_size: int,
    patterns: List[str],
    image_format: str,
) -> List[str]:
    """
    Sinh dummy images vào target_dir và trả về danh sách path dạng string.
    """
    ensure_dir(target_dir)
    saved_paths: List[str] = []

    ext = normalize_ext(image_format)

    for idx in range(count):
        pattern = patterns[idx % len(patterns)]
        arr = make_random_image_array(image_size=image_size, pattern=pattern)
        filename = f"{prefix}_{idx:04d}.{ext}"
        out_path = target_dir / filename
        save_image(arr, out_path)
        saved_paths.append(normalize_path(out_path))

    return saved_paths


def write_debug_manifest(
    output_dir: Path,
    content_paths: List[str],
    style_paths: List[str],
    image_size: int,
    seed: int,
) -> Path:
    manifest = {
        "mode": "debug",
        "image_size": image_size,
        "seed": seed,
        "num_content": len(content_paths),
        "num_style": len(style_paths),
        "content_dir": normalize_path(output_dir / "content"),
        "style_dir": normalize_path(output_dir / "style"),
        "content_files": content_paths,
        "style_files": style_paths,
        "notes": (
            "Dummy RGB images for early smoke test. "
            "Used to unblock model/train/infer pipeline in phase 0/1."
        ),
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


def prepare_debug_data(
    output_dir: Path,
    image_size: int,
    num_content: int,
    num_style: int,
    seed: int,
    image_format: str,
) -> Dict[str, str]:
    set_seed(seed)

    ensure_dir(output_dir)
    content_dir = output_dir / "content"
    style_dir = output_dir / "style"

    content_patterns = ["noise", "gradient", "blocks", "stripes"]
    style_patterns = ["blocks", "stripes", "gradient", "noise"]

    content_paths = generate_debug_images(
        target_dir=content_dir,
        prefix="content",
        count=num_content,
        image_size=image_size,
        patterns=content_patterns,
        image_format=image_format,
    )

    style_paths = generate_debug_images(
        target_dir=style_dir,
        prefix="style",
        count=num_style,
        image_size=image_size,
        patterns=style_patterns,
        image_format=image_format,
    )

    manifest_path = write_debug_manifest(
        output_dir=output_dir,
        content_paths=content_paths,
        style_paths=style_paths,
        image_size=image_size,
        seed=seed,
    )

    return {
        "mode": "debug",
        "output_dir": normalize_path(output_dir),
        "content_dir": normalize_path(content_dir),
        "style_dir": normalize_path(style_dir),
        "manifest_path": normalize_path(manifest_path),
        "num_content": str(len(content_paths)),
        "num_style": str(len(style_paths)),
    }


# =========================================================
# REAL MODE SKELETON (GĐ 2)
# =========================================================
def collect_image_paths(root_dir: Optional[Path]) -> List[Path]:
    """
    Quét đệ quy toàn bộ file ảnh hợp lệ trong root_dir.
    """
    if root_dir is None:
        return []

    if not root_dir.exists() or not root_dir.is_dir():
        return []

    image_paths = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    image_paths.sort()
    return image_paths


def load_rgb_image(image_path: Path) -> Image.Image:
    """
    Đọc ảnh và ép về RGB.
    """
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        return rgb_img.copy()


def resize_image(image: Image.Image, image_size: int) -> Image.Image:
    """
    Resize về ảnh vuông kích thước chuẩn.
    """
    return image.resize((image_size, image_size), RESAMPLE_BICUBIC)


def save_pil_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()

    if ext in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def build_real_filename(domain_name: str, index: int, image_format: str) -> str:
    ext = normalize_ext(image_format)
    return f"{domain_name}_{index:05d}.{ext}"


def process_real_domain(
    source_dir: Optional[Path],
    output_dir: Path,
    domain_name: str,
    image_size: int,
    image_format: str,
) -> Dict[str, object]:
    """
    Skeleton cho domain thật:
    - quét ảnh
    - đọc ảnh
    - convert RGB
    - resize
    - lưu path chuẩn hóa sang processed

    Chưa split train/val/test ở GĐ 2.
    """
    domain_output_dir = output_dir / domain_name
    ensure_dir(domain_output_dir)

    if source_dir is None:
        return {
            "domain_name": domain_name,
            "source_dir": None,
            "output_dir": normalize_path(domain_output_dir),
            "num_found": 0,
            "num_processed": 0,
            "num_failed": 0,
            "files": [],
            "failed_files": [],
            "warning": f"{domain_name}: source_dir is None, skipped.",
        }

    image_paths = collect_image_paths(source_dir)

    processed_files: List[Dict[str, str]] = []
    failed_files: List[Dict[str, str]] = []

    for idx, src_path in enumerate(image_paths):
        try:
            image = load_rgb_image(src_path)
            image = resize_image(image, image_size=image_size)

            filename = build_real_filename(
                domain_name=domain_name,
                index=idx,
                image_format=image_format,
            )
            dst_path = domain_output_dir / filename
            save_pil_image(image, dst_path)

            processed_files.append(
                {
                    "source_path": normalize_path(src_path),
                    "processed_path": normalize_path(dst_path),
                    "relative_processed_path": dst_path.relative_to(output_dir).as_posix(),
                }
            )
        except Exception as e:
            failed_files.append(
                {
                    "source_path": normalize_path(src_path),
                    "error": str(e),
                }
            )

    warning = None
    if not source_dir.exists():
        warning = f"{domain_name}: source directory does not exist, skipped."
    elif len(image_paths) == 0:
        warning = f"{domain_name}: no valid images found."

    return {
        "domain_name": domain_name,
        "source_dir": normalize_path(source_dir),
        "output_dir": normalize_path(domain_output_dir),
        "num_found": len(image_paths),
        "num_processed": len(processed_files),
        "num_failed": len(failed_files),
        "files": processed_files,
        "failed_files": failed_files,
        "warning": warning,
    }


def write_real_manifest(
    output_dir: Path,
    image_size: int,
    seed: int,
    image_format: str,
    content_summary: Dict[str, object],
    style_summaries: List[Dict[str, object]],
) -> Path:
    total_processed = int(content_summary["num_processed"]) + sum(
        int(item["num_processed"]) for item in style_summaries
    )
    total_failed = int(content_summary["num_failed"]) + sum(
        int(item["num_failed"]) for item in style_summaries
    )

    manifest = {
        "mode": "real",
        "stage": "giai_doan_2_real_mode_skeleton",
        "image_size": image_size,
        "seed": seed,
        "image_format": normalize_ext(image_format),
        "output_dir": normalize_path(output_dir),
        "content": content_summary,
        "styles": style_summaries,
        "total_processed": total_processed,
        "total_failed": total_failed,
        "notes": [
            "Stage 2 skeleton only.",
            "Current responsibilities: read image, convert RGB, resize, normalize output paths.",
            "Train/val/test split will be completed in a later stage (Giai đoạn 5).",
        ],
    }

    manifest_path = output_dir / "manifest_real.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


def prepare_real_data(
    output_dir: Path,
    content_dir: Optional[Path],
    style_anime_dir: Optional[Path],
    style_watercolor_dir: Optional[Path],
    style_sketch_dir: Optional[Path],
    image_size: int,
    seed: int,
    image_format: str,
) -> Dict[str, str]:
    """
    GĐ 2:
    Chuẩn bị skeleton cho dữ liệu thật:
    - content -> output_dir/content
    - style_anime -> output_dir/style_anime
    - style_watercolor -> output_dir/style_watercolor
    - style_sketch -> output_dir/style_sketch

    Chưa làm split train/val/test ở giai đoạn này.
    """
    set_seed(seed)
    ensure_dir(output_dir)

    content_summary = process_real_domain(
        source_dir=content_dir,
        output_dir=output_dir,
        domain_name="content",
        image_size=image_size,
        image_format=image_format,
    )

    style_summaries = [
        process_real_domain(
            source_dir=style_anime_dir,
            output_dir=output_dir,
            domain_name="style_anime",
            image_size=image_size,
            image_format=image_format,
        ),
        process_real_domain(
            source_dir=style_watercolor_dir,
            output_dir=output_dir,
            domain_name="style_watercolor",
            image_size=image_size,
            image_format=image_format,
        ),
        process_real_domain(
            source_dir=style_sketch_dir,
            output_dir=output_dir,
            domain_name="style_sketch",
            image_size=image_size,
            image_format=image_format,
        ),
    ]

    manifest_path = write_real_manifest(
        output_dir=output_dir,
        image_size=image_size,
        seed=seed,
        image_format=image_format,
        content_summary=content_summary,
        style_summaries=style_summaries,
    )

    total_style_processed = sum(int(item["num_processed"]) for item in style_summaries)
    total_style_failed = sum(int(item["num_failed"]) for item in style_summaries)

    return {
        "mode": "real",
        "output_dir": normalize_path(output_dir),
        "content_dir": normalize_path(output_dir / "content"),
        "style_anime_dir": normalize_path(output_dir / "style_anime"),
        "style_watercolor_dir": normalize_path(output_dir / "style_watercolor"),
        "style_sketch_dir": normalize_path(output_dir / "style_sketch"),
        "manifest_path": normalize_path(manifest_path),
        "num_content": str(content_summary["num_processed"]),
        "num_style_total": str(total_style_processed),
        "num_failed_total": str(int(content_summary["num_failed"]) + total_style_failed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data for AdaIN project (debug mode + real mode skeleton)."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        choices=["debug", "real"],
        help="debug: tạo dummy data | real: xử lý khung dữ liệu thật",
    )

    # Shared args
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_data",
        help="Thư mục output",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Kích thước ảnh vuông, ví dụ 256",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed để tái lập dữ liệu",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Định dạng ảnh output",
    )

    # Debug-mode args
    parser.add_argument(
        "--num_content",
        type=int,
        default=12,
        help="Số ảnh content dummy cần tạo",
    )
    parser.add_argument(
        "--num_style",
        type=int,
        default=12,
        help="Số ảnh style dummy cần tạo",
    )

    # Real-mode args
    parser.add_argument(
        "--content_dir",
        type=str,
        default=None,
        help="Thư mục raw content images",
    )
    parser.add_argument(
        "--style_anime_dir",
        type=str,
        default=None,
        help="Thư mục raw style anime images",
    )
    parser.add_argument(
        "--style_watercolor_dir",
        type=str,
        default=None,
        help="Thư mục raw style watercolor images",
    )
    parser.add_argument(
        "--style_sketch_dir",
        type=str,
        default=None,
        help="Thư mục raw style sketch images",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.mode == "debug":
        result = prepare_debug_data(
            output_dir=output_dir,
            image_size=args.image_size,
            num_content=args.num_content,
            num_style=args.num_style,
            seed=args.seed,
            image_format=args.image_format,
        )

        print("=" * 60)
        print("DONE: Generated debug dummy data successfully.")
        print(f"Mode         : {result['mode']}")
        print(f"Output dir   : {result['output_dir']}")
        print(f"Content dir  : {result['content_dir']}")
        print(f"Style dir    : {result['style_dir']}")
        print(f"Manifest     : {result['manifest_path']}")
        print(f"Num content  : {result['num_content']}")
        print(f"Num style    : {result['num_style']}")
        print("=" * 60)
        return

    result = prepare_real_data(
        output_dir=output_dir,
        content_dir=Path(args.content_dir) if args.content_dir else None,
        style_anime_dir=Path(args.style_anime_dir) if args.style_anime_dir else None,
        style_watercolor_dir=Path(args.style_watercolor_dir) if args.style_watercolor_dir else None,
        style_sketch_dir=Path(args.style_sketch_dir) if args.style_sketch_dir else None,
        image_size=args.image_size,
        seed=args.seed,
        image_format=args.image_format,
    )

    print("=" * 60)
    print("DONE: Prepared real-mode skeleton successfully.")
    print(f"Mode              : {result['mode']}")
    print(f"Output dir        : {result['output_dir']}")
    print(f"Content dir       : {result['content_dir']}")
    print(f"Style anime dir   : {result['style_anime_dir']}")
    print(f"Style watercolor  : {result['style_watercolor_dir']}")
    print(f"Style sketch dir  : {result['style_sketch_dir']}")
    print(f"Manifest          : {result['manifest_path']}")
    print(f"Num content       : {result['num_content']}")
    print(f"Num style total   : {result['num_style_total']}")
    print(f"Num failed total  : {result['num_failed_total']}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# ---------------------------
# Ví dụ chạy debug mode
# python -m src.data.prepare_data --mode debug --output_dir debug_data --image_size 256
# python -m src.data.prepare_data --mode debug --output_dir debug_data --image_size 256 --num_content 20 --num_style 20
#
# Ví dụ chạy real mode skeleton (GĐ 2)
# python -m src.data.prepare_data --mode real --content_dir data/raw/content --style_anime_dir data/raw/style_anime --style_watercolor_dir data/raw/style_watercolor --style_sketch_dir data/raw/style_sketch --output_dir data/processed --image_size 256