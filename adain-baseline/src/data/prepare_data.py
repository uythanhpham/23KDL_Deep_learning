"""Dọn dẹp, chuẩn hóa ảnh từ raw -> processed."""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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

    ext = image_format.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"

    for idx in range(count):
        pattern = patterns[idx % len(patterns)]
        arr = make_random_image_array(image_size=image_size, pattern=pattern)
        filename = f"{prefix}_{idx:04d}.{ext}"
        out_path = target_dir / filename
        save_image(arr, out_path)
        saved_paths.append(str(out_path))

    return saved_paths


def write_manifest(
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
        "content_dir": str(output_dir / "content"),
        "style_dir": str(output_dir / "style"),
        "content_files": content_paths,
        "style_files": style_paths,
        "notes": (
            "Dummy RGB images for early smoke test. "
            "Used to unblock model/train/infer pipeline in phase 0."
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

    manifest_path = write_manifest(
        output_dir=output_dir,
        content_paths=content_paths,
        style_paths=style_paths,
        image_size=image_size,
        seed=seed,
    )

    return {
        "output_dir": str(output_dir),
        "content_dir": str(content_dir),
        "style_dir": str(style_dir),
        "manifest_path": str(manifest_path),
        "num_content": str(len(content_paths)),
        "num_style": str(len(style_paths)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dummy/debug data for AdaIN project."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        choices=["debug"],
        help="Giai đoạn 0 chỉ cần debug mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_data",
        help="Thư mục output, ví dụ: debug_data",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Kích thước ảnh vuông, ví dụ 256",
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed để tái lập dữ liệu giả",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Định dạng ảnh output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)

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
    print(f"Output dir   : {result['output_dir']}")
    print(f"Content dir  : {result['content_dir']}")
    print(f"Style dir    : {result['style_dir']}")
    print(f"Manifest     : {result['manifest_path']}")
    print(f"Num content  : {result['num_content']}")
    print(f"Num style    : {result['num_style']}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# python -m src.data.prepare_data --mode debug --output_dir debug_data --image_size 256
# python -m src.data.prepare_data --mode debug --output_dir debug_data --image_size 256 --num_content 20 --num_style 20