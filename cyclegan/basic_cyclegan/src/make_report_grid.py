from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def list_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])


def make_grid(images, labels, out_path: Path, thumb=256):
    n = len(images)
    canvas = Image.new("RGB", (thumb * n, thumb + 36), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img_path, label) in enumerate(zip(images, labels)):
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((thumb, thumb), Image.BICUBIC)
        x = i * thumb + (thumb - img.width) // 2
        y = 30 + (thumb - img.height) // 2
        canvas.paste(img, (x, y))
        draw.text((i * thumb + 8, 8), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Tạo grid ảnh inference để đưa vào báo cáo.")
    parser.add_argument("--content_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--out", default="report_grid.jpg")
    parser.add_argument("--max_items", type=int, default=5)
    args = parser.parse_args()

    content_imgs = list_images(Path(args.content_dir))[: args.max_items]
    output_imgs = list_images(Path(args.output_dir))[: args.max_items]
    n = min(len(content_imgs), len(output_imgs))
    if n == 0:
        raise RuntimeError("Không đủ ảnh để tạo grid.")

    # Tạo từng hàng: input, output
    for i in range(n):
        make_grid(
            [content_imgs[i], output_imgs[i]],
            ["Input photo", "CycleGAN output"],
            Path(args.out).with_name(f"{Path(args.out).stem}_{i:02d}.jpg"),
        )
    print("[Done] Đã tạo grid báo cáo.")


if __name__ == "__main__":
    main()
