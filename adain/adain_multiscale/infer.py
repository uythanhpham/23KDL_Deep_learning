"""Inference AdaIN Multi-Scale.

Chạy từ thư mục adain_multiscale:

    python infer.py --checkpoint checkpoints/best_model.pth \
        --content_dir ../../data/archive/testA --style_dir ../../data/archive/testB \
        --output_dir outputs/infer

Mặc định pair_mode=cycle: mỗi ảnh content ghép 1 ảnh style (lấy dư tuần hoàn) →
output 1-1 với content, dùng được ngay cho scripts/evaluate_all.sh.
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from Model.adain_multiscale import AdaINStyleTransfer

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def denormalize(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(STD).view(1, 3, 1, 1).to(device)
    return (tensor * std + mean).clamp(0, 1)


def list_images(folder: str) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in VALID_EXTENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference AdaIN Multi-Scale")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--style_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--pair_mode", type=str, default="cycle", choices=["cycle", "all"],
                        help="cycle: 1 style/content (output 1-1, dùng cho evaluate); all: tích chéo content×style")
    parser.add_argument("--max_images", type=int, default=0, help="0 = chạy hết")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Thiết bị: {device}")

    model = AdaINStyleTransfer().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = get_transform(args.size)

    contents = list_images(args.content_dir)
    styles = list_images(args.style_dir)
    if args.max_images > 0:
        contents = contents[: args.max_images]

    def load(path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            return transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        for i, c_path in enumerate(contents):
            c_img = load(c_path)
            if args.pair_mode == "cycle":
                s_path = styles[i % len(styles)]
                output = denormalize(model(c_img, load(s_path), alpha=args.alpha), device)
                save_image(output, output_dir / f"{c_path.stem}_stylized.jpg")
            else:
                for s_path in styles:
                    output = denormalize(model(c_img, load(s_path), alpha=args.alpha), device)
                    save_image(output, output_dir / f"result_{c_path.stem}_{s_path.stem}.jpg")
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(contents)}")

    print(f"[Thành công] Ảnh đã lưu tại: {output_dir}")


if __name__ == "__main__":
    main()
