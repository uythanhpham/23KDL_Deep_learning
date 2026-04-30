"Inference AdaIN checkpoint."
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from src.models.adain import AdaINStyleTransfer

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transform(size: int = 512) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

def load_model(checkpoint: str, device: torch.device) -> AdaINStyleTransfer:
    model = AdaINStyleTransfer().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "decoder_state_dict" in ckpt:
        model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def list_images(folder: str | Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--style_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_content", type=int, default=10)
    parser.add_argument("--max_style", type=int, default=10)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] device={device}")
    model = load_model(args.checkpoint, device)
    transform = get_transform(args.size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contents = list_images(args.content_dir)[: args.max_content]
    styles = list_images(args.style_dir)[: args.max_style]
    print(f"content={len(contents)} | style={len(styles)}")

    with torch.no_grad():
        for c_path in contents:
            c_img = transform(Image.open(c_path).convert("RGB")).unsqueeze(0).to(device)
            for s_path in styles:
                s_img = transform(Image.open(s_path).convert("RGB")).unsqueeze(0).to(device)
                output = model(c_img, s_img, alpha=args.alpha)
                output = denormalize(output).clamp(0, 1)
                out_name = f"{c_path.stem}__{s_path.stem}__a{args.alpha:.2f}.jpg"
                save_image(output, output_dir / out_name, normalize=False)
                print(f"[Save] {out_name}")
    print(f"[DONE] {output_dir.resolve()}")

if __name__ == "__main__":
    main()
