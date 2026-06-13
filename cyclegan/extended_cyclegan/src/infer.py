from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.datasets import SingleImageDataset, load_jsonl_palettes
from src.models.cyclegan import CycleGANModel
from src.utils.config import load_config
from src.utils.misc import get_device


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer Palette-guided CycleGAN một chiều")
    parser.add_argument("--config", type=str, default="configs/train_vangogh.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--direction", type=str, default="A2B", choices=["A2B", "B2A"])
    parser.add_argument("--input_dir", type=str, default=None, help="Mặc định: testA nếu A2B, testB nếu B2A")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=0, help="0 = chạy hết")
    parser.add_argument("--palette_jsonl", type=str, default=None,
                        help="Override file palette bank; mặc định lấy từ config (palette_art nếu A2B, palette_photo nếu B2A)")
    parser.add_argument("--palette_index", type=int, default=-1,
                        help="Chọn palette thứ i trong bank (theo thứ tự file); -1 = chọn ngẫu nhiên với seed cố định")
    parser.add_argument("--seed", type=int, default=42, help="Seed khi chọn palette ngẫu nhiên")
    return parser


def pick_palette(cfg: dict, args: argparse.Namespace) -> torch.Tensor:
    """Chọn 1 vector palette (1, 24) cho cả lượt infer.

    A2B cần palette nghệ thuật (p_target), B2A cần palette photo (p_photo).
    Fallback vector 0 nếu không có bank — nhất quán với hành vi gán mặc định lúc train.
    """
    if args.palette_jsonl is not None:
        bank_path = args.palette_jsonl
    else:
        key = "palette_art" if args.direction == "A2B" else "palette_photo"
        bank_path = cfg["data"].get(key, "")

    palettes = load_jsonl_palettes(bank_path) if bank_path else {}
    if not palettes:
        print("[Warning] Không nạp được palette bank — dùng vector 0 (24-d). "
              "Chỉ định --palette_jsonl hoặc cập nhật config để có kết quả đúng.")
        return torch.zeros(1, 24)

    names = sorted(palettes.keys())
    if args.palette_index >= 0:
        name = names[args.palette_index % len(names)]
    else:
        name = random.Random(args.seed).choice(names)
    print(f"[Infer] palette='{name}' (từ {bank_path})")
    return palettes[name].unsqueeze(0)


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    style = cfg["data"]["style"]
    device = get_device(cfg["train"].get("device", "auto"))

    if args.input_dir is None:
        split = cfg["data"].get("testA", "testA") if args.direction == "A2B" else cfg["data"].get("testB", "testB")
        input_dir = Path(cfg["data"]["root"]) / style / split
    else:
        input_dir = Path(args.input_dir)

    if args.output_dir is None:
        output_dir = Path(cfg["output"]["root"]) / "inference" / style / args.direction
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SingleImageDataset(
        image_dir=input_dir,
        image_size=int(cfg["train"]["image_size"]),
        crop_size=int(cfg["train"]["crop_size"]),
        exts=cfg["data"].get("image_extensions"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = CycleGANModel(cfg, device)
    model.load_checkpoint(args.checkpoint, strict=True, load_discriminators=False)
    model.eval()

    palette = pick_palette(cfg, args).to(device)

    print(f"[Infer] style={style}, direction={args.direction}, device={device}")
    print(f"[Infer] input={input_dir}")
    print(f"[Infer] output={output_dir}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Infer")):
            if args.max_images > 0 and i >= args.max_images:
                break
            image = batch["image"].to(device)
            src_path = Path(batch["path"][0])

            if args.direction == "A2B":
                out = model.infer_A2B(image, palette)
            else:
                out = model.infer_B2A(image, palette)

            save_path = output_dir / f"{src_path.stem}_{args.direction}.jpg"
            save_image(denorm(out[0]), save_path)

    print("[Done] Infer xong.")


if __name__ == "__main__":
    main()
