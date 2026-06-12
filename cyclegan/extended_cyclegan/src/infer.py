from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.datasets import SingleImageDataset
from src.models.cyclegan import CycleGANModel
from src.utils.config import load_config
from src.utils.misc import get_device


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer CycleGAN một chiều")
    parser.add_argument("--config", type=str, default="configs/train_monet.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--direction", type=str, default="A2B", choices=["A2B", "B2A"])
    parser.add_argument("--input_dir", type=str, default=None, help="Mặc định: testA nếu A2B, testB nếu B2A")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=0, help="0 = chạy hết")
    return parser


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
                out = model.infer_A2B(image)
            else:
                out = model.infer_B2A(image)

            save_path = output_dir / f"{src_path.stem}_{args.direction}.jpg"
            save_image(denorm(out[0]), save_path)

    print("[Done] Infer xong.")


if __name__ == "__main__":
    main()
