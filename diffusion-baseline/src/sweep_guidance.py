"""
========================================================================
sweep_guidance.py — Quét guidance_scale cho Diffusion (img2img)
========================================================================
Sinh kết quả stylized trên CÙNG một cặp (Content, Style) ở NHIỀU mức
guidance_scale rồi ghép thành 1 ảnh lưới có nhãn — dùng cho hình
"Sweep guidance_scale" trong báo cáo.

Điểm quan trọng: cố định seed TRƯỚC MỖI lần sample → nhiễu khởi tạo giống
nhau ở mọi mức guidance, nên khác biệt giữa các ảnh CHỈ do guidance_scale.

Cách dùng (chạy từ thư mục diffusion-baseline):
    python -m src.sweep_guidance \
        --checkpoint checkpoints/best_model.pth \
        --content ../data/archive/testA/vangogh_testA_000000.jpg \
        --style   ../data/archive/testB/vangogh_testB_000005.jpg \
        --guidance_scales 1,2,3,5 \
        --strength 0.6 --ddim_steps 50 --guidance_rescale 0.7 \
        --out outputs/sweep_guidance

Trên Kaggle: trỏ --checkpoint vào /kaggle/input/... và --content/--style vào
dataset tương ứng. Kết quả: outputs/sweep_guidance/grid_guidance.png (ảnh để
chèn vào báo cáo) + các ảnh đơn out_s<scale>.png.
========================================================================
"""
import os
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
from PIL import Image, ImageDraw

from src.models.unet import UNet
from src.models.style_encoder import StyleEncoder
from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.ddim import DDIMSampler
from src.infer import load_image, denormalize, load_model  # tái dùng đúng pipeline với infer.py


def to_pil(t: torch.Tensor) -> Image.Image:
    """Tensor (1,3,H,W) miền [-1,1] -> PIL.Image RGB."""
    arr = denormalize(t.detach().cpu())[0].permute(1, 2, 0).numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_labeled_grid(panels, cell=256, band=28, pad=8):
    """Ghép các (nhãn, PIL.Image) thành 1 hàng có dải nhãn phía trên mỗi ô (PIL thuần)."""
    n = len(panels)
    W = n * cell + (n + 1) * pad
    H = band + cell + 2 * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(panels):
        img = img.resize((cell, cell), Image.BILINEAR)
        x = pad + i * (cell + pad)
        canvas.paste(img, (x, band + pad))
        # canh giữa nhãn theo bề rộng ô
        try:
            tw = draw.textlength(label)
        except Exception:
            tw = len(label) * 6
        draw.text((x + (cell - tw) / 2, pad), label, fill=(0, 0, 0))
    return canvas


def main():
    p = argparse.ArgumentParser(description="Quét guidance_scale (Diffusion img2img)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--content", required=True, help="Đường dẫn 1 ảnh content")
    p.add_argument("--style", required=True, help="Đường dẫn 1 ảnh style (tranh Van Gogh)")
    p.add_argument("--model_config", default="configs/model.yaml")
    p.add_argument("--guidance_scales", default="1,2,3,5",
                   help="Danh sách guidance_scale, phân tách bằng dấu phẩy")
    p.add_argument("--strength", type=float, default=0.6)
    p.add_argument("--guidance_rescale", type=float, default=0.7)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/sweep_guidance")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    scales = [float(s) for s in args.guidance_scales.split(",") if s.strip()]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    model, style_encoder = load_model(args.checkpoint, model_cfg, device)

    # Cảnh báo nếu null_style chưa train CFG → mọi guidance_scale sẽ cho cùng 1 ảnh
    null_norm = float(style_encoder.null_style.detach().abs().sum())
    if null_norm <= 1e-6:
        print("[sweep] ⚠ null_style chưa train CFG (|L1|≈0). Các mức guidance>1 "
              "sẽ KHÔNG khác s=1. Cần checkpoint train kèm style_dropout>0.")

    scheduler = DDPMScheduler(**model_cfg["diffusion"], device=device)
    sampler = DDIMSampler(scheduler, ddim_steps=args.ddim_steps)
    null_emb = style_encoder.null_style.detach().unsqueeze(0).to(device)

    c = load_image(args.content, args.image_size)
    s = load_image(args.style, args.image_size)

    with torch.no_grad():
        style_emb = style_encoder.encode_style(s.to(device))

    outputs = []
    for g in scales:
        torch.manual_seed(args.seed)              # CỐ ĐỊNH seed trước mỗi lần → cô lập guidance
        with torch.no_grad():
            o = sampler.sample_img2img(
                model, c, style_emb, device,
                strength=args.strength,
                null_style_emb=null_emb,
                guidance_scale=g,
                guidance_rescale=args.guidance_rescale,
            ).cpu()
        outputs.append((g, o))
        to_pil(o).save(out_dir / f"out_s{g:g}.png")
        print(f"[sweep] ✓ guidance_scale={g:g}")

    # ----- Ghép lưới có nhãn: Content | Style | s=... -----
    panels = [("Content", to_pil(c)), ("Style (Van Gogh)", to_pil(s))]
    panels += [(f"s = {g:g}", to_pil(o)) for g, o in outputs]

    grid = make_labeled_grid(panels)
    grid_path = out_dir / "grid_guidance.png"
    grid.save(grid_path)
    print(f"[sweep] ✓ Lưới đã lưu: {grid_path}")
    print(f"[sweep]   Chèn vào báo cáo: \\includegraphics[width=\\linewidth]{{figures/grid_guidance.png}}")


if __name__ == "__main__":
    main()
