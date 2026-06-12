"""
========================================================================
infer.py — CLI SINH ẢNH (INFERENCE) cho Diffusion  [RIÊNG cho diffusion]
========================================================================
Sinh N ảnh stylized rồi lưu 2 thư mục KHỚP TÊN 1-1:
    <out_dir>/content/0000.png   (ảnh content đã resize — đầu vào model)
    <out_dir>/output/0000.png    (ảnh output stylized)
→ Đưa thẳng sang CLI chung `src.evaluate` để chấm điểm.

Tách rời INFERENCE (per-model) khỏi EVALUATE (dùng chung 3 model).
CycleGAN/AdaIN tự có file infer riêng; chỉ cần xuất ra cùng cấu trúc content/ + output/.

Cách dùng:
    python -m src.infer \
        --checkpoint <ckpt.pth> --content_dir <testA> --style_dir <testB> \
        --out_dir outputs/eval_out --num_samples 200 \
        --strength 0.8 --guidance_scale 2.0 --guidance_rescale 0.7 --ddim_steps 50
========================================================================
"""
import os
import argparse
import random
from pathlib import Path

import yaml
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.models.unet import UNet
from src.models.style_encoder import StyleEncoder
from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.ddim import DDIMSampler

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in VALID_EXT)


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1,1]
    ])
    return tf(img).unsqueeze(0)


def denormalize(t):
    return (t.clamp(-1.0, 1.0) + 1.0) / 2.0


def load_model(checkpoint, model_cfg, device):
    model = UNet(**model_cfg["model"]).to(device)
    se = StyleEncoder(**model_cfg["style_encoder"]).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ("ema_model" in ckpt or "model" in ckpt):
        sd = ckpt.get("ema_model", ckpt.get("model"))
        model.load_state_dict(sd)
        if "style_encoder" in ckpt:
            se.load_state_dict(ckpt["style_encoder"], strict=False)  # ckpt cũ chưa có null_style
        print(f"[infer] Loaded {'EMA' if 'ema_model' in ckpt else 'model'} | epoch={ckpt.get('epoch','?')}")
    else:
        model.load_state_dict(ckpt)
        print("[infer] Loaded raw state_dict (⚠ không có style_encoder)")
    model.eval(); se.eval()
    return model, se


def main():
    p = argparse.ArgumentParser(description="Sinh ảnh stylized (Diffusion) cho evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--content_dir", required=True)
    p.add_argument("--style_dir", required=True, help="Tập Van Gogh (để bốc style ngẫu nhiên)")
    p.add_argument("--out_dir", default="outputs/eval_out")
    p.add_argument("--model_config", default="configs/model.yaml")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--mode", choices=["img2img", "noise"], default="img2img")
    p.add_argument("--strength", type=float, default=0.8)
    p.add_argument("--guidance_scale", type=float, default=2.0)
    p.add_argument("--guidance_rescale", type=float, default=0.7)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    model, style_encoder = load_model(args.checkpoint, model_cfg, device)
    null_norm = float(style_encoder.null_style.detach().abs().sum())
    g = args.guidance_scale
    if null_norm <= 1e-6 and g != 1.0:
        print(f"[infer] ⚠ null_style chưa train CFG (|L1|≈0) → ép guidance_scale=1.0")
        g = 1.0

    scheduler = DDPMScheduler(**model_cfg["diffusion"], device=device)
    sampler = DDIMSampler(scheduler, ddim_steps=args.ddim_steps)
    null_emb = style_encoder.null_style.detach().unsqueeze(0).to(device)

    c_dir = Path(args.out_dir) / "content"; c_dir.mkdir(parents=True, exist_ok=True)
    o_dir = Path(args.out_dir) / "output";  o_dir.mkdir(parents=True, exist_ok=True)

    content_paths = list_images(args.content_dir)
    style_paths = list_images(args.style_dir)
    if not content_paths or not style_paths:
        raise FileNotFoundError("content_dir/style_dir rỗng.")

    n = min(args.num_samples, len(content_paths))
    random.seed(args.seed)
    print(f"[infer] Sinh {n} ảnh | mode={args.mode} strength={args.strength} "
          f"guidance={g} rescale={args.guidance_rescale} ddim={args.ddim_steps}")

    for i in range(n):
        cpath = content_paths[i % len(content_paths)]
        spath = random.choice(style_paths)            # đa dạng style → phủ phân phối Van Gogh
        c = load_image(cpath, args.image_size)
        s = load_image(spath, args.image_size)
        torch.manual_seed(args.seed + i)              # đa dạng seed
        with torch.no_grad():
            style_emb = style_encoder.encode_style(s.to(device))
            if args.mode == "img2img":
                o = sampler.sample_img2img(model, c, style_emb, device, strength=args.strength,
                                           null_style_emb=null_emb, guidance_scale=g,
                                           guidance_rescale=args.guidance_rescale).cpu()
            else:
                B, C, H, W = c.shape
                o = sampler.sample(model, (B, C, H, W), style_emb, device,
                                   null_style_emb=null_emb, guidance_scale=g,
                                   guidance_rescale=args.guidance_rescale).cpu()
        save_image(denormalize(c), c_dir / f"{i:04d}.png")
        save_image(denormalize(o), o_dir / f"{i:04d}.png")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n}")

    print(f"[infer] ✓ Xong. content→ {c_dir} | output→ {o_dir}")
    print(f"[infer]   Chấm điểm: python -m src.evaluate --pred_dir {o_dir} --ref_dir {c_dir} "
          f"--style_dir {args.style_dir} --model_name diffusion")


if __name__ == "__main__":
    main()
