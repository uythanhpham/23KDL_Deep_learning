"""
==========================================================
LOCAL EVALUATE — Chạy trên máy local (CPU/GPU)
==========================================================
Đặt file này trong thư mục diffusion-baseline/ rồi chạy:
    python evaluate_local.py

Yêu cầu: pip install lpips scikit-image pytorch-fid
==========================================================
"""

import os
import sys
import shutil
import json
import time
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# =====================================================================
# CẤU HÌNH — Chỉnh sửa đường dẫn ở đây
# =====================================================================
CHECKPOINT   = r"diffusion-baseline/checkpoints/best_model.pth"
CONTENT_DIR  = r"data/archive/testA"
STYLE_DIR    = r"data/archive/testB"
OUTPUT_DIR   = r"diffusion-baseline/outputs/eval_local"
NUM_SAMPLES  = 15
SEED         = 42

# =====================================================================
# IMPORT CÁC MODULE TỪ PROJECT
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models.unet import UNet
from src.models.style_encoder import StyleEncoder
from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.ddim import DDIMSampler

# Đọc config từ repo
import yaml
with open(os.path.join(PROJECT_ROOT, "configs/model.yaml"), "r", encoding="utf-8") as f:
    model_cfg = yaml.safe_load(f)
with open(os.path.join(PROJECT_ROOT, "configs/sample.yaml"), "r", encoding="utf-8") as f:
    sample_cfg = yaml.safe_load(f)["sample"]

# Lấy thông số sampling từ config
IMAGE_SIZE  = sample_cfg.get("image_size", 256)
MODE        = sample_cfg.get("mode", "content_to_stylized")
SAMPLER     = sample_cfg.get("sampler", "ddim")
DDIM_STEPS  = sample_cfg.get("ddim_steps", 100)
STRENGTH    = sample_cfg.get("strength", 0.6)


# =====================================================================
# HÀM TIỆN ÍCH
# =====================================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_image(path, image_size):
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return tf(img).unsqueeze(0)

def denormalize(tensor):
    return (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0

def get_image_paths(directory, limit=None):
    valid = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in valid
    ])
    return paths[:limit] if limit else paths


# =====================================================================
# BƯỚC 1: LOAD MODEL
# =====================================================================
def load_model(device):
    print("\n" + "="*60)
    print("BƯỚC 1: LOAD MODEL")
    print("="*60)

    model = UNet(**model_cfg["model"]).to(device)
    style_encoder = StyleEncoder(**model_cfg["style_encoder"]).to(device)

    print(f"Đang load checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)

    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
        print("✓ Loaded EMA model")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("✓ Loaded model")
    else:
        model.load_state_dict(ckpt)
        print("✓ Loaded raw state_dict")

    model.eval()
    style_encoder.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ UNet: {n_params:.2f}M params")
    return model, style_encoder


# =====================================================================
# BƯỚC 2: SINH ẢNH
# =====================================================================
def generate_images(model, style_encoder, device):
    print("\n" + "="*60)
    print(f"BƯỚC 2: SINH {NUM_SAMPLES} ẢNH")
    print(f"Mode: {MODE} | Sampler: {SAMPLER} {DDIM_STEPS} steps | Strength: {STRENGTH}")
    print("="*60)

    scheduler = DDPMScheduler(**model_cfg["diffusion"], device=device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    content_paths = get_image_paths(CONTENT_DIR, NUM_SAMPLES)
    style_paths   = get_image_paths(STYLE_DIR, NUM_SAMPLES)

    output_paths = []
    ref_paths = []

    for i in range(NUM_SAMPLES):
        c_path = content_paths[i % len(content_paths)]
        s_path = style_paths[i % len(style_paths)]

        c_tensor = load_image(c_path, IMAGE_SIZE)
        s_tensor = load_image(s_path, IMAGE_SIZE)

        print(f"  [{i+1}/{NUM_SAMPLES}] Content: {os.path.basename(c_path)} | Style: {os.path.basename(s_path)}")

        with torch.no_grad():
            style_emb = style_encoder.encode_style(s_tensor.to(device))
            B, C, H, W = c_tensor.shape

            if MODE == "content_to_stylized":
                T = scheduler.num_timesteps
                t_max = max(1, min(int(T * STRENGTH), T - 1))
                x_t, _ = scheduler.add_noise(c_tensor.to(device), torch.tensor([t_max], device=device))

                for t_val in tqdm(reversed(range(t_max)), desc=f"  Denoising {i+1}", total=t_max, leave=False):
                    t_batch = torch.full((1,), t_val, device=device, dtype=torch.long)
                    eps = model(x_t, t_batch, style_emb)
                    x_t = scheduler.step(eps, t_val, x_t)
                output = x_t.cpu()

            else:  # noise_to_stylized
                if SAMPLER == "ddim":
                    sampler = DDIMSampler(scheduler, ddim_steps=DDIM_STEPS)
                    output = sampler.sample(model, (B, C, H, W), style_emb, device).cpu()
                else:
                    output = scheduler.sample(model, (B, C, H, W), style_emb, device).cpu()

        # Lưu output
        out_path = os.path.join(OUTPUT_DIR, f"output_{i:03d}.png")
        save_image(denormalize(output), out_path)
        output_paths.append(out_path)
        ref_paths.append(c_path)

        # Lưu grid (content | style | output)
        grid = make_grid(torch.cat([
            denormalize(c_tensor),
            denormalize(s_tensor),
            denormalize(output)
        ], dim=0), nrow=3, padding=2)
        save_image(grid, os.path.join(OUTPUT_DIR, f"grid_{i:03d}.png"))

    print(f"\n✓ Đã sinh {NUM_SAMPLES} ảnh tại: {OUTPUT_DIR}")
    return output_paths, ref_paths


# =====================================================================
# BƯỚC 3: TÍNH METRICS
# =====================================================================
def compute_metrics(output_paths, ref_paths):
    print("\n" + "="*60)
    print("BƯỚC 3: ĐÁNH GIÁ METRICS")
    print("="*60)

    # Import metrics
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage.metrics import mean_squared_error as mse_fn
        has_skimage = True
    except ImportError:
        has_skimage = False
        print("[CẢNH BÁO] Thiếu scikit-image → bỏ qua SSIM/RMSE")

    try:
        import lpips
        lpips_model = lpips.LPIPS(net='vgg').eval()
        has_lpips = True
    except ImportError:
        has_lpips = False
        print("[CẢNH BÁO] Thiếu lpips → bỏ qua LPIPS")

    ssim_scores, rmse_scores, lpips_scores = [], [], []

    transform_lpips = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    for i, (pred_path, ref_path) in enumerate(zip(output_paths, ref_paths)):
        pred_img = Image.open(pred_path).convert("RGB")
        ref_img  = Image.open(ref_path).convert("RGB")

        if pred_img.size != ref_img.size:
            ref_img = ref_img.resize(pred_img.size, Image.Resampling.LANCZOS)

        if has_skimage:
            pred_np = np.array(pred_img)
            ref_np  = np.array(ref_img)
            ssim_val = ssim_fn(ref_np, pred_np, channel_axis=-1, data_range=255)
            rmse_val = np.sqrt(mse_fn(ref_np, pred_np))
            ssim_scores.append(ssim_val)
            rmse_scores.append(rmse_val)

        if has_lpips:
            pred_t = transform_lpips(pred_img).unsqueeze(0)
            ref_t  = transform_lpips(ref_img).unsqueeze(0)
            with torch.no_grad():
                lpips_val = lpips_model(ref_t, pred_t).item()
                lpips_scores.append(lpips_val)

    results = {"num_images": len(output_paths)}
    results["avg_ssim"]  = round(np.mean(ssim_scores), 4)  if ssim_scores  else "N/A"
    results["avg_rmse"]  = round(np.mean(rmse_scores), 4)  if rmse_scores  else "N/A"
    results["avg_lpips"] = round(np.mean(lpips_scores), 4) if lpips_scores else "N/A"
    return results


# =====================================================================
# BƯỚC 4: HIỂN THỊ KẾT QUẢ
# =====================================================================
def display_results(results):
    def comment(metric, val):
        if isinstance(val, str): return "-"
        if metric == "SSIM":  return "Tốt" if val > 0.7 else "Chấp nhận" if val >= 0.5 else "Cần train thêm"
        if metric == "RMSE":  return "Tốt" if val < 20  else "Chấp nhận" if val <= 40  else "Cần train thêm"
        if metric == "LPIPS": return "Tốt" if val < 0.2 else "Chấp nhận" if val <= 0.4 else "Cần train thêm"
        return "-"

    border = "+" + "-"*60 + "+"
    sep    = "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*28 + "+"

    print("\n" + border)
    print(f"| {'KẾT QUẢ ĐÁNH GIÁ':^58} |")
    print(sep)
    print(f"| {'Metric':^12} | {'Value':^12} | {'Nhận xét':^26} |")
    print(sep)

    for name, key in [("SSIM ↑", "avg_ssim"), ("RMSE ↓", "avg_rmse"), ("LPIPS ↓", "avg_lpips")]:
        val = results.get(key, "N/A")
        if isinstance(val, (int, float)):
            print(f"| {name:<12} | {val:^12.4f} | {comment(name.split()[0], val):<26} |")
        else:
            print(f"| {name:<12} | {'N/A':^12} | {'-':<26} |")

    print(border)
    print(f"| {'Số ảnh':^12} | {results['num_images']:^12} | {'Mode: ' + MODE:<26} |")
    print(border)

    # Lưu JSON
    json_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Đã lưu metrics tại: {json_path}")


# =====================================================================
# BƯỚC 5: HIỂN THỊ ẢNH (matplotlib)
# =====================================================================
def show_grids():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[CẢNH BÁO] Thiếu matplotlib → bỏ qua hiển thị ảnh")
        return

    grid_files = sorted(Path(OUTPUT_DIR).glob("grid_*.png"))
    if not grid_files:
        print("Không tìm thấy grid ảnh.")
        return

    n = min(8, len(grid_files))
    print(f"\n✓ Hiển thị {n} grid ảnh...")

    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
    fig.patch.set_facecolor("#1a1a2e")
    plt.rcParams["text.color"] = "white"

    if n == 1: axes = [axes]

    for col, title in enumerate(["Content", "Style", "Output"]):
        axes[0][col].set_title(title, fontsize=13, color="white", pad=12)

    for row in range(n):
        img = Image.open(grid_files[row]).convert("RGB")
        W = img.width // 3
        crops = [
            img.crop((0, 0, W, img.height)),
            img.crop((W, 0, 2*W, img.height)),
            img.crop((2*W, 0, img.width, img.height)),
        ]
        for col, crop in enumerate(crops):
            axes[row][col].imshow(np.array(crop))
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            for spine in axes[row][col].spines.values(): spine.set_visible(False)

        axes[row][0].set_ylabel(f"#{row+1}", fontsize=11, color="white",
                                 rotation=0, labelpad=25, va="center")

    fig.suptitle(f"Style-guided Diffusion | {MODE} | {SAMPLER.upper()} {DDIM_STEPS} steps",
                 fontsize=15, fontweight='bold', color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "visual_comparison.png")
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: {save_path}")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    start = time.time()
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, style_encoder = load_model(device)
    output_paths, ref_paths = generate_images(model, style_encoder, device)
    results = compute_metrics(output_paths, ref_paths)
    results["eval_time_s"] = round(time.time() - start, 2)
    display_results(results)
    show_grids()

    print(f"\n🏁 Tổng thời gian: {results['eval_time_s']}s")
