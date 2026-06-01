import os
import argparse
import yaml
import random
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.models.unet import UNet
from src.models.style_encoder import StyleEncoder
from src.diffusion.scheduler import DDPMScheduler
from src.diffusion.ddim import DDIMSampler

# =====================================================================
# CÁC HÀM TIỆN ÍCH
# =====================================================================
def set_seed(seed: int):
    """Cố định seed để ảnh sinh ra mang tính tái lập."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_image(path: str, image_size: int, normalize_mode: str = "diffusion") -> torch.Tensor:
    """Đọc ảnh từ ổ cứng và chuẩn hóa thành Tensor shape (1, 3, H, W)."""
    img = Image.open(path).convert("RGB")
    
    # Ở giai đoạn Inference, ta ưu tiên CenterCrop để giữ trọng tâm ảnh
    transform_list = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ]
    
    if normalize_mode == "diffusion":
        # Chuẩn hóa về miền [-1, 1] cho UNet (và cho StyleEncoder vì nó tự convert bên trong)
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    elif normalize_mode == "imagenet":
        # Chuẩn hóa ImageNet nếu cần đưa trực tiếp vào VGG (hiện không dùng trực tiếp)
        transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        
    tf = transforms.Compose(transform_list)
    return tf(img).unsqueeze(0)  # Thêm chiều Batch (B=1)

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Đưa Tensor từ miền [-1, 1] của Diffusion về lại miền [0, 1] để lưu thành ảnh."""
    return (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0

def save_comparison_grid(content: torch.Tensor, style: torch.Tensor, output: torch.Tensor, save_path: str):
    """Lưu lưới 3 ảnh (Content, Style, Output) để dễ dàng so sánh bằng mắt."""
    # Đưa tất cả về [0, 1] trước khi gộp
    c_img = denormalize(content)
    s_img = denormalize(style)
    o_img = denormalize(output)
    
    grid = make_grid(torch.cat([c_img, s_img, o_img], dim=0), nrow=3, padding=2)
    save_image(grid, save_path)

def load_config(sample_yaml: str, model_yaml: str) -> dict:
    """Gộp cấu hình model và sample."""
    with open(sample_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(model_yaml, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    cfg.update(model_cfg)
    return cfg

def get_image_paths(path: str) -> list:
    """Hỗ trợ xử lý nếu đường dẫn là 1 file cụ thể HOẶC 1 thư mục chứa nhiều ảnh."""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1].lower() in valid_exts]
        return sorted(paths)
    return []

# =====================================================================
# HÀM LOAD MODEL VÀ SAMPLING LÕI
# =====================================================================
def load_model(checkpoint_path: str, model_cfg: dict, style_cfg: dict, device: str):
    """Khởi tạo mô hình và tải trọng số tốt nhất."""
    model = UNet(**model_cfg).to(device)
    style_encoder = StyleEncoder(**style_cfg).to(device)
    
    print(f"[*] Đang tải Checkpoint từ: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Luôn ưu tiên dùng EMA Model vì nó cho chất lượng ảnh mượt mà, sắc nét hơn
    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
        print("✓ Đã load trọng số từ EMA Model (Chất lượng cao).")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("✓ Đã load trọng số từ Model gốc.")
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    style_encoder.eval()
    return model, style_encoder

def run_sampling(model, style_encoder, scheduler, content_t, style_t, sampler_type, ddim_steps, mode, device, strength=0.6):
    """
    Thực thi quá trình sinh ảnh dựa trên Mode yêu cầu.
    
    Args:
        strength: Mức độ nhiễu thêm vào content (0.0 = giữ nguyên, 1.0 = nhiễu thuần).
                  Giá trị thấp hơn giữ cấu trúc content tốt hơn.
    """
    B, C, H, W = content_t.shape if content_t is not None else style_t.shape
    
    with torch.no_grad():
        # Trích xuất vector phong cách (Style Embedding)
        style_emb = style_encoder.encode_style(style_t.to(device))
        
        # -------------------------------------------------------------
        # MODE A: TỪ NHIỄU THUẦN TÚY (Chỉ cần Style, bỏ qua Content)
        # -------------------------------------------------------------
        if mode == "noise_to_stylized":
            if sampler_type == "ddim":
                sampler = DDIMSampler(scheduler, ddim_steps=ddim_steps)
                output = sampler.sample(model, (B, C, H, W), style_emb, device)
            else:
                output = scheduler.sample(model, (B, C, H, W), style_emb, device)
            return output.cpu()
            
        # -------------------------------------------------------------
        # MODE B: TỪ ẢNH GỐC (Giữ cấu trúc Content, đắp Style)
        # -------------------------------------------------------------
        elif mode == "content_to_stylized":
            assert content_t is not None, "Chế độ content_to_stylized yêu cầu phải có ảnh content!"
            # Sử dụng DDIM img2img thay vì DDPM step-by-step (nhanh gấp ~12x)
            sampler = DDIMSampler(scheduler, ddim_steps=ddim_steps)
            output = sampler.sample_img2img(model, content_t, style_emb, device, strength=strength)
            return output.cpu()

# =====================================================================
# HÀM CHÍNH (MAIN PROCESS)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Sinh ảnh với Style-guided Diffusion")
    parser.add_argument("--sample_config", type=str, default="configs/sample.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml")
    args = parser.parse_args()
    
    # 1. Khởi tạo cấu hình và thiết bị
    cfg = load_config(args.sample_config, args.model_config)
    s_cfg = cfg["sample"]
    set_seed(s_cfg.get("seed", 42))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Khởi động Sampling trên thiết bị: {device.upper()}")
    
    # 2. Tải Mô hình & Scheduler
    model, style_encoder = load_model(s_cfg["checkpoint"], cfg["model"], cfg["style_encoder"], device)
    scheduler = DDPMScheduler(**cfg["diffusion"], device=device)
    
    os.makedirs(s_cfg["output_dir"], exist_ok=True)
    
    # 3. Lọc danh sách file Content và Style
    content_paths = get_image_paths(s_cfg["content_path"])
    style_paths = get_image_paths(s_cfg["style_path"])
    
    # Nếu Mode là noise_to_stylized thì chỉ quan tâm style_paths
    if len(style_paths) == 0:
        raise FileNotFoundError(f"Không tìm thấy ảnh style nào tại: {s_cfg['style_path']}")
        
    num_samples = max(len(content_paths), len(style_paths))
    image_size = s_cfg.get("image_size", 256)
    strength = s_cfg.get("strength", 0.6)
    
    print("\n" + "="*50)
    print(f"BẮT ĐẦU SINH ẢNH (Mode: {s_cfg['mode']} | Sampler: {s_cfg['sampler']} | Strength: {strength})")
    print("="*50 + "\n")
    
    # 4. Vòng lặp lấy mẫu (Inference Loop)
    for i in range(num_samples):
        # Chọn cặp ảnh (nếu 1 bên ít hơn sẽ xoay vòng - modulo)
        c_path = content_paths[i % len(content_paths)] if len(content_paths) > 0 else None
        s_path = style_paths[i % len(style_paths)]
        
        # Load Content và Style bằng hệ số chuẩn hóa của Diffusion [-1, 1]
        c_tensor = load_image(c_path, image_size, "diffusion") if c_path else None
        s_tensor = load_image(s_path, image_size, "diffusion")
        
        print(f"[{i+1}/{num_samples}] Content: {os.path.basename(c_path) if c_path else 'NOISE'} | Style: {os.path.basename(s_path)}")
        
        # Sinh ảnh
        output_t = run_sampling(
            model=model,
            style_encoder=style_encoder,
            scheduler=scheduler,
            content_t=c_tensor,
            style_t=s_tensor,
            sampler_type=s_cfg["sampler"],
            ddim_steps=s_cfg["ddim_steps"],
            mode=s_cfg["mode"],
            device=device,
            strength=strength
        )
        
        # Lưu kết quả đầu ra độc lập
        out_name = f"output_{i:03d}.png"
        out_path = os.path.join(s_cfg["output_dir"], out_name)
        save_image(denormalize(output_t), out_path)
        
        # Lưu ảnh đối chiếu (Grid 3 ảnh) nếu được yêu cầu
        if s_cfg.get("save_grid", False) and c_tensor is not None:
            grid_name = f"grid_{i:03d}.png"
            grid_path = os.path.join(s_cfg["output_dir"], grid_name)
            save_comparison_grid(c_tensor, s_tensor, output_t, grid_path)
            
    # 5. Hoàn tất
    print(f"\n[✓] Đã sinh xong {num_samples} ảnh tại thư mục: {s_cfg['output_dir']}")

if __name__ == "__main__":
    main()