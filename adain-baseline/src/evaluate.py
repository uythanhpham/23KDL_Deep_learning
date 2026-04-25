import os
import time
import json
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import lpips

# Import từ local
from src.models.adain import AdaIN

def calculate_metrics(pred_path, ref_path, lpips_fn, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    try:
        pred_img = transform(Image.open(pred_path).convert('RGB')).unsqueeze(0).to(device)
        ref_img = transform(Image.open(ref_path).convert('RGB')).unsqueeze(0).to(device)

        lpips_val = lpips_fn(pred_img, ref_img).item()
        pred_np = pred_img.squeeze().cpu().numpy().transpose(1, 2, 0)
        ref_np = ref_img.squeeze().cpu().numpy().transpose(1, 2, 0)

        s_val = ssim(pred_np, ref_np, data_range=1, channel_axis=2)
        r_val = np.sqrt(mean_squared_error(pred_np, ref_np))
        return lpips_val, s_val, r_val
    except Exception as e:
        print(f"  [!] Lỗi khi xử lý file: {e}")
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    # Đường dẫn
    pred_dir = "outputs/infer_final_watercolor"
    ref_dir  = "data/processed/test/content"
    
    # Lấy danh sách ảnh THỰC TẾ đang có trong folder output
    files = [f for f in os.listdir(pred_dir) if f.endswith(('.jpg', '.png'))]
    num_files = len(files)

    if num_files == 0:
        print("[Lỗi] Thư mục output trống rỗng. Hãy gen ít nhất vài tấm đã nhé!")
        return

    print(f"=========================================")
    print(f"[Eval] Tìm thấy {num_files} ảnh đã gen. Bắt đầu đánh giá...")
    print(f"=========================================")

    lpips_list, ssim_list, rmse_list = [], [], []
    success_count = 0
    start_time = time.time()

    for i, f in enumerate(files):
        # Tách tên: result_content_00000_style_watercolor_00000.jpg
        parts = f.split('_')
        if len(parts) < 3: continue
        
        content_id = f"{parts[1]}_{parts[2]}" # content_00000
        
        # Thử tìm file gốc .png hoặc .jpg
        ref_file = ""
        for ext in ['.png', '.jpg', '.jpeg']:
            if os.path.exists(os.path.join(ref_dir, content_id + ext)):
                ref_file = content_id + ext
                break

        if ref_file:
            res = calculate_metrics(os.path.join(pred_dir, f), os.path.join(ref_dir, ref_file), lpips_fn, device)
            if res:
                lpips_list.append(res[0]); ssim_list.append(res[1]); rmse_list.append(res[2])
                success_count += 1
        
        if (i + 1) % 10 == 0 or (i + 1) == num_files:
            print(f"  > Đã xong {i+1}/{num_files} ảnh...")

    # Tính toán kết quả cuối
    metrics = {
        "images_evaluated": success_count,
        "avg_lpips": float(np.mean(lpips_list)) if lpips_list else 0,
        "avg_ssim": float(np.mean(ssim_list)) if ssim_list else 0,
        "avg_rmse": float(np.mean(rmse_list)) if rmse_list else 0,
        "eval_time_per_image": (time.time() - start_time) / success_count if success_count > 0 else 0
    }

    os.makedirs("outputs/eval", exist_ok=True)
    with open("outputs/eval/metrics.json", "w") as f_json:
        json.dump(metrics, f_json, indent=4)

    print(f"=========================================")
    print(f"[Thành công] Đã đánh giá {success_count} ảnh.")
    print(f"Kết quả: SSIM={metrics['avg_ssim']:.4f}, LPIPS={metrics['avg_lpips']:.4f}")
    print(f"=========================================")

if __name__ == "__main__":
    main()