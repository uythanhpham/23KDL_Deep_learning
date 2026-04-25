import os
import time
import json
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import lpips
from src.models.adain import AdaINStyleTransfer as AdaIN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AdaIN Style Transfer Results")
    parser.add_argument('--pred_dir', type=str, required=True, help="Thư mục chứa ảnh đã gen")
    parser.add_argument('--ref_dir', type=str, required=True, help="Thư mục chứa ảnh content gốc")
    parser.add_argument('--output_file', type=str, default='outputs/eval/metrics.json', help="Nơi lưu file JSON kết quả")
    return parser.parse_args()

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
        print(f"  [!] Lỗi file {os.path.basename(pred_path)}: {e}")
        return None

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    print(f"=========================================")
    print(f"[Eval] Đang đánh giá thư mục: {args.pred_dir}")
    
    files = [f for f in os.listdir(args.pred_dir) if f.endswith(('.jpg', '.png'))]
    if not files:
        print(f"[Lỗi] Thư mục {args.pred_dir} không có ảnh nào.")
        return

    lpips_list, ssim_list, rmse_list = [], [], []
    success_count = 0
    start_time = time.time()

    for i, f in enumerate(files):
        # Logic tách tên: result_content_00000_style_xxx...
        parts = f.split('_')
        if len(parts) < 3: continue
        content_id = f"{parts[1]}_{parts[2]}" 
        
        ref_file = ""
        for ext in ['.png', '.jpg', '.jpeg']:
            if os.path.exists(os.path.join(args.ref_dir, content_id + ext)):
                ref_file = content_id + ext
                break

        if ref_file:
            res = calculate_metrics(os.path.join(args.pred_dir, f), os.path.join(args.ref_dir, ref_file), lpips_fn, device)
            if res:
                lpips_list.append(res[0]); ssim_list.append(res[1]); rmse_list.append(res[2])
                success_count += 1
        
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  > Tiến độ: {i+1}/{len(files)} ảnh...")

    if success_count > 0:
        metrics = {
            "pred_dir": args.pred_dir,
            "images_evaluated": success_count,
            "avg_lpips": float(np.mean(lpips_list)),
            "avg_ssim": float(np.mean(ssim_list)),
            "avg_rmse": float(np.mean(rmse_list)),
            "eval_time_total_s": time.time() - start_time
        }
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f_json:
            json.dump(metrics, f_json, indent=4)
        print(f"[Thành công] Báo cáo lưu tại: {args.output_file}")
    else:
        print("[Lỗi] Không có ảnh nào được đánh giá thành công.")

if __name__ == "__main__":
    main()