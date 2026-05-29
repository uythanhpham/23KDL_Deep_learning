import os
import time
import json
import argparse
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# =====================================================================
# NHẬP THƯ VIỆN ĐÁNH GIÁ (KÈM BẪY LỖI NẾU THIẾU)
# =====================================================================
try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import mean_squared_error as mse_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[CẢNH BÁO] Không tìm thấy thư viện scikit-image. Tính năng SSIM và RMSE sẽ bị vô hiệu hóa.")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[CẢNH BÁO] Không tìm thấy thư viện lpips. Tính năng LPIPS (Perceptual Loss) sẽ bị vô hiệu hóa.")

try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("[CẢNH BÁO] Không tìm thấy thư viện pytorch-fid. Tính năng FID Score sẽ bị vô hiệu hóa.")


# =====================================================================
# HÀM TÍNH TOÁN CÁC CHỈ SỐ: SSIM, RMSE, LPIPS
# =====================================================================
def calculate_metrics(pred_dir: str, ref_dir: str, device: str) -> dict:
    global HAS_LPIPS
    """
    So sánh ảnh sinh ra (pred) và ảnh gốc (ref/content) để đo lường
    mức độ giữ lại cấu trúc nội dung.
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    
    # Lấy danh sách ảnh, bỏ qua các ảnh grid
    pred_paths = sorted([p for p in glob(os.path.join(pred_dir, "*")) if p.lower().endswith(valid_exts) and "grid" not in os.path.basename(p)])
    ref_paths = sorted([p for p in glob(os.path.join(ref_dir, "*")) if p.lower().endswith(valid_exts)])
    
    num_images = min(len(pred_paths), len(ref_paths))
    if num_images == 0:
        return {"num_images": 0, "avg_ssim": "N/A", "avg_rmse": "N/A", "avg_lpips": "N/A"}

    # Khởi tạo mô hình LPIPS nếu có
    lpips_model = None
    if HAS_LPIPS:
        try:
            # Sử dụng mạng VGG mặc định của LPIPS
            lpips_model = lpips.LPIPS(net='vgg').to(device)
            lpips_model.eval()
        except Exception as e:
            print(f"[LỖI] Không thể load mô hình LPIPS: {e}")
            HAS_LPIPS = False

    # Pipeline chuyển ảnh cho LPIPS (chuẩn hóa về miền [-1, 1])
    transform_lpips = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    ssim_scores, rmse_scores, lpips_scores = [], [], []

    print(f"[*] Đang tính toán SSIM, RMSE, LPIPS cho {num_images} cặp ảnh...")
    
    # Duyệt qua từng cặp ảnh (ghép theo thứ tự index)
    for i in range(num_images):
        pred_img = Image.open(pred_paths[i]).convert('RGB')
        ref_img = Image.open(ref_paths[i]).convert('RGB')
        
        # Đảm bảo 2 ảnh cùng kích thước trước khi đo
        if pred_img.size != ref_img.size:
            ref_img = ref_img.resize(pred_img.size, Image.Resampling.LANCZOS)
            
        # Tính SSIM & RMSE (Yêu cầu Numpy array)
        if HAS_SKIMAGE:
            pred_np = np.array(pred_img)
            ref_np = np.array(ref_img)
            
            try:
                # channel_axis=-1 báo cho hàm biết ảnh có cấu trúc (H, W, C)
                ssim_val = ssim_fn(ref_np, pred_np, channel_axis=-1, data_range=255)
                rmse_val = np.sqrt(mse_fn(ref_np, pred_np))
                ssim_scores.append(ssim_val)
                rmse_scores.append(rmse_val)
            except Exception as e:
                print(f"[LỖI] Tính SSIM/RMSE thất bại tại ảnh {i}: {e}")
                
        # Tính LPIPS (Yêu cầu PyTorch Tensor trên GPU/CPU)
        if HAS_LPIPS and lpips_model is not None:
            pred_t = transform_lpips(pred_img).unsqueeze(0).to(device)
            ref_t = transform_lpips(ref_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                try:
                    # LPIPS trả về 1 tensor chứa giá trị loss
                    lpips_val = lpips_model(ref_t, pred_t).item()
                    lpips_scores.append(lpips_val)
                except Exception as e:
                    print(f"[LỖI] Tính LPIPS thất bại tại ảnh {i}: {e}")

    # Tổng hợp kết quả trung bình
    results = {"num_images": num_images}
    results["avg_ssim"] = round(np.mean(ssim_scores), 4) if len(ssim_scores) > 0 else "N/A"
    results["avg_rmse"] = round(np.mean(rmse_scores), 4) if len(rmse_scores) > 0 else "N/A"
    results["avg_lpips"] = round(np.mean(lpips_scores), 4) if len(lpips_scores) > 0 else "N/A"
    
    return results


# =====================================================================
# HÀM TÍNH TOÁN FID SCORE (Đo khoảng cách phân phối ảnh)
# =====================================================================
def compute_fid_score(pred_dir: str, ref_dir: str, device: str) -> float:
    """
    Đo khoảng cách Frechet Inception Distance giữa tập ảnh sinh ra và tập gốc.
    """
    if not HAS_FID:
        return "N/A"
        
    print(f"[*] Đang tính toán FID Score giữa 2 thư mục...")
    try:
        # Pytorch_FID dùng string "cuda" thay vì torch.device
        dev_str = "cuda:0" if "cuda" in device else "cpu"
        
        fid_value = fid_score.calculate_fid_given_paths(
            paths=[ref_dir, pred_dir],
            batch_size=8,
            device=dev_str,
            dims=2048 # Trích xuất feature từ layer Pool3 của InceptionV3
        )
        return round(fid_value, 4)
    except Exception as e:
        print(f"[LỖI] Lỗi trong quá trình tính FID: {e}")
        return "N/A"


# =====================================================================
# HÀM CHÍNH
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Style-guided Diffusion")
    parser.add_argument("--pred_dir", type=str, required=True, help="Thư mục chứa ảnh đã sinh (Output)")
    parser.add_argument("--ref_dir", type=str, required=True, help="Thư mục chứa ảnh gốc (Content)")
    parser.add_argument("--style_dir", type=str, default=None, help="[Dự trữ] Thư mục chứa ảnh style để đo Style Metric trong tương lai")
    parser.add_argument("--output_file", type=str, default="outputs/eval/metrics.json", help="File JSON lưu kết quả")
    args = parser.parse_args()

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang chạy Evaluate trên thiết bị: {device.upper()}")
    
    # Đảm bảo thư mục output tồn tại
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 1. Tính SSIM, RMSE, LPIPS (Đo lường từng cặp ảnh)
    metrics_dict = calculate_metrics(args.pred_dir, args.ref_dir, device)
    
    # 2. Tính FID (Đo lường toàn bộ phân phối ảnh)
    fid_val = compute_fid_score(args.pred_dir, args.ref_dir, device)
    
    # 3. Tổng hợp thời gian
    eval_time = round(time.time() - start_time, 2)
    
    # 4. Gom dữ liệu để xuất JSON
    final_report = {
        "pred_dir": args.pred_dir,
        "ref_dir": args.ref_dir,
        "num_images": metrics_dict["num_images"],
        "avg_ssim": metrics_dict["avg_ssim"],
        "avg_rmse": metrics_dict["avg_rmse"],
        "avg_lpips": metrics_dict["avg_lpips"],
        "fid": fid_val,
        "eval_time_s": eval_time
    }
    
    # Lưu JSON
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)

    # 5. In bảng kết quả ra Terminal
    print("\n" + "="*50)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ (EVALUATION METRICS)")
    print("="*50)
    print(f" Số lượng cặp ảnh : {final_report['num_images']}")
    print(f" Thư mục Predicted: {final_report['pred_dir']}")
    print(f" Thư mục Reference: {final_report['ref_dir']}")
    print("-" * 50)
    print(f" SSIM (↑ Tốt hơn)   : {final_report['avg_ssim']}")
    print(f" RMSE (↓ Tốt hơn)   : {final_report['avg_rmse']}")
    print(f" LPIPS (↓ Tốt hơn)  : {final_report['avg_lpips']}")
    print(f" FID (↓ Tốt hơn)    : {final_report['fid']}")
    print("-" * 50)
    print(f" Thời gian chạy     : {final_report['eval_time_s']} giây")
    print(f" Đã lưu báo cáo tại : {args.output_file}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()