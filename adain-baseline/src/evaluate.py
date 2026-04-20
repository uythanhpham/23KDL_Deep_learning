"""Chạy test set và tính metric."""
"""
[GĐ 2] Giai đoạn 2 — Hoàn thiện khung song song
File: src/evaluate.py
Mục đích: Skeleton cho quá trình evaluate. Đọc ảnh từ infer output và ảnh gốc (content/style),
tính toán các dummy metrics (sau này sẽ thay bằng hàm loss thật), và lưu kết quả ra file JSON.
"""

import os
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate skeleton cho AdaIN (Smoke Test)")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Đường dẫn file config')
    parser.add_argument('--pred_dir', type=str, default='outputs/infer_smoke', help='Thư mục chứa ảnh do mô hình sinh ra')
    parser.add_argument('--ref_dir', type=str, default='debug_data', help='Thư mục chứa ảnh tham chiếu gốc (content/style)')
    parser.add_argument('--output_file', type=str, default='outputs/eval/metrics_smoke.json', help='File JSON lưu kết quả')
    return parser.parse_args()

def dummy_compute_metrics(pred_path, ref_dir):
    """
    Hàm tính toán metric giả (Mock).
    Ở GĐ 4, ta sẽ thay hàm này bằng PerceptualLoss, ContentLoss... import từ src.losses
    """
    # Giả lập các chỉ số đánh giá cho phong cách (style transfer)
    return {
        "content_loss": 1.234,
        "style_loss": 2.345,
        "total_loss": 3.579,
        "psnr": 24.5  # Peak Signal-to-Noise Ratio giả định
    }

def main():
    args = parse_args()
    
    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)
    
    print(f"\n[Eval] Bắt đầu chạy Evaluate (Smoke Test)...")
    print(f"[Eval] Thư mục ảnh dự đoán : {pred_dir}")
    print(f"[Eval] Thư mục ảnh gốc      : {ref_dir}")
    
    # 1. Kiểm tra thư mục đầu vào
    if not pred_dir.exists():
        print(f"[Lỗi] Không tìm thấy thư mục chứa ảnh dự đoán: {pred_dir}")
        print("Gợi ý: Hãy chạy lại luồng Infer ở GĐ1 để sinh ảnh trước.")
        return
        
    pred_files = list(pred_dir.glob('*.jpg')) + list(pred_dir.glob('*.png'))
    if not pred_files:
        print(f"[Lỗi] Không có tấm ảnh nào trong {pred_dir} để đánh giá.")
        return
        
    # 2. Quét file và tính toán metric
    results = {}
    total_metrics = {"content_loss": 0.0, "style_loss": 0.0, "total_loss": 0.0, "psnr": 0.0}
    
    for img_path in pred_files:
        img_name = img_path.name
        print(f"  -> Đang đo lường cho: {img_name}")
        
        # Lấy metric giả
        metrics = dummy_compute_metrics(img_path, ref_dir)
        results[img_name] = metrics
        
        # Cộng dồn để lát tính trung bình
        for k, v in metrics.items():
            total_metrics[k] += v
            
    # 3. Tính chỉ số trung bình (Average) cho toàn bộ batch ảnh
    num_imgs = len(pred_files)
    avg_metrics = {k: round(v / num_imgs, 4) for k, v in total_metrics.items()}
    
    report = {
        "average_metrics": avg_metrics,
        "detailed_results": results
    }
    
    # 4. Lưu kết quả ra file JSON
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print(f"\n[Thành công] Đã lưu báo cáo đánh giá JSON tại: {args.output_file}")
    print(f"[Eval] Kết quả trung bình: {avg_metrics}\n")

if __name__ == '__main__':
    main()