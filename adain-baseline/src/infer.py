"""Dùng model sinh thử 1 ảnh."""
"""
Dùng model sinh thử 1 ảnh.
[GĐ 1] Luồng infer skeleton: load content/style -> gọi model (mock/untrained) -> lưu ảnh output.
"""

import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
import yaml

# Cố gắng import model thật, nếu chưa có sẽ dùng Mock Model để đảm bảo luồng chạy không bị crash
try:
    from src.models.adain import AdaIN
except ImportError:
    import torch.nn as nn
    class AdaIN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, content, style, *args, **kwargs):
            # Mock behavior: Trả về ảnh hòa trộn 50% content và 50% style để test luồng
            return (content + style) / 2.0

def parse_args():
    parser = argparse.ArgumentParser(description="Inference skeleton cho AdaIN (Smoke Test)")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Đường dẫn đến file config')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mock.pth', help='Đường dẫn đến model checkpoint')
    parser.add_argument('--content_dir', type=str, default='debug_data/content', help='Thư mục chứa ảnh content mock')
    parser.add_argument('--style_dir', type=str, default='debug_data/style', help='Thư mục chứa ảnh style mock')
    parser.add_argument('--output_dir', type=str, default='outputs/infer_smoke', help='Thư mục lưu ảnh output')
    return parser.parse_args()

def load_image(image_path, size=512):
    """Load ảnh từ đường dẫn, resize, crop và chuyển thành Tensor"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    # Thêm batch dimension (1, C, H, W)
    return transform(img).unsqueeze(0)

def main():
    args = parse_args()
    
    # 1. Khởi tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Đang sử dụng device: {device}")
    
    # 2. Chuẩn bị model
    model = AdaIN().to(device)
    model.eval()
    
    # Load checkpoint nếu tồn tại (cho các giai đoạn sau)
    if os.path.exists(args.checkpoint):
        print(f"[Infer] Tìm thấy checkpoint, đang load từ: {args.checkpoint}")
        # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        # (Tạm comment phần load weight thực tế ở Giai đoạn mock)
    else:
        print(f"[Infer] Không tìm thấy checkpoint tại {args.checkpoint}. Chạy với Mock Model/Untrained weights.")
        
    # 3. Chuẩn bị dữ liệu đầu vào
    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    
    if not content_dir.exists() or not style_dir.exists():
        print(f"[Lỗi] Không tìm thấy thư mục dữ liệu: {content_dir} hoặc {style_dir}")
        return

    content_files = list(content_dir.glob('*.*'))
    style_files = list(style_dir.glob('*.*'))
    
    if not content_files or not style_files:
        print("[Lỗi] Thư mục content hoặc style không có ảnh để test.")
        return
        
    # Lấy ra cặp ảnh đầu tiên để chạy smoke test
    content_img_path = content_files[0]
    style_img_path = style_files[0]
    print(f"[Infer] Xử lý: Content='{content_img_path.name}' | Style='{style_img_path.name}'")
    
    # Load thành tensor
    content_tensor = load_image(content_img_path).to(device)
    style_tensor = load_image(style_img_path).to(device)
    
    # 4. Inference
    with torch.no_grad():
        output_tensor = model(content_tensor, style_tensor)
        
    # 5. Lưu ảnh output
    output_filename = f"output_{content_img_path.stem}_{style_img_path.stem}.jpg"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Kẹp giá trị pixel về [0, 1] trước khi lưu
    output_tensor = torch.clamp(output_tensor, 0, 1)
    save_image(output_tensor, output_path)
    print(f"[Thành công] Đã lưu ảnh đầu ra tại: {output_path}")

if __name__ == '__main__':
    main()