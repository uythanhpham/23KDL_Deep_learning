import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from src.models.adain import AdaINStyleTransfer as AdaIN

def get_transform():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])

def main():
    parser = argparse.ArgumentParser(description="GĐ 7: Inference thật với Best Checkpoint")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--content_dir', type=str, required=True)
    parser.add_argument('--style_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Infer] Thiết bị sử dụng: {device}")

    # 1. Khởi tạo model và load trọng số thật
    model = AdaIN().to(device)
    print(f"[Infer] Đang nạp 'não' từ: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    # Xử lý nếu checkpoint lưu dưới dạng dict
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    transform = get_transform()

    # 2. Lấy danh sách ảnh
    contents = sorted([f for f in os.listdir(args.content_dir) if f.endswith(('.jpg', '.png'))])
    styles = sorted([f for f in os.listdir(args.style_dir) if f.endswith(('.jpg', '.png'))])

    print(f"[Infer] Bắt đầu xử lý batch: {len(contents)} ảnh content x {len(styles)} ảnh style")

    # 3. Chạy Inference (Ghép cặp hoặc chạy theo lô tùy ý, ở đây ta chạy 1-1)
    # 3. Chạy Batch Inference (Ghép chéo 1 content với tất cả style)
    with torch.no_grad():
        for c_name in contents:
            c_path = os.path.join(args.content_dir, c_name)
            c_img = transform(Image.open(c_path).convert('RGB')).unsqueeze(0).to(device)
            
            for s_name in styles:
                s_path = os.path.join(args.style_dir, s_name)
                s_img = transform(Image.open(s_path).convert('RGB')).unsqueeze(0).to(device)
                
                output = model(c_img, s_img)
                
                out_name = f"result_{c_name.split('.')[0]}_{s_name.split('.')[0]}.jpg"
                save_image(output, os.path.join(args.output_dir, out_name))
                print(f"  -> Đã tạo: {out_name}")

    print("\n[Thành công] Toàn bộ ảnh đã được 'lên đồ' tại: " + args.output_dir)

if __name__ == '__main__':
    main()