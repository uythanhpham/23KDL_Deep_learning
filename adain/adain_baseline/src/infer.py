import os
import argparse
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image
from .models.adain import AdaINStyleTransfer as AdaIN

# Định nghĩa các hằng số chuẩn của ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transform(size=512):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        #  FIX: Bắt buộc Normalize để VGG19 hiểu được ảnh
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def denormalize(tensor, device):
    """Đảo ngược Normalize để đưa về dải [0, 1] trước khi lưu ảnh"""
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(STD).view(1, 3, 1, 1).to(device)
    return tensor * std + mean

def main():
    parser = argparse.ArgumentParser(description="GĐ 7: Inference thật với Best Checkpoint")
    parser.add_argument('--config',   type=str,   default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--content_dir', type=str, required=True)
    parser.add_argument('--style_dir',   type=str, required=True)
    parser.add_argument('--output_dir',   type=str, required=True)
    parser.add_argument('--alpha',        type=float, default=1.0)
    parser.add_argument('--size',         type=int,   default=512)
    parser.add_argument('--pair_mode',    type=str,   default='all', choices=['all', 'cycle'],
                        help="all: tích chéo content×style; cycle: 1 style/content (output 1-1, dùng cho scripts/evaluate_all.sh)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Infer] Thiết bị: {device}")

    # 1. Load model
    model = AdaIN().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    transform = get_transform(args.size)

    # 2. Lấy danh sách ảnh
    contents = sorted([f for f in os.listdir(args.content_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    styles   = sorted([f for f in os.listdir(args.style_dir)   if f.endswith(('.jpg', '.png', '.jpeg'))])

    # 3. Batch inference
    with torch.no_grad():
        for c_idx, c_name in enumerate(contents):
            c_img = transform(
                Image.open(os.path.join(args.content_dir, c_name)).convert('RGB')
            ).unsqueeze(0).to(device)

            # cycle: mỗi content ghép đúng 1 style (lấy dư tuần hoàn) → output 1-1
            if args.pair_mode == 'cycle':
                pair_styles = [styles[c_idx % len(styles)]]
            else:
                pair_styles = styles

            for s_name in pair_styles:
                s_img = transform(
                    Image.open(os.path.join(args.style_dir, s_name)).convert('RGB')
                ).unsqueeze(0).to(device)

                # Chạy model
                output = model(c_img, s_img, alpha=args.alpha)

                # FIX: Denormalize trước khi lưu
                output = denormalize(output, device)
                # Clamp để tránh noise (giá trị vượt quá 0-1)
                #output = torch.clamp(output, 0, 1) bỏ dòng này sau debug

                if args.pair_mode == 'cycle':
                    out_name = f"{Path(c_name).stem}_stylized.jpg"
                else:
                    out_name = f"result_{Path(c_name).stem}_{Path(s_name).stem}.jpg"
                save_image(output, os.path.join(args.output_dir, out_name), normalize=False)
                print(f"  -> Đã lưu: {out_name}")

    print(f"\n[Thành công] Ảnh đã lưu tại: {args.output_dir}")

if __name__ == '__main__':
    main()