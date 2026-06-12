import os
import argparse
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# Định nghĩa các đuôi file ảnh hợp lệ
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def is_image_file(filename: str) -> bool:
    """Kiểm tra xem file có phải là file ảnh hợp lệ không."""
    return any(filename.lower().endswith(ext) for ext in VALID_EXTENSIONS)

def scan_image_files(directory: str | Path) -> List[Path]:
    """Quét đệ quy và trả về danh sách các đường dẫn file ảnh trong thư mục."""
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Thư mục không tồn tại: {dir_path}")
    
    image_paths = [p for p in dir_path.rglob('*') if p.is_file() and is_image_file(p.name)]
    image_paths.sort()
    return image_paths

def set_seed(seed: int) -> None:
    """Cố định seed ngẫu nhiên cho tính tái lập (reproducibility)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_json(filepath: str | Path) -> Dict:
    """Đọc file JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# =====================================================================
# GIẢI THÍCH: TẠI SAO CẢ CONTENT VÀ STYLE ĐỀU DÙNG [-1, 1]?
# =====================================================================
# Cả Content lẫn Style đều được chuẩn hóa về miền [-1, 1].
# Lý do: StyleEncoder._to_vgg_input() đã tự chuyển đổi từ [-1, 1] 
# sang chuẩn ImageNet bên trong. Nếu DataLoader normalize theo ImageNet
# trước, rồi _to_vgg_input() lại normalize thêm lần nữa → SAI.
# Vì vậy, cả 2 transform đều dùng Normalize([0.5]*3, [0.5]*3) để 
# đưa về [-1, 1], và StyleEncoder sẽ tự xử lý phần còn lại.
# =====================================================================

def get_content_transform(image_size: int) -> transforms.Compose:
    """Pipeline biến đổi cho ảnh Content (Đầu vào UNet)."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.12), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Đưa về [-1, 1]
    ])

def get_style_transform(image_size: int) -> transforms.Compose:
    """Pipeline biến đổi cho ảnh Style — cũng chuẩn hóa về [-1, 1] như Content."""
    return transforms.Compose([
        transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1] — StyleEncoder._to_vgg_input() sẽ tự chuyển sang ImageNet
    ])

class StyleDiffusionDataset(Dataset):
    """
    Dataset cho Style-guided Diffusion.
    Ảnh content được load tuần tự theo index, ảnh style được bốc ngẫu nhiên 
    (không cần map 1-1 vì ta muốn mô hình học cách inject style bất kỳ vào content bất kỳ).
    """
    def __init__(self, content_dir: str, style_dir: str, content_transform: transforms.Compose, style_transform: transforms.Compose, seed: int = 42):
        self.content_paths = scan_image_files(content_dir)
        self.style_paths = scan_image_files(style_dir)
        self.content_transform = content_transform
        self.style_transform = style_transform
        
        # Đảm bảo data không bị rỗng
        assert len(self.content_paths) > 0, f"Không tìm thấy ảnh content tại {content_dir}"
        assert len(self.style_paths) > 0, f"Không tìm thấy ảnh style tại {style_dir}"
        
        # Cố định trạng thái random nội bộ của class (tùy chọn nhưng an toàn)
        self.rng = random.Random(seed)
        
    def __len__(self) -> int:
        # Số bước trong 1 epoch dựa trên số lượng ảnh content
        return len(self.content_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        content_path = self.content_paths[idx]
        style_path = self.rng.choice(self.style_paths) # Lấy ngẫu nhiên 1 ảnh style
        
        content_img = Image.open(content_path).convert("RGB")
        style_img = Image.open(style_path).convert("RGB")
        
        if self.content_transform:
            content_tensor = self.content_transform(content_img)
            
        if self.style_transform:
            style_tensor = self.style_transform(style_img)
            
        return {
            "content": content_tensor,
            "style": style_tensor,
            "content_path": str(content_path),
            "style_path": str(style_path)
        }

def build_dataloaders(
    content_dir: str, 
    style_dir: str, 
    image_size: int, 
    batch_size: int, 
    val_split: float = 0.1, 
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Khởi tạo Train và Validation DataLoaders."""
    set_seed(seed)
    
    c_transform = get_content_transform(image_size)
    s_transform = get_style_transform(image_size)
    
    full_dataset = StyleDiffusionDataset(
        content_dir=content_dir, 
        style_dir=style_dir, 
        content_transform=c_transform, 
        style_transform=s_transform,
        seed=seed
    )
    
    total_images = len(full_dataset)
    print(f"Dataset có tổng cộng {total_images} ảnh content và {len(full_dataset.style_paths)} ảnh style.")
    
    # kwargs chung cho DataLoader
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True if torch.cuda.is_available() else False
    }
    
    if val_split > 0.0:
        val_size = int(total_images * val_split)
        train_size = total_images - val_size
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
        return train_loader, val_loader
    else:
        train_loader = DataLoader(full_dataset, shuffle=True, drop_last=True, **loader_kwargs)
        return train_loader, None

def main():
    parser = argparse.ArgumentParser(description="Smoke test Dataset cho Style-guided Diffusion")
    parser.add_argument("--content_dir", type=str, required=True, help="Thư mục chứa ảnh content")
    parser.add_argument("--style_dir", type=str, required=True, help="Thư mục chứa ảnh style")
    parser.add_argument("--image_size", type=int, default=64, help="Kích thước ảnh vuông")
    parser.add_argument("--batch_size", type=int, default=4, help="Kích thước batch")
    parser.add_argument("--num_workers", type=int, default=0, help="Số luồng đọc dữ liệu")
    parser.add_argument("--smoke_test", action="store_true", help="Chạy chế độ kiểm thử nhanh")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        print("Đang khởi tạo DataLoaders...")
        train_loader, _ = build_dataloaders(
            content_dir=args.content_dir,
            style_dir=args.style_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            val_split=0.1,
            num_workers=args.num_workers
        )
        
        batch = next(iter(train_loader))
        
        c_tensor = batch["content"]
        s_tensor = batch["style"]
        B, C, H, W = args.batch_size, 3, args.image_size, args.image_size
        
        # Kiểm tra shape
        assert c_tensor.shape == (B, C, H, W), f"Lỗi shape content: {c_tensor.shape}"
        assert s_tensor.shape == (B, C, H, W), f"Lỗi shape style: {s_tensor.shape}"
        
        # Kiểm tra range — CẢ HAI đều phải thuộc [-1.1, 1.1]
        c_min, c_max = c_tensor.min().item(), c_tensor.max().item()
        assert c_min >= -1.1 and c_max <= 1.1, f"Lỗi range content: [{c_min:.2f}, {c_max:.2f}]"
        
        s_min, s_max = s_tensor.min().item(), s_tensor.max().item()
        assert s_min >= -1.1 and s_max <= 1.1, f"Lỗi range style: [{s_min:.2f}, {s_max:.2f}]"
        
        print("\n--- THÔNG TIN BATCH ---")
        print(f"Content: shape={c_tensor.shape}, range=[{c_min:.4f}, {c_max:.4f}]")
        print(f"Style  : shape={s_tensor.shape}, range=[{s_min:.4f}, {s_max:.4f}]")
        print("PASS: DataLoader OK")

if __name__ == "__main__":
    main()