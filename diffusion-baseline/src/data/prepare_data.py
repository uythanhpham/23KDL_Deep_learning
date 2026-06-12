import os
import argparse
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List

def make_random_image_array(image_size: int, pattern: str) -> np.ndarray:
    """
    Sinh mảng numpy ảnh RGB (H, W, 3) dạng uint8 dựa trên pattern.
    """
    H = W = image_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    
    if pattern == "noise":
        # Nhiễu ngẫu nhiên (mô phỏng texture phức tạp)
        img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        
    elif pattern == "gradient":
        # Gradient mượt mà (mô phỏng khối cấu trúc, bầu trời, background)
        x = np.linspace(0, 255, W)
        y = np.linspace(0, 255, H)
        xv, yv = np.meshgrid(x, y)
        base = ((xv + yv) / 2).astype(np.uint8)
        img[:, :, 0] = base           # Kênh R
        img[:, :, 1] = 255 - base     # Kênh G
        img[:, :, 2] = 128            # Kênh B cố định
        
    elif pattern == "blocks":
        # Lưới 4x4 màu ngẫu nhiên (mô phỏng bố cục đối tượng, hình khối)
        block_h, block_w = max(1, H // 4), max(1, W // 4)
        for i in range(4):
            for j in range(4):
                color = np.random.randint(0, 256, 3, dtype=np.uint8)
                end_h = (i+1)*block_h if i < 3 else H
                end_w = (j+1)*block_w if j < 3 else W
                img[i*block_h:end_h, j*block_w:end_w] = color
                
    elif pattern == "stripes":
        # Sọc dọc (mô phỏng các nét vẽ, hoạ tiết lặp lại)
        stripe_width = max(1, W // 8)
        for i in range(8):
            color = np.random.randint(0, 256, 3, dtype=np.uint8)
            end_w = (i+1)*stripe_width if i < 7 else W
            img[:, i*stripe_width:end_w] = color
            
    return img

def generate_images(target_dir: str, prefix: str, count: int, image_size: int, patterns: List[str], fmt: str) -> List[str]:
    """
    Sinh hàng loạt ảnh và lưu vào thư mục đích, luân phiên các pattern.
    exist_ok=True giúp script có thể chạy nhiều lần mà không bị lỗi (idempotent).
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    
    for i in range(count):
        # Luân phiên chọn pattern từ danh sách
        pattern = patterns[i % len(patterns)]
        
        # Sinh ảnh dạng array và chuyển thành PIL Image
        img_array = make_random_image_array(image_size, pattern)
        img = Image.fromarray(img_array, mode="RGB")
        
        filename = f"{prefix}_{i:04d}.{fmt}"
        file_path = target_path / filename
        
        # Ghi đè file nếu đã tồn tại
        img.save(file_path)
        image_files.append(filename)
        
    return image_files

def write_manifest(output_dir: str, content_files: List[str], style_files: List[str], image_size: int, seed: int):
    """
    Tạo và lưu file manifest.json để ghi nhận metadata cho quá trình huấn luyện Style-guided.
    """
    manifest_data = {
        "mode": "style_guided_diffusion",
        "image_size": image_size,
        "seed": seed,
        "content": {
            "dir": "debug_data/content",
            "num_images": len(content_files),
            "files": content_files
        },
        "style": {
            "dir": "debug_data/style",
            "num_images": len(style_files),
            "files": style_files
        },
        "notes": "Ảnh dummy để smoke test. Thay bằng data thật khi train."
    }
    
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Sinh dummy RGB images cho Style-guided Diffusion smoke test.")
    parser.add_argument("--output_dir", type=str, default="debug_data", help="Thư mục gốc chứa data")
    parser.add_argument("--image_size", type=int, default=64, help="Kích thước ảnh vuông (H=W)")
    parser.add_argument("--num_images", type=int, default=20, help="Số lượng ảnh cần sinh cho mỗi loại")
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái tạo kết quả")
    parser.add_argument("--image_format", type=str, default="png", help="Định dạng ảnh lưu trữ")
    
    args = parser.parse_args()
    
    # 1. Cố định seed
    if args.seed is not None:
        np.random.seed(args.seed)
        
    output_path = Path(args.output_dir)
    content_dir = output_path / "content"
    style_dir = output_path / "style"
    
    # 2. Định nghĩa các pattern đặc trưng 
    # Content mô phỏng cấu trúc hình học rõ ràng
    content_patterns = ["gradient", "blocks"]
    # Style mô phỏng texture, chất liệu hoặc nét cọ phân tán
    style_patterns = ["noise", "stripes"]
    
    # 3. Sinh dữ liệu Content
    content_files = generate_images(
        target_dir=content_dir,
        prefix="content",
        count=args.num_images,
        image_size=args.image_size,
        patterns=content_patterns,
        fmt=args.image_format
    )
    
    # 4. Sinh dữ liệu Style
    style_files = generate_images(
        target_dir=style_dir,
        prefix="style",
        count=args.num_images,
        image_size=args.image_size,
        patterns=style_patterns,
        fmt=args.image_format
    )
    
    # 5. Lưu manifest file
    write_manifest(
        output_dir=output_path,
        content_files=content_files,
        style_files=style_files,
        image_size=args.image_size,
        seed=args.seed
    )
    
    # 6. Thông báo thành công
    print(f"DONE: Đã sinh {len(content_files)} content + {len(style_files)} style ảnh tại {args.output_dir}")

if __name__ == "__main__":
    main()