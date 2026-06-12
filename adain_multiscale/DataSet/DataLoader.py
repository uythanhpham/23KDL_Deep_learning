"""Chuẩn bị Dataset và DataLoader hỗ trợ ghép cặp Content và Style."""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path

class StyleTransferDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 256, pair_mode: str = "cycle") -> None:
        """
        Args:
            root_dir: Đường dẫn gốc, chứa 2 thư mục con 'content' và 'style'.
            image_size: Kích thước ảnh cần resize (H, W).
            pair_mode: Cách ghép cặp ảnh ('cycle' hoặc 'random').
        """
        super().__init__()
        self.root_path = Path(root_dir)
        self.content_dir = self.root_path / "content"
        self.style_dir = self.root_path / "style"
        self.pair_mode = pair_mode

        # Định nghĩa các đuôi mở rộng của file ảnh hợp lệ
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Đọc toàn bộ danh sách ảnh từ 2 thư mục con
        self.content_paths = sorted([
            p for p in self.content_dir.glob("*") if p.suffix.lower() in valid_extensions
        ])
        self.style_paths = sorted([
            p for p in self.style_dir.glob("*") if p.suffix.lower() in valid_extensions
        ])

        # Kiểm tra tính toàn vẹn của dữ liệu đầu vào
        if len(self.content_paths) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh hợp lệ trong thư mục content: {self.content_dir}")
        if len(self.style_paths) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh hợp lệ trong thư mục style: {self.style_dir}")

        # Pipeline tiền xử lý ảnh chuẩn cho kiến trúc VGG
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # Chuẩn hóa theo phân phối chuẩn ImageNet của VGG19 mã hóa gốc
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        """Độ dài của Dataset được tính dựa trên số lượng ảnh Content làm gốc."""
        return len(self.content_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 1. Đọc và biến đổi ảnh Content
        c_path = self.content_paths[idx]
        c_img = Image.open(c_path).convert("RGB")
        c_tensor = self.transform(c_img)

        # 2. Xử lý ghép cặp với ảnh Style theo cơ chế 'cycle' hoặc 'random'
        if self.pair_mode == "cycle":
            # Lấy dư tuần hoàn nếu số lượng ảnh style ít hoặc nhiều hơn ảnh content
            s_idx = idx % len(self.style_paths)
        else:
            # Chọn ngẫu nhiên một ảnh style bất kỳ trong tập dữ liệu
            s_idx = torch.randint(0, len(self.style_paths), (1,)).item()

        s_path = self.style_paths[s_idx]
        s_img = Image.open(s_path).convert("RGB")
        s_tensor = self.transform(s_img)

        return {
            "content": c_tensor,
            "style": s_tensor
        }


def build_dataloaders(
    root_dir: str,
    image_size: int = 256,
    pair_mode: str = "cycle",
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader]:
    """
    Khởi tạo cấu trúc dữ liệu, chia tập Train/Val phân rã và trả về cặp bộ nạp dữ liệu.
    
    Returns:
        train_loader, val_loader
    """
    # Khởi tạo dataset toàn cục chứa toàn bộ ảnh
    full_dataset = StyleTransferDataset(
        root_dir=root_dir,
        image_size=image_size,
        pair_mode=pair_mode
    )

    # Tính toán kích thước phân chia tập Train và Validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Phân tách ngẫu nhiên không trùng lặp bằng hàm của PyTorch
    # Cố định generator seed nếu bạn muốn kết quả chia tập giống nhau giữa các lần chạy
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Khởi tạo bộ nạp dữ liệu Train DataLoader (Bắt buộc shuffle để trộn đặc trưng)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Tối ưu hóa tốc độ đẩy dữ liệu lên GPU RAM nhanh hơn
        drop_last=True    # Bỏ batch cuối nếu bị lẻ, giữ nguyên kích thước batch ổn định
    )

    # Khởi tạo bộ nạp dữ liệu Validation DataLoader (Không cần shuffle)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader