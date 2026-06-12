from __future__ import annotations
from pathlib import Path
import torch
from torchvision.utils import make_grid
from PIL import Image, ImageDraw

def denorm(x: torch.Tensor) -> torch.Tensor:
    """Chuyển đổi dải màu từ [-1, 1] của Tanh về dải [0, 1] để hiển thị."""
    return (x.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)

@torch.no_grad()
def save_sample_grid(
    real_A: torch.Tensor,
    fake_B: torch.Tensor,
    rec_A: torch.Tensor,
    real_B: torch.Tensor,
    fake_A: torch.Tensor,
    rec_B: torch.Tensor,
    p_target: torch.Tensor,
    out_path: str | Path,
    nrow: int = 3,
) -> None:
    """Lưu lưới ảnh kết quả huấn luyện kèm thanh màu palette minh họa bên dưới."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Gom 6 thành phần ảnh của hệ thống CycleGAN
    images = [
        denorm(real_A[0]), denorm(fake_B[0]), denorm(rec_A[0]),
        denorm(real_B[0]), denorm(fake_A[0]), denorm(rec_B[0])
    ]
    grid = make_grid(images, nrow=nrow, padding=2)
    
    # 2. Biến đổi cấu trúc Tensor thành PIL Image
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    
    # 3. Tạo một bảng canvas trắng mới, cơi nới chiều cao thêm 40 pixel
    canvas = Image.new("RGB", (im.width, im.height + 40), "white")
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # 4. Phân rã vector p_target
    palette_colors = p_target.view(-1, 4)[:, :3]
    bar_width = canvas.width // len(palette_colors)
    
    # 5. Vẽ các ô chữ nhật màu sắc
    for i, color in enumerate(palette_colors):
        rgb = [int(torch.clamp(c * 255, 0, 255)) for c in color]
        draw.rectangle(
            [i * bar_width, im.height, (i + 1) * bar_width, im.height + 40], 
            fill=tuple(rgb)
        )
        
    # 6. Lưu file
    canvas.save(out_path)