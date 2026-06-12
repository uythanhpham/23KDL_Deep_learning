import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbedding(nn.Module):
    """
    Tạo embedding cho các bước thời gian (timesteps) dựa trên hàm sin và cos.
    
    Lý do sử dụng Sinusoidal Embedding:
    Trong Diffusion Model, mạng UNet chia sẻ chung trọng số (weights) cho mọi bước khử nhiễu.
    Để mô hình phân biệt được nó đang ở bước thứ t nào (nhiễu nhiều hay ít), ta cần "tiêm" 
    thông tin thời gian vào. Sinusoidal embedding giúp biến đổi số nguyên t thành một vector 
    liên tục, cho phép mô hình dễ dàng học được tính chất tương đối (relative distance) 
    giữa các bước thời gian thông qua các tần số khác nhau.
    
    Công thức:
    $$PE_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/dim}}\right)$$
    $$PE_{(t, 2i+1)} = \cos\left(\frac{t}{10000^{2i/dim}}\right)$$
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, "Dimension (dim) phải là số chẵn để chia đều cho sin và cos."
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor chứa các timestep, shape (B,)
        Returns:
            Tọa độ embedding của các timestep, shape (B, dim)
        """
        half_dim = self.dim // 2
        
        # Tính toán mẫu số bằng e mũ logarit để đảm bảo độ ổn định tính toán (numerical stability)
        # Công thức tương đương: 1 / (10000 ** (2i / dim)) = exp( - (2i / dim) * log(10000) )
        emb = math.log(10000.0) / half_dim
        # arange từ 0 đến half_dim - 1
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        
        # Nhân timestep t với các tần số
        # t.shape = (B,), emb.shape = (half_dim,) -> emb.shape sau khi nhân = (B, half_dim)
        emb = t[:, None].float() * emb[None, :]
        
        # Ghép cặp kết quả của hàm sin và cos
        # shape kết quả: (B, half_dim * 2) = (B, dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb

class TimestepEmbedding(nn.Module):
    """
    Module hoàn chỉnh xử lý Timestep Embedding cho UNet.
    Bọc SinusoidalPositionEmbedding bên trong một mạng MLP 2 lớp với hàm kích hoạt SiLU.
    Việc đi qua MLP giúp embedding có thêm khả năng biểu diễn phi tuyến tính trước 
    khi được cộng/nhân vào các feature map của hình ảnh.
    """
    def __init__(self, dim: int = 256, time_embed_dim: int = 1024):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(dim)
        
        # Mạng MLP theo đúng chuẩn kiến trúc DDPM
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor timestep, shape (B,)
        Returns:
            Embedding vector đã qua MLP, shape (B, time_embed_dim)
        """
        # (B,) -> (B, dim)
        x = self.sinusoidal(t)
        # (B, dim) -> (B, time_embed_dim)
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    # ==========================================
    # SMOKE TEST
    # ==========================================
    print("=== TEST TIMESTEP EMBEDDINGS ===")
    
    # 1. Khởi tạo fake timesteps t với batch_size = 4
    batch_size = 4
    t = torch.randint(0, 1000, (batch_size,))
    print(f"Input timesteps t: {t} | Shape: {t.shape}")
    
    # 2. Định nghĩa cấu hình
    dim = 256
    time_embed_dim = dim * 4
    
    # 3. Chạy qua module Sinusoidal (để kiểm tra step trung gian)
    sin_embed = SinusoidalPositionEmbedding(dim=dim)
    out_sin = sin_embed(t)
    assert out_sin.shape == (batch_size, dim), f"Lỗi shape Sinusoidal: {out_sin.shape}"
    print(f"✓ Sinusoidal embedding shape: {out_sin.shape} (B, dim)")
    
    # 4. Chạy qua module TimestepEmbedding tổng
    ts_embed = TimestepEmbedding(dim=dim, time_embed_dim=time_embed_dim)
    out_ts = ts_embed(t)
    
    # 5. Kiểm tra kết quả
    assert out_ts.shape == (batch_size, time_embed_dim), f"Lỗi shape TimestepEmbedding: {out_ts.shape}"
    print(f"✓ TimestepEmbedding shape (MLP output): {out_ts.shape} (B, time_embed_dim)")
    print("=== PASS ===")