import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    """
    Self-Attention block cho Diffusion Model (chủ yếu được chèn ở các độ phân giải thấp của UNet).
    
    Giải thích cơ chế trong Diffusion:
    - Trong khi các lớp Convolution chỉ nhìn được các đặc trưng cục bộ (local features) thông qua 
      kernel size, Self-Attention giúp mỗi pixel trên feature map có thể "nhìn" và tính toán độ 
      tương quan với toàn bộ các pixel khác (global context).
    - Điều này cực kỳ quan trọng trong Diffusion để mô hình giữ được sự đồng nhất của bức ảnh 
      (ví dụ: mắt trái và mắt phải của một khuôn mặt phải đối xứng dù nằm xa nhau).
    - Dựa trên công thức cốt lõi: $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "Số channels phải chia hết cho num_heads"
        
        # Pre-norm: DDPM tiêu chuẩn thường dùng GroupNorm(32 groups). 
        # Cấu hình an toàn để tránh lỗi nếu channels không chia hết cho 32.
        num_groups = 32 if channels % 32 == 0 else (8 if channels % 8 == 0 else 1)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        
        # Mạng Multi-head Attention tích hợp sẵn của PyTorch
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads, 
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor đầu vào, shape (B, C, H, W)
        Returns:
            Tensor đầu ra sau khi qua attention và cộng residual, shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. Pre-norm
        h = self.norm(x)
        
        # 2. Reshape: Flatten không gian 2D (H, W) thành chuỗi 1D (Sequence Length = H * W)
        # Từ (B, C, H, W) -> (B, C, H*W) -> Đảo chiều thành (B, H*W, C)
        h = h.view(B, C, H * W).transpose(1, 2)
        
        # 3. Tính Self-Attention (Query, Key, Value đều chính là h)
        # need_weights=False giúp tiết kiệm bộ nhớ tính toán
        attn_out, _ = self.attention(h, h, h, need_weights=False)
        
        # 4. Reshape lại về ảnh 2D: (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        # 5. Residual Connection
        return x + attn_out


if __name__ == "__main__":
    # ==========================================
    # SMOKE TEST
    # ==========================================
    print("=== TEST ATTENTION BLOCK ===")
    
    # 1. Khởi tạo fake input tensor
    B, C, H, W = 2, 128, 16, 16
    x = torch.randn(B, C, H, W)
    print(f"Input shape: {x.shape}")
    
    # 2. Chạy qua block
    attn = AttentionBlock(channels=C, num_heads=4)
    out = attn(x)
    print(f"Output shape: {out.shape}")
    
    # 3. Kiểm tra tính hợp lệ
    assert out.shape == x.shape, f"Lỗi shape: Cần {x.shape} nhưng nhận {out.shape}"
    print("PASS: AttentionBlock shape OK")