import torch
import torch.nn as nn
from typing import List, Tuple

from src.models.embeddings import TimestepEmbedding
from src.models.attention import AttentionBlock

"""
========================================================================
GIẢI THÍCH: AdaIN (Adaptive Instance Normalization) vs GroupNorm
========================================================================
1. GroupNorm (Trong bản Unconditional): 
   - Chuẩn hóa feature map dựa trên các nhóm kênh tĩnh. Điều này giúp mô hình 
     ổn định trong việc sinh ảnh ngẫu nhiên nhưng không có cơ chế linh hoạt 
     để nhận điều kiện (condition) từ bên ngoài.

2. AdaIN (Trong bản Style-guided):
   - Đầu tiên, nó dùng InstanceNorm để chuẩn hóa độc lập từng kênh của từng ảnh, 
     qua đó "xóa bỏ" thông tin phong cách (style) ban đầu của feature map.
   - Tiếp theo, một lớp Linear (style_proj) sẽ chiếu vector `style_emb` thành 
     2 bộ tham số `gamma` (scale) và `beta` (shift).
   - Cuối cùng, công thức `gamma * h + beta` áp đặt phong cách mới vào mạng. 
     Nói cách khác, AdaIN là "cây kim" dùng để bơm thẳng style vào mạch máu của UNet.
========================================================================
"""

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization — inject style vào feature map
    """
    def __init__(self, num_channels: int, style_dim: int):
        super().__init__()
        # affine=False để không học gamma/beta mặc định, mà sẽ lấy từ style_emb
        self.norm = nn.InstanceNorm2d(num_channels, affine=False)
        self.style_proj = nn.Linear(style_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) | style_emb: (B, style_dim)
        h = self.norm(x)
        p = self.style_proj(style_emb)                 # (B, 2C)
        gamma, beta = p.chunk(2, dim=1)                # Mỗi phần có shape (B, C)
        
        # Thêm các chiều không gian để broadcast với (B, C, H, W)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)      # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)        # (B, C, 1, 1)
        
        return gamma * h + beta


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, style_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.adain1 = AdaIN(out_ch, style_dim)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.adain2 = AdaIN(out_ch, style_dim)
        
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        # Nhánh 1: Conv -> AdaIN
        h = self.conv1(x)
        h = self.adain1(h, style_emb)
        
        # Inject timestep vào sau Activation
        t_proj = self.time_proj(self.act(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h) + t_proj
        
        # Nhánh 2: Conv -> AdaIN -> Dropout
        h = self.conv2(h)
        h = self.adain2(h, style_emb)
        h = self.act(self.dropout(h))
        
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, style_dim: int, num_res: int, use_attn: bool, dropout: float = 0.1):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        curr_in = in_ch
        for _ in range(num_res):
            self.res_blocks.append(ResidualBlock(curr_in, out_ch, time_dim, style_dim, dropout))
            self.attn_blocks.append(AttentionBlock(out_ch) if use_attn else nn.Identity())
            curr_in = out_ch
            
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, style_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x, t_emb, style_emb)
            x = attn(x)
            skips.append(x)
            
        out = self.downsample(x)
        return out, skips


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, style_dim: int, num_res: int, use_attn: bool, dropout: float = 0.1):
        super().__init__()
        # ConvTranspose2d thực hiện tăng độ phân giải (Upsample) ngay tại đầu Block
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        curr_in = in_ch
        for _ in range(num_res):
            self.res_blocks.append(ResidualBlock(curr_in + skip_ch, out_ch, time_dim, style_dim, dropout))
            self.attn_blocks.append(AttentionBlock(out_ch) if use_attn else nn.Identity())
            curr_in = out_ch

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor], t_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        for res, attn, skip in zip(self.res_blocks, self.attn_blocks, reversed(skips)):
            # Cắt ghép kênh từ Skip Connection
            x = torch.cat([x, skip], dim=1)
            x = res(x, t_emb, style_emb)
            x = attn(x)
            
        return x


class Bottleneck(nn.Module):
    def __init__(self, channels: int, time_dim: int, style_dim: int, dropout: float = 0.1):
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_dim, style_dim, dropout)
        self.attn = AttentionBlock(channels)
        self.res2 = ResidualBlock(channels, channels, time_dim, style_dim, dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        h = self.res1(x, t_emb, style_emb)
        h = self.attn(h)
        h = self.res2(h, t_emb, style_emb)
        return h


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, channel_mults: List[int] = [1, 2, 4],
                 num_res_blocks: int = 2, attention_resolutions: List[int] = [16], dropout: float = 0.1, style_dim: int = 512):
        super().__init__()
        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(dim=base_channels, time_embed_dim=time_embed_dim)
        
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # --- Khởi tạo nhánh Down ---
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        current_res = 64
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            use_attn = current_res in attention_resolutions
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_embed_dim, style_dim, num_res_blocks, use_attn, dropout)
            )
            in_ch = out_ch
            current_res //= 2
            
        # --- Nút thắt Bottleneck ---
        self.bottleneck = Bottleneck(in_ch, time_embed_dim, style_dim, dropout)
        
        # --- Khởi tạo nhánh Up ---
        self.up_blocks = nn.ModuleList()
        
        for mult in reversed(channel_mults):
            skip_ch = base_channels * mult
            out_ch = base_channels * mult
            use_attn = current_res in attention_resolutions
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, time_embed_dim, style_dim, num_res_blocks, use_attn, dropout)
            )
            in_ch = out_ch
            current_res *= 2
            
        self.output_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_conv(x)
        
        all_skips = []
        for down in self.down_blocks:
            h, skips = down(h, t_emb, style_emb)
            all_skips.append(skips)
            
        h = self.bottleneck(h, t_emb, style_emb)
        
        for up, skips in zip(self.up_blocks, reversed(all_skips)):
            h = up(h, skips, t_emb, style_emb)
            
        return self.output_conv(h)


if __name__ == "__main__":
    # ==========================================
    # SMOKE TEST KIỂM TRA MÔ HÌNH UNET (STYLE-GUIDED)
    # ==========================================
    print("=== BẮT ĐẦU SMOKE TEST UNET ===")
    
    # 1. Khởi tạo mô hình
    model = UNet(style_dim=512)
    model.eval()
    
    # 2. Sinh dữ liệu giả lập (Batch size = 2)
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    se = torch.randn(2, 512)
    
    # Forward pass lần 1
    with torch.no_grad():
        out = model(x, t, se)
        
    assert out.shape == x.shape, f"FAIL: Expected shape {x.shape}, got {out.shape}"
    
    # 3. Kiểm tra tính nhạy cảm của AdaIN
    # Tạo một vector style hoàn toàn mới
    se2 = torch.randn(2, 512)
    with torch.no_grad():
        out2 = model(x, t, se2)
        
    # Đảm bảo output thay đổi khi đưa phong cách mới vào
    assert not torch.allclose(out, out2), "FAIL: AdaIN không hoạt động! Style embedding thay đổi nhưng output vẫn giữ nguyên."
    
    # 4. In kết quả
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ UNet: {out.shape} | Params: {total_params:.2f}M")
    print("✓ AdaIN conditioning hoạt động (Output nhạy cảm với Style Embedding)")
    print("=== SMOKE 2C★: PASS ===")