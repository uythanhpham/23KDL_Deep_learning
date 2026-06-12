import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

class DDPMScheduler(nn.Module):
    """
    Bộ định thời (Scheduler) cho Denoising Diffusion Probabilistic Models (DDPM).
    Xử lý cả quá trình thêm nhiễu (Forward Process) và khử nhiễu (Reverse Process).
    """
    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = "linear", 
                 beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Lựa chọn lịch trình phương sai (Variance schedule)
        if beta_schedule == "linear":
            betas = self._linear_schedule(beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = self._cosine_schedule()
        else:
            raise ValueError(f"Không hỗ trợ beta_schedule: {beta_schedule}")
            
        # Tính toán các hệ số alpha
        alphas = 1.0 - betas
        # Tính tích lũy (cumulative product) của alpha
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Đăng ký các tensor dưới dạng buffer (không học) để PyTorch tự động quản lý thiết bị (device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # Tính trước thêm biến cho bước khử nhiễu (step)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

    def _linear_schedule(self, beta_start: float, beta_end: float) -> torch.Tensor:
        """Lịch trình tuyến tính cơ bản của DDPM nguyên thủy."""
        return torch.linspace(beta_start, beta_end, self.num_timesteps)

    def _cosine_schedule(self, s: float = 0.008) -> torch.Tensor:
        """
        Lịch trình Cosine (Nichol & Dhariwal 2021).
        Tránh việc ảnh bị phá hủy quá nhanh ở các bước t nhỏ.
        Công thức: $f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$
        """
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clamp giá trị ở 0.999 để tránh bất ổn định toán học
        return torch.clip(betas, 0.0001, 0.999)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Hàm hỗ trợ trích xuất hệ số tương ứng với timestep t và reshape để broadcast với x_t.
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        # Reshape thành (B, 1, 1, 1) để nhân trực tiếp với tensor ảnh (B, C, H, W)
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quá trình khuếch tán thuận (Forward Process). Thêm nhiễu vào ảnh gốc.
        Công thức: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        sqrt_ah_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mah_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        x_t = sqrt_ah_t * x0 + sqrt_1mah_t * noise
        return x_t, noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """
        [PHƯƠNG THỨC MỚI] Ước lượng lại ảnh $x_0$ từ $x_t$ và nhiễu dự đoán $\epsilon_\theta$.
        Sử dụng cực kỳ hiệu quả để tính Style Loss / Content Loss ngay trên ảnh dự đoán.
        Công thức: $\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$
        """
        sqrt_ah_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mah_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        x0_pred = (x_t - sqrt_1mah_t * eps_pred) / sqrt_ah_t
        # Clamp lại về vùng không gian thực [-1.0, 1.0]
        return x0_pred.clamp(-1.0, 1.0)

    def step(self, eps_pred: torch.Tensor, t_scalar: int, x_t: torch.Tensor) -> torch.Tensor:
        """
        Quá trình lấy mẫu ngược (Reverse Process) cho 1 bước duy nhất.
        Dự đoán $x_{t-1}$ từ $x_t$.
        """
        t = torch.tensor([t_scalar], device=x_t.device)
        
        alpha_t = self._extract(self.alphas, t, x_t.shape)
        sqrt_1mah_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        
        # Tính kỳ vọng (mean) mu
        # $\mu = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$
        mu = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_1mah_t) * eps_pred)
        
        if t_scalar == 0:
            return mu
        else:
            # Tính độ lệch chuẩn sigma
            # $\sigma = \sqrt{ \beta_t \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} }$
            ah_t = self._extract(self.alphas_cumprod, t, x_t.shape)
            ah_t_prev = self._extract(self.alphas_cumprod_prev, t, x_t.shape)
            sigma = torch.sqrt(beta_t * (1.0 - ah_t_prev) / (1.0 - ah_t))
            
            noise = torch.randn_like(x_t)
            return mu + sigma * noise

    def sample(self, model: nn.Module, shape: Tuple[int, ...], style_emb: torch.Tensor, device: str) -> torch.Tensor:
        """
        Vòng lặp lấy mẫu (Sampling loop) từ nhiễu thuần túy ra ảnh sắc nét.
        Có đưa `style_emb` vào làm điều kiện (Conditioning).
        """
        model.eval()
        x = torch.randn(shape, device=device)
        
        with torch.no_grad():
            for i in reversed(range(self.num_timesteps)):
                # Tạo batch timestep
                t_batch = torch.full((shape[0],), i, device=device, dtype=torch.long)
                
                # Dự đoán nhiễu (Truyền thêm style_emb vào UNet)
                eps_pred = model(x, t_batch, style_emb)
                
                # Khử nhiễu từng bước một
                x = self.step(eps_pred, i, x)
                
        return x.clamp(-1.0, 1.0)

if __name__ == "__main__":
    # ==========================================
    # SMOKE TEST SCHEDULER
    # ==========================================
    print("=== BẮT ĐẦU SMOKE TEST SCHEDULER ===")
    
    sched = DDPMScheduler(num_timesteps=1000, beta_schedule="linear")
    
    # Giả lập dữ liệu
    x0 = torch.randn(2, 3, 64, 64)
    t = torch.tensor([100, 500])
    
    # 1. Forward process (thêm nhiễu)
    x_t, noise = sched.add_noise(x0, t)
    
    # 2. Backward prediction
    # Chú ý: Ở đây ta truyền `noise` thực sự (chính là ground truth epsilon) vào hàm dự đoán.
    # Về mặt toán học, hàm predict_x0 phải khử hoàn toàn độ nhiễu và trả lại x0 gốc.
    x0_rec = sched.predict_x0(x_t, t, noise)
    
    # 3. Tính toán sai số khôi phục (reconstruction error)
    err = (x0_rec - x0.clamp(-1.0, 1.0)).abs().mean().item()
    
    assert err < 0.01, f"FAIL: Reconstruction error quá lớn {err:.4f}, công thức toán học bị sai lệch."
    print(f"✓ predict_x0 error: {err:.6f} (≈0 khi dùng ground-truth noise)")
    print("=== SMOKE 3A: PASS ===")