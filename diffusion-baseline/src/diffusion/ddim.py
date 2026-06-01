import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, List

from src.diffusion.scheduler import DDPMScheduler

class DDIMSampler(nn.Module):
    """
    Bộ lấy mẫu DDIM (Song et al. 2020) giúp tăng tốc độ sinh ảnh.
    Thay vì phải đi lùi từng bước một (ví dụ 1000 bước như DDPM), DDIM cho phép
    nhảy cóc (sub-sequence) qua các bước, giảm xuống còn 50-100 bước mà vẫn giữ được
    chất lượng hình ảnh.
    """
    def __init__(self, scheduler: DDPMScheduler, ddim_steps: int = 50, eta: float = 0.0):
        super().__init__()
        self.scheduler = scheduler
        self.ddim_steps = ddim_steps
        self.eta = eta
        
        # Tạo chuỗi timestep nhảy cóc
        self.timesteps = self._make_schedule(ddim_steps)

    def _make_schedule(self, ddim_steps: int) -> torch.Tensor:
        """
        Tạo chuỗi các bước thời gian (timesteps) trải đều từ T-1 về 0.
        Ví dụ: T=1000, ddim_steps=50 -> [980, 960, ..., 20, 0]
        """
        T = self.scheduler.num_timesteps
        step_ratio = T // ddim_steps
        # Tạo danh sách các timestep lùi dần, lấy đúng số lượng ddim_steps
        ts = list(reversed(range(0, T, step_ratio)))[:ddim_steps]
        return torch.tensor(ts, dtype=torch.long)

    def step(self, eps_pred: torch.Tensor, t: int, t_prev: int, x_t: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện một bước lùi DDIM từ t xuống t_prev.
        Công thức tính toán dựa trên định lý Bayes ngầm định (Implicit model).
        """
        device = x_t.device
        
        # Trích xuất \bar{\alpha}_t và \bar{\alpha}_{t-1}
        ah_t = self.scheduler.alphas_cumprod[t].to(device)
        if t_prev >= 0:
            ah_prev = self.scheduler.alphas_cumprod[t_prev].to(device)
        else:
            # Khi t_prev < 0 (bước cuối cùng về ảnh gốc x_0), \bar{\alpha}_0 quy ước bằng 1
            ah_prev = torch.tensor(1.0, device=device)
            
        # 1. Dự đoán lại ảnh gốc x_0 từ x_t
        # $\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$
        x0_pred = (x_t - torch.sqrt(1.0 - ah_t) * eps_pred) / torch.sqrt(ah_t)
        x0_pred = x0_pred.clamp(-1.0, 1.0)
        
        # 2. Tính toán độ nhiễu sigma cho thành phần ngẫu nhiên (dựa trên eta)
        # $\sigma = \eta \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$
        sigma = self.eta * torch.sqrt((1 - ah_prev) / (1 - ah_t)) * torch.sqrt(1 - ah_t / ah_prev)
        
        # 3. Tính toán hướng trỏ về x_t (Direction pointing to x_t)
        # $d_t = \sqrt{1 - \bar{\alpha}_{t-1} - \sigma^2} \epsilon_\theta$
        dir_xt = torch.sqrt(1.0 - ah_prev - sigma**2) * eps_pred
        
        # 4. Thêm nhiễu (nếu eta > 0)
        noise = sigma * torch.randn_like(x_t) if self.eta > 0.0 else 0.0
        
        # Tổng hợp lại thành x_{t-1}
        # $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + d_t + \sigma z$
        x_prev = torch.sqrt(ah_prev) * x0_pred + dir_xt + noise
        
        return x_prev

    def sample(self, model: nn.Module, shape: Tuple[int, ...], style_emb: torch.Tensor, device: str) -> torch.Tensor:
        """
        Vòng lặp lấy mẫu DDIM hoàn chỉnh.
        """
        model.eval()
        x = torch.randn(shape, device=device)
        ts = self.timesteps.tolist()
        
        with torch.no_grad():
            for i, t_val in enumerate(tqdm(ts, desc="DDIM Sampling")):
                # Tạo batch chứa timestep hiện tại
                t_batch = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
                
                # Model dự đoán nhiễu
                eps_pred = model(x, t_batch, style_emb)
                
                # Xác định timestep tiếp theo (đi lùi về 0)
                t_prev = ts[i + 1] if (i + 1) < len(ts) else -1
                
                # Thực hiện bước lùi
                x = self.step(eps_pred, t_val, t_prev, x)
                
        return x.clamp(-1.0, 1.0)

    def sample_img2img(self, model: nn.Module, x0: torch.Tensor, style_emb: torch.Tensor, 
                       device: str, strength: float = 0.6) -> torch.Tensor:
        """
        DDIM Img2Img: Thêm nhiễu vào ảnh gốc rồi khử nhiễu bằng DDIM (nhảy cóc).
        Thay vì đi 600 bước DDPM tuần tự, DDIM chỉ cần 30-50 bước.
        
        Args:
            x0: Ảnh content gốc, shape (B, C, H, W), miền [-1, 1]
            strength: Mức độ nhiễu (0.0 = giữ nguyên, 1.0 = nhiễu thuần)
        """
        model.eval()
        ts = self.timesteps.tolist()  # Danh sách timestep nhảy cóc [980, 960, ..., 0]
        
        # Tìm vị trí bắt đầu dựa trên strength
        # strength=0.6 → bắt đầu từ 60% trong chuỗi DDIM steps
        start_idx = max(0, int(len(ts) * (1.0 - strength)))
        ts_sub = ts[start_idx:]  # Chỉ lấy phần timestep từ start_idx trở đi
        
        if len(ts_sub) == 0:
            return x0.clamp(-1.0, 1.0)
        
        # Thêm nhiễu vào x0 tại timestep bắt đầu
        t_start = ts_sub[0]
        t_tensor = torch.full((x0.shape[0],), t_start, device=device, dtype=torch.long)
        
        ah_t = self.scheduler.alphas_cumprod[t_start].to(device)
        noise = torch.randn_like(x0)
        x = torch.sqrt(ah_t) * x0.to(device) + torch.sqrt(1.0 - ah_t) * noise
        
        with torch.no_grad():
            for i, t_val in enumerate(tqdm(ts_sub, desc="DDIM Img2Img")):
                t_batch = torch.full((x0.shape[0],), t_val, device=device, dtype=torch.long)
                eps_pred = model(x, t_batch, style_emb)
                
                # Timestep tiếp theo trong chuỗi con
                if (i + 1) < len(ts_sub):
                    t_prev = ts_sub[i + 1]
                else:
                    t_prev = -1
                
                x = self.step(eps_pred, t_val, t_prev, x)
        
        return x.clamp(-1.0, 1.0)


if __name__ == "__main__":
    # ==========================================
    # SMOKE TEST KIỂM TRA DDIM SAMPLER
    # ==========================================
    print("=== BẮT ĐẦU SMOKE TEST DDIM ===")
    
    # 1. Khởi tạo Scheduler và DDIM Sampler
    sched = DDPMScheduler(num_timesteps=1000, beta_schedule="linear")
    ddim_steps = 50
    sampler = DDIMSampler(scheduler=sched, ddim_steps=ddim_steps, eta=0.0)
    
    # Kiểm tra số lượng bước sinh ảnh
    assert len(sampler.timesteps) == ddim_steps, f"FAIL: Expected {ddim_steps} steps, got {len(sampler.timesteps)}"
    print(f"✓ Timesteps length: {len(sampler.timesteps)} (Mô hình sẽ sinh ảnh trong {ddim_steps} bước)")
    
    # 2. Tạo một Dummy Model để test vòng lặp sample mà không cần load UNet thật
    class DummyModel(nn.Module):
        def forward(self, x, t, style_emb):
            # Luôn dự đoán ra một nhiễu ngẫu nhiên cùng kích thước
            return torch.randn_like(x) * 0.1
            
    dummy_model = DummyModel()
    
    # 3. Chạy luồng Sample
    B, C, H, W = 2, 3, 32, 32
    style_dummy = torch.randn(B, 512)
    device = "cpu"
    
    out_img = sampler.sample(model=dummy_model, shape=(B, C, H, W), style_emb=style_dummy, device=device)
    
    # 4. Xác thực kết quả
    assert out_img.shape == (B, C, H, W), f"FAIL: Shape vỡ, output {out_img.shape}"
    out_min, out_max = out_img.min().item(), out_img.max().item()
    assert out_min >= -1.0 and out_max <= 1.0, f"FAIL: Tensor bị tràn ra khỏi [-1, 1], range: [{out_min:.2f}, {out_max:.2f}]"
    
    print(f"✓ Sample shape: {out_img.shape}")
    print(f"✓ Output range: [{out_min:.4f}, {out_max:.4f}] (Đã được clamp hợp lệ)")
    print("=== SMOKE 3B: PASS ===")