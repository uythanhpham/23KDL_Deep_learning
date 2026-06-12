import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

"""
========================================================================
GIẢI THÍCH: StyleDiffusionLoss — Module tính tổng Loss cho Style-guided DDPM
========================================================================
Gồm 3 thành phần:
  1. noise_loss  : MSE giữa nhiễu dự đoán và nhiễu thực (mục tiêu cốt lõi DDPM)
  2. style_loss  : Gram-matrix loss ép x0_pred có phong cách giống style image
  3. content_loss: Feature-matching loss ép x0_pred giữ cấu trúc nội dung

QUAN TRỌNG: x0_pred phải GIỮ NGUYÊN gradient graph (KHÔNG dùng detach/no_grad)
để gradient từ style_loss và content_loss chảy ngược về UNet qua predict_x0.
Tương tự, style_emb KHÔNG được bọc no_grad — nếu detach thì MLP của StyleEncoder
không nhận được gradient nào và sẽ đứng yên ở trọng số random.

t_mask: Chỉ tính perceptual loss ở các timestep nhỏ (t < T//5) vì ở đó
x0_pred đủ sạch để VGG trích xuất feature có ý nghĩa.

────────────────────────────────────────────────────────────────────────
VÌ SAO LÀ MỘT nn.Module (chứ không phải hàm)?
Để hỗ trợ ĐA GPU bằng nn.DataParallel, ta cần gói TOÀN BỘ phép tính
(encode_style → UNet → predict_x0 → perceptual) vào trong forward của một
module rồi DataParallel chính module này. Khi đó mỗi GPU tự tính trọn vẹn
trên phần dữ liệu của nó; KHÔNG có tensor-có-gradient nào được tạo bên ngoài
rồi scatter xuyên GPU (đây chính là thứ gây 'illegal memory access' khi bọc
DataParallel quanh riêng UNet còn style_emb lại tính ở ngoài).
========================================================================
"""

def noise_prediction_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor) -> torch.Tensor:
    """
    Hàm mất mát chính của DDPM (Mean Squared Error).
    Đo lường sai số giữa nhiễu mà mô hình dự đoán (eps_pred) và nhiễu thực tế (eps_true).
    """
    return F.mse_loss(eps_pred, eps_true)


class StyleDiffusionLoss(nn.Module):
    """
    Gói model (UNet) + style_encoder + scheduler để tính 3 thành phần loss
    TRỌN VẸN trên 1 thiết bị mỗi lần forward → tương thích nn.DataParallel.

    forward() trả về 3 tensor shape (1,) (giá trị trung bình trên phần batch
    của replica). DataParallel sẽ gather theo dim 0 → (num_gpus,); người gọi
    chỉ cần .mean() rồi nhân loss_weights ở bên ngoài.
    """
    def __init__(self, model: nn.Module, style_encoder: nn.Module, scheduler: nn.Module,
                 t_mask_ratio: int = 5, amp_enabled: bool = False, style_dropout: float = 0.0):
        super().__init__()
        self.model = model
        self.style_encoder = style_encoder
        self.scheduler = scheduler
        self.t_mask_ratio = t_mask_ratio      # t < T // t_mask_ratio mới tính perceptual loss
        self.amp_enabled = amp_enabled         # bật autocast NGAY trong forward (an toàn với DataParallel)
        self.style_dropout = style_dropout     # xác suất thay style_emb bằng null (cho CFG)

    def forward(self, x0: torch.Tensor, style: torch.Tensor,
                t: torch.Tensor, noise: torch.Tensor):
        # autocast đặt bên trong forward để propagate đúng sang các worker thread
        # của DataParallel (không phụ thuộc autocast ở thread chính).
        with autocast('cuda', enabled=self.amp_enabled):
            T = self.scheduler.num_timesteps
            B = x0.shape[0]

            # 1. Style embedding — KHÔNG no_grad (để MLP nhận gradient)
            style_emb = self.style_encoder.encode_style(style)

            # 1b. STYLE DROPOUT (cho Classifier-Free Guidance):
            #     thay style_emb của một số mẫu bằng null_style → model học cả nhánh
            #     có-style lẫn không-style. Mẫu bị drop KHÔNG tính perceptual loss
            #     (vì lúc đó ta không muốn ép style).
            if self.style_dropout > 0.0:
                drop = torch.rand(B, device=x0.device) < self.style_dropout
                null = self.style_encoder.null_style.to(style_emb.dtype)
                style_emb = torch.where(drop.unsqueeze(1), null.unsqueeze(0), style_emb)
            else:
                drop = torch.zeros(B, dtype=torch.bool, device=x0.device)

            # 2. Forward diffusion (dùng noise truyền vào để add_noise nhất quán khi scatter)
            x_t, eps_true = self.scheduler.add_noise(x0, t, noise)

            # 3. UNet dự đoán nhiễu
            eps_pred = self.model(x_t, t, style_emb)

            # 4. Noise loss (tính trên TẤT CẢ mẫu, kể cả mẫu null)
            loss_noise = noise_prediction_loss(eps_pred, eps_true)

            # 5. Dự đoán lại x0 (giữ gradient để perceptual loss chảy về UNet)
            x0_pred = self.scheduler.predict_x0(x_t, t, eps_pred)

            # 6+7. Perceptual loss: chỉ ở t thấp VÀ mẫu KHÔNG bị drop
            percep_mask = (t < T // self.t_mask_ratio) & (~drop)
            loss_style = torch.zeros((), device=x0.device)
            loss_content = torch.zeros((), device=x0.device)

            if percep_mask.any():
                # Style loss (Gram matrix)
                pred_style_in = self.style_encoder._to_vgg_input(x0_pred[percep_mask])
                pf = self.style_encoder.vgg(pred_style_in)["style_feats"]
                with torch.no_grad():
                    tf = self.style_encoder.vgg(
                        self.style_encoder._to_vgg_input(style[percep_mask])
                    )["style_feats"]
                loss_style = self.style_encoder.compute_style_loss(pf, tf)

                # Content loss (feature matching)
                pc = self.style_encoder.encode_content(x0_pred[percep_mask])
                with torch.no_grad():
                    tc = self.style_encoder.encode_content(x0[percep_mask])
                loss_content = self.style_encoder.compute_content_loss(pc, tc)

        # Trả về shape (1,) để DataParallel gather theo dim 0
        return loss_noise.reshape(1), loss_style.reshape(1), loss_content.reshape(1)


if __name__ == "__main__":
    # ==========================================
    # SMOKE TEST KIỂM TRA LUỒNG TÍNH LOSS
    # ==========================================
    print("=== BẮT ĐẦU SMOKE TEST DIFFUSION LOSS ===")

    from src.models.unet import UNet
    from src.models.style_encoder import StyleEncoder
    from src.diffusion.scheduler import DDPMScheduler

    # 1. Khởi tạo giả lập
    device = "cpu"
    model = UNet(style_dim=512).to(device)
    enc = StyleEncoder(style_dim=512).to(device)
    sched = DDPMScheduler(num_timesteps=1000, beta_schedule="cosine").to(device)
    loss_module = StyleDiffusionLoss(model, enc, sched, amp_enabled=False).to(device)

    # 2. Tạo batch dữ liệu thô (miền [-1, 1]) + t/noise sinh BÊN NGOÀI (như trainer)
    B = 4
    x0    = torch.randn(B, 3, 64, 64)
    style = torch.randn(B, 3, 64, 64)
    t     = torch.randint(0, 1000, (B,))
    noise = torch.randn_like(x0)
    weights = {"noise": 1.0, "style": 500.0, "content": 0.01}

    # 3. Tính Loss (mô phỏng đúng cách trainer dùng)
    ln, ls, lc = loss_module(x0, style, t, noise)
    assert ln.shape == torch.Size([1]), f"FAIL: noise loss shape {ln.shape}"
    total = weights["noise"] * ln.mean() + weights["style"] * ls.mean() + weights["content"] * lc.mean()
    assert total.shape == torch.Size([]), f"FAIL: Total loss shape {total.shape}"

    # 4. Backprop
    total.backward()

    # 5. UNet phải nhận gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "FAIL: UNet không nhận được gradient!"

    # 6. VGG phải đóng băng tuyệt đối (0 leak gradient)
    for name, p in enc.vgg.named_parameters():
        assert p.grad is None, f"FAIL: Leak gradient tại layer {name} của VGG!"

    # 7. [REGRESSION FIX #1] MLP của StyleEncoder PHẢI nhận gradient
    mlp_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.mlp.parameters())
    assert mlp_has_grad, "FAIL: MLP của StyleEncoder KHÔNG nhận gradient (lỗi #1 chưa được fix)!"

    print(f"✓ Output shape (ln,ls,lc) = {ln.shape},{ls.shape},{lc.shape} | total scalar OK")
    print(f"✓ Backpropagation thành công — gradient chảy đến UNet")
    print(f"✓ VGG frozen (0 leak gradient) OK")
    print(f"✓ MLP StyleEncoder NHẬN gradient (fix #1 OK)")
    print(f"✓ raw: noise={ln.item():.4f} style={ls.item():.6f} content={lc.item():.4f}")
    print("=== SMOKE 4A: PASS ===")
