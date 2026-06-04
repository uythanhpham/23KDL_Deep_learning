import torch
import torch.nn.functional as F

"""
========================================================================
GIẢI THÍCH: style_diffusion_loss — Hàm tính tổng Loss cho Style-guided DDPM
========================================================================
Gồm 3 thành phần:
  1. noise_loss  : MSE giữa nhiễu dự đoán và nhiễu thực (mục tiêu cốt lõi DDPM)
  2. style_loss  : Gram-matrix loss ép x0_pred có phong cách giống style image
  3. content_loss: Feature-matching loss ép x0_pred giữ cấu trúc nội dung

QUAN TRỌNG: x0_pred phải GIỮA NGUYÊN gradient graph (KHÔNG dùng detach/no_grad)
để gradient từ style_loss và content_loss chảy ngược về UNet qua predict_x0.
Nếu cắt gradient → UNet chỉ học noise prediction thuần, bỏ qua style conditioning.

t_mask: Chỉ tính perceptual loss ở các timestep nhỏ (t < T//2) vì ở đó 
x0_pred đủ sạch để VGG trích xuất feature có ý nghĩa.
========================================================================
"""

def noise_prediction_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor) -> torch.Tensor:
    """
    Hàm mất mát chính của DDPM (Mean Squared Error).
    Đo lường sai số giữa nhiễu mà mô hình dự đoán (eps_pred) và nhiễu thực tế (eps_true).
    """
    return F.mse_loss(eps_pred, eps_true)

def style_diffusion_loss(model, style_encoder, scheduler, batch: dict, weights: dict, device: str) -> tuple[torch.Tensor, dict]:
    """
    Tính tổng các hàm Loss cho Style-guided Diffusion.
    Trả về Tensor tổng (để gọi .backward()) và từ điển (dict) chứa các giá trị loss để log.
    """
    # Đưa ảnh vào thiết bị tính toán
    x0 = batch["content"].to(device)
    style = batch["style"].to(device)
    
    B = x0.shape[0]
    T = scheduler.num_timesteps
    
    # ---------------------------------------------------------
    # 1. Trích xuất Style Embedding
    #    KHÔNG dùng torch.no_grad() ở đây! Nếu detach, MLP của StyleEncoder
    #    sẽ không nhận gradient nào (noise_loss bị chặn tại style_emb, còn
    #    style/content loss đi qua VGG chứ không qua MLP) → MLP đứng yên ở
    #    trọng số random. Bỏ no_grad để gradient từ noise_loss chảy về MLP.
    #    (VGG bên trong vẫn frozen — requires_grad=False — nên autograd không
    #     lưu activation của VGG, chỉ thêm phần MLP rất nhỏ, không gây OOM.)
    # ---------------------------------------------------------
    style_emb = style_encoder.encode_style(style)
        
    # ---------------------------------------------------------
    # 2. Forward Diffusion (Thêm nhiễu ngẫu nhiên)
    # ---------------------------------------------------------
    t = torch.randint(0, T, (B,), device=device)
    x_t, eps_true = scheduler.add_noise(x0, t)
    
    # ---------------------------------------------------------
    # 3. UNet dự đoán nhiễu (Dựa trên x_t, thời gian t, và điều kiện style)
    # ---------------------------------------------------------
    eps_pred = model(x_t, t, style_emb)
    
    # 4. Tính toán Noise Loss (Mục tiêu cốt lõi)
    loss_noise = noise_prediction_loss(eps_pred, eps_true)
    
    # ---------------------------------------------------------
    # 5. Dự đoán lại ảnh gốc x_0 — KHÔNG DETACH, KHÔNG NO_GRAD
    #    để gradient từ perceptual loss chảy ngược về UNet
    # ---------------------------------------------------------
    x0_pred = scheduler.predict_x0(x_t, t, eps_pred)
    
    # Mask lọc ảnh ở giai đoạn ÍT nhiễu (t < T//5 ≈ t<200).
    # Chỉ ở t thấp, x0_pred mới đủ sạch để VGG trích được style/content có nghĩa;
    # nếu lấy tới T//2 thì x0_pred còn rất nhiễu → gradient perceptual bị bẩn.
    # Lưu ý: t vẫn phân bố ĐỀU trên [0, T) cho noise_loss (không làm lệch DDPM),
    # mask này chỉ áp cho perceptual loss.
    t_mask = (t < T // 5)
        
    # Khởi tạo loss phụ bằng 0.0 an toàn
    loss_style = torch.tensor(0.0, device=device)
    loss_content = torch.tensor(0.0, device=device)
    
    # ---------------------------------------------------------
    # 6. Tính Style Loss (Dựa trên Gram Matrix)
    # ---------------------------------------------------------
    if t_mask.any() and weights.get("style", 0.0) > 0:
        # Chuyển x0_pred và ảnh style chuẩn sang miền ImageNet để đưa vào VGG
        pred_style_in = style_encoder._to_vgg_input(x0_pred[t_mask])
        true_style_in = style_encoder._to_vgg_input(style[t_mask])
        
        pf = style_encoder.vgg(pred_style_in)["style_feats"]
        # Target không cần gradient
        with torch.no_grad():
            tf = style_encoder.vgg(true_style_in)["style_feats"]
        loss_style = style_encoder.compute_style_loss(pf, tf)
        
    # ---------------------------------------------------------
    # 7. Tính Content Loss (Giữ nguyên cấu trúc ban đầu)
    # ---------------------------------------------------------
    if t_mask.any() and weights.get("content", 0.0) > 0:
        pc = style_encoder.encode_content(x0_pred[t_mask])
        # x0 gốc không cần tính gradient, dùng làm mỏ neo (target)
        with torch.no_grad():
            tc = style_encoder.encode_content(x0[t_mask])
        loss_content = style_encoder.compute_content_loss(pc, tc)
        
    # ---------------------------------------------------------
    # 8. Tổng hợp toàn bộ Loss (Weighted Sum)
    # ---------------------------------------------------------
    w_noise   = weights.get("noise", 1.0)
    w_style   = weights.get("style", 0.0)
    w_content = weights.get("content", 0.0)

    total_loss = (w_noise * loss_noise
                + w_style * loss_style
                + w_content * loss_content)

    # Từ điển hỗ trợ log. Ngoài giá trị THÔ (raw) của từng loss, log thêm phần
    # ĐÓNG GÓP CÓ TRỌNG SỐ (weighted) để biết style có thực sự ảnh hưởng tới total
    # hay không — dùng để tinh chỉnh loss_weights sao cho style_w ~ noise_w.
    info = {
        "noise_loss": loss_noise.item(),
        "style_loss": loss_style.item(),
        "content_loss": loss_content.item(),
        "total_loss": total_loss.item(),
        # Đóng góp có trọng số vào total_loss
        "noise_w": (w_noise * loss_noise).item(),
        "style_w": (w_style * loss_style).item(),
        "content_w": (w_content * loss_content).item(),
    }

    return total_loss, info


if __name__ == "__main__":
    # ==========================================
    # SMOKE TEST KIỂM TRA LUỒNG TÍNH LOSS
    # ==========================================
    print("=== BẮT ĐẦU SMOKE TEST DIFFUSION LOSS ===")
    
    import os
    import sys
    # Import các module đã xây dựng từ trước
    from src.models.unet import UNet
    from src.models.style_encoder import StyleEncoder
    from src.diffusion.scheduler import DDPMScheduler
    
    # 1. Khởi tạo giả lập
    device = "cpu"
    model = UNet(style_dim=512).to(device)
    enc = StyleEncoder(style_dim=512).to(device)
    sched = DDPMScheduler(num_timesteps=1000, beta_schedule="cosine").to(device)
    
    # 2. Tạo batch dữ liệu thô (miền [-1, 1])
    batch = {
        "content": torch.randn(4, 3, 64, 64),
        "style": torch.randn(4, 3, 64, 64)
    }
    weights = {"noise": 1.0, "style": 0.5, "content": 0.1}
    
    # 3. Tính Loss
    total, info = style_diffusion_loss(model, enc, sched, batch, weights, device)
    
    # 4. Xác minh shape (phải là Scalar vô hướng)
    assert total.shape == torch.Size([]), f"FAIL: Total loss shape {total.shape}"
    
    # 5. Chạy lan truyền ngược (Backpropagation)
    total.backward()
    
    # 6. Kiểm tra: UNet phải nhận gradient từ tất cả loss components
    has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "FAIL: UNet không nhận được gradient!"
    
    # 7. Kiểm tra an toàn: Đảm bảo mạng VGG19 không bị rò rỉ gradient (phải đóng băng tuyệt đối)
    for name, p in enc.vgg.named_parameters():
        assert p.grad is None, f"FAIL: Leak gradient tại layer {name} của VGG!"

    # 8. [REGRESSION CHO FIX #1] MLP của StyleEncoder PHẢI nhận gradient.
    #    Trước khi fix (encode_style nằm trong torch.no_grad) thì grad sẽ là None
    #    → MLP đứng yên ở trọng số random. Sau fix phải có gradient khác 0.
    mlp_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.mlp.parameters()
    )
    assert mlp_has_grad, "FAIL: MLP của StyleEncoder KHÔNG nhận gradient (lỗi #1 chưa được fix)!"

    print(f"✓ Output shape Scalar OK")
    print(f"✓ Backpropagation thành công — gradient chảy đến UNet")
    print(f"✓ VGG frozen (0 leak gradient) OK")
    print(f"✓ MLP StyleEncoder NHẬN gradient (fix #1 OK)")
    print(f"✓ Info dict: {info}")
    print("=== SMOKE 4A: PASS ===")