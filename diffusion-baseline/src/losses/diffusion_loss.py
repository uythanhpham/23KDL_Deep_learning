import torch
import torch.nn.functional as F

"""
========================================================================
GIẢI THÍCH TRICK: t_mask (Lọc Timestep cho Perceptual Loss)
========================================================================
Trong Diffusion, ở các bước timestep lớn (ví dụ t > 700), ảnh x_t gần 
như là nhiễu thuần túy (pure noise). Lúc này, mạng UNet dự đoán x_0 
(x0_pred) sẽ cực kỳ mờ và thiếu chính xác.
Nếu ta ép mạng VGG19 (vốn được train trên ảnh nét rõ) phải trích xuất 
Gram Matrix từ x0_pred quá mờ này, Loss sinh ra sẽ rất "ảo" và làm nhiễu 
quá trình huấn luyện.
=> TRICK: Ta chỉ kích hoạt Style Loss và Content Loss khi timestep t 
nằm ở khoảng nhỏ (ví dụ t < T/4). Lúc này ảnh x_t chỉ bị nhiễu nhẹ, 
x0_pred đủ sắc nét để VGG đánh giá phong cách và cấu trúc một cách chuẩn xác.
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
    # Xóa ảnh vào thiết bị tính toán
    x0 = batch["content"].to(device)
    style = batch["style"].to(device)
    
    B = x0.shape[0]
    T = scheduler.num_timesteps
    
    # ---------------------------------------------------------
    # 1. Trích xuất Style Embedding (Đóng băng VGG)
    # ---------------------------------------------------------
    with torch.no_grad():
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
    # 5. Dự đoán lại ảnh gốc x_0 để chuẩn bị tính Perceptual Loss
    # ---------------------------------------------------------
    with torch.no_grad():
        # Dự đoán x0_pred nhưng tách khỏi đồ thị gradient (detach)
        x0_pred = scheduler.predict_x0(x_t, t, eps_pred.detach())
        # Tạo mask lọc các ảnh ở giai đoạn ít nhiễu
        t_mask = (t < T // 4)
        
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
        tf = style_encoder.vgg(true_style_in)["style_feats"]
        loss_style = style_encoder.compute_style_loss(pf, tf)
        
    # ---------------------------------------------------------
    # 7. Tính Content Loss (Giữ nguyên cấu trúc ban đầu)
    # ---------------------------------------------------------
    if t_mask.any() and weights.get("content", 0.0) > 0:
        pc = style_encoder.encode_content(x0_pred[t_mask])
        # x0 gốc không cần tính gradient, dùng làm mỏ neo (target)
        tc = style_encoder.encode_content(x0[t_mask].detach())
        loss_content = style_encoder.compute_content_loss(pc, tc)
        
    # ---------------------------------------------------------
    # 8. Tổng hợp toàn bộ Loss (Weighted Sum)
    # ---------------------------------------------------------
    total_loss = (weights.get("noise", 1.0) * loss_noise 
                + weights.get("style", 0.0) * loss_style 
                + weights.get("content", 0.0) * loss_content)
                
    # Từ điển hỗ trợ log lên màn hình / Weights & Biases
    info = {
        "noise_loss": loss_noise.item(),
        "style_loss": loss_style.item(),
        "content_loss": loss_content.item(),
        "total_loss": total_loss.item()
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
    sched = DDPMScheduler(num_timesteps=1000, beta_schedule="linear").to(device)
    
    # 2. Tạo batch dữ liệu thô (miền [-1, 1])
    batch = {
        "content": torch.randn(4, 3, 64, 64),
        "style": torch.randn(4, 3, 64, 64)
    }
    weights = {"noise": 1.0, "style": 0.1, "content": 0.01}
    
    # 3. Tính Loss
    total, info = style_diffusion_loss(model, enc, sched, batch, weights, device)
    
    # 4. Xác minh shape (phải là Scalar vô hướng)
    assert total.shape == torch.Size([]), f"FAIL: Total loss shape {total.shape}"
    
    # 5. Chạy lan truyền ngược (Backpropagation)
    total.backward()
    
    # 6. Kiểm tra an toàn: Đảm bảo mạng VGG19 không bị rò rỉ gradient (phải đóng băng tuyệt đối)
    for name, p in enc.vgg.named_parameters():
        assert p.grad is None, f"FAIL: Leak gradient tại layer {name} của VGG!"
        
    print(f"✓ Output shape Scalar OK")
    print(f"✓ Backpropagation thành công (không crash)")
    print(f"✓ VGG frozen (0 leak gradient) OK")
    print(f"✓ Info dict: {info}")
    print("=== SMOKE 4A: PASS ===")