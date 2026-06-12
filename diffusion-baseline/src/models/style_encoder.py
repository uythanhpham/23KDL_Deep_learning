import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict
from torch.amp import autocast

def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    Tính ma trận Gram (Gram Matrix) của một feature map.
    """
    # Bọc bằng context tắt autocast để bắt buộc tính toán trên float32 an toàn
    with autocast('cuda', enabled=False):
        feat = feat.float()
        B, C, H, W = feat.shape
        # Reshape: (B, C, H, W) -> (B, C, H*W)
        feat_reshaped = feat.view(B, C, H * W)
        # Nhân ma trận với ma trận chuyển vị của nó (B, C, H*W) x (B, H*W, C) -> (B, C, C)
        G = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
        # Chuẩn hóa để tránh giá trị quá lớn
        return G / (C * H * W)

class VGGFeatureExtractor(nn.Module):
    """
    Trích xuất đặc trưng từ mạng VGG19 đã được pre-train trên ImageNet.
    VGG luôn được đóng băng (frozen - không cập nhật weights) và ở chế độ eval().
    """
    def __init__(self, style_layers=[3, 8, 17, 26], content_layers=[17]):
        super().__init__()
        # Load VGG19 pretrained
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # Chia mạng VGG thành 4 phần (slices) dựa trên các index layer yêu cầu
        self.slice_0 = nn.Sequential()
        self.slice_1 = nn.Sequential()
        self.slice_2 = nn.Sequential()
        self.slice_3 = nn.Sequential()

        for x in range(4):
            self.slice_0.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice_1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice_3.add_module(str(x), vgg_pretrained_features[x])

        # VGG luôn ở trạng thái eval() và không tính gradient
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        # x yêu cầu đã được normalize theo chuẩn ImageNet
        h_relu1 = self.slice_0(x)
        h_relu2 = self.slice_1(h_relu1)
        h_relu3 = self.slice_2(h_relu2)
        h_relu4 = self.slice_3(h_relu3)
        
        # style_feats gồm 4 layers: [3, 8, 17, 26]
        style_feats = [h_relu1, h_relu2, h_relu3, h_relu4]
        # content_feats lấy layer 17 (chính là đầu ra của slice_2)
        content_feats = [h_relu3]

        return {"style_feats": style_feats, "content_feats": content_feats}


class StyleEncoder(nn.Module):
    """
    Mã hóa ảnh Style thành vector style_emb (để inject vào AdaIN) 
    và cung cấp các hàm tính Loss cho Content/Style.
    """
    def __init__(self, style_dim: int = 512, **kwargs):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        
        # Tính tổng số kênh cho MLP:
        # 4 lớp VGG có số kênh lần lượt là [64, 128, 256, 512]
        # Mỗi kênh ta lấy cả Mean và Std -> nhân đôi -> 2 * (64 + 128 + 256 + 512) = 1920
        self.mlp = nn.Sequential(
            nn.Linear(1920, 1024),
            nn.SiLU(),
            nn.Linear(1024, style_dim)
        )

        # Vector "KHÔNG style" (null) cho Classifier-Free Guidance.
        # Lúc train: thỉnh thoảng thay style_emb bằng vector này (style dropout) để model
        # học cả nhánh có-style lẫn null. Lúc sinh: dùng làm nhánh unconditional.
        # Là nn.Parameter nên được train cùng MLP và lưu trong checkpoint.
        self.null_style = nn.Parameter(torch.zeros(style_dim))

    def _to_vgg_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        [QUAN TRỌNG] Dùng ở nhiều chỗ để chuyển đổi Tensor từ miền [-1, 1] 
        của UNet Diffusion sang miền chuẩn hóa của ImageNet để đưa vào VGG.
        """
        # Đưa từ [-1, 1] về [0, 1]
        x = x.float()
        x01 = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        
        # ImageNet Mean và Std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device, dtype=x.dtype)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device, dtype=x.dtype)
        
        return (x01 - mean) / std

    def encode_style(self, style_img: torch.Tensor) -> torch.Tensor:
        """Trích xuất ra vector style_emb kích thước (B, style_dim)"""
        vgg_in = self._to_vgg_input(style_img)
        feats = self.vgg(vgg_in)["style_feats"]
        
        # Tính Mean và Std trên không gian (H, W) tức là dim [2, 3] cho từng feature map
        stats = [torch.cat([f.mean(dim=[2, 3]), f.std(dim=[2, 3])], dim=1) for f in feats]
        
        # Nối tất cả các stats lại với nhau -> (B, 1920)
        concat = torch.cat(stats, dim=1)
        
        # Đi qua MLP để nén/chiếu về kích thước style_dim yêu cầu
        return self.mlp(concat)

    def encode_content(self, content_img: torch.Tensor) -> torch.Tensor:
        """Trích xuất feature map gốc để tính Content Loss"""
        vgg_in = self._to_vgg_input(content_img)
        # Chỉ trả về feature đầu tiên trong list content_feats
        return self.vgg(vgg_in)["content_feats"][0]

    def compute_style_loss(self, pred_feats: List[torch.Tensor], target_feats: List[torch.Tensor]) -> torch.Tensor:
        """Tính tổng MSE Loss của Gram Matrix giữa các features dự đoán và mục tiêu"""
        loss = 0.0
        for p, t in zip(pred_feats, target_feats):
            loss += F.mse_loss(gram_matrix(p), gram_matrix(t))
        return loss

    def compute_content_loss(self, pred_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        """Tính MSE Loss cơ bản để ép cấu trúc không gian của ảnh"""
        return F.mse_loss(pred_feat, target_feat)


if __name__ == "__main__":
    # ==========================================
    # SMOKE TEST STYLE ENCODER
    # ==========================================
    enc = StyleEncoder(512).eval()
    
    # Tạo fake image cho style và content trong miền [-1, 1]
    si = torch.randn(2, 3, 256, 256)
    ci = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        se = enc.encode_style(si)
        cf = enc.encode_content(ci)
        
    assert se.shape == torch.Size([2, 512]), f"FAIL: style_emb shape {se.shape}"
    # Mạng VGG layer 17 (relu3_3) có 256 channels
    assert cf.shape[1] == 256, f"FAIL: content_feat channel size {cf.shape[1]}"
    
    # Kiểm tra đảm bảo toàn bộ VGG đã bị đóng băng gradient
    for p in enc.vgg.parameters(): 
        assert not p.requires_grad, "FAIL: VGG parameter requires_grad là True!"
        
    print(f"✓ style_emb: {se.shape} | content_feat: {cf.shape}")
    print("✓ VGG frozen OK")
    print("=== SMOKE 2D: PASS ===")