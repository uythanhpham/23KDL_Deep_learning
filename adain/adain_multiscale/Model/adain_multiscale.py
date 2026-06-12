from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models

# 1. Hàm tính thống kê
def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    b, c = feat.shape[:2]
    feat_var = feat.view(b, c, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

# 2. Phép biến đổi AdaIN
def adain(content_feat: torch.Tensor, style_mean: torch.Tensor, style_std: torch.Tensor):
    c_mean, c_std = calc_mean_std(content_feat)
    return style_std * (content_feat - c_mean) / c_std + style_mean

# 3. Encoder (VGG19 - Cố định)
class VGGEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = vgg[:2]
        self.slice2 = vgg[2:7]
        self.slice3 = vgg[7:12]
        self.slice4 = vgg[12:21]
        for p in self.parameters(): p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4

# 4. Decoder (Multi-layer Style Injection)
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Cấu trúc Decoder phân tầng để tiêm style vào từng cấp độ
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU())
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU())
        self.conv5 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Upsample(scale_factor=2), nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU())
        self.conv7 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Upsample(scale_factor=2), nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU())
        self.output = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3))

    def forward(self, content_feats, style_stats):
        h1, h2, h3, h4 = content_feats
        
        # Tiêm phong cách tại mỗi tầng (Multi-layer injection)
        x = adain(h4, style_stats[3][0], style_stats[3][1])
        x = self.conv1(x)
        
        x = adain(x, style_stats[2][0], style_stats[2][1])
        x = self.conv2(x); x = self.conv3(x); x = self.conv4(x); x = self.conv5(x)
        
        x = adain(x, style_stats[1][0], style_stats[1][1])
        x = self.conv6(x); x = self.conv7(x)
        
        x = adain(x, style_stats[0][0], style_stats[0][1])
        x = self.conv8(x)
        
        return self.output(x)

# 5. Full Model
class AdaINStyleTransfer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = Decoder()

    def forward(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        with torch.no_grad():
            c_feats = self.encoder(content)
            s_feats = self.encoder(style)
            # Tính stats cho cả 4 tầng
            s_stats = [calc_mean_std(f) for f in s_feats]
            
        # Truyền cả feature map của content và bộ stats của style vào decoder
        out = self.decoder(c_feats, s_stats)
        
        # Alpha blending để cân bằng độ mạnh của style
        return alpha * out + (1.0 - alpha) * content