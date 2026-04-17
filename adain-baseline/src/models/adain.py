# 23KDL_Deep_learning/adain-baseline/src/models/adain.py

"""Chứa Encoder VGG, lớp AdaIN, và Decoder."""
import torch
import torch.nn as nn
import torchvision.models as models


def adain(content_feat, style_feat, eps=1e-5):
    """
    Adaptive Instance Normalization
    content_feat, style_feat: (B, C, H, W)
    AdaIN(x, y) = σ(y) * ((x − μ(x)) / σ(x)) + μ(y)

    FIX: tách batch_size của content và style riêng để tránh lỗi
    khi 2 tensor có batch size khác nhau.
    """
    b_c, channels, height, width = content_feat.size()
    b_s = style_feat.size(0)

    # content statistics
    c_mean = content_feat.view(b_c, channels, -1).mean(dim=2).view(b_c, channels, 1, 1)
    c_std  = content_feat.view(b_c, channels, -1).std(dim=2).view(b_c, channels, 1, 1) + eps

    # style statistics
    s_mean = style_feat.view(b_s, channels, -1).mean(dim=2).view(b_s, channels, 1, 1)
    s_std  = style_feat.view(b_s, channels, -1).std(dim=2).view(b_s, channels, 1, 1) + eps

    # normalize content rồi scale theo style
    normalized = (content_feat - c_mean) / c_std
    return s_std * normalized + s_mean


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # FIX: dùng weights= thay pretrained= (tránh DeprecationWarning từ torchvision >= 0.13)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Slices cho relu1_1, relu2_1, relu3_1, relu4_1
        self.slice1 = nn.Sequential(*list(vgg)[:2])    # layer 0-1  : relu1_1 (64ch)
        self.slice2 = nn.Sequential(*list(vgg)[2:7])   # layer 2-6  : relu2_1 (128ch)
        self.slice3 = nn.Sequential(*list(vgg)[7:12])  # layer 7-11 : relu3_1 (256ch)
        self.slice4 = nn.Sequential(*list(vgg)[12:21]) # layer 12-20: relu4_1 (512ch)

        for p in self.parameters():
            p.requires_grad = False  # Freeze encoder

    def forward(self, x, return_all=False):
        h1 = self.slice1(x)   # (B,  64, H,   W  )
        h2 = self.slice2(h1)  # (B, 128, H/2, W/2)
        h3 = self.slice3(h2)  # (B, 256, H/4, W/4)
        h4 = self.slice4(h3)  # (B, 512, H/8, W/8)
        # return_all=True dùng khi tính perceptual loss (cần cả 4 feature maps)
        return (h1, h2, h3, h4) if return_all else h4


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),  # RGB output
        )

    def forward(self, x):
        return self.net(x)


class AdaINStyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = Decoder()

    def forward(self, content, style, alpha=1.0):
        # 1. Encode
        content_feat = self.encoder(content)  # (B, 512, H/8, W/8)
        style_feat   = self.encoder(style)    # (B, 512, H/8, W/8)

        # 2. AdaIN
        t = adain(content_feat, style_feat)   # (B, 512, H/8, W/8)

        # 3. Interpolate (alpha=1.0: full style, alpha=0.0: giữ nguyên content)
        t = alpha * t + (1 - alpha) * content_feat

        # 4. Decode → ảnh, clamp về [0, 1] để đồng nhất với range của dataset
        # FIX: thêm clamp tránh giá trị âm hoặc > 1 khi tính loss / visualize
        return self.decoder(t).clamp(0, 1)    # (B, 3, H, W)
