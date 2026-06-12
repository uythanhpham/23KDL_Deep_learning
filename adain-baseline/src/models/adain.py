"AdaIN model: VGG19 encoder cố định + AdaIN + decoder trainable."
from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    b, c = feat.shape[:2]
    feat_var = feat.view(b, c, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adain(content_feat: torch.Tensor, style_feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    c_mean, c_std = calc_mean_std(content_feat, eps)
    s_mean, s_std = calc_mean_std(style_feat, eps)
    return (content_feat - c_mean) / c_std * s_std + s_mean

def _make_vgg19_features() -> nn.Sequential:
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    for i, layer in enumerate(vgg):
        if isinstance(layer, nn.ReLU):
            vgg[i] = nn.ReLU(inplace=False)
    return vgg

class VGGEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg = _make_vgg19_features()
        self.slice1 = nn.Sequential(*list(vgg)[:2])     # relu1_1
        self.slice2 = nn.Sequential(*list(vgg)[2:7])    # relu2_1
        self.slice3 = nn.Sequential(*list(vgg)[7:12])   # relu3_1
        self.slice4 = nn.Sequential(*list(vgg)[12:21])  # relu4_1
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor, return_all: bool = False):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return (h1, h2, h3, h4) if return_all else h4

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AdaINStyleTransfer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = Decoder()

    def forward(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        with torch.no_grad():
            content_feat = self.encoder(content)
            style_feat = self.encoder(style)
            t = adain(content_feat, style_feat)
            t = alpha * t + (1.0 - alpha) * content_feat
        return self.decoder(t)
