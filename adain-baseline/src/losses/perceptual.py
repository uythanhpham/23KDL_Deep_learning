"Content/style perceptual loss cho AdaIN."
from __future__ import annotations
import torch
import torch.nn.functional as F

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    b, c = feat.shape[:2]
    feat_var = feat.view(b, c, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def content_loss(output_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output_feat, target_feat)

def style_loss(output_feats, style_feats) -> torch.Tensor:
    loss = 0.0
    for out_f, style_f in zip(output_feats, style_feats):
        out_mean, out_std = calc_mean_std(out_f)
        style_mean, style_std = calc_mean_std(style_f)
        loss = loss + F.mse_loss(out_mean, style_mean)
        loss = loss + F.mse_loss(out_std, style_std)
    return loss

def perceptual_loss(output_feats, style_feats, adain_target_feat: torch.Tensor, lambda_style: float = 10.0):
    c_loss = content_loss(output_feats[-1], adain_target_feat)
    s_loss = style_loss(output_feats, style_feats)
    total = c_loss + lambda_style * s_loss
    return total, c_loss, s_loss
