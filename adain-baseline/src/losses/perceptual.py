# 23KDL_Deep_learning/adain-baseline/src/losses/perceptual.py
"""Chấm điểm content loss và style loss."""

import torch.nn as nn  # FIX: import đúng, bỏ `from torch.nn.modules import loss`


def compute_mean_std(feat, eps=1e-5):
    """Tính mean và std theo spatial (H, W) cho mỗi (batch, channel)."""
    batch_size, channels = feat.size()[:2]
    mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    std  = feat.view(batch_size, channels, -1).std(dim=2).view(batch_size, channels, 1, 1) + eps
    return mean, std


def content_loss(output_feat, adain_feat):
    """
    Content loss: MSE giữa feature map của ảnh output (sau decode + encode lại)
    và feature map sau AdaIN (t).
    Cả 2 đều ở relu4_1 → shape (B, 512, H/8, W/8).
    """
    # FIX: đổi tên tham số cho rõ nghĩa hơn (adain_feat thay vì target_feat)
    return nn.MSELoss()(output_feat, adain_feat)


def style_loss(output_feats, style_feats):
    """
    Style loss: tổng MSE của mean và std tại 4 layer relu của VGG
    (relu1_1, relu2_1, relu3_1, relu4_1).

    output_feats: tuple (h1, h2, h3, h4) từ encoder(output_image, return_all=True)
    style_feats : tuple (h1, h2, h3, h4) từ encoder(style_image,  return_all=True)
    """
    total = 0.0
    for out_f, style_f in zip(output_feats, style_feats):
        out_mean,   out_std   = compute_mean_std(out_f)
        style_mean, style_std = compute_mean_std(style_f)
        total += nn.MSELoss()(out_mean, style_mean)
        total += nn.MSELoss()(out_std,  style_std)
    return total


def perceptual_loss(output_feats, style_feats, adain_feat, lambda_style=10.0):
    """
    Tổng loss = content_loss + lambda_style * style_loss

    Args:
        output_feats : tuple (h1,h2,h3,h4) — encoder(output, return_all=True)
        style_feats  : tuple (h1,h2,h3,h4) — encoder(style,  return_all=True)
        adain_feat   : tensor (B,512,H,W)  — feature sau AdaIN (t) trong forward()
        lambda_style : trọng số style loss, mặc định 10.0 theo paper

    Returns:
        total_loss, c_loss, s_loss  (trả về riêng để log dễ theo dõi)
    """
    # output relu4_1 là phần tử cuối của tuple
    c_loss = content_loss(output_feats[-1], adain_feat)
    s_loss = style_loss(output_feats, style_feats)
    total  = c_loss + lambda_style * s_loss
    return total, c_loss, s_loss
