# 23KDL_Deep_learning/adain-baseline/src/losses/perceptual.py

import torch
import torch.nn.functional as F


def compute_mean_std(feat, eps=1e-5):
    """
    feat: [B, C, H, W]
    trả về mean/std theo từng channel
    """
    b, c = feat.size()[:2]
    feat_ = feat.view(b, c, -1)
    mean = feat_.mean(dim=2).view(b, c, 1, 1)
    var = feat_.var(dim=2, unbiased=False).view(b, c, 1, 1)
    std = (var + eps).sqrt()
    return mean, std


def channel_correlation_matrix(feat):
    """
    feat: [B, C, H, W]
    trả về ma trận tương quan kênh [B, C, C]
    """
    b, c, h, w = feat.shape
    x = feat.view(b, c, -1)                 # [B, C, N]
    x = x - x.mean(dim=2, keepdim=True)     # center để ổn định hơn
    n = x.size(2)
    mat = torch.bmm(x, x.transpose(1, 2)) / max(n, 1)
    return mat


def _base_loss(x, y, use_smooth_l1=True):
    if use_smooth_l1:
        return F.smooth_l1_loss(x, y)
    return F.l1_loss(x, y)


def content_loss(output_feat, adain_feat, use_smooth_l1=True):
    """
    Content loss: so trực tiếp feature của ảnh output encode lại với t
    """
    target = adain_feat.detach()
    return _base_loss(output_feat, target, use_smooth_l1=use_smooth_l1)


def style_loss(
    output_feat,
    adain_feat,
    matrix_weight=1.0,
    stat_weight=0.25,
    use_smooth_l1=True,
):
    """
    Style loss kiểu custom theo yêu cầu:
    - so ma trận tương quan của output_feat với t
    - cộng thêm mean/std của output_feat với t để ổn định hơn
    """
    target = adain_feat.detach()

    mat_out = channel_correlation_matrix(output_feat)
    mat_t = channel_correlation_matrix(target)
    loss_matrix = _base_loss(mat_out, mat_t, use_smooth_l1=use_smooth_l1)

    mean_out, std_out = compute_mean_std(output_feat)
    mean_t, std_t = compute_mean_std(target)
    loss_stat = (
        _base_loss(mean_out, mean_t, use_smooth_l1=use_smooth_l1)
        + _base_loss(std_out, std_t, use_smooth_l1=use_smooth_l1)
    )

    total_style = matrix_weight * loss_matrix + stat_weight * loss_stat
    return total_style, loss_matrix, loss_stat


def perceptual_loss(
    output_feat,
    adain_feat,
    lambda_content=1.0,
    lambda_style=3.0,
    matrix_weight=1.0,
    stat_weight=0.25,
    use_smooth_l1=True,
):
    """
    Tổng loss custom:
        total = lambda_content * content_loss
              + lambda_style  * style_loss

    Ở đây cả content và style đều neo vào t theo đúng yêu cầu.
    """
    c_loss = content_loss(
        output_feat=output_feat,
        adain_feat=adain_feat,
        use_smooth_l1=use_smooth_l1,
    )

    s_loss, s_matrix, s_stat = style_loss(
        output_feat=output_feat,
        adain_feat=adain_feat,
        matrix_weight=matrix_weight,
        stat_weight=stat_weight,
        use_smooth_l1=use_smooth_l1,
    )

    total = lambda_content * c_loss + lambda_style * s_loss
    return total, c_loss, s_loss, s_matrix, s_stat