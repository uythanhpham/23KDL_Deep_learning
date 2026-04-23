import torch
import torch.nn.functional as F


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    # img: [B, C, H, W]
    loss_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return loss_h + loss_w


def adain_target_reconstruction_loss(
    encoder,
    output_img: torch.Tensor,
    t: torch.Tensor,
    lambda_mse: float = 10.0,
    lambda_l1: float = 0.5,
    lambda_tv: float = 1e-5,
):
    """
    Mục tiêu:
        z = encoder(output_img)
        ép z ~= t

    output_img: ảnh decoder sinh ra, shape [B, 3, H, W]
    t: AdaIN target feature, shape [B, C, h, w]
    """

    # encode lại ảnh generate về cùng feature space với t
    z = encoder(output_img)

    # target feature không cho backprop ngược vào nhánh target
    t_det = t.detach()

    loss_mse = F.mse_loss(z, t_det)
    loss_l1 = F.l1_loss(z, t_det)
    loss_tv = total_variation_loss(output_img)

    total_loss = (
        lambda_mse * loss_mse
        + lambda_l1 * loss_l1
        + lambda_tv * loss_tv
    )

    return total_loss, loss_mse, loss_l1, loss_tv