import torch
from src.models.adain import adain
from src.losses.perceptual import perceptual_loss


class AdaINTrainer:
    def __init__(
        self,
        model,
        optimizer,
        lambda_content=1.0,
        lambda_style=3.0,
        style_matrix_weight=1.0,
        style_stat_weight=0.25,
        loss_use_smooth_l1=True,
        device="cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.style_matrix_weight = style_matrix_weight
        self.style_stat_weight = style_stat_weight
        self.loss_use_smooth_l1 = loss_use_smooth_l1
        self.device = device

    def _forward_and_compute_loss(self, content, style):
        # 1) encode
        content_feat = self.model.encoder(content)   # [B, 512, H/8, W/8]
        style_feat = self.model.encoder(style)       # [B, 512, H/8, W/8]

        # 2) AdaIN target
        t = adain(content_feat, style_feat)

        # 3) decode
        output = self.model.decoder(t).clamp(0, 1)

        # 4) encode lại ảnh output về cùng feature space với t
        output_feat = self.model.encoder(output)

        # 5) custom t-anchored loss
        total_loss, loss_c, loss_s, loss_s_mat, loss_s_stat = perceptual_loss(
            output_feat=output_feat,
            adain_feat=t,
            lambda_content=self.lambda_content,
            lambda_style=self.lambda_style,
            matrix_weight=self.style_matrix_weight,
            stat_weight=self.style_stat_weight,
            use_smooth_l1=self.loss_use_smooth_l1,
        )

        return total_loss, loss_c, loss_s, loss_s_mat, loss_s_stat

    def train_step(self, content, style):
        self.model.train()
        self.optimizer.zero_grad()

        total_loss, loss_c, loss_s, loss_s_mat, loss_s_stat = self._forward_and_compute_loss(content, style)

        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "content_loss": loss_c.item(),
            "style_loss": loss_s.item(),
            "style_matrix_loss": loss_s_mat.item(),
            "style_stat_loss": loss_s_stat.item(),
        }

    @torch.no_grad()
    def validate_step(self, content, style):
        self.model.eval()

        total_loss, loss_c, loss_s, loss_s_mat, loss_s_stat = self._forward_and_compute_loss(content, style)

        return {
            "total_loss": total_loss.item(),
            "content_loss": loss_c.item(),
            "style_loss": loss_s.item(),
            "style_matrix_loss": loss_s_mat.item(),
            "style_stat_loss": loss_s_stat.item(),
        }