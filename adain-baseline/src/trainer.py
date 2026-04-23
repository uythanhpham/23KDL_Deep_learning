import torch
from src.models.adain import adain
from src.losses.perceptual import adain_target_reconstruction_loss


class AdaINTrainer:
    def __init__(
        self,
        model,
        optimizer,
        lambda_mse=10.0,
        lambda_l1=0.5,
        lambda_tv=1e-5,
        device="cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.lambda_tv = lambda_tv
        self.device = device

    def _forward_and_compute_loss(self, content, style):
        # 1) encode content/style
        content_feat = self.model.encoder(content)   # [B, 512, H/8, W/8]
        style_feat = self.model.encoder(style)       # [B, 512, H/8, W/8]

        # 2) AdaIN target
        t = adain(content_feat, style_feat)

        # 3) decode
        output = self.model.decoder(t).clamp(0, 1)

        # 4) loss mới: encode lại output và ép sát t
        total_loss, loss_mse, loss_l1, loss_tv = adain_target_reconstruction_loss(
            encoder=self.model.encoder,
            output_img=output,
            t=t,
            lambda_mse=self.lambda_mse,
            lambda_l1=self.lambda_l1,
            lambda_tv=self.lambda_tv,
        )

        return total_loss, loss_mse, loss_l1, loss_tv

    def train_step(self, content, style):
        self.model.train()
        self.optimizer.zero_grad()

        total_loss, loss_mse, loss_l1, loss_tv = self._forward_and_compute_loss(content, style)

        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "mse_loss": loss_mse.item(),
            "l1_loss": loss_l1.item(),
            "tv_loss": loss_tv.item(),
        }

    @torch.no_grad()
    def validate_step(self, content, style):
        self.model.eval()

        total_loss, loss_mse, loss_l1, loss_tv = self._forward_and_compute_loss(content, style)

        return {
            "total_loss": total_loss.item(),
            "mse_loss": loss_mse.item(),
            "l1_loss": loss_l1.item(),
            "tv_loss": loss_tv.item(),
        }