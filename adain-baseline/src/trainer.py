# 23KDL_Deep_learning/adain-baseline/src/trainer.py
"""Chạy vòng lặp train/validate và lưu checkpoint."""
# NOTE ****:
import torch
from src.models.adain import adain
from src.losses.perceptual import perceptual_loss

class AdaINTrainer:
    def __init__(self, model, optimizer, lambda_style=10.0, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lambda_style = lambda_style
        self.device = device

    def train_step(self, content, style):
        """Thực hiện 1 bước forward -> loss -> backward -> update"""
        self.model.train()
        self.optimizer.zero_grad()

        # --- Forward ---
        content_feat = self.model.encoder(content)
        style_feats = self.model.encoder(style, return_all=True)
        style_feat = style_feats[-1]

        t = adain(content_feat, style_feat)
        output = self.model.decoder(t).clamp(0, 1)

        output_feats = self.model.encoder(output, return_all=True)

        # --- Loss Calculation ---
        total_loss, loss_c, loss_s = perceptual_loss(
            output_feats, style_feats, t, lambda_style=self.lambda_style
        )

        # --- Backward & Update ---
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "content_loss": loss_c.item(),
            "style_loss": loss_s.item()
        }
