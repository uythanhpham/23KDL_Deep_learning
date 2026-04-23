# 23KDL_Deep_learning/adain-baseline/src/trainer.py
"""Chạy vòng lặp t_trainrain/validat_traine và lưu checkpoint_train."""
# NOTE ****:
import torch
from adain_baseline.src.models.adain import adain
from adain_baseline.src.losses.perceptual import perceptual_loss

class AdaINTrainer:
    def __init__(self, model, optimizer, lambda_style=10.0, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lambda_style = lambda_style
        self.device = device

    def train_step(self, train_content, train_style, valid_content, valid_style):
        """Thực hiện 1 bước forward -> loss -> backward -> update"""
        # === TRAIN ===
        self.model.train()
        self.optimizer.zero_grad()

        train_content_feat = self.model.encoder(train_content)
        train_style_feats  = self.model.encoder(train_style, return_all=True)
        train_style_feat   = train_style_feats[-1]
        t_train            = adain(train_content_feat, train_style_feat)
        train_output       = self.model.decoder(t_train)
        train_output_feats = self.model.encoder(train_output, return_all=True)

        train_total_loss, train_loss_c, train_loss_s = perceptual_loss(
            train_output_feats, train_style_feats, t_train,
            lambda_style=self.lambda_style
        )

        train_total_loss.backward()   # ← backward before validation
        self.optimizer.step()

        # === VALIDATION ===
        self.model.eval()
        with torch.no_grad():
            valid_content_feat = self.model.encoder(valid_content)
            valid_style_feats  = self.model.encoder(valid_style, return_all=True)
            valid_style_feat   = valid_style_feats[-1]
            t_valid            = adain(valid_content_feat, valid_style_feat)
            valid_output       = self.model.decoder(t_valid)
            valid_output_feats = self.model.encoder(valid_output, return_all=True)

            valid_total_loss, valid_loss_c, valid_loss_s = perceptual_loss(
                valid_output_feats, valid_style_feats, t_valid,
                lambda_style=self.lambda_style
            )

        return {
            "train_total_loss":   train_total_loss.item(),
            "train_content_loss": train_loss_c.item(),
            "train_style_loss":   train_loss_s.item(),
            "valid_total_loss":   valid_total_loss.item(),
            "valid_content_loss": valid_loss_c.item(),
            "valid_style_loss":   valid_loss_s.item(),
        }
