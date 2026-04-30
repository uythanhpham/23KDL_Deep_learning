"""Memory-safe Trainer cho AdaIN decoder."""

from __future__ import annotations

import gc
from contextlib import nullcontext
import torch

from src.models.adain import adain
from src.losses.perceptual import perceptual_loss


class AdaINTrainer:
    def __init__(self, model, optimizer, lambda_style=10.0, device="cpu", use_amp=True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lambda_style = lambda_style
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        self.model.encoder.eval()
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            except TypeError:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except TypeError:
            return torch.cuda.amp.autocast(enabled=True)

    def train_step(self, content, style):
        self.model.decoder.train()
        content = content.to(self.device, non_blocking=False)
        style = style.to(self.device, non_blocking=False)
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast():
            with torch.no_grad():
                content_feat = self.model.encoder(content)
                style_feats = self.model.encoder(style, return_all=True)
                style_feat = style_feats[-1]
                target = adain(content_feat, style_feat).detach()
                style_feats = tuple(f.detach() for f in style_feats)

            output = self.model.decoder(target)
            output_feats = self.model.encoder(output, return_all=True)
            total_loss, c_loss, s_loss = perceptual_loss(
                output_feats=output_feats,
                style_feats=style_feats,
                adain_target_feat=target,
                lambda_style=self.lambda_style,
            )

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        result = {
            "total_loss": float(total_loss.detach().item()),
            "content_loss": float(c_loss.detach().item()),
            "style_loss": float(s_loss.detach().item()),
        }

        del content, style, content_feat, style_feats, style_feat, target
        del output, output_feats, total_loss, c_loss, s_loss
        return result

    @torch.no_grad()
    def val_step(self, content, style):
        self.model.decoder.eval()
        content = content.to(self.device, non_blocking=False)
        style = style.to(self.device, non_blocking=False)

        with self._autocast():
            content_feat = self.model.encoder(content)
            style_feats = self.model.encoder(style, return_all=True)
            style_feat = style_feats[-1]
            target = adain(content_feat, style_feat)

            output = self.model.decoder(target)
            output_feats = self.model.encoder(output, return_all=True)
            total_loss, c_loss, s_loss = perceptual_loss(
                output_feats=output_feats,
                style_feats=style_feats,
                adain_target_feat=target,
                lambda_style=self.lambda_style,
            )

        result = {
            "total_loss": float(total_loss.detach().item()),
            "content_loss": float(c_loss.detach().item()),
            "style_loss": float(s_loss.detach().item()),
        }

        del content, style, content_feat, style_feats, style_feat, target
        del output, output_feats, total_loss, c_loss, s_loss
        return result


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
