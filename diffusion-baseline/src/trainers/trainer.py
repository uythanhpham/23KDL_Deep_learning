import copy
import torch
import torch.nn.functional as F                          # ← thêm import F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from src.losses.diffusion_loss import style_diffusion_loss

class DiffusionTrainer:
    def __init__(self, model, style_encoder, scheduler, optimizer,
                 device, grad_clip=1.0, ema_decay=0.9999,
                 loss_weights=None, mixed_precision=False):   # ← thêm mixed_precision
        
        self.model          = model
        self.style_encoder  = style_encoder
        self.scheduler      = scheduler
        self.optimizer      = optimizer
        self.device         = device
        self.grad_clip      = grad_clip
        self.ema_decay      = ema_decay
        self.mixed_precision = mixed_precision               # ← lưu lại
        self.scaler         = GradScaler(enabled=mixed_precision)  # ← khởi tạo ở đây

        if loss_weights is None:
            self.loss_weights = {"noise": 1.0, "style": 0.1, "content": 0.01}
        else:
            self.loss_weights = loss_weights

        self.style_encoder.eval()
        for p in self.style_encoder.parameters():
            p.requires_grad = False

        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def _update_ema(self):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                if model_p.requires_grad:
                    ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1.0 - self.ema_decay)

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        with autocast(enabled=self.mixed_precision):
            total_loss, info = style_diffusion_loss(
                model=self.model,
                style_encoder=self.style_encoder,
                scheduler=self.scheduler,
                batch=batch,
                weights=self.loss_weights,
                device=self.device
            )

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
        self.scaler.step(self.optimizer)    # ← chỉ gọi 1 lần
        self.scaler.update()
        # ← bỏ optimizer.step() và clip_grad_norm_() thừa

        self._update_ema()

        return {
            "train_loss":   info["total_loss"],
            "noise_loss":   info["noise_loss"],
            "style_loss":   info["style_loss"],
            "content_loss": info["content_loss"],
            "grad_norm":    grad_norm,
        }

    def val_step(self, batch: dict) -> dict:
        self.model.eval()
        with torch.no_grad():
            x0        = batch["content"].to(self.device)
            style     = batch["style"].to(self.device)
            style_emb = self.style_encoder.encode_style(style)

            t = torch.randint(0, self.scheduler.num_timesteps,
                              (x0.shape[0],), device=self.device)
            x_t, eps_true = self.scheduler.add_noise(x0, t)
            eps_pred      = self.model(x_t, t, style_emb)
            val_loss      = F.mse_loss(eps_pred, eps_true).item()  # ← F đã import

        return {"val_loss": val_loss, "noise_loss": val_loss,
                "style_loss": 0.0, "content_loss": 0.0}

    def save_checkpoint(self, path: str, epoch: int):
        torch.save({
            "epoch":        epoch,
            "model":        self.model.state_dict(),
            "ema_model":    self.ema_model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "loss_weights": self.loss_weights,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.ema_model.load_state_dict(ckpt["ema_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "loss_weights" in ckpt:
            self.loss_weights = ckpt["loss_weights"]
        return ckpt["epoch"]