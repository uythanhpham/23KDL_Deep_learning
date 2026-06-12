import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler
from src.losses.diffusion_loss import StyleDiffusionLoss


def unwrap(m):
    """Trả về module gốc, gỡ lớp DataParallel/DDP nếu có."""
    return m.module if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else m


class DiffusionTrainer:
    def __init__(self, model, style_encoder, scheduler, optimizer,
                 device, grad_clip=1.0, ema_decay=0.9999,
                 loss_weights=None, mixed_precision=False, loss_module=None):

        self.model          = model           # UNet GỐC (không bọc DataParallel) — cho EMA/val/save
        self.style_encoder  = style_encoder
        self.scheduler      = scheduler
        self.optimizer      = optimizer
        self.device         = device
        self.grad_clip      = grad_clip
        self.ema_decay      = ema_decay
        self.mixed_precision = mixed_precision
        self.scaler         = GradScaler('cuda', enabled=mixed_precision)

        if loss_weights is None:
            self.loss_weights = {"noise": 1.0, "style": 0.1, "content": 0.01}
        else:
            self.loss_weights = loss_weights

        # loss_module: StyleDiffusionLoss (có thể đã bọc nn.DataParallel ở train.py).
        # Đây là thứ được gọi lúc train — mỗi GPU tự tính trọn vẹn loss.
        if loss_module is None:
            loss_module = StyleDiffusionLoss(model, style_encoder, scheduler,
                                             amp_enabled=mixed_precision)
        self.loss_module = loss_module

        # Tham số được tối ưu = UNet + MLP của StyleEncoder (VGG frozen, bỏ qua).
        # Dùng cho clip_grad_norm_ (gồm CẢ style_encoder, không chỉ UNet như bản cũ).
        self.trainable_params = [p for p in (list(model.parameters()) +
                                             list(style_encoder.parameters()))
                                 if p.requires_grad]

        # EMA giữ bản UNet gốc để lưu/eval gọn gàng
        self.ema_model = copy.deepcopy(unwrap(model))
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def _update_ema(self):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), unwrap(self.model).parameters()):
                if model_p.requires_grad:
                    ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1.0 - self.ema_decay)

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        x0    = batch["content"].to(self.device)
        style = batch["style"].to(self.device)
        B     = x0.shape[0]
        T     = self.scheduler.num_timesteps
        # Sinh t, noise BÊN NGOÀI rồi truyền vào loss_module để DataParallel scatter
        # nhất quán theo dim 0 với (x0, style). autocast nằm TRONG loss_module.forward.
        t     = torch.randint(0, T, (B,), device=self.device)
        noise = torch.randn_like(x0)

        ln, ls, lc = self.loss_module(x0, style, t, noise)
        # DataParallel gather theo dim 0 → (num_gpus,); .mean() để gộp về scalar.
        loss_noise, loss_style, loss_content = ln.mean(), ls.mean(), lc.mean()

        w = self.loss_weights
        w_noise   = w.get("noise", 1.0)
        w_style   = w.get("style", 0.0)
        w_content = w.get("content", 0.0)
        total_loss = (w_noise * loss_noise
                    + w_style * loss_style
                    + w_content * loss_content)

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = clip_grad_norm_(self.trainable_params, self.grad_clip).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self._update_ema()

        return {
            "train_loss":   total_loss.item(),
            "noise_loss":   loss_noise.item(),
            "style_loss":   loss_style.item(),
            "content_loss": loss_content.item(),
            "style_w":      (w_style * loss_style).item(),     # đóng góp style đã nhân weight
            "content_w":    (w_content * loss_content).item(),
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
        # Lưu state_dict của UNet gốc (gỡ DataParallel) để evaluate_local.py / sample.py
        # vốn dựng UNet thường có thể load thẳng, không cần xử lý prefix "module."
        torch.save({
            "epoch":        epoch,
            "model":        unwrap(self.model).state_dict(),
            "ema_model":    self.ema_model.state_dict(),
            "style_encoder": self.style_encoder.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "loss_weights": self.loss_weights,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        unwrap(self.model).load_state_dict(ckpt["model"])
        self.ema_model.load_state_dict(ckpt["ema_model"])
        if "style_encoder" in ckpt:
            # strict=False: checkpoint cũ (trước khi thêm CFG) chưa có 'null_style' →
            # tham số null_style giữ giá trị khởi tạo (zeros) và sẽ học khi train tiếp.
            missing, unexpected = self.style_encoder.load_state_dict(ckpt["style_encoder"], strict=False)
            if missing:
                print(f"[load_checkpoint] style_encoder thiếu key (giữ init): {list(missing)}")
        # Optimizer: bọc try/except vì khi BẬT CFG, optimizer có thêm param 'null_style'
        # → số param lệch với optimizer state cũ → load_state_dict báo lỗi. Khi đó bỏ qua
        # (dùng optimizer mới khởi tạo; momentum Adam tự ấm lại sau vài chục step).
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError) as e:
            print(f"[load_checkpoint] BỎ QUA optimizer state (lệch param, vd thêm null_style): {e}")
        # KHÔNG ghi đè loss_weights bằng giá trị trong checkpoint: khi resume để đổi
        # cấu hình loss (vd style 0.5 → 500), ta muốn dùng loss_weights MỚI từ config.
        # Giữ self.loss_weights (đã set lúc khởi tạo trainer) làm nguồn chân lý.
        if "loss_weights" in ckpt and ckpt["loss_weights"] != self.loss_weights:
            print(f"[load_checkpoint] loss_weights trong ckpt={ckpt['loss_weights']} "
                  f"→ BỎ QUA, dùng config hiện tại={self.loss_weights}")
        return ckpt["epoch"]