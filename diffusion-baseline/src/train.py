import os
import argparse
import yaml
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Chống memory fragmentation — phải set TRƯỚC khi PyTorch allocate bất kỳ tensor nào
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from src.data.datasets       import build_dataloaders
from src.models.unet          import UNet
from src.models.style_encoder import StyleEncoder
from src.diffusion.scheduler  import DDPMScheduler
from src.trainers.trainer     import DiffusionTrainer

import logging


# =====================================================================
# LOGGER
# =====================================================================
def setup_logger(log_file="checkpoints/training.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("DiffusionTrain")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()


# =====================================================================
# EARLY STOPPING
# =====================================================================
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 verbose: bool = True, save_path: str = "checkpoints/best_model.pth"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.save_path  = save_path
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"[EarlyStopping] Bộ đếm: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                logger.info(f"[EarlyStopping] Loss giảm {self.best_loss:.4f} → {val_loss:.4f}. Lưu model...")
            self.best_loss = val_loss
            self._save(val_loss, model)
            self.counter = 0
        return self.early_stop

    def _save(self, val_loss: float, model: torch.nn.Module):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)


# =====================================================================
# UTILS
# =====================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


def load_config(train_yaml: str, model_yaml: str) -> dict:
    with open(train_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(model_yaml, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    cfg.update(model_cfg)
    return cfg


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Style-guided Diffusion")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml")
    parser.add_argument("--model_config",  type=str, default="configs/model.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint để resume training")
    args = parser.parse_args()

    # 1. Load config trước tiên
    cfg = load_config(args.train_config, args.model_config)

    # 2. Seed
    set_seed(cfg["train"]["seed"])

    # 3. Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[*] Thiết bị: {device.upper()}")
    if device == "cuda":
        logger.info(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"[*] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"[*] Mixed precision: {cfg['train']['mixed_precision']}")

    # 4. DataLoaders
    train_loader, val_loader = build_dataloaders(
        content_dir = cfg["data"]["content_dir"],
        style_dir   = cfg["data"]["style_dir"],
        image_size  = cfg["data"]["image_size"],
        batch_size  = cfg["data"]["batch_size"],
        val_split   = cfg["data"]["val_split"],
        num_workers = cfg["data"]["num_workers"],
        seed        = cfg["train"]["seed"],
    )
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)

    # 5. Models
    model         = UNet(**cfg["model"]).to(device)
    style_encoder = StyleEncoder(**cfg["style_encoder"]).to(device)
    style_encoder.eval()
    scheduler     = DDPMScheduler(**cfg["diffusion"], device=device)
    optimizer     = Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"[*] UNet params: {n_params:.2f}M")

    # 6. Trainer
    trainer = DiffusionTrainer(
        model           = model,
        style_encoder   = style_encoder,
        scheduler       = scheduler,
        optimizer       = optimizer,
        device          = device,
        grad_clip       = cfg["train"]["grad_clip"],
        ema_decay       = cfg["train"]["ema_decay"],
        loss_weights    = cfg["loss_weights"],
        mixed_precision = cfg["train"]["mixed_precision"],
    )

    # 6b. Resume từ checkpoint (nếu có)
    start_epoch = 1
    if args.resume:
        logger.info(f"[*] Đang resume từ: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        if isinstance(ckpt, dict) and "model" in ckpt:
            # Format đầy đủ: {epoch, model, ema_model, optimizer, loss_weights}
            start_epoch = trainer.load_checkpoint(args.resume) + 1
            logger.info(f"[*] ✓ Resume ĐẦY ĐỦ (epoch + optimizer + EMA)")
        else:
            # Format raw state_dict (ví dụ: best_model.pth từ EarlyStopping)
            state_dict = ckpt if isinstance(ckpt, dict) else ckpt
            trainer.model.load_state_dict(state_dict)
            trainer.ema_model.load_state_dict(state_dict)
            # Optimizer giữ nguyên mới khởi tạo (không có state cũ để load)
            start_epoch = cfg["train"].get("resume_epoch", 1)
            logger.info(f"[*] ⚠️ Resume từ RAW state_dict (chỉ có weights)")
            logger.info(f"[*]   → Optimizer được reset, EMA copy từ model")
            logger.info(f"[*]   → Bắt đầu từ epoch {start_epoch}")
        
        del ckpt
        torch.cuda.empty_cache()
        logger.info(f"[*] Resume thành công! Bắt đầu từ epoch {start_epoch}")

    # 7. LR Scheduler — Cosine Annealing để hội tụ nhanh trong thời gian giới hạn
    epochs = cfg["train"]["epochs"]
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # Nếu resume, fast-forward LR scheduler
    for _ in range(start_epoch - 1):
        lr_scheduler.step()
    logger.info(f"[*] LR Scheduler: CosineAnnealingLR (T_max={epochs}, eta_min=1e-6)")

    # 8. Early Stopping
    best_model_path = os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pth")
    early_stop = EarlyStopping(
        patience  = cfg["early_stopping"]["patience"],
        min_delta = float(cfg["early_stopping"]["min_delta"]),
        save_path = best_model_path,
    )

    log_every  = cfg["train"]["log_every"]
    save_every = cfg["train"]["save_every"]
    tl_len     = len(train_loader)

    # Ước lượng thời gian
    est_time_per_epoch = tl_len * 0.5 / 60  # ~0.5s per step
    est_total = est_time_per_epoch * epochs
    logger.info(f"[*] Steps/epoch: {tl_len} | Ước lượng: ~{est_time_per_epoch:.1f} phút/epoch | ~{est_total:.1f} phút tổng")

    logger.info("\n" + "="*50)
    logger.info("BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN")
    logger.info("="*50 + "\n")

    # 9. Training loop
    for epoch in range(start_epoch, epochs + 1):

        # ── TRAIN ──────────────────────────────────────────
        train_losses = []
        model.train()

        for step_i, batch in enumerate(train_loader):
            r = trainer.train_step(batch)
            train_losses.append(r["train_loss"])

            if (step_i + 1) % log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"[E{epoch}/{epochs}] S{step_i+1}/{tl_len} "
                    f"| Total:{r['train_loss']:.4f} "
                    f"| Noise:{r['noise_loss']:.4f} "
                    f"| Style:{r['style_loss']:.4f} "
                    f"| Content:{r['content_loss']:.4f} "
                    f"| Grad:{r['grad_norm']:.3f} "
                    f"| LR:{current_lr:.6f}"
                )

        avg_train = sum(train_losses) / len(train_losses)
        torch.cuda.empty_cache()   # ← giải phóng VRAM sau train

        # ── VALIDATION ─────────────────────────────────────
        if val_loader and len(val_loader) > 0:
            val_losses = [trainer.val_step(b)["val_loss"] for b in val_loader]
            avg_val    = sum(val_losses) / len(val_losses)
            monitor    = avg_val
            logger.info(f"\n[Epoch {epoch}] Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        else:
            monitor = avg_train
            logger.info(f"\n[Epoch {epoch}] Train: {avg_train:.4f} | (No Val) | LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        torch.cuda.empty_cache()   # ← giải phóng VRAM sau val

        # ── LR SCHEDULER STEP ─────────────────────────────
        lr_scheduler.step()

        # ── CHECKPOINT ─────────────────────────────────────
        if epoch % save_every == 0:
            ckpt_path = os.path.join(
                cfg["train"]["checkpoint_dir"], f"epoch_{epoch:04d}.pth"
            )
            trainer.save_checkpoint(ckpt_path, epoch)
            logger.info(f"[*] Checkpoint: {ckpt_path}")

        # ── EARLY STOPPING ─────────────────────────────────
        if early_stop.step(monitor, trainer.ema_model):
            logger.info(f"\n[!] Early Stopping tại Epoch {epoch}!")
            break

    # 10. Lưu model cuối
    final_path = os.path.join(cfg["train"]["checkpoint_dir"], "last_model.pth")
    trainer.save_checkpoint(final_path, epoch)
    logger.info(f"[*] Lưu model cuối: {final_path}")
    logger.info(f"[*] Best model: {best_model_path}")
    logger.info("\n✅ Hoàn thành huấn luyện!")


if __name__ == "__main__":
    main()