import os
import argparse
import yaml
import random
import numpy as np
import torch
from torch.optim import Adam

from src.data.datasets import build_dataloaders
from src.models.unet import UNet
from src.models.style_encoder import StyleEncoder
from src.diffusion.scheduler import DDPMScheduler
from src.trainers.trainer import DiffusionTrainer



import logging

def setup_logger(log_file="checkpoints/training.log"):
    """Thiết lập logger để ghi đồng thời ra console và file."""
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
# CLASS: EARLY STOPPING (Cơ chế dừng sớm)
# =====================================================================
class EarlyStopping:
    """
    Theo dõi Validation Loss, nếu không giảm sau một số epoch (patience) thì dừng huấn luyện.
    Đồng thời tự động lưu lại mô hình có Validation Loss thấp nhất.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True, save_path: str = "checkpoints/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Kiểm tra và cập nhật trạng thái Early Stopping."""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Bộ đếm: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"[EarlyStopping] Validation loss giảm từ {self.best_loss:.4f} xuống {val_loss:.4f}. Đang lưu mô hình...")
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
        return self.early_stop

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """Lưu state_dict của mô hình (thường là EMA model cho chất lượng tốt nhất)."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)


# =====================================================================
# CÁC HÀM TIỆN ÍCH (UTILS)
# =====================================================================
def set_seed(seed: int):
    """Cố định seed ngẫu nhiên cho toàn bộ hệ thống để tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def load_config(train_yaml: str, model_yaml: str) -> dict:
    """Tải và gộp cấu hình từ 2 file YAML."""
    with open(train_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(model_yaml, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
        
    # Gộp config model vào config train chung
    cfg.update(model_cfg)
    return cfg


# =====================================================================
# HÀM CHÍNH (MAIN)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Style-guided Diffusion")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml")
    parser.add_argument("--model_config",  type=str, default="configs/model.yaml")
    args = parser.parse_args()

    # 1. Load config TRƯỚC — các bước khác phụ thuộc vào cfg
    cfg = load_config(args.train_config, args.model_config)   # ← cfg load trước tiên

    # 2. Set seed
    set_seed(cfg["train"]["seed"])

    # 3. Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị: {device.upper()}")

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

    # 5-7. Models + Scheduler
    model         = UNet(**cfg["model"]).to(device)
    style_encoder = StyleEncoder(**cfg["style_encoder"]).to(device)
    style_encoder.eval()
    scheduler     = DDPMScheduler(**cfg["diffusion"], device=device)
    optimizer     = Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    # 8. Trainer — truyền mixed_precision vào đây, không tạo scaler ở ngoài
    trainer = DiffusionTrainer(
        model            = model,
        style_encoder    = style_encoder,
        scheduler        = scheduler,
        optimizer        = optimizer,
        device           = device,
        grad_clip        = cfg["train"]["grad_clip"],
        ema_decay        = cfg["train"]["ema_decay"],
        loss_weights     = cfg["loss_weights"],
        mixed_precision  = cfg["train"]["mixed_precision"],  # ← truyền vào trainer
    )

    # 9. Early Stopping
    best_model_path = os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pth")
    early_stop = EarlyStopping(
        patience  = cfg["early_stopping"]["patience"],
        min_delta = float(cfg["early_stopping"]["min_delta"]),
        save_path = best_model_path,
    )

    epochs    = cfg["train"]["epochs"]
    log_every = cfg["train"]["log_every"]
    save_every= cfg["train"]["save_every"]
    tl_len    = len(train_loader)

    print("\n" + "="*50)
    print("BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN")
    print("="*50 + "\n")

    for epoch in range(1, epochs + 1):
        # Train
        train_losses = []
        for step_i, batch in enumerate(train_loader):
            r = trainer.train_step(batch)   # ← KHÔNG wrap autocast ở đây nữa
            train_losses.append(r["train_loss"])
            if (step_i + 1) % log_every == 0:
                logger.info(
                    f"[E{epoch}/{epochs}] S{step_i+1}/{tl_len} "
                    f"| Total:{r['train_loss']:.4f} | Noise:{r['noise_loss']:.4f} "
                    f"| Style:{r['style_loss']:.4f} | Content:{r['content_loss']:.4f} "
                    f"| Grad:{r['grad_norm']:.3f}"
                )

        avg_train = sum(train_losses) / len(train_losses)

        # Validation
        if val_loader and len(val_loader) > 0:
            val_losses   = [trainer.val_step(b)["val_loss"] for b in val_loader]
            avg_val      = sum(val_losses) / len(val_losses)
            monitor_loss = avg_val
            logger.info(f"\n[Epoch {epoch}] Train: {avg_train:.4f} | Val: {avg_val:.4f}\n")
        else:
            monitor_loss = avg_train
            logger.info(f"\n[Epoch {epoch}] Train: {avg_train:.4f} | (No Val)\n")

        # Checkpoint
        if epoch % save_every == 0:
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"epoch_{epoch:04d}.pth")
            trainer.save_checkpoint(ckpt_path, epoch)
            logger.info(f"[*] Checkpoint: {ckpt_path}")

        if early_stop.step(monitor_loss, trainer.ema_model):
            logger.info(f"\n[!] Early Stopping tại Epoch {epoch}!")
            break

    final_path = os.path.join(cfg["train"]["checkpoint_dir"], "last_model.pth")
    trainer.save_checkpoint(final_path, epoch)
    logger.info(f"[*] Lưu model cuối: {final_path}")
    logger.info("\n✅ Hoàn thành huấn luyện!")

if __name__ == "__main__":
    main()