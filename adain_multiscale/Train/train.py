"""Điều phối huấn luyện Attention AdaIN: Hỗ trợ hàm train_pipeline linh hoạt giúp resume model, optimizer và scheduler."""

import argparse
import csv
from Model.adain_multiscale import calc_mean_std
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, StepLR
from pathlib import Path

#from adain_baseline.src.data.datasets import build_dataloaders
#from adain_baseline.src.models.adain import AdaINStyleTransfer
#from adain_baseline.src.trainer import AdaINTrainer

from .trainer import AdaINTrainer
from Loss.loss import StyleTransferLoss
from Model.adain_multiscale import AdaINStyleTransfer, calc_mean_std
from Data.datasets import build_dataloaders

import torchvision.utils as vutils
def save_visualization(model, content, style, path, device):
    model.eval()
    with torch.no_grad():
        # 1. Trích xuất đặc trưng
        c_feats = model.encoder(content.to(device))
        s_feats = model.encoder(style.to(device))
        
        # 2. CẦN THIẾT: Tính style_stats từ s_feats (giống hệt cách train)
        s_stats = [calc_mean_std(f) for f in s_feats]
        
        # 3. Truyền c_feats và s_stats vào decoder
        output = model.decoder(c_feats, s_stats)
        
        # 4. Denormalize để lưu ảnh
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        out_denorm = output * std + mean
        
        # 5. Ghép ảnh (Content - Style - Result)
        grid = vutils.make_grid([
            content[0].cpu(), 
            style[0].cpu(), 
            torch.clamp(out_denorm[0].cpu(), 0, 1)
        ], nrow=3)
        vutils.save_image(grid, path)
# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        verbose: bool = True,
        model: nn.Module = None,
        save_path: str = "best_model.pth"
    ):
        self.patience    = patience
        self.min_delta   = min_delta
        self.verbose     = verbose
        self.model       = model
        self.save_path   = save_path
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"[EarlyStopping] Val loss cải thiện: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter   = 0
            if self.model is not None:
                torch.save(self.model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"[EarlyStopping] Đã lưu best model: {self.save_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Không cải thiện. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("[EarlyStopping] Dừng training sớm!")
        return self.should_stop


# ─────────────────────────────────────────────
# Hàm Core Pipeline Huấn Luyện Linh Hoạt
# ─────────────────────────────────────────────
def train_pipeline(
    model, optimizer, train_loader, val_loader, device, checkpoint_dir,
    epochs=100, start_epoch=1, scheduler=None, lambda_style=10.0, lambda_content=1.0,
    patience=15, min_delta=1e-4, resume_path=None
) -> None:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    csv_log_file = checkpoint_path / "history.csv"

    # 1. Khôi phục từ checkpoint (Resume logic)
    if resume_path and Path(resume_path).exists():
        print(f"==> Đang khôi phục từ: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Kiểm tra cấu trúc dict để tránh KeyError
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            if scheduler and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"==> Đã resume thành công từ epoch {start_epoch-1}")
        else:
            model.load_state_dict(checkpoint)
            print("==> Cảnh báo: Checkpoint thuần, chỉ nạp weights.")

    # 2. Thiết lập log
    if start_epoch == 1 or not csv_log_file.exists():
        with open(csv_log_file, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_total", "train_content", "train_style", "val_total", "val_content", "val_style"])

    trainer = AdaINTrainer(model=model, optimizer=optimizer, lambda_style=lambda_style, lambda_content=lambda_content, device=device)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, model=model, save_path=str(checkpoint_path / "best_model.pth"))

    print(f"\n>>> Bắt đầu huấn luyện từ Epoch {start_epoch} -> {epochs} <<<")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_t_loss, epoch_t_c, epoch_t_s = 0.0, 0.0, 0.0
        current_lr = optimizer.param_groups[0]['lr']

        # Training
        for train_batch in train_loader:
            t_loss = trainer.train_step(train_batch["content"].to(device), train_batch["style"].to(device))
            epoch_t_loss += t_loss["total_loss"]
            epoch_t_c += t_loss["content_loss"]
            epoch_t_s += t_loss["style_loss"]
        
        # Validation
        model.eval()
        epoch_v_loss, epoch_v_c, epoch_v_s = 0.0, 0.0, 0.0
        with torch.no_grad():
            for v_batch in val_loader:
                v_loss = trainer.validate(v_batch["content"].to(device), v_batch["style"].to(device))
                epoch_v_loss += v_loss["total_loss"]
                epoch_v_c += v_loss["content_loss"]
                epoch_v_s += v_loss["style_loss"]

        num_t = len(train_loader)
        num_v = len(val_loader)
        
        # Ghi log CSV
        with open(csv_log_file, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, current_lr, epoch_t_loss/num_t, epoch_t_c/num_t, epoch_t_s/num_t, 
                                    epoch_v_loss/num_v, epoch_v_c/num_v, epoch_v_s/num_v])
                # 1. Tạo thư mục samples
        sample_dir = checkpoint_path / "samples"
        sample_dir.mkdir(exist_ok=True)

        # 2. Tạo ảnh thử nghiệm mỗi 5 epoch
        if epoch % 5 == 0 or epoch == 1:
            sample_dir = checkpoint_path / "samples"
            sample_dir.mkdir(exist_ok=True)
            sample_batch = next(iter(val_loader))
            save_visualization( # Đảm bảo tên hàm ở đây khớp với hàm bạn đã định nghĩa
                model, 
                sample_batch["content"][:1], 
                sample_batch["style"][:1], 
                sample_dir / f"epoch_{epoch}.png", 
                device
            )
        # Lưu checkpoint & Early Stopping
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoint_path / f"epoch_{epoch}.pth")
        
        if scheduler: scheduler.step()
        if early_stopping.step(epoch_v_loss/num_v): break
