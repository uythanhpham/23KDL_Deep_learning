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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    parser.add_argument("--model_config", type=str, default="configs/model.yaml")
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.train_config, args.model_config)
    
    # 2. Set seed để tái lập
    set_seed(cfg["train"]["seed"])
    
    # 3. Chọn Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Khởi động quá trình huấn luyện trên thiết bị: {device.upper()}")
    
    # 4. Khởi tạo DataLoaders
    train_loader, val_loader = build_dataloaders(
        content_dir=cfg["data"]["content_dir"],
        style_dir=cfg["data"]["style_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["train"]["seed"]
    )
    
    # Đảm bảo thư mục lưu checkpoint tồn tại
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    
    # 5. Khởi tạo UNet
    model = UNet(**cfg["model"]).to(device)
    
    # 6. Khởi tạo Style Encoder (Luôn để ở chế độ eval)
    style_encoder = StyleEncoder(**cfg["style_encoder"]).to(device)
    style_encoder.eval()
    
    # 7. Khởi tạo Scheduler (Định thời Diffusion)
    scheduler = DDPMScheduler(**cfg["diffusion"], device=device)
    
    # 8. Khởi tạo Optimizer (Chỉ tối ưu trọng số của UNet)
    optimizer = Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    
    # 9. Khởi tạo Trainer
    trainer = DiffusionTrainer(
        model=model,
        style_encoder=style_encoder,
        scheduler=scheduler,
        optimizer=optimizer,
        device=device,
        grad_clip=cfg["train"]["grad_clip"],
        ema_decay=cfg["train"]["ema_decay"],
        loss_weights=cfg["loss_weights"]
    )
    
    # 10. Khởi tạo Early Stopping
    best_model_path = os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pth")
    early_stop = EarlyStopping(
        patience=cfg["early_stopping"]["patience"],
        min_delta=float(cfg["early_stopping"]["min_delta"]),
        save_path=best_model_path
    )
    
    # Lấy các thiết lập vòng lặp từ config
    epochs = cfg["train"]["epochs"]
    log_every = cfg["train"]["log_every"]
    save_every = cfg["train"]["save_every"]
    tl_len = len(train_loader)
    
    print("\n" + "="*50)
    print("BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN")
    print("="*50 + "\n")
    
    # 11. Vòng lặp Epoch
    for epoch in range(1, epochs + 1):
        # -------------------------------------------------------------
        # HUẤN LUYỆN (TRAINING)
        # -------------------------------------------------------------
        train_losses = []
        for step_i, batch in enumerate(train_loader):
            r = trainer.train_step(batch)
            train_losses.append(r["train_loss"])
            
            # Log tiến độ theo từng cụm step
            if (step_i + 1) % log_every == 0:
                print(f"[E{epoch}/{epochs}] S{step_i+1}/{tl_len} "
                      f"| Total:{r['train_loss']:.4f} | Noise:{r['noise_loss']:.4f} "
                      f"| Style:{r['style_loss']:.4f} | Content:{r['content_loss']:.4f} "
                      f"| Grad:{r['grad_norm']:.3f}")
                      
        avg_train = sum(train_losses) / len(train_losses)
        
        # -------------------------------------------------------------
        # ĐÁNH GIÁ (VALIDATION)
        # -------------------------------------------------------------
        if val_loader is not None and len(val_loader) > 0:
            val_losses = [trainer.val_step(b)["val_loss"] for b in val_loader]
            avg_val = sum(val_losses) / len(val_losses)
        else:
            avg_val = 0.0 # Bỏ qua nếu không dùng tập validation
            
        # Log tóm tắt kết thúc Epoch
        print(f"\n[Epoch {epoch}] Train:{avg_train:.4f} | Val:{avg_val:.4f}\n")
        
        # -------------------------------------------------------------
        # LƯU CHECKPOINT VÀ KIỂM TRA EARLY STOPPING
        # -------------------------------------------------------------
        if epoch % save_every == 0:
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"epoch_{epoch:04d}.pth")
            trainer.save_checkpoint(ckpt_path, epoch)
            print(f"[*] Đã lưu checkpoint định kỳ tại: {ckpt_path}")
            
        # Kiểm tra Early Stopping bằng EMA Model (vì EMA model cho chất lượng ảnh tốt hơn mô hình thường)
        if early_stop.step(avg_val, trainer.ema_model):
            print(f"\n[!] Kích hoạt Early Stopping tại Epoch {epoch}. Ngừng huấn luyện để tránh Overfitting!")
            break
            
    print("\nHoàn thành huấn luyện!")

if __name__ == "__main__":
    main()