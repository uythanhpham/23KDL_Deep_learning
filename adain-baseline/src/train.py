# 23KDL_Deep_learning/adain-baseline/src/train.py
"""Điều phối: đọc config, chuẩn bị model, data, optimizer."""
import argparse
import torch
import torch.nn as nn                          # ← FIX: thêm import nn
from torch.optim import Adam
from pathlib import Path

from adain_baseline.src.data.datasets import build_dataloaders
from adain_baseline.src.models.adain import AdaINStyleTransfer
from adain_baseline.src.trainer import AdaINTrainer


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
# Argparse
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train AdaIN Style Transfer")
    parser.add_argument("--root_dir",       type=str,   default="adain_baseline/debug_data")
    parser.add_argument("--image_size",     type=int,   default=256)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--lambda_style",   type=float, default=10.0)
    parser.add_argument("--epochs",         type=int,   default=5)
    parser.add_argument("--val_split",      type=float, default=0.2)
    parser.add_argument("--num_workers",    type=int,   default=2)
    parser.add_argument("--checkpoint_dir", type=str,   default="adain_baseline/checkpoints")
    parser.add_argument("--patience",       type=int,   default=5)
    parser.add_argument("--min_delta",      type=float, default=1e-4)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang chạy trên thiết bị: {device} ---")

    # 1. Data
    train_loader, val_loader = build_dataloaders(
        root_dir=args.root_dir,
        image_size=args.image_size,
        pair_mode="cycle",
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Model & Optimizer
    model     = AdaINStyleTransfer().to(device)
    optimizer = Adam(model.decoder.parameters(), lr=args.lr)

    # 3. Checkpoint dir — tạo trước khi dùng          ← FIX: chuyển lên trước EarlyStopping
    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # 4. Trainer & Early Stopping
    trainer = AdaINTrainer(
        model=model,
        optimizer=optimizer,
        lambda_style=args.lambda_style,
        device=device,
    )
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True,
        model=model,
        save_path=str(checkpoint_path / "best_model.pth"),  # ← FIX: dùng checkpoint_path
    )

    # 5. Vòng lặp train
    print(f"Bắt đầu huấn luyện — tối đa {args.epochs} epoch...")
    val_iter = iter(val_loader)

    for epoch in range(args.epochs):
        epoch_train_loss = 0.0
        loss_dict = {}                                       # ← FIX: init để tránh UnboundLocalError

        for batch_idx, train_batch in enumerate(train_loader):
            content_images = train_batch["content"].to(device)
            style_images   = train_batch["style"].to(device)

            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter  = iter(val_loader)
                val_batch = next(val_iter)

            val_content = val_batch["content"].to(device)
            val_style   = val_batch["style"].to(device)

            loss_dict = trainer.train_step(
                content_images, style_images,
                val_content,    val_style,
            )

            epoch_train_loss += loss_dict["train_total_loss"]

            if (batch_idx + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] | Step [{batch_idx+1}/{len(train_loader)}] "
                    f"| Train Loss: {loss_dict['train_total_loss']:.4f} "
                    f"(C: {loss_dict['train_content_loss']:.4f}, S: {loss_dict['train_style_loss']:.4f}) "
                    f"| Val Loss: {loss_dict['valid_total_loss']:.4f}"
                )

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss   = loss_dict.get("valid_total_loss", float("inf"))

        print(
            f"==> Epoch {epoch+1} | "
            f"Avg Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss (last step): {avg_val_loss:.4f}"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path / f"adain_epoch_{epoch+1}.pth",
        )

        if early_stopping.step(avg_val_loss):
            print(f"Dừng sớm tại epoch {epoch+1}.")
            break

    print("Hoàn thành quá trình huấn luyện!")


if __name__ == "__main__":
    main()
