"""Entry script huấn luyện AdaIN Multi-Scale.

Chạy từ thư mục adain_multiscale:

    python run_train.py --root_dir ../../data/adain --epochs 100
"""
import argparse

import torch

from DataSet.DataLoader import build_dataloaders
from Model.adain_multiscale import AdaINStyleTransfer
from Train.train import train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AdaIN Multi-Scale Style Transfer")
    parser.add_argument("--root_dir", type=str, default="../../data/adain",
                        help="Thư mục chứa 2 thư mục con 'content' và 'style' (tạo bằng scripts/setup_data.sh)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--pair_mode", type=str, default="cycle", choices=["cycle", "random"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_style", type=float, default=10.0)
    parser.add_argument("--lambda_content", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint để train tiếp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Thiết bị: {device}")

    model = AdaINStyleTransfer().to(device)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    train_loader, val_loader = build_dataloaders(
        root_dir=args.root_dir,
        image_size=args.image_size,
        pair_mode=args.pair_mode,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    train_pipeline(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        scheduler=None,
        lambda_style=args.lambda_style,
        lambda_content=args.lambda_content,
        patience=args.patience,
        min_delta=args.min_delta,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
