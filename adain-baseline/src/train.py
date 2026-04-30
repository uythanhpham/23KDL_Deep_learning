"""Train AdaIN decoder theo iteration, bản memory-safe cho GPU VRAM thấp."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import Adam

from src.data.datasets import build_dataloaders, set_seed
from src.models.adain import AdaINStyleTransfer
from src.trainer import AdaINTrainer, cleanup_cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Train AdaIN decoder memory-safe")
    parser.add_argument("--train_content_dir", type=str, default=r"F:\data\processed\adain\train\content")
    parser.add_argument("--train_style_dir", type=str, default=r"F:\data\processed\adain\train\style")
    parser.add_argument("--val_content_dir", type=str, default=r"F:\data\processed\adain\val\content")
    parser.add_argument("--val_style_dir", type=str, default=r"F:\data\processed\adain\val\style")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pair_mode", type=str, default="random", choices=["random", "cycle"])
    parser.add_argument("--max_iter", type=int, default=160000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=5e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_style", type=float, default=10.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints/adain_crop128_amp")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=3)
    parser.add_argument("--empty_cache_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def adjust_learning_rate(optimizer, base_lr, lr_decay, iteration):
    lr = base_lr / (1.0 + iteration * lr_decay)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def save_checkpoint(path, model, optimizer, iteration, args, best_val=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iter": iteration,
        "model_state_dict": model.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "best_val": best_val,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("iter", 0)) if isinstance(ckpt, dict) else 0


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and not args.no_amp and device.type == "cuda")

    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[AMP] {use_amp}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        train_content_dir=args.train_content_dir,
        train_style_dir=args.train_style_dir,
        val_content_dir=args.val_content_dir,
        val_style_dir=args.val_style_dir,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pair_mode=args.pair_mode,
    )

    model = AdaINStyleTransfer().to(device)
    optimizer = Adam(model.decoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    trainer = AdaINTrainer(model=model, optimizer=optimizer, lambda_style=args.lambda_style, device=device, use_amp=use_amp)

    start_iter = 0
    best_val = float("inf")
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer, device)
        print(f"[Resume] Loaded {args.resume} at iter={start_iter}")

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    start_time = time.time()
    cleanup_cuda()

    print("[Train] Start")
    print(f"max_iter={args.max_iter}, batch_size={args.batch_size}, resize={args.resize_size}, crop={args.crop_size}, amp={use_amp}")

    for it in range(start_iter + 1, args.max_iter + 1):
        lr = adjust_learning_rate(optimizer, args.lr, args.lr_decay, it)
        batch, train_iter = next_or_restart(train_iter, train_loader)
        loss = trainer.train_step(batch["content"], batch["style"])
        del batch

        if it % args.empty_cache_every == 0:
            cleanup_cuda()

        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - start_time
            if device.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_text = f"mem={mem_alloc:.2f}G/{mem_reserved:.2f}G"
            else:
                mem_text = "mem=cpu"
            print(f"Iter [{it:06d}/{args.max_iter}] lr={lr:.8f} loss={loss['total_loss']:.4f} C={loss['content_loss']:.4f} S={loss['style_loss']:.4f} {mem_text} time={elapsed/60:.1f}m")

        if it % args.val_every == 0:
            cleanup_cuda()
            val_total = val_c = val_s = 0.0
            for _ in range(args.val_batches):
                vbatch, val_iter = next_or_restart(val_iter, val_loader)
                vloss = trainer.val_step(vbatch["content"], vbatch["style"])
                del vbatch
                val_total += vloss["total_loss"]
                val_c += vloss["content_loss"]
                val_s += vloss["style_loss"]
            val_total /= args.val_batches
            val_c /= args.val_batches
            val_s /= args.val_batches
            print(f"[Val] iter={it} loss={val_total:.4f} C={val_c:.4f} S={val_s:.4f}")
            if val_total < best_val:
                best_val = val_total
                save_checkpoint(save_dir / "best_model.pth", model, optimizer, it, args, best_val=best_val)
                print(f"[Save] best_model.pth | best_val={best_val:.4f}")
            cleanup_cuda()

        if it % args.save_every == 0:
            save_checkpoint(save_dir / f"adain_iter_{it:06d}.pth", model, optimizer, it, args, best_val=best_val)
            print(f"[Save] adain_iter_{it:06d}.pth")

    save_checkpoint(save_dir / "last_model.pth", model, optimizer, args.max_iter, args, best_val=best_val)
    print("[DONE] Training completed.")


if __name__ == "__main__":
    main()
