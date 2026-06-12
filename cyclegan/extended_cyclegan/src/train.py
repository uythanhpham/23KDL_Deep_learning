from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import UnpairedImageDataset
from src.models.cyclegan import CycleGANModel
from src.utils.config import load_config, save_config
from src.utils.misc import append_csv, format_losses, get_device, set_seed
from src.utils.visualize import save_sample_grid


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CycleGAN theo cấu trúc <root>/<style>/trainA/trainB tích hợp Palette Loss"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_monet.yaml",
        help="Path tới file config YAML",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        choices=["monet", "vangogh", "ukiyoe", "cezanne"],
        help="Override data.style trong config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path tới checkpoint .pth nếu muốn train tiếp",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs",
    )
    parser.add_argument(
        "--epochs_decay",
        type=int,
        default=None,
        help="Override train.epochs_decay",
    )
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=None,
        help="Override train.max_steps_per_epoch. 0 = dùng hết dataset",
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=None,
        help="Override train.save_every_steps. 0 = chỉ lưu cuối epoch",
    )
    return parser


def lambda_lr(epoch_index: int, n_epochs: int, n_epochs_decay: int) -> float:
    """Lịch learning rate giống CycleGAN official:

    - Giữ nguyên LR trong n_epochs đầu.
    - Sau đó decay tuyến tính về 0 trong n_epochs_decay epoch.

    epoch_index của LambdaLR bắt đầu từ 0.
    """
    if epoch_index < n_epochs:
        return 1.0

    decay_progress = epoch_index - n_epochs
    return max(0.0, 1.0 - decay_progress / float(max(1, n_epochs_decay)))


def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """Khi resume checkpoint được load từ CPU/GPU khác, state của optimizer có thể

    nằm sai device. Hàm này chuyển tensor state về device hiện tại.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_checkpoint(
    *,
    model: CycleGANModel,
    latest_path: Path,
    epoch: int,
    step: int | None,
    optim_G: torch.optim.Optimizer,
    optim_D: torch.optim.Optimizer,
    scheduler_G: torch.optim.lr_scheduler.LambdaLR,
    scheduler_D: torch.optim.lr_scheduler.LambdaLR,
    scaler_G: torch.cuda.amp.GradScaler,
    scaler_D: torch.cuda.amp.GradScaler,
    total_epochs: int,
    n_epochs: int,
    n_epochs_decay: int,
    mid_epoch: bool,
) -> None:
    """Lưu checkpoint đầy đủ:

    - weights G/D
    - optimizer state
    - scheduler state
    - AMP scaler state
    - epoch/step metadata

    Lưu ra latest.pth để resume hoặc backup.
    """
    extra = {
        "scheduler_G": scheduler_G.state_dict(),
        "scheduler_D": scheduler_D.state_dict(),
        "scaler_G": scaler_G.state_dict(),
        "scaler_D": scaler_D.state_dict(),
        "total_epochs": total_epochs,
        "n_epochs": n_epochs,
        "n_epochs_decay": n_epochs_decay,
        "mid_epoch": mid_epoch,
    }

    if step is not None:
        extra["step"] = step

    model.save_checkpoint(
        latest_path,
        epoch=epoch,
        optim_G=optim_G,
        optim_D=optim_D,
        extra=extra,
    )


def main() -> None:
    args = build_argparser().parse_args()

    cfg = load_config(args.config)

    if args.style is not None:
        cfg["data"]["style"] = args.style
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.epochs_decay is not None:
        cfg["train"]["epochs_decay"] = args.epochs_decay
    if args.max_steps_per_epoch is not None:
        cfg["train"]["max_steps_per_epoch"] = args.max_steps_per_epoch
    if args.save_every_steps is not None:
        cfg["train"]["save_every_steps"] = args.save_every_steps

    style = cfg["data"]["style"]

    set_seed(int(cfg["train"].get("seed", 42)))
    device = get_device(cfg["train"].get("device", "auto"))

    out_root = Path(cfg["output"]["root"])
    ckpt_dir = out_root / "checkpoints" / style
    sample_dir = out_root / "samples" / style
    log_dir = out_root / "logs" / style

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, ckpt_dir / "config_used.yaml")

    # Khởi tạo Dataset kèm việc nạp dictionary 'data' để đọc các tệp JSONL Palette
    dataset = UnpairedImageDataset(
        root=cfg["data"]["root"],
        style=style,
        cfg_data=cfg["data"],  # Bổ sung tham số để datasets.py xử lý nạp JSONL
        phase="train",
        dir_A=cfg["data"].get("trainA", "trainA"),
        dir_B=cfg["data"].get("trainB", "trainB"),
        image_size=int(cfg["train"]["image_size"]),
        crop_size=int(cfg["train"]["crop_size"]),
        exts=cfg["data"].get("image_extensions"),
        serial_batches=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print(f"[Train] style={style}", flush=True)
    print(f"[Train] data_root={cfg['data']['root']}", flush=True)
    print(f"[Train] trainA={dataset.dir_A} ({len(dataset.paths_A)} ảnh)", flush=True)
    print(f"[Train] trainB={dataset.dir_B} ({len(dataset.paths_B)} ảnh)", flush=True)
    print(f"[Train] device={device}", flush=True)
    print(f"[Train] output={out_root.resolve()}", flush=True)
    print(f"[Train] checkpoint_dir={ckpt_dir.resolve()}", flush=True)

    model = CycleGANModel(cfg, device)

    optim_G = torch.optim.Adam(
        list(model.G_A2B.parameters()) + list(model.G_B2A.parameters()),
        lr=float(cfg["train"]["lr"]),
        betas=(float(cfg["train"]["beta1"]), float(cfg["train"]["beta2"])),
    )

    optim_D = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=float(cfg["train"]["lr"]),
        betas=(float(cfg["train"]["beta1"]), float(cfg["train"]["beta2"])),
    )

    n_epochs = int(cfg["train"]["epochs"])
    n_epochs_decay = int(cfg["train"]["epochs_decay"])
    total_epochs = n_epochs + n_epochs_decay

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optim_G,
        lr_lambda=lambda epoch_index: lambda_lr(
            epoch_index, n_epochs, n_epochs_decay
        ),
    )

    scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optim_D,
        lr_lambda=lambda epoch_index: lambda_lr(
            epoch_index, n_epochs, n_epochs_decay
        ),
    )

    use_amp = bool(cfg["train"].get("use_amp", False)) and device.type == "cuda"

    scaler_G = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_D = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[Train] use_amp={use_amp}", flush=True)

    start_epoch = 1

    if args.resume is not None:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint resume: {resume_path}"
            )

        ckpt = model.load_checkpoint(
            resume_path,
            strict=True,
            load_discriminators=True,
        )

        if "optim_G" in ckpt:
            optim_G.load_state_dict(ckpt["optim_G"])
            move_optimizer_to_device(optim_G, device)

        if "optim_D" in ckpt:
            optim_D.load_state_dict(ckpt["optim_D"])
            move_optimizer_to_device(optim_D, device)

        if "scheduler_G" in ckpt:
            scheduler_G.load_state_dict(ckpt["scheduler_G"])
        else:
            print(
                "[Warning] Checkpoint không có scheduler_G. LR scheduler sẽ được khởi tạo lại.",
                flush=True,
            )

        if "scheduler_D" in ckpt:
            scheduler_D.load_state_dict(ckpt["scheduler_D"])
        else:
            print(
                "[Warning] Checkpoint không có scheduler_D. LR scheduler sẽ được khởi tạo lại.",
                flush=True,
            )

        if use_amp and "scaler_G" in ckpt:
            scaler_G.load_state_dict(ckpt["scaler_G"])

        if use_amp and "scaler_D" in ckpt:
            scaler_D.load_state_dict(ckpt["scaler_D"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1

        print(f"[Resume] checkpoint={resume_path}", flush=True)
        print(
            f"[Resume] tiếp tục từ epoch {start_epoch}/{total_epochs}", flush=True
        )

        if bool(ckpt.get("mid_epoch", False)):
            print(
                "[Resume] Checkpoint này được lưu giữa epoch. "
                "Code sẽ tiếp tục từ epoch kế tiếp theo logic hiện tại.",
                flush=True,
            )

    if start_epoch > total_epochs:
        print(
            f"[Done] Checkpoint đã ở epoch {start_epoch - 1}, vượt hoặc bằng total_epochs={total_epochs}.",
            flush=True,
        )
        return

    max_steps = int(cfg["train"].get("max_steps_per_epoch", 0))
    save_every_steps = int(cfg["train"].get("save_every_steps", 500))
    save_every_epochs = int(cfg["train"].get("save_every_epochs", 5))
    sample_every_epochs = int(cfg["train"].get("sample_every_epochs", 1))

    print(f"[Train] total_epochs={total_epochs}", flush=True)
    print(f"[Train] max_steps_per_epoch={max_steps}", flush=True)
    print(f"[Train] save_every_steps={save_every_steps}", flush=True)
    print(f"[Train] save_every_epochs={save_every_epochs}", flush=True)
    print(f"[Train] sample_every_epochs={sample_every_epochs}", flush=True)

    log_path = log_dir / "train_log.csv"

    fieldnames = [
        "epoch",
        "step",
        "lr",
        "G_total",
        "G_A2B",
        "G_B2A",
        "cycle_A",
        "cycle_B",
        "idt_A",
        "idt_B",
        "palette_loss",  # Thêm trường log lỗi bảng màu định hướng vào CSV
        "D_A",
        "D_B",
    ]

    latest_path = ckpt_dir / "latest.pth"

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()

        print(f"\n[Epoch {epoch}/{total_epochs}] start", flush=True)

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            leave=True,
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=1.0,
        )

        last_tensors = None
        losses_float = None

        for step, batch in enumerate(pbar, start=1):
            if max_steps > 0 and step > max_steps:
                break

            # --- SỬA LỖI HỔNG 3: TRÍCH XUẤT HÌNH ẢNH VÀ CÁC VECTOR PALETTE 24 CHIỀU ---
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)
            p_A = batch["p_A"].to(device, non_blocking=True)
            p_B = batch["p_B"].to(device, non_blocking=True)
            p_target = batch["p_target"].to(device, non_blocking=True)

            # ---------------------------------------------------------
            # 1) Update Generators: G_A2B và G_B2A
            # ---------------------------------------------------------
            model.set_requires_grad([model.D_A, model.D_B], False)
            optim_G.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Bơm đầy đủ 5 tham số đầu vào cho hàm tính loss thế hệ mới
                loss_G, out = model.compute_generator_loss(
                    real_A, real_B, p_A, p_B, p_target
                )

            scaler_G.scale(loss_G).backward()
            scaler_G.step(optim_G)
            scaler_G.update()

            fake_A = out["fake_A"].detach()
            fake_B = out["fake_B"].detach()

            # ---------------------------------------------------------
            # 2) Update Discriminators: D_A và D_B
            # ---------------------------------------------------------
            model.set_requires_grad([model.D_A, model.D_B], True)
            optim_D.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_D_A, loss_D_B = model.compute_discriminator_losses(
                    real_A=real_A,
                    real_B=real_B,
                    fake_A=fake_A,
                    fake_B=fake_B,
                )
                loss_D = loss_D_A + loss_D_B

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optim_D)
            scaler_D.update()

            losses_float = {
                "G_total": float(out["G_total"].detach().cpu()),
                "G_A2B": float(out["G_A2B"].detach().cpu()),
                "G_B2A": float(out["G_B2A"].detach().cpu()),
                "cycle_A": float(out["cycle_A"].detach().cpu()),
                "cycle_B": float(out["cycle_B"].detach().cpu()),
                "idt_A": float(out["idt_A"].detach().cpu()),
                "idt_B": float(out["idt_B"].detach().cpu()),
                "palette_loss": float(out["palette_loss"].detach().cpu()),  # Đưa giá trị lỗi Palette vào bộ nhớ Log
                "D_A": float(loss_D_A.detach().cpu()),
                "D_B": float(loss_D_B.detach().cpu()),
            }

            current_lr = float(optim_G.param_groups[0]["lr"])

            pbar.set_postfix(
                {
                    "lr": f"{current_lr:.7f}",
                    "G": f"{losses_float['G_total']:.3f}",
                    "Pal": f"{losses_float['palette_loss']:.3f}",  # Hiển thị độ lệch màu ngay trên thanh Progress bar
                    "D_A": f"{losses_float['D_A']:.3f}",
                    "D_B": f"{losses_float['D_B']:.3f}",
                }
            )

            append_csv(
                log_path,
                {
                    "epoch": epoch,
                    "step": step,
                    "lr": current_lr,
                    **losses_float,
                },
                fieldnames=fieldnames,
            )

            last_tensors = (real_A, real_B, out, p_target)

            # ---------------------------------------------------------
            # Lưu checkpoint giữa epoch để tránh mất dữ liệu trên Kaggle/Colab
            # ---------------------------------------------------------
            if save_every_steps > 0 and step % save_every_steps == 0:
                save_training_checkpoint(
                    model=model,
                    latest_path=latest_path,
                    epoch=epoch,
                    step=step,
                    optim_G=optim_G,
                    optim_D=optim_D,
                    scheduler_G=scheduler_G,
                    scheduler_D=scheduler_D,
                    scaler_G=scaler_G,
                    scaler_D=scaler_D,
                    total_epochs=total_epochs,
                    n_epochs=n_epochs,
                    n_epochs_decay=n_epochs_decay,
                    mid_epoch=True,
                )

                print(
                    f"\n[Save] step checkpoint: epoch={epoch}, step={step}, path={latest_path}",
                    flush=True,
                )

        if losses_float is None:
            raise RuntimeError(
                "Không có batch nào được train trong epoch này. "
                "Kiểm tra dataset, batch_size, drop_last hoặc max_steps_per_epoch."
            )

        scheduler_G.step()
        scheduler_D.step()

        print(f"[Epoch {epoch}] {format_losses(losses_float)}", flush=True)

        if last_tensors is not None and epoch % sample_every_epochs == 0:
            real_A, real_B, out, p_target_sample = last_tensors

            save_sample_grid(
                real_A=real_A,
                fake_B=out["fake_B"],
                rec_A=out["rec_A"],
                real_B=real_B,
                fake_A=out["fake_A"],
                rec_B=out["rec_B"],
                p_target=p_target_sample,
                out_path=sample_dir / f"epoch_{epoch:03d}.jpg",
            )

            print(
                f"[Save] sample: {sample_dir / f'epoch_{epoch:03d}.jpg'}",
                flush=True,
            )

        # ---------------------------------------------------------
        # Luôn lưu checkpoint cuối epoch.
        # ---------------------------------------------------------
        save_training_checkpoint(
            model=model,
            latest_path=latest_path,
            epoch=epoch,
            step=None,
            optim_G=optim_G,
            optim_D=optim_D,
            scheduler_G=scheduler_G,
            scheduler_D=scheduler_D,
            scaler_G=scaler_G,
            scaler_D=scaler_D,
            total_epochs=total_epochs,
            n_epochs=n_epochs,
            n_epochs_decay=n_epochs_decay,
            mid_epoch=False,
        )

        print(f"[Save] latest checkpoint: {latest_path}", flush=True)

        if epoch % save_every_epochs == 0 or epoch == total_epochs:
            epoch_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
            shutil.copy2(latest_path, epoch_path)
            print(f"[Save] epoch checkpoint: {epoch_path}", flush=True)

    print("[Done] Train xong.", flush=True)
    print(f"[Checkpoint] {ckpt_dir / 'latest.pth'}", flush=True)
    print(f"[Samples] {sample_dir}", flush=True)
    print(f"[Log] {log_path}", flush=True)


if __name__ == "__main__":
    main()