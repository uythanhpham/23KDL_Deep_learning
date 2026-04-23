from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.optim import Adam

from src.models.adain import AdaINStyleTransfer
from src.trainer import AdaINTrainer
from src.data import datasets as datasets_module


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AdaIN on Kaggle / local")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Đường dẫn tới file YAML config, ví dụ: configs/config.yml",
    )
    return parser.parse_args()


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise ValueError("Nội dung config YAML phải là một dictionary.")

    return cfg


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_save_dir(cfg: Dict[str, Any]) -> Path:
    save_dir = cfg.get("save_dir", "/kaggle/working/checkpoints")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_required(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Thiếu key bắt buộc trong config: '{key}'")
    return cfg[key]


def get_optional(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------
# Dataset / Dataloader factory
# ---------------------------------------------------------
def build_dataset_for_split(cfg: Dict[str, Any], split: str):
    mode = str(get_optional(cfg, "mode", "real")).strip().lower()
    image_size = int(get_optional(cfg, "image_size", 256))

    if mode == "debug":
        root_dir = get_required(cfg, "debug_root_dir")

        if not hasattr(datasets_module, "build_debug_dataset"):
            raise AttributeError(
                "Không tìm thấy hàm build_debug_dataset trong src.data.datasets"
            )

        return datasets_module.build_debug_dataset(
            root_dir=root_dir,
            image_size=image_size,
        )

    if mode == "real":
        real_root_dir = get_required(cfg, "real_root_dir")
        style_domain = get_optional(cfg, "style_domain", "all")

        if hasattr(datasets_module, "build_real_dataset"):
            return datasets_module.build_real_dataset(
                real_root_dir=real_root_dir,
                split=split,
                style_domain=style_domain,
                image_size=image_size,
            )

        if hasattr(datasets_module, "build_dataset"):
            return datasets_module.build_dataset(
                mode=mode,
                real_root_dir=real_root_dir,
                split=split,
                style_domain=style_domain,
                image_size=image_size,
            )

        raise AttributeError(
            "Không tìm thấy build_real_dataset hoặc build_dataset trong src.data.datasets"
        )

    raise ValueError(f"mode không hợp lệ: '{mode}'")


def build_dataloader_for_split(dataset, cfg: Dict[str, Any], split: str):
    if not hasattr(datasets_module, "build_dataloader"):
        raise AttributeError(
            "Không tìm thấy hàm build_dataloader trong src.data.datasets"
        )

    batch_size = int(get_optional(cfg, "batch_size", 8))
    num_workers = int(get_optional(cfg, "num_workers", 2))
    shuffle = True if split == "train" else False

    return datasets_module.build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# ---------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------
def make_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    config: Dict[str, Any],
    history: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "config": deepcopy(config),
        "history": deepcopy(history),
    }


def save_checkpoint(
    save_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    config: Dict[str, Any],
    history: Dict[str, Any],
) -> None:
    payload = make_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_loss=best_loss,
        config=config,
        history=history,
    )
    torch.save(payload, save_path)


# ---------------------------------------------------------
# Main train loop
# ---------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    device = get_device()
    save_dir = resolve_save_dir(cfg)

    print("=" * 80)
    print("AdaIN Training")
    print("=" * 80)
    print(f"Config file : {Path(args.config).resolve()}")
    print(f"Device      : {device}")
    print(f"Mode        : {get_optional(cfg, 'mode', 'real')}")
    print(f"Save dir    : {save_dir}")
    print("=" * 80)

    save_json(cfg, save_dir / "resolved_config.json")

    # Build train / val
    train_dataset = build_dataset_for_split(cfg, split="train")
    val_dataset = build_dataset_for_split(cfg, split="val")

    train_dataloader = build_dataloader_for_split(train_dataset, cfg, split="train")
    val_dataloader = build_dataloader_for_split(val_dataset, cfg, split="val")

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset rỗng. Hãy kiểm tra lại dữ liệu đầu vào.")

    if len(val_dataset) == 0:
        raise RuntimeError("Val dataset rỗng. Hãy kiểm tra lại dữ liệu đầu vào.")

    print(f"[INFO] Train dataset size : {len(train_dataset)}")
    print(f"[INFO] Val dataset size   : {len(val_dataset)}")
    print(f"[INFO] Train batches      : {len(train_dataloader)}")
    print(f"[INFO] Val batches        : {len(val_dataloader)}")
    print(f"[INFO] Batch size         : {get_optional(cfg, 'batch_size', 8)}")
    print(f"[INFO] Image size         : {get_optional(cfg, 'image_size', 256)}")

    # Model / optimizer / trainer
    model = AdaINStyleTransfer().to(device)

    lr = float(get_optional(cfg, "lr", 1e-4))
    lambda_mse = float(get_optional(cfg, "lambda_mse", 10.0))
    lambda_l1 = float(get_optional(cfg, "lambda_l1", 0.5))
    lambda_tv = float(get_optional(cfg, "lambda_tv", 1e-5))
    epochs = int(get_optional(cfg, "epochs", 5))

    use_early_stopping = bool(get_optional(cfg, "early_stopping", False))
    rise_patience = int(get_optional(cfg, "rise_patience", 3))
    log_every = int(get_optional(cfg, "log_every", 10))

    prev_val_loss = None
    rise_counter = 0

    optimizer = Adam(model.decoder.parameters(), lr=lr)

    trainer = AdaINTrainer(
        model=model,
        optimizer=optimizer,
        lambda_mse=lambda_mse,
        lambda_l1=lambda_l1,
        lambda_tv=lambda_tv,
        device=device,
    )

    history: Dict[str, Any] = {
        "train_epoch_losses": [],
        "train_epoch_mse_losses": [],
        "train_epoch_l1_losses": [],
        "train_epoch_tv_losses": [],
        "val_epoch_losses": [],
        "val_epoch_mse_losses": [],
        "val_epoch_l1_losses": [],
        "val_epoch_tv_losses": [],
        "epochs": epochs,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lambda_mse": lambda_mse,
        "lambda_l1": lambda_l1,
        "lambda_tv": lambda_tv,
        "lr": lr,
    }

    best_loss = float("inf")

    print("\n[INFO] Bắt đầu huấn luyện...\n")

    for epoch in range(1, epochs + 1):
        model.train()

        epoch_total_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_tv_loss = 0.0

        epoch_start_time = time.time()

        # =====================
        # TRAIN
        # =====================
        for batch_idx, batch in enumerate(train_dataloader, start=1):
            if "content" not in batch or "style" not in batch:
                raise KeyError(
                    "Batch phải chứa key 'content' và 'style'. "
                    f"Nhưng batch hiện có keys: {list(batch.keys())}"
                )

            content_images = batch["content"].to(device, non_blocking=True)
            style_images = batch["style"].to(device, non_blocking=True)

            loss_dict = trainer.train_step(content_images, style_images)

            total_loss = float(loss_dict["total_loss"])
            mse_loss = float(loss_dict["mse_loss"])
            l1_loss = float(loss_dict["l1_loss"])
            tv_loss = float(loss_dict["tv_loss"])

            epoch_total_loss += total_loss
            epoch_mse_loss += mse_loss
            epoch_l1_loss += l1_loss
            epoch_tv_loss += tv_loss

            if batch_idx % log_every == 0 or batch_idx == len(train_dataloader):
                print(
                    f"[TRAIN] Epoch [{epoch}/{epochs}] "
                    f"Step [{batch_idx}/{len(train_dataloader)}] "
                    f"| total: {total_loss:.4f} "
                    f"| mse: {mse_loss:.4f} "
                    f"| l1: {l1_loss:.4f} "
                    f"| tv: {tv_loss:.6f}"
                )

        num_train_batches = len(train_dataloader)
        avg_train_total_loss = epoch_total_loss / num_train_batches
        avg_train_mse_loss = epoch_mse_loss / num_train_batches
        avg_train_l1_loss = epoch_l1_loss / num_train_batches
        avg_train_tv_loss = epoch_tv_loss / num_train_batches

        # =====================
        # VALIDATION
        # =====================
        model.eval()

        val_total_loss = 0.0
        val_mse_loss = 0.0
        val_l1_loss = 0.0
        val_tv_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                if "content" not in batch or "style" not in batch:
                    raise KeyError(
                        "Val batch phải chứa key 'content' và 'style'. "
                        f"Nhưng batch hiện có keys: {list(batch.keys())}"
                    )

                content_images = batch["content"].to(device, non_blocking=True)
                style_images = batch["style"].to(device, non_blocking=True)

                if not hasattr(trainer, "validate_step"):
                    raise AttributeError(
                        "AdaINTrainer hiện chưa có validate_step(). "
                        "Bạn cần thêm validate_step vào src/trainer.py."
                    )

                loss_dict = trainer.validate_step(content_images, style_images)

                val_total_loss += float(loss_dict["total_loss"])
                val_mse_loss += float(loss_dict["mse_loss"])
                val_l1_loss += float(loss_dict["l1_loss"])
                val_tv_loss += float(loss_dict["tv_loss"])

        num_val_batches = len(val_dataloader)
        avg_val_total_loss = val_total_loss / num_val_batches
        avg_val_mse_loss = val_mse_loss / num_val_batches
        avg_val_l1_loss = val_l1_loss / num_val_batches
        avg_val_tv_loss = val_tv_loss / num_val_batches

        epoch_time = time.time() - epoch_start_time

        history["train_epoch_losses"].append(avg_train_total_loss)
        history["train_epoch_mse_losses"].append(avg_train_mse_loss)
        history["train_epoch_l1_losses"].append(avg_train_l1_loss)
        history["train_epoch_tv_losses"].append(avg_train_tv_loss)

        history["val_epoch_losses"].append(avg_val_total_loss)
        history["val_epoch_mse_losses"].append(avg_val_mse_loss)
        history["val_epoch_l1_losses"].append(avg_val_l1_loss)
        history["val_epoch_tv_losses"].append(avg_val_tv_loss)

        print("-" * 80)
        print(
            f"Epoch {epoch}/{epochs} hoàn tất "
            f"| train_total: {avg_train_total_loss:.4f} "
            f"| val_total: {avg_val_total_loss:.4f} "
            f"| train_mse: {avg_train_mse_loss:.4f} "
            f"| val_mse: {avg_val_mse_loss:.4f} "
            f"| train_l1: {avg_train_l1_loss:.4f} "
            f"| val_l1: {avg_val_l1_loss:.4f} "
            f"| train_tv: {avg_train_tv_loss:.6f} "
            f"| val_tv: {avg_val_tv_loss:.6f} "
            f"| time: {epoch_time:.2f}s"
        )
        print("-" * 80)

        is_best = avg_val_total_loss < best_loss
        if is_best:
            best_loss = avg_val_total_loss

        # Save checkpoint từng epoch
        epoch_ckpt_path = save_dir / f"adain_epoch_{epoch:03d}.pth"
        save_checkpoint(
            save_path=epoch_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_loss=best_loss,
            config=cfg,
            history=history,
        )

        # Save last checkpoint
        last_ckpt_path = save_dir / "last.pth"
        save_checkpoint(
            save_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_loss=best_loss,
            config=cfg,
            history=history,
        )

        # Save best checkpoint theo val loss
        if is_best:
            best_ckpt_path = save_dir / "best.pth"
            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_loss,
                config=cfg,
                history=history,
            )
            print(f"[INFO] Đã cập nhật best checkpoint với val loss = {best_loss:.4f}")

        # =====================
        # EARLY STOPPING: val loss tăng liên tục
        # =====================
        if prev_val_loss is not None and avg_val_total_loss > prev_val_loss:
            rise_counter += 1
            print(f"[INFO] Val loss tăng. rise_counter = {rise_counter}/{rise_patience}")
        else:
            rise_counter = 0

        prev_val_loss = avg_val_total_loss

        save_json(history, save_dir / "train_history.json")

        if use_early_stopping and rise_counter >= rise_patience:
            print(f"[EARLY STOPPING] Dừng vì val loss tăng liên tục {rise_patience} epoch.")
            break

    history["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(history, save_dir / "train_history.json")

    print("\n[INFO] Hoàn thành quá trình huấn luyện!")
    print(f"[INFO] Best val loss  : {best_loss:.4f}")
    print(f"[INFO] Checkpoints    : {save_dir}")
    print(f"[INFO] History file   : {save_dir / 'train_history.json'}")


if __name__ == "__main__":
    main()