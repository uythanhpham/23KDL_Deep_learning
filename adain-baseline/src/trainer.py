# 23KDL_Deep_learning/adain-baseline/src/train.py
"""
Train script theo hướng production-lite / Kaggle-friendly.

Chức năng:
- Parse --config
- Đọc config.yaml
- Hỗ trợ mode = debug | real
- Tạo dataset / dataloader
- Khởi tạo model / optimizer / trainer
- Train theo epoch
- Log loss
- Save checkpoint vào save_dir (mặc định Kaggle-friendly)

Ví dụ:
    python -m src.train --config configs/config.yaml
"""

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
        help="Đường dẫn tới file YAML config, ví dụ: configs/config.yaml",
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
    save_dir = cfg.get("save_dir", "/content/drive/MyDrive/Nam3_ki_2/TH_DL/project/AdaIn/checkpoints")
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
def build_dataset_from_config(cfg: Dict[str, Any]):
    """
    Tạo dataset dựa trên mode trong config.

    Hỗ trợ:
    - mode = "debug" -> build_debug_dataset(...)
    - mode = "real"  -> build_real_dataset(...) hoặc build_dataset(...)

    Kỳ vọng các builder ở src.data.datasets có chữ ký gần giống:
    - build_debug_dataset(root_dir=..., image_size=..., ...)
    - build_real_dataset(real_root_dir=..., split=..., style_domain=..., image_size=..., ...)
      hoặc
    - build_dataset(mode=..., root_dir=..., split=..., style_domain=..., image_size=..., ...)
    """
    mode = str(get_optional(cfg, "mode", "real")).strip().lower()
    image_size = int(get_optional(cfg, "image_size", 256))

    if mode == "debug":
        root_dir = get_required(cfg, "debug_root_dir")

        if not hasattr(datasets_module, "build_debug_dataset"):
            raise AttributeError(
                "Không tìm thấy hàm build_debug_dataset trong src.data.datasets"
            )

        dataset = datasets_module.build_debug_dataset(
            root_dir=root_dir,
            image_size=image_size,
        )
        return dataset

    if mode == "real":
        real_root_dir = get_required(cfg, "real_root_dir")
        split = str(get_optional(cfg, "split", "train"))
        style_domain = get_optional(cfg, "style_domain", None)

        if hasattr(datasets_module, "build_real_dataset"):
            dataset = datasets_module.build_real_dataset(
                real_root_dir=real_root_dir,
                split=split,
                style_domain=style_domain,
                image_size=image_size,
            )
            return dataset

        if hasattr(datasets_module, "build_dataset"):
            dataset = datasets_module.build_dataset(
                mode=mode,
                root_dir=real_root_dir,
                split=split,
                style_domain=style_domain,
                image_size=image_size,
            )
            return dataset

        raise AttributeError(
            "Không tìm thấy build_real_dataset hoặc build_dataset trong src.data.datasets"
        )

    raise ValueError(
        f"mode không hợp lệ: '{mode}'. Chỉ chấp nhận 'debug' hoặc 'real'."
    )


def build_dataloader_from_config(dataset, cfg: Dict[str, Any]):
    if not hasattr(datasets_module, "build_dataloader"):
        raise AttributeError(
            "Không tìm thấy hàm build_dataloader trong src.data.datasets"
        )

    batch_size = int(get_optional(cfg, "batch_size", 8))
    num_workers = int(get_optional(cfg, "num_workers", 2))
    shuffle = bool(get_optional(cfg, "shuffle", True))

    dataloader = datasets_module.build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


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

    dataset = build_dataset_from_config(cfg)
    dataloader = build_dataloader_from_config(dataset, cfg)

    if len(dataset) == 0:
        raise RuntimeError("Dataset rỗng. Hãy kiểm tra lại dữ liệu đầu vào.")

    print(f"[INFO] Dataset size     : {len(dataset)}")
    print(f"[INFO] Num batches      : {len(dataloader)}")
    print(f"[INFO] Batch size       : {get_optional(cfg, 'batch_size', 8)}")
    print(f"[INFO] Image size       : {get_optional(cfg, 'image_size', 256)}")

    model = AdaINStyleTransfer().to(device)

    lr = float(get_optional(cfg, "lr", 1e-4))
    lambda_style = float(get_optional(cfg, "lambda_style", 10.0))
    epochs = int(get_optional(cfg, "epochs", 5))

    optimizer = Adam(model.decoder.parameters(), lr=lr)

    trainer = AdaINTrainer(
        model=model,
        optimizer=optimizer,
        lambda_style=lambda_style,
        device=device,
    )

    history: Dict[str, Any] = {
        "epoch_losses": [],
        "epoch_content_losses": [],
        "epoch_style_losses": [],
        "epochs": epochs,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    best_loss = float("inf")

    print("\n[INFO] Bắt đầu huấn luyện...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_content_loss = 0.0
        epoch_style_loss = 0.0

        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader, start=1):
            if "content" not in batch or "style" not in batch:
                raise KeyError(
                    "Batch phải chứa key 'content' và 'style'. "
                    f"Nhưng batch hiện có keys: {list(batch.keys())}"
                )

            content_images = batch["content"].to(device, non_blocking=True)
            style_images = batch["style"].to(device, non_blocking=True)

            loss_dict = trainer.train_step(content_images, style_images)

            total_loss = float(loss_dict["total_loss"])
            content_loss = float(loss_dict["content_loss"])
            style_loss = float(loss_dict["style_loss"])

            epoch_total_loss += total_loss
            epoch_content_loss += content_loss
            epoch_style_loss += style_loss

            log_every = int(get_optional(cfg, "log_every", 10))
            if batch_idx % log_every == 0 or batch_idx == len(dataloader):
                print(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Step [{batch_idx}/{len(dataloader)}] "
                    f"| total: {total_loss:.4f} "
                    f"| content: {content_loss:.4f} "
                    f"| style: {style_loss:.4f}"
                )

        num_batches = len(dataloader)
        avg_total_loss = epoch_total_loss / num_batches
        avg_content_loss = epoch_content_loss / num_batches
        avg_style_loss = epoch_style_loss / num_batches
        epoch_time = time.time() - epoch_start_time

        history["epoch_losses"].append(avg_total_loss)
        history["epoch_content_losses"].append(avg_content_loss)
        history["epoch_style_losses"].append(avg_style_loss)

        print("-" * 80)
        print(
            f"Epoch {epoch}/{epochs} hoàn tất "
            f"| avg_total: {avg_total_loss:.4f} "
            f"| avg_content: {avg_content_loss:.4f} "
            f"| avg_style: {avg_style_loss:.4f} "
            f"| time: {epoch_time:.2f}s"
        )
        print("-" * 80)

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

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
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
            print(f"[INFO] Đã cập nhật best checkpoint với loss = {best_loss:.4f}")

        save_json(history, save_dir / "train_history.json")

    history["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(history, save_dir / "train_history.json")

    print("\n[INFO] Hoàn thành quá trình huấn luyện!")
    print(f"[INFO] Best loss      : {best_loss:.4f}")
    print(f"[INFO] Checkpoints    : {save_dir}")
    print(f"[INFO] History file   : {save_dir / 'train_history.json'}")


if __name__ == "__main__":
    main()