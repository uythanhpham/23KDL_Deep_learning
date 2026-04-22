from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.models.adain import AdaINStyleTransfer, adain
from src.losses.perceptual import perceptual_loss
from src.data import datasets as datasets_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate test perceptual loss từ best checkpoint cho AdaIN"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Đường dẫn config YAML. Có thể bỏ qua nếu truyền tay các tham số."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Đường dẫn tới best.pth"
    )
    parser.add_argument(
        "--real_root_dir",
        type=str,
        required=True,
        help="Root processed data, ví dụ .../2_data/processed"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Mặc định là test"
    )
    parser.add_argument(
        "--style_domain",
        type=str,
        default="all",
        choices=["anime", "sketch", "watercolor", "all"],
        help="Chọn style test"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Kích thước resize"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size test"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Num workers"
    )
    parser.add_argument(
        "--lambda_style",
        type=float,
        default=10.0,
        help="Hệ số style loss"
    )
    parser.add_argument(
        "--lambda_content",
        type=float,
        default=1.0,
        help="Hệ số content loss"
    )
    parser.add_argument(
        "--style_matrix_weight",
        type=float,
        default=1.0,
        help="Trọng số phần ma trận tương quan"
    )
    parser.add_argument(
        "--style_stat_weight",
        type=float,
        default=0.25,
        help="Trọng số phần mean/std"
    )
    parser.add_argument(
        "--loss_use_smooth_l1",
        type=lambda x: str(x).lower() in ["1", "true", "yes", "y"],
        default=True,
        help="Có dùng SmoothL1 cho loss hay không"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/eval/test_loss.json",
        help="File JSON lưu kết quả"
    )

    return parser.parse_args()


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML phải là dictionary.")

    return cfg


def get_optional(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    print(f"[Eval] Đang load checkpoint từ: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[Eval] Đã load ckpt['model_state_dict'] thành công.")
        return ckpt

    if isinstance(ckpt, dict):
        try:
            model.load_state_dict(ckpt)
            print("[Eval] Đã load state_dict trực tiếp thành công.")
            return {"raw_state_dict": True}
        except Exception as e:
            raise RuntimeError(f"Checkpoint tồn tại nhưng không load được. Lỗi: {e}") from e

    raise RuntimeError(
        "Định dạng checkpoint không hợp lệ. "
        "Kỳ vọng: payload dict có key 'model_state_dict' hoặc state_dict trần."
    )


def build_test_dataloader(
    real_root_dir: str,
    split: str,
    style_domain: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
):
    if not hasattr(datasets_module, "build_real_dataset"):
        raise AttributeError("Không tìm thấy build_real_dataset trong src.data.datasets")

    if not hasattr(datasets_module, "build_dataloader"):
        raise AttributeError("Không tìm thấy build_dataloader trong src.data.datasets")

    dataset = datasets_module.build_real_dataset(
        real_root_dir=real_root_dir,
        split=split,
        style_domain=style_domain,
        image_size=image_size,
        enable_hflip=False,
        enable_random_crop=False,
    )

    dataloader = datasets_module.build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataset, dataloader


def evaluate_test_loss(
    model,
    dataloader,
    device,
    lambda_content,
    lambda_style,
    style_matrix_weight,
    style_stat_weight,
    loss_use_smooth_l1,
):
    model.eval()

    total_loss_sum = 0.0
    content_loss_sum = 0.0
    style_loss_sum = 0.0
    style_matrix_loss_sum = 0.0
    style_stat_loss_sum = 0.0
    num_batches = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if "content" not in batch or "style" not in batch:
                raise KeyError(
                    f"Test batch phải có key 'content' và 'style'. "
                    f"Nhưng batch hiện có keys: {list(batch.keys())}"
                )

            content = batch["content"].to(device, non_blocking=True)
            style = batch["style"].to(device, non_blocking=True)

            content_feat = model.encoder(content)
            style_feat = model.encoder(style)

            t = adain(content_feat, style_feat)
            output = model.decoder(t).clamp(0, 1)
            output_feat = model.encoder(output)

            total_loss, loss_c, loss_s, loss_s_mat, loss_s_stat = perceptual_loss(
                output_feat=output_feat,
                adain_feat=t,
                lambda_content=lambda_content,
                lambda_style=lambda_style,
                matrix_weight=style_matrix_weight,
                stat_weight=style_stat_weight,
                use_smooth_l1=loss_use_smooth_l1,
            )

            batch_size = content.size(0)

            total_loss_sum += float(total_loss.item()) * batch_size
            content_loss_sum += float(loss_c.item()) * batch_size
            style_loss_sum += float(loss_s.item()) * batch_size
            style_matrix_loss_sum += float(loss_s_mat.item()) * batch_size
            style_stat_loss_sum += float(loss_s_stat.item()) * batch_size

            num_batches += 1
            num_samples += batch_size

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[Eval] Batch {batch_idx + 1}/{len(dataloader)} "
                    f"| total={total_loss.item():.4f} "
                    f"| content={loss_c.item():.4f} "
                    f"| style={loss_s.item():.4f} "
                    f"| style_mat={loss_s_mat.item():.4f} "
                    f"| style_stat={loss_s_stat.item():.4f}"
                )

    if num_samples == 0:
        raise RuntimeError("Test dataloader rỗng, không có sample nào để evaluate.")

    avg_total_loss = total_loss_sum / num_samples
    avg_content_loss = content_loss_sum / num_samples
    avg_style_loss = style_loss_sum / num_samples
    avg_style_matrix_loss = style_matrix_loss_sum / num_samples
    avg_style_stat_loss = style_stat_loss_sum / num_samples

    return {
        "num_batches": num_batches,
        "num_samples": num_samples,
        "avg_total_loss": avg_total_loss,
        "avg_content_loss": avg_content_loss,
        "avg_style_loss": avg_style_loss,
        "avg_style_matrix_loss": avg_style_matrix_loss,
        "avg_style_stat_loss": avg_style_stat_loss,
    }


def main() -> None:
    args = parse_args()

    cfg = {}
    if args.config is not None:
        cfg = load_yaml_config(args.config)

    checkpoint = args.checkpoint
    real_root_dir = args.real_root_dir
    split = args.split
    style_domain = args.style_domain
    image_size = args.image_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    lambda_content = args.lambda_content
    lambda_style = args.lambda_style
    style_matrix_weight = args.style_matrix_weight
    style_stat_weight = args.style_stat_weight
    loss_use_smooth_l1 = args.loss_use_smooth_l1
    output_file = args.output_file

    if args.config is not None:
        real_root_dir = real_root_dir or get_optional(cfg, "real_root_dir", None)
        image_size = image_size or int(get_optional(cfg, "image_size", 256))
        batch_size = batch_size or int(get_optional(cfg, "batch_size", 4))
        num_workers = num_workers or int(get_optional(cfg, "num_workers", 2))

        lambda_content = float(get_optional(cfg, "lambda_content", lambda_content))
        lambda_style = float(get_optional(cfg, "lambda_style", lambda_style))
        style_matrix_weight = float(get_optional(cfg, "style_matrix_weight", style_matrix_weight))
        style_stat_weight = float(get_optional(cfg, "style_stat_weight", style_stat_weight))
        loss_use_smooth_l1 = bool(get_optional(cfg, "loss_use_smooth_l1", loss_use_smooth_l1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("AdaIN Evaluate Test Loss")
    print("=" * 80)
    print(f"Device              : {device}")
    print(f"Checkpoint          : {checkpoint}")
    print(f"Real root           : {real_root_dir}")
    print(f"Split               : {split}")
    print(f"Style domain        : {style_domain}")
    print(f"Image size          : {image_size}")
    print(f"Batch size          : {batch_size}")
    print(f"Num workers         : {num_workers}")
    print(f"Lambda content      : {lambda_content}")
    print(f"Lambda style        : {lambda_style}")
    print(f"Style matrix weight : {style_matrix_weight}")
    print(f"Style stat weight   : {style_stat_weight}")
    print(f"Use SmoothL1        : {loss_use_smooth_l1}")
    print("=" * 80)

    model = AdaINStyleTransfer().to(device)
    ckpt_meta = load_checkpoint_into_model(model, checkpoint, device)

    dataset, dataloader = build_test_dataloader(
        real_root_dir=real_root_dir,
        split=split,
        style_domain=style_domain,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"[Eval] Dataset length : {len(dataset)}")
    print(f"[Eval] Num batches    : {len(dataloader)}")

    metrics = evaluate_test_loss(
        model=model,
        dataloader=dataloader,
        device=device,
        lambda_content=lambda_content,
        lambda_style=lambda_style,
        style_matrix_weight=style_matrix_weight,
        style_stat_weight=style_stat_weight,
        loss_use_smooth_l1=loss_use_smooth_l1,
    )

    result = {
        "checkpoint": str(checkpoint),
        "real_root_dir": str(real_root_dir),
        "split": split,
        "style_domain": style_domain,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lambda_content": lambda_content,
        "lambda_style": lambda_style,
        "style_matrix_weight": style_matrix_weight,
        "style_stat_weight": style_stat_weight,
        "loss_use_smooth_l1": loss_use_smooth_l1,
        "checkpoint_best_loss": ckpt_meta.get("best_loss") if isinstance(ckpt_meta, dict) else None,
        "checkpoint_epoch": ckpt_meta.get("epoch") if isinstance(ckpt_meta, dict) else None,
        "metrics": metrics,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Hoàn tất evaluate test loss!")
    print(f"[INFO] avg_total_loss         : {metrics['avg_total_loss']:.6f}")
    print(f"[INFO] avg_content_loss       : {metrics['avg_content_loss']:.6f}")
    print(f"[INFO] avg_style_loss         : {metrics['avg_style_loss']:.6f}")
    print(f"[INFO] avg_style_matrix_loss  : {metrics['avg_style_matrix_loss']:.6f}")
    print(f"[INFO] avg_style_stat_loss    : {metrics['avg_style_stat_loss']:.6f}")
    print(f"[INFO] Output file            : {output_path}")


if __name__ == "__main__":
    main()