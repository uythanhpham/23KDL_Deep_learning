# 23KDL_Deep_learning/adain-baseline/src/infer.py
"""
Dùng model sinh thử 1 ảnh.

Luồng infer:
- đọc config
- load content/style image
- khởi tạo model
- load checkpoint
- chạy inference
- lưu ảnh output
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


# =========================================================
# Cố gắng import model thật.
# Nếu chưa có / import lỗi thì fallback sang mock model
# để smoke test luồng infer không bị crash.
# =========================================================
try:
    from src.models.adain import AdaINStyleTransfer
except ImportError:
    import torch.nn as nn

    class AdaINStyleTransfer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, content, style, *args, **kwargs):
            # Mock behavior: hòa trộn content + style để test pipeline
            return (content + style) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference cho AdaIN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Đường dẫn tới file config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Đường dẫn tới checkpoint",
    )
    parser.add_argument(
        "--content_path",
        type=str,
        default=None,
        help="Đường dẫn tới 1 ảnh content cụ thể. Nếu không truyền thì lấy ảnh đầu tiên trong content_dir",
    )
    parser.add_argument(
        "--style_path",
        type=str,
        default=None,
        help="Đường dẫn tới 1 ảnh style cụ thể. Nếu không truyền thì lấy ảnh đầu tiên trong style_dir",
    )
    parser.add_argument(
        "--content_dir",
        type=str,
        default="debug_data/content",
        help="Thư mục chứa ảnh content",
    )
    parser.add_argument(
        "--style_dir",
        type=str,
        default="debug_data/style",
        help="Thư mục chứa ảnh style",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/infer",
        help="Thư mục lưu ảnh output",
    )
    return parser.parse_args()


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"[Cảnh báo] Không tìm thấy config tại: {config_path}. Sẽ dùng default args.")
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    if not isinstance(cfg, dict):
        raise ValueError("Nội dung config YAML phải là một dictionary.")

    return cfg


def get_optional(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def load_image(image_path: str | Path, size: int = 512) -> torch.Tensor:
    """
    Load ảnh từ đường dẫn -> resize -> center crop -> tensor -> thêm batch dim.
    Output shape: (1, C, H, W)
    """
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    return transform(img).unsqueeze(0)


def find_first_image_in_dir(folder: str | Path) -> Path | None:
    folder = Path(folder)
    if not folder.exists():
        return None

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
        files.extend(folder.glob(ext.upper()))

    files = sorted(files)
    return files[0] if files else None


def resolve_input_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """
    Ưu tiên:
    - nếu user truyền --content_path / --style_path thì dùng trực tiếp
    - nếu không thì lấy ảnh đầu tiên từ content_dir / style_dir
    """
    if args.content_path is not None:
        content_path = Path(args.content_path)
    else:
        content_path = find_first_image_in_dir(args.content_dir)

    if args.style_path is not None:
        style_path = Path(args.style_path)
    else:
        style_path = find_first_image_in_dir(args.style_dir)

    if content_path is None or not content_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh content. content_path={args.content_path}, content_dir={args.content_dir}"
        )

    if style_path is None or not style_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh style. style_path={args.style_path}, style_dir={args.style_dir}"
        )

    return content_path, style_path


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"[Infer] Không tìm thấy checkpoint tại {checkpoint_path}. Chạy với trọng số hiện tại.")
        return

    print(f"[Infer] Đang load checkpoint từ: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Trường hợp checkpoint là payload dict do train.py save
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[Infer] Đã load ckpt['model_state_dict'] thành công.")
        return

    # Trường hợp checkpoint là state_dict trần
    if isinstance(ckpt, dict):
        try:
            model.load_state_dict(ckpt)
            print("[Infer] Đã load state_dict trực tiếp thành công.")
            return
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint tồn tại nhưng không load được vào model. Lỗi: {e}"
            ) from e

    raise RuntimeError(
        "Định dạng checkpoint không hợp lệ. "
        "Kỳ vọng là payload dict có key 'model_state_dict' hoặc state_dict trần."
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Đang sử dụng device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(get_optional(cfg, "image_size", 512))

    # 1. Chuẩn bị input paths
    content_img_path, style_img_path = resolve_input_paths(args)
    print(
        f"[Infer] Xử lý: Content='{content_img_path.name}' | Style='{style_img_path.name}'"
    )

    # 2. Chuẩn bị model
    model = AdaINStyleTransfer().to(device)
    model.eval()

    # 3. Load checkpoint đúng format mà train.py đang save
    load_checkpoint_into_model(model, args.checkpoint, device)

    # 4. Load ảnh thành tensor
    content_tensor = load_image(content_img_path, size=image_size).to(device)
    style_tensor = load_image(style_img_path, size=image_size).to(device)

    # 5. Inference
    with torch.no_grad():
        output_tensor = model(content_tensor, style_tensor)

    # 6. Clamp về [0, 1] rồi lưu
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

    output_filename = f"output_{content_img_path.stem}_{style_img_path.stem}.png"
    output_path = output_dir / output_filename

    save_image(output_tensor, output_path)
    print(f"[Thành công] Đã lưu ảnh đầu ra tại: {output_path}")


if __name__ == "__main__":
    main()