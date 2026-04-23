from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.models.adain import AdaINStyleTransfer

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STYLE_FOLDER_BY_DOMAIN = {
    "anime": "style_anime",
    "sketch": "style_sketch",
    "watercolor": "style_watercolor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load best.pth và generate ảnh stylized từ content cố định hoặc random content, "
            "đồng thời cho phép chọn style domain / style ảnh cụ thể."
        )
    )
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Đường dẫn config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth", help="Đường dẫn best.pth")
    parser.add_argument(
        "--real_root_dir",
        type=str,
        default=None,
        help="Root processed data, ví dụ: E:/.../data/processed",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Split dữ liệu muốn lấy content/style",
    )
    parser.add_argument("--image_size", type=int, default=None, help="Resize ảnh vuông")
    parser.add_argument("--alpha", type=float, default=1.0, help="Mức độ style: 0.0 -> giữ content, 1.0 -> full style")
    parser.add_argument("--seed", type=int, default=42, help="Seed random")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/generated_from_best",
        help="Thư mục lưu ảnh output",
    )

    # Content source
    parser.add_argument("--content_path", type=str, default=None, help="1 ảnh content cụ thể")
    parser.add_argument("--content_dir", type=str, default=None, help="Folder content override")
    parser.add_argument(
        "--random_content",
        action="store_true",
        help="Bật chọn random content từ content_dir / processed/<split>/content",
    )
    parser.add_argument(
        "--num_contents",
        type=int,
        default=1,
        help="Số lượng content cần lấy. Nếu không random thì lấy từ đầu folder.",
    )

    # Style source
    parser.add_argument(
        "--style_paths",
        nargs="*",
        default=None,
        help="Danh sách ảnh style cụ thể. Nếu truyền thì ưu tiên hơn style domain.",
    )
    parser.add_argument("--style_dir", type=str, default=None, help="Folder style override")
    parser.add_argument(
        "--style_domains",
        nargs="*",
        default=None,
        help="Danh sách style domain muốn dùng, ví dụ: anime sketch watercolor hoặc all",
    )
    parser.add_argument(
        "--random_style",
        action="store_true",
        help="Bật chọn random style trong mỗi style domain / style_dir",
    )
    parser.add_argument(
        "--num_styles_per_domain",
        type=int,
        default=1,
        help="Số style cần lấy cho mỗi domain khi dùng --style_domains",
    )

    return parser.parse_args()


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_optional(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTS


def scan_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if is_image_file(p)])


def load_image(path: str | Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    print(f"[Load] Checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[Load] Đã load ckpt['model_state_dict'] thành công.")
        return

    if isinstance(ckpt, dict):
        try:
            model.load_state_dict(ckpt)
            print("[Load] Đã load state_dict trực tiếp thành công.")
            return
        except Exception as e:
            raise RuntimeError(f"Checkpoint tồn tại nhưng không load được vào model. Lỗi: {e}") from e

    raise RuntimeError("Định dạng checkpoint không hợp lệ.")


def normalize_style_domains(style_domains: Optional[Sequence[str]], default_domain: str) -> List[str]:
    raw = list(style_domains) if style_domains else [default_domain]
    normalized: List[str] = []
    for item in raw:
        x = str(item).strip().lower()
        if x in {"style_anime", "anime"}:
            normalized.append("anime")
        elif x in {"style_sketch", "sketch"}:
            normalized.append("sketch")
        elif x in {"style_watercolor", "watercolor"}:
            normalized.append("watercolor")
        elif x == "all":
            normalized.extend(["anime", "sketch", "watercolor"])
        else:
            raise ValueError(f"Style domain không hợp lệ: {item}")
    # unique but keep order
    out: List[str] = []
    seen = set()
    for x in normalized:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def choose_items(paths: List[Path], k: int, use_random: bool, rng: random.Random) -> List[Path]:
    if len(paths) == 0:
        return []
    k = max(1, min(k, len(paths)))
    if use_random:
        return rng.sample(paths, k)
    return paths[:k]


def resolve_content_paths(
    real_root_dir: Path,
    split: str,
    content_path: Optional[str],
    content_dir: Optional[str],
    random_content: bool,
    num_contents: int,
    rng: random.Random,
) -> List[Path]:
    if content_path is not None:
        p = Path(content_path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy content_path: {p}")
        return [p]

    folder = Path(content_dir) if content_dir is not None else (real_root_dir / split / "content")
    files = scan_images(folder)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy ảnh content trong folder: {folder}")
    return choose_items(files, num_contents, random_content, rng)


def resolve_style_entries(
    real_root_dir: Path,
    split: str,
    style_paths: Optional[Sequence[str]],
    style_dir: Optional[str],
    style_domains: Optional[Sequence[str]],
    default_style_domain: str,
    random_style: bool,
    num_styles_per_domain: int,
    rng: random.Random,
) -> List[Tuple[Path, str]]:
    if style_paths:
        entries: List[Tuple[Path, str]] = []
        for raw in style_paths:
            p = Path(raw)
            if not p.exists():
                raise FileNotFoundError(f"Không tìm thấy style_path: {p}")
            entries.append((p, "custom"))
        return entries

    if style_dir is not None:
        folder = Path(style_dir)
        files = scan_images(folder)
        if not files:
            raise FileNotFoundError(f"Không tìm thấy ảnh style trong folder: {folder}")
        chosen = choose_items(files, num_styles_per_domain, random_style, rng)
        return [(p, folder.name) for p in chosen]

    domains = normalize_style_domains(style_domains, default_style_domain)
    entries: List[Tuple[Path, str]] = []
    for domain in domains:
        folder = real_root_dir / split / STYLE_FOLDER_BY_DOMAIN[domain]
        files = scan_images(folder)
        if not files:
            raise FileNotFoundError(f"Không tìm thấy ảnh style cho domain '{domain}' trong folder: {folder}")
        chosen = choose_items(files, num_styles_per_domain, random_style, rng)
        for p in chosen:
            entries.append((p, domain))
    return entries


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    real_root_dir = Path(str(args.real_root_dir or get_optional(cfg, "real_root_dir", "data/processed")))
    split = str(args.split or get_optional(cfg, "split", "train"))
    image_size = int(args.image_size or get_optional(cfg, "image_size", 256))
    default_style_domain = str(get_optional(cfg, "style_domain", "anime"))

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Generate images from best.pth")
    print("=" * 80)
    print(f"Device        : {device}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Real root     : {real_root_dir}")
    print(f"Split         : {split}")
    print(f"Image size    : {image_size}")
    print(f"Alpha         : {args.alpha}")
    print("=" * 80)

    content_paths = resolve_content_paths(
        real_root_dir=real_root_dir,
        split=split,
        content_path=args.content_path,
        content_dir=args.content_dir,
        random_content=args.random_content,
        num_contents=args.num_contents,
        rng=rng,
    )
    style_entries = resolve_style_entries(
        real_root_dir=real_root_dir,
        split=split,
        style_paths=args.style_paths,
        style_dir=args.style_dir,
        style_domains=args.style_domains,
        default_style_domain=default_style_domain,
        random_style=args.random_style,
        num_styles_per_domain=args.num_styles_per_domain,
        rng=rng,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AdaINStyleTransfer().to(device)
    model.eval()
    load_checkpoint_into_model(model, args.checkpoint, device)

    print(f"[Info] Số content sẽ dùng : {len(content_paths)}")
    print(f"[Info] Số style sẽ dùng   : {len(style_entries)}")

    total_saved = 0
    with torch.no_grad():
        for content_idx, content_path in enumerate(content_paths, start=1):
            content_tensor = load_image(content_path, size=image_size).to(device)
            print(f"\n[Content {content_idx}/{len(content_paths)}] {content_path}")

            for style_idx, (style_path, style_tag) in enumerate(style_entries, start=1):
                style_tensor = load_image(style_path, size=image_size).to(device)
                output_tensor = model(content_tensor, style_tensor, alpha=args.alpha)
                output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

                output_name = (
                    f"out_c-{safe_name(content_path)}"
                    f"_s-{safe_name(style_path)}"
                    f"_tag-{style_tag}"
                    f"_a-{args.alpha:.2f}.png"
                )
                output_path = output_dir / output_name
                save_image(output_tensor, output_path)
                total_saved += 1

                print(
                    f"  -> [{style_idx}/{len(style_entries)}] "
                    f"style={style_path.name} | tag={style_tag} | saved={output_path}"
                )

    print("\n" + "=" * 80)
    print(f"[Done] Tổng số ảnh đã lưu: {total_saved}")
    print(f"[Done] Output dir        : {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
