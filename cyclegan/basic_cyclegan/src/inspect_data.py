from __future__ import annotations

import argparse
from pathlib import Path

from src.data.datasets import list_images


def build_argparser():
    parser = argparse.ArgumentParser(description="Kiểm tra cấu trúc dataset CycleGAN")
    parser.add_argument("--root", type=str, default="D:/data/processed/cyclegan")
    parser.add_argument("--styles", nargs="+", default=["monet", "vangogh", "ukiyoe", "cezanne"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root)
    print(f"[Inspect] root={root}")

    ok = True
    for style in args.styles:
        print(f"\n=== {style} ===")
        for split in ["trainA", "trainB", "testA", "testB"]:
            path = root / style / split
            try:
                imgs = list_images(path)
                print(f"[OK] {path}: {len(imgs)} ảnh")
            except Exception as exc:
                ok = False
                print(f"[LỖI] {path}: {exc}")

    if not ok:
        raise SystemExit("Dataset chưa đúng. Hãy kiểm tra lại path/split ở trên.")
    print("\n[OK] Dataset CycleGAN đã đúng cấu trúc cơ bản.")


if __name__ == "__main__":
    main()
