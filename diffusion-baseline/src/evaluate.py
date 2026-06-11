"""
========================================================================
evaluate.py — CLI ĐÁNH GIÁ CHUNG (model-agnostic)
========================================================================
Entry point duy nhất để chấm điểm output của BẤT KỲ model nào (Diffusion/CycleGAN/AdaIN).
Chỉ cần các thư mục ảnh — lõi tính toán nằm ở src/eval/metrics.py.

Cách dùng:
    python -m src.evaluate \
        --pred_dir  <thư mục OUTPUT của model> \
        --ref_dir   <thư mục CONTENT gốc> \
        --style_dir <thư mục ảnh Van Gogh tham chiếu>   # cho FID/KID
        --model_name diffusion

Metric:
  - content (per-image, ref vs pred): LPIPS↓, DINOv2 cos↑, CLIP style↑  (đọc theo TRADE-OFF)
  - distribution (pred vs style_dir): FID↓, KID↓  (KID là metric phân phối chính)
Xuất: <output_dir>/per_image.csv  +  --output_file (JSON tổng hợp).
========================================================================
"""
import os
import json
import time
import argparse

import torch

from src.eval import metrics as M


def main():
    parser = argparse.ArgumentParser(description="Đánh giá chung style transfer (model-agnostic)")
    parser.add_argument("--pred_dir", type=str, required=True, help="Thư mục ảnh OUTPUT của model")
    parser.add_argument("--ref_dir", type=str, required=True, help="Thư mục ảnh CONTENT gốc (ghép 1-1 với output)")
    parser.add_argument("--style_dir", type=str, default=None, help="Thư mục ảnh Van Gogh tham chiếu (để tính FID/KID)")
    parser.add_argument("--model_name", type=str, default="model", help="Tên model (để gắn vào báo cáo khi gộp)")
    parser.add_argument("--output_file", type=str, default="outputs/eval/summary.json", help="File JSON tổng hợp")
    parser.add_argument("--csv_file", type=str, default=None, help="File CSV per-image (mặc định cạnh output_file)")
    parser.add_argument("--no_dino", action="store_true", help="Bỏ DINO (nếu thiếu timm)")
    parser.add_argument("--no_clip", action="store_true", help="Bỏ CLIP (nếu thiếu open_clip)")
    args = parser.parse_args()

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Evaluate trên: {device.upper()} | model = {args.model_name}")

    out_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    csv_path = args.csv_file or os.path.join(out_dir, "per_image.csv")

    # 1) Per-image: LPIPS / DINOv2 / CLIP (content vs output)
    print("[*] Tính per-image (LPIPS / DINOv2 / CLIP)...")
    rows = M.content_metrics(args.ref_dir, args.pred_dir, device=device,
                             use_dino=not args.no_dino, use_clip=not args.no_clip)
    M.save_csv(rows, csv_path)
    summ = M.summarize(rows)

    # 2) Phân phối: FID / KID (output vs tập Van Gogh)
    dist = {}
    if args.style_dir:
        print("[*] Tính FID / KID (vs tập Van Gogh)...")
        dist = M.distribution_metrics(args.pred_dir, args.style_dir, device=device)
    else:
        print("[!] Bỏ qua FID/KID (chưa truyền --style_dir).")

    report = {
        "model_name": args.model_name,
        "pred_dir": args.pred_dir,
        "ref_dir": args.ref_dir,
        "style_dir": args.style_dir,
        "num_images": len(rows),
        **summ,
        **dist,
        "eval_time_s": round(time.time() - start, 2),
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 3) In bảng
    def fmt(x, n=4):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) else "N/A"

    print("\n" + "=" * 56)
    print(f"  KẾT QUẢ ĐÁNH GIÁ — {args.model_name}  ({len(rows)} ảnh)")
    print("=" * 56)
    print("  [Content — đọc theo TRADE-OFF, không minimize đơn lẻ]")
    print(f"    LPIPS(content,output)   ↓ : {fmt(summ.get('lpips_mean'))}")
    print(f"    DINOv2 cos(content,out) ↑ : {fmt(summ.get('dino_cos_mean'))}")
    print("  [Style]")
    print(f"    CLIP style score        ↑ : {fmt(summ.get('clip_style_mean'))}")
    print(f"    FID (vs Van Gogh)       ↓ : {fmt(dist.get('fid'), 3)}")
    print(f"    KID (vs Van Gogh) ⭐    ↓ : {fmt(dist.get('kid_mean'), 5)} ± {fmt(dist.get('kid_std'), 5)}")
    print("-" * 56)
    print(f"  CSV per-image : {csv_path}")
    print(f"  JSON tổng hợp : {args.output_file}")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
