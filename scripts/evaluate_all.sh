#!/bin/bash
# Đánh giá THỐNG NHẤT output của cả 5 model bằng chung một bộ metric
# (LPIPS + DINOv2 + CLIP style + FID/KID — lõi: diffusion-baseline/src/eval/metrics.py).
#
# RÀNG BUỘC GHÉP CẶP: mỗi thư mục output phải chứa đúng 1 ảnh cho 1 ảnh content,
# và tên file khi sort phải trùng thứ tự với thư mục content (metrics.py ghép cặp
# theo thứ tự sort — pair_by_order). Infer mỗi model trên cùng tập testA.
#
# Cách dùng: chạy từ repo gốc, override đường dẫn output qua biến môi trường:
#   ADAIN_OUT=... ADAIN_MS_OUT=... CYCLEGAN_OUT=... PALETTE_CG_OUT=... DIFFUSION_OUT=... \
#       bash scripts/evaluate_all.sh
# Thêm EXTRA_FLAGS="--no_dino --no_clip" nếu thiếu timm/open_clip.
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONTENT_DIR="${CONTENT_DIR:-$ROOT_DIR/data/archive/testA}"   # content gốc (ghép 1-1 với output)
STYLE_DIR="${STYLE_DIR:-$ROOT_DIR/data/archive/testB}"       # tranh Van Gogh tham chiếu → FID/KID
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/outputs/eval}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Đường dẫn output mặc định của từng model (theo layout chuẩn của mỗi project)
ADAIN_OUT="${ADAIN_OUT:-$ROOT_DIR/adain/adain_baseline/outputs/infer}"
ADAIN_MS_OUT="${ADAIN_MS_OUT:-$ROOT_DIR/adain/adain_multiscale/outputs/infer}"
CYCLEGAN_OUT="${CYCLEGAN_OUT:-$ROOT_DIR/cyclegan/basic_cyclegan/outputs/inference/vangogh/A2B}"
PALETTE_CG_OUT="${PALETTE_CG_OUT:-$ROOT_DIR/cyclegan/extended_cyclegan/outputs/inference/vangogh/A2B}"
DIFFUSION_OUT="${DIFFUSION_OUT:-$ROOT_DIR/diffusion-baseline/outputs/eval_out/output}"

mkdir -p "$EVAL_DIR"

evaluate_one() {
    local name="$1"
    local pred_dir="$2"
    if [ ! -d "$pred_dir" ] || [ -z "$(ls -A "$pred_dir" 2>/dev/null)" ]; then
        echo "[Bỏ qua] $name: chưa có output tại $pred_dir"
        return 0
    fi
    echo "=== Đánh giá $name ($pred_dir) ==="
    (cd "$ROOT_DIR/diffusion-baseline" && python -m src.evaluate \
        --pred_dir "$pred_dir" \
        --ref_dir "$CONTENT_DIR" \
        --style_dir "$STYLE_DIR" \
        --model_name "$name" \
        --output_file "$EVAL_DIR/${name}.json" \
        $EXTRA_FLAGS)
}

evaluate_one "adain_baseline"   "$ADAIN_OUT"
evaluate_one "adain_multiscale" "$ADAIN_MS_OUT"
evaluate_one "cyclegan_basic"   "$CYCLEGAN_OUT"
evaluate_one "cyclegan_palette" "$PALETTE_CG_OUT"
evaluate_one "diffusion"        "$DIFFUSION_OUT"

echo
echo "[Done] Kết quả JSON tại: $EVAL_DIR/ (mỗi model 1 file + per_image.csv cạnh đó)"
