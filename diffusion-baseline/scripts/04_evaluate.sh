#!/bin/bash
set -e
echo "========================================="
echo "[GĐ 4] Đánh giá chất lượng ảnh (CLI chung)"
echo "========================================="

# Có thể override bằng biến môi trường, ví dụ:
#   PRED_DIR=outputs/eval_out/output REF_DIR=outputs/eval_out/content STYLE_DIR=data/style ./scripts/04_evaluate.sh
PRED_DIR="${PRED_DIR:-outputs/eval_out/output}"   # thư mục OUTPUT của model
REF_DIR="${REF_DIR:-outputs/eval_out/content}"    # thư mục CONTENT (ghép 1-1 với output)
STYLE_DIR="${STYLE_DIR:-data/style}"              # tập Van Gogh tham chiếu → FID/KID
MODEL_NAME="${MODEL_NAME:-diffusion}"
OUTPUT="${OUTPUT:-outputs/eval/summary.json}"

mkdir -p "$(dirname "$OUTPUT")"

# LPIPS/DINOv2/CLIP (ref vs pred) + FID/KID (pred vs style). Lõi: src/eval/metrics.py
python -m src.evaluate \
    --pred_dir "$PRED_DIR" \
    --ref_dir "$REF_DIR" \
    --style_dir "$STYLE_DIR" \
    --model_name "$MODEL_NAME" \
    --output_file "$OUTPUT"

echo "DONE: summary tại $OUTPUT (+ per_image.csv cùng thư mục)"
