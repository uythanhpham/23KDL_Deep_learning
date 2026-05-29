#!/bin/bash
set -e
echo "========================================="
echo "[GĐ 4] Đánh giá chất lượng ảnh"
echo "========================================="
PRED_DIR="outputs/samples"
REF_DIR="data/content"
OUTPUT="outputs/eval/metrics.json"
mkdir -p outputs/eval
python -m src.evaluate \
    --pred_dir $PRED_DIR --ref_dir $REF_DIR --output_file $OUTPUT
echo "DONE: metrics tại $OUTPUT"
