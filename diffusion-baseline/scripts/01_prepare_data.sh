#!/bin/bash
set -e
echo "========================================="
echo "[GĐ 1] Chuẩn bị debug data (content + style)"
echo "========================================="
OUTPUT_DIR="debug_data"
IMAGE_SIZE=64
NUM_IMAGES=20
python -m src.data.prepare_data \
    --output_dir $OUTPUT_DIR --image_size $IMAGE_SIZE --num_images $NUM_IMAGES
echo "DONE: $OUTPUT_DIR/content + $OUTPUT_DIR/style"
