#!/usr/bin/env bash
# [GĐ 4] Giai đoạn 4 — Chuẩn hóa entrypoint Evaluate
# Mục đích: Script chạy đánh giá (evaluate) chuẩn hóa cho dự án AdaIN.

echo "========================================="
echo "[GĐ 4] Bắt đầu chạy Evaluate Smoke Test"
echo "========================================="

# Đảm bảo script dừng ngay nếu có lỗi
set -e

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CONFIG_FILE="configs/config.yaml"
PRED_DIR="outputs/infer_smoke"
REF_DIR="debug_data"
OUTPUT_FILE="outputs/eval/metrics_smoke.json"

# --- LỆNH CHẠY ---
python -m src.evaluate \
    --config "$CONFIG_FILE" \
    --pred_dir "$PRED_DIR" \
    --ref_dir "$REF_DIR" \
    --output_file "$OUTPUT_FILE"

echo "========================================="
echo "[Thành công] Hoàn tất quá trình Đánh giá!"
echo "Báo cáo được lưu tại: $OUTPUT_FILE"
echo "========================================="
