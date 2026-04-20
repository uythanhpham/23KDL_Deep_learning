#!/usr/bin/env bash
# [GĐ 3] Giai đoạn 3 — Smoke test core
# Mục đích: Script chạy inference chuẩn hóa cho dự án AdaIN. 
# Sau này có model thật chỉ cần đổi tham số CHECKPOINT ở đây.

echo "========================================="
echo "[GĐ 3] Bắt đầu chạy Inference Smoke Test"
echo "========================================="

# Đảm bảo script dừng ngay lập tức nếu có bất kỳ lỗi nào xảy ra
set -e

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CONFIG_FILE="configs/config.yaml"
CHECKPOINT="checkpoints/mock.pth"
CONTENT_DIR="debug_data/content"
STYLE_DIR="debug_data/style"
OUTPUT_DIR="outputs/infer_smoke"

# --- LỆNH CHẠY ---
python -m src.infer \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT" \
    --content_dir "$CONTENT_DIR" \
    --style_dir "$STYLE_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "========================================="
echo "[Thành công] Hoàn tất quá trình Inference!"
echo "Ảnh kết quả được lưu tại: $OUTPUT_DIR"
echo "========================================="