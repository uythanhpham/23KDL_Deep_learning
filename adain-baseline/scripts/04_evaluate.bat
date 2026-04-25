@echo off
chcp 65001 >nul

echo =========================================
echo [GĐ 8] Bắt đầu tính toán Metrics
echo =========================================

:: --- CẤU HÌNH ĐƯỜNG DẪN (Bạn chỉ cần sửa tại đây) ---
:: Thư mục ảnh bạn vừa gen ra
set PRED_DIR=outputs/infer_final_watercolor

:: Thư mục chứa ảnh content gốc để so sánh
set REF_DIR=data/processed/test/content

:: Tên file kết quả (nên đặt tên theo style để dễ phân biệt)
set OUTPUT_JSON=outputs/eval/metrics_watercolor.json

:: --- THIẾT LẬP MÔI TRƯỜNG ---
set PYTHONPATH=.

:: --- LỆNH CHẠY (Không cần đụng vào phần này) ---
python -m src.evaluate ^
    --pred_dir "%PRED_DIR%" ^
    --ref_dir "%REF_DIR%" ^
    --output_file "%OUTPUT_JSON%"

if %ERRORLEVEL% NEQ 0 (
    echo [Lỗi] Quá trình đánh giá bị dừng do có lỗi!
    exit /b %ERRORLEVEL%
)

echo =========================================
echo [Thành công] Đánh giá hoàn tất!
echo Kết quả tại: %OUTPUT_JSON%
echo =========================================