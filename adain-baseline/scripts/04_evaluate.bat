@echo off
chcp 65001 >nul

echo =========================================
echo [GĐ 8] Bắt đầu tính toán Metrics (SSIM, LPIPS)
echo =========================================

:: --- THIẾT LẬP MÔI TRƯỜNG ---
:: Đảm bảo Python nhận diện được thư mục src
set PYTHONPATH=.

:: --- LỆNH CHẠY ---
python -m src.evaluate

if %ERRORLEVEL% NEQ 0 (
    echo [Lỗi] Quá trình đánh giá bị dừng do có lỗi!
    exit /b %ERRORLEVEL%
)

echo =========================================
echo [Thành công] Đánh giá hoàn tất!
echo Xem kết quả tại: outputs/eval/metrics.json
echo =========================================