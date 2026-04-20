@echo off
chcp 65001 >nul
:: [GĐ 4] Giai đoạn 4 — Chuẩn hóa entrypoint Evaluate (Windows)

echo =========================================
echo [GĐ 4] Bắt đầu chạy Evaluate Smoke Test
echo =========================================

:: --- CẤU HÌNH ĐƯỜNG DẪN ---
set CONFIG_FILE=configs/config.yaml
set PRED_DIR=outputs/infer_smoke
set REF_DIR=debug_data
set OUTPUT_FILE=outputs/eval/metrics_smoke.json

:: --- LỆNH CHẠY ---
python -m src.evaluate ^
    --config "%CONFIG_FILE%" ^
    --pred_dir "%PRED_DIR%" ^
    --ref_dir "%REF_DIR%" ^
    --output_file "%OUTPUT_FILE%"

if %ERRORLEVEL% NEQ 0 (
    echo [Lỗi] Quá trình chạy bị dừng do có lỗi!
    exit /b %ERRORLEVEL%
)

echo =========================================
echo [Thành công] Hoàn tất quá trình Đánh giá!
echo Báo cáo được lưu tại: %OUTPUT_FILE%
echo =========================================