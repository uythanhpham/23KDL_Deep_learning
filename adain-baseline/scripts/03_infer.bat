@echo off
chcp 65001 >nul

echo =========================================
echo [GĐ 7] Bắt đầu chạy Inference Thật
echo =========================================

:: --- CẤU HÌNH ĐƯỜNG DẪN ---
set CONFIG_FILE=configs/config.yaml
set CHECKPOINT=src/checkpoints/best_model.pth
set CONTENT_DIR=data/processed/test/content
set STYLE_DIR=data/processed/test/style_watercolor
set OUTPUT_DIR=outputs/infer_final_watercolor

:: --- LỆNH CHẠY ---
python -m src.infer ^
    --config "%CONFIG_FILE%" ^
    --checkpoint "%CHECKPOINT%" ^
    --content_dir "%CONTENT_DIR%" ^
    --style_dir "%STYLE_DIR%" ^
    --output_dir "%OUTPUT_DIR%"

if %ERRORLEVEL% NEQ 0 (
    echo [Lỗi] Quá trình chạy bị dừng do có lỗi!
    exit /b %ERRORLEVEL%
)

echo =========================================
echo [Thành công] Hoàn tất! Kết quả tại: %OUTPUT_DIR%
echo =========================================