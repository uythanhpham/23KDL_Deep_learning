@echo off
chcp 65001 >nul
:: [GĐ 3] Giai đoạn 3 — Smoke test core (Phiên bản Windows)
:: Lệnh chcp 65001 giúp hiển thị tiếng Việt không bị lỗi font trong Terminal

echo =========================================
echo [GĐ 3] Bắt đầu chạy Inference Smoke Test
echo =========================================

:: --- CẤU HÌNH ĐƯỜNG DẪN ---
set CONFIG_FILE=configs/config.yaml
<<<<<<< HEAD
set CHECKPOINT=checkpoints/adain_epoch_7.pth
set CONTENT_DIR=data/processed/test/content
set STYLE_DIR=data/processed/test/style_anime
set OUTPUT_DIR=outputs/infer_final_anime
=======
set CHECKPOINT=src/checkpoints/adain_epoch_7.pth
set CONTENT_DIR=data/processed/test/content
set STYLE_DIR=data/processed/test/style_watercolor
set OUTPUT_DIR=outputs/infer_final_watercolor

>>>>>>> role/eval

:: --- LỆNH CHẠY ---
:: Lưu ý: Windows dùng dấu ^ để xuống dòng thay vì dấu \ như Linux
python -m src.infer ^
    --config "%CONFIG_FILE%" ^
    --checkpoint "%CHECKPOINT%" ^
    --content_dir "%CONTENT_DIR%" ^
    --style_dir "%STYLE_DIR%" ^
    --output_dir "%OUTPUT_DIR%"

:: Bắt lỗi nếu lệnh python chạy thất bại
if %ERRORLEVEL% NEQ 0 (
    echo [Lỗi] Quá trình chạy bị dừng do có lỗi!
    exit /b %ERRORLEVEL%
)

echo =========================================
echo [Thành công] Hoàn tất quá trình Inference!
echo Ảnh kết quả được lưu tại: %OUTPUT_DIR%
echo =========================================