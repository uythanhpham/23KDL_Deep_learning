@echo off
setlocal

cd /d "%~dp0\.."

set CHECKPOINT=outputs\checkpoints\vangogh\latest.pth
set INPUT_DIR=D:\data\processed\cyclegan\vangogh\testA
set OUTPUT_DIR=outputs\inference\vangogh\A2B

echo [Infer] vangogh photo -^> style
python -m src.infer ^
  --config "configs/train_vangogh.yaml" ^
  --checkpoint "%CHECKPOINT%" ^
  --direction A2B ^
  --input_dir "%INPUT_DIR%" ^
  --output_dir "%OUTPUT_DIR%" ^
  --max_images 50

pause
