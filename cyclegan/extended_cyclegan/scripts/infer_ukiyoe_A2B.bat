@echo off
setlocal

cd /d "%~dp0\.."

set CHECKPOINT=outputs\checkpoints\ukiyoe\latest.pth
set INPUT_DIR=D:\data\processed\cyclegan\ukiyoe\testA
set OUTPUT_DIR=outputs\inference\ukiyoe\A2B

echo [Infer] ukiyoe photo -^> style
python -m src.infer ^
  --config "configs/train_ukiyoe.yaml" ^
  --checkpoint "%CHECKPOINT%" ^
  --direction A2B ^
  --input_dir "%INPUT_DIR%" ^
  --output_dir "%OUTPUT_DIR%" ^
  --max_images 50

pause
