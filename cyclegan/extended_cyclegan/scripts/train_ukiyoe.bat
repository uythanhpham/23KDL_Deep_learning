@echo off
setlocal

cd /d "%~dp0\.."

echo [Train] CycleGAN style=ukiyoe
python -m src.train --config "configs/train_ukiyoe.yaml"

pause
