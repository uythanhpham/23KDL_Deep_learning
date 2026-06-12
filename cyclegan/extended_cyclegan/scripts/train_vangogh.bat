@echo off
setlocal

cd /d "%~dp0\.."

echo [Train] CycleGAN style=vangogh
python -m src.train --config "configs/vangogh_local.yaml"

pause
