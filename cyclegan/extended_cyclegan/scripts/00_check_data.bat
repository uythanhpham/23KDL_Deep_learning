@echo off
setlocal

cd /d "%~dp0\.."

echo [Check] Kiem tra dataset CycleGAN...
python -m src.inspect_data --root "D:/data/processed/cyclegan"

pause
