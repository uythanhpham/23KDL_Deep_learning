:: E:\Nam3_ki2\TH DL\PROJECT\23KDL_Deep_learning\adain-baseline\scripts\01_prepare_data.bat
@echo off
setlocal

echo ==============================================
echo [AdaIN] Prepare debug/mock data
echo ==============================================

cd /d "%~dp0.."

:: CHỈNH THAM SỐ TẠO ẢNH CHỖ NÀY NHA!!!!!!!!!!!!!!!!!!!!!!!!!
if "%PYTHON_BIN%"=="" set PYTHON_BIN=python
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=debug_data
if "%IMAGE_SIZE%"=="" set IMAGE_SIZE=256
if "%NUM_CONTENT%"=="" set NUM_CONTENT=12
if "%NUM_STYLE%"=="" set NUM_STYLE=12
if "%SEED%"=="" set SEED=42
if "%IMAGE_FORMAT%"=="" set IMAGE_FORMAT=png
::_________________________________________________________

echo Project root : %cd%
echo Python bin   : %PYTHON_BIN%
echo Output dir   : %OUTPUT_DIR%
echo Image size   : %IMAGE_SIZE%
echo Num content  : %NUM_CONTENT%
echo Num style    : %NUM_STYLE%
echo Seed         : %SEED%
echo Image format : %IMAGE_FORMAT%
echo ----------------------------------------------

%PYTHON_BIN% -m src.data.prepare_data ^
  --mode debug ^
  --output_dir %OUTPUT_DIR% ^
  --image_size %IMAGE_SIZE% ^
  --num_content %NUM_CONTENT% ^
  --num_style %NUM_STYLE% ^
  --seed %SEED% ^
  --image_format %IMAGE_FORMAT%

if errorlevel 1 (
    echo ----------------------------------------------
    echo [ERROR] prepare_data.py chạy thất bại.
    exit /b 1
)

echo ----------------------------------------------
echo [DONE] debug/mock data generated successfully.
echo ==============================================

endlocal

:: .\scripts\01_prepare_data.bat