@echo off
echo ══════════════════════════════════════════════════
echo   Zora Server - One-Time Setup
echo ══════════════════════════════════════════════════
echo.

:: Create weights directory
if not exist weights mkdir weights

:: Install Python dependencies
echo [1/2] Installing Python packages (this may take a few minutes)...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: pip install failed. Make sure Python is installed and in PATH.
    echo Download Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo [2/2] Checking for model weights...
if exist weights\model_best.pth (
    echo Model weights found!
) else (
    echo No model weights found at weights\model_best.pth
    echo.
    echo You need to copy the trained model weights here.
    echo Options:
    echo   1. Copy from Kaggle notebook output
    echo   2. Run the Kaggle notebook and download the weights
    echo.
)

echo.
echo ══════════════════════════════════════════════════
echo   Setup complete! Run 'start_server.bat' to launch.
echo ══════════════════════════════════════════════════
pause
