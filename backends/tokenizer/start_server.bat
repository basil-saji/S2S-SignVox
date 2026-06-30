@echo off
echo.
echo ╔══════════════════════════════════════════════════╗
echo ║       Zora - Sign Language Translation Server    ║
echo ╚══════════════════════════════════════════════════╝
echo.

if not exist weights\model_best.pth (
    echo ⚠️  WARNING: No model weights found at weights\model_best.pth
    echo    The server will start but predictions will be random.
    echo    Copy model_best.pth to the weights folder first.
    echo.
)

echo Starting server on http://localhost:8000 ...
echo.
python server.py
pause
