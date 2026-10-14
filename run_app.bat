@echo off
REM Fish Classifier - Web UI Batch Script
REM Launches Gradio web interface

echo ============================================
echo Fish Species Classifier - Web UI
echo ============================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Auto-detect latest model in models/ folder
python app.py

echo.
pause