@echo off
REM Fish Classifier - GPU Training Batch Script
REM Runs training with GPU acceleration on Windows

echo ============================================
echo Fish Species Classifier - GPU Training
echo ============================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv && venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate venv and run training
call venv\Scripts\activate.bat

echo [INFO] Using GPU: NVIDIA GeForce RTX 4060 Laptop GPU
echo [INFO] Model: MobileNetV3-Small (~10MB weights)
echo [INFO] Output: models/mobilenet_v3_small_fish_classifier.pth
echo.

python train.py ^
  --data_dir dataset ^
  --model_name mobilenet_v3_small ^
  --batch_size 64 ^
  --num_workers 2 ^
  --epochs 15 ^
  --lr 0.001

echo.
echo ============================================
echo Training complete!
echo Model saved to: models\mobilenet_v3_small_fish_classifier.pth
echo ============================================
pause