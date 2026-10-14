@echo off
REM Fish Classifier - Main Menu
REM Choose what to run

:MENU
cls
echo ============================================
echo  FISH SPECIES CLASSIFIER - MAIN MENU
echo ============================================
echo.
echo [1] Train Model (GPU) - MobileNetV3-Small
echo [2] Train Model (GPU) - ResNet18
echo [3] Train Model (CPU) - fallback
echo [4] Launch Web UI (Gradio)
echo [5] Download Model from Hugging Face
echo [6] Install/Update Dependencies
echo [7] Exit
echo.
set /p CHOICE=Enter choice [1-7]: 

if "%CHOICE%"=="1" goto TRAIN_SMALL
if "%CHOICE%"=="2" goto TRAIN_RESNET
if "%CHOICE%"=="3" goto TRAIN_CPU
if "%CHOICE%"=="4" goto RUN_APP
if "%CHOICE%"=="5" goto DOWNLOAD
if "%CHOICE%"=="6" goto INSTALL
if "%CHOICE%"=="7" goto EXIT

echo Invalid choice. Press any key to try again...
pause
goto MENU

:TRAIN_SMALL
cls
echo Starting MobileNetV3-Small training on GPU...
call venv\Scripts\activate.bat
python train.py --data_dir dataset --model_name mobilenet_v3_small --batch_size 64 --num_workers 2 --epochs 15 --lr 0.001
pause
goto MENU

:TRAIN_RESNET
cls
echo Starting ResNet18 training on GPU...
call venv\Scripts\activate.bat
python train.py --data_dir dataset --model_name resnet18 --batch_size 32 --num_workers 2 --epochs 15 --lr 0.001
pause
goto MENU

:TRAIN_CPU
cls
echo Starting CPU training (no CUDA)...
set CUDA_VISIBLE_DEVICES=-1
call venv\Scripts\activate.bat
python train.py --data_dir dataset --model_name mobilenet_v3_small --batch_size 16 --num_workers 0 --epochs 15 --lr 0.001
pause
goto MENU

:RUN_APP
cls
echo Launching Gradio Web UI...
call venv\Scripts\activate.bat
python app.py
pause
goto MENU

:DOWNLOAD
cls
echo Downloading model from Hugging Face Hub...
call venv\Scripts\activate.bat
python download_model.py
pause
goto MENU

:INSTALL
cls
echo Installing/Updating dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Done!
pause
goto MENU

:EXIT
echo Goodbye!
timeout /t 2 >nul
exit