@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "ENV_DIR=D:\anaconda3\envs\simpleclick-dinov3"
set "PYTHON=%ENV_DIR%\python.exe"
set "CHECKPOINT=weights\simpleclick_models\cocolvis_vit_base.pth"
set "DINOV3_WEIGHTS=weights\dinov3\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

if not exist "%PYTHON%" (
    echo Python environment not found:
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%demo.py" (
    echo SimpleClick project not found:
    echo %PROJECT_DIR%
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%%CHECKPOINT%" (
    echo SimpleClick checkpoint not found:
    echo %PROJECT_DIR%%CHECKPOINT%
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%%DINOV3_WEIGHTS%" (
    echo DINOv3 ViT-L weights not found:
    echo %PROJECT_DIR%%DINOV3_WEIGHTS%
    pause
    exit /b 1
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%PATH%"
set "PYTHONUTF8=1"

pushd "%PROJECT_DIR%"
"%PYTHON%" demo.py ^
    --checkpoint="%CHECKPOINT%" ^
    --dinov3-model=dinov3_vitl16 ^
    --dinov3-weights="%DINOV3_WEIGHTS%" ^
    --gpu 0 ^
    --no-granularity-refine-model
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo SimpleClick exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
