@echo off
setlocal
set "PROJECT_DIR=%~dp0..\simpleclick+dinov3"
set "ENV_DIR=D:\anaconda3\envs\simpleclick-dinov3"
set "PYTHON=%ENV_DIR%\python.exe"
set "CHECKPOINT=weights\simpleclick_models\cocolvis_vit_base.pth"
set "GRANULARITY_REFINE_CHECKPOINT=experiments\granularity_refine\best.pth"

if not exist "%PYTHON%" (
    echo Python environment not found:
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%\demo.py" (
    echo SimpleClick project not found:
    echo %PROJECT_DIR%
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%\%CHECKPOINT%" (
    echo SimpleClick checkpoint not found:
    echo %PROJECT_DIR%\%CHECKPOINT%
    pause
    exit /b 1
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%PATH%"
set "PYTHONUTF8=1"

if /i "%~1"=="--check" (
    echo SimpleClick GUI launcher is ready.
    exit /b 0
)

pushd "%PROJECT_DIR%"
if exist "%PROJECT_DIR%\%GRANULARITY_REFINE_CHECKPOINT%" (
    "%PYTHON%" demo.py --checkpoint="%CHECKPOINT%" --granularity-refine-checkpoint="%GRANULARITY_REFINE_CHECKPOINT%" --dinov3-model=dinov3_vitl16 --gpu 0
) else (
    "%PYTHON%" demo.py --checkpoint="%CHECKPOINT%" --dinov3-model=dinov3_vitl16 --gpu 0
)
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo SimpleClick exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
