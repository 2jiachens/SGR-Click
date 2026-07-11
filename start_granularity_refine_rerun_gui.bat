@echo off
setlocal

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if defined SIMPLECLICK_DINOV3_ENV (
    set "ENV_DIR=%SIMPLECLICK_DINOV3_ENV%"
) else (
    set "ENV_DIR=D:\anaconda3\envs\simpleclick-dinov3"
)

set "PYTHON=%ENV_DIR%\python.exe"
set "CHECKPOINT=weights\simpleclick_models\cocolvis_vit_base.pth"
set "DINOV3_MODEL=dinov3_vitl16"
set "BASE_GRANULARITY_CHECKPOINT="

if defined GRANULARITY_REFINE_RERUN_CHECKPOINT (
    set "BASE_GRANULARITY_CHECKPOINT=%GRANULARITY_REFINE_RERUN_CHECKPOINT%"
) else if exist "%PROJECT_DIR%\experiments\granularity_refine_rerun\best.pth" (
    set "BASE_GRANULARITY_CHECKPOINT=experiments\granularity_refine_rerun\best.pth"
) else if exist "%PROJECT_DIR%\experiments\granularity_refine_rerun\latest.pth" (
    set "BASE_GRANULARITY_CHECKPOINT=experiments\granularity_refine_rerun\latest.pth"
)

if not exist "%PYTHON%" (
    echo Python environment not found:
    echo %PYTHON%
    echo.
    echo You can override it by setting SIMPLECLICK_DINOV3_ENV.
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%\granularity_recovery\demo_granularity_recovery.py" (
    echo Granularity Recovery demo not found:
    echo %PROJECT_DIR%\granularity_recovery\demo_granularity_recovery.py
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%\%CHECKPOINT%" (
    echo SimpleClick checkpoint not found:
    echo %PROJECT_DIR%\%CHECKPOINT%
    pause
    exit /b 1
)

if "%BASE_GRANULARITY_CHECKPOINT%"=="" (
    echo Rerun base checkpoint not found under:
    echo %PROJECT_DIR%\experiments\granularity_refine_rerun
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%\%BASE_GRANULARITY_CHECKPOINT%" (
    echo Rerun base checkpoint not found:
    echo %PROJECT_DIR%\%BASE_GRANULARITY_CHECKPOINT%
    pause
    exit /b 1
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%PATH%"
set "PYTHONUTF8=1"
set "PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%"

if /i "%~1"=="--check" (
    echo Granularity Recovery GUI launcher is ready.
    echo Using rerun base checkpoint: %BASE_GRANULARITY_CHECKPOINT%
    exit /b 0
)

pushd "%PROJECT_DIR%"
"%PYTHON%" -m granularity_recovery.demo_granularity_recovery ^
  --checkpoint="%CHECKPOINT%" ^
  --base-granularity-checkpoint="%BASE_GRANULARITY_CHECKPOINT%" ^
  --dinov3-model=%DINOV3_MODEL% ^
  --gpu 0
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Granularity Recovery demo exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
