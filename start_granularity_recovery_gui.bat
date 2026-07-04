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
set "RECOVERY_CHECKPOINT="

if defined BASE_GRANULARITY_CHECKPOINT (
    set "BASE_GRANULARITY_CHECKPOINT=%BASE_GRANULARITY_CHECKPOINT%"
) else if exist "%PROJECT_DIR%\experiments\granularity_refine\best.pth" (
    set "BASE_GRANULARITY_CHECKPOINT=experiments\granularity_refine\best.pth"
) else if exist "%PROJECT_DIR%\experiments\granularity_refine\latest.pth" (
    set "BASE_GRANULARITY_CHECKPOINT=experiments\granularity_refine\latest.pth"
)

if defined GRANULARITY_RECOVERY_CHECKPOINT (
    set "RECOVERY_CHECKPOINT=%GRANULARITY_RECOVERY_CHECKPOINT%"
) else if exist "%PROJECT_DIR%\experiments\granularity_recovery\best.pth" (
    set "RECOVERY_CHECKPOINT=experiments\granularity_recovery\best.pth"
) else if exist "%PROJECT_DIR%\experiments\granularity_recovery\latest.pth" (
    set "RECOVERY_CHECKPOINT=experiments\granularity_recovery\latest.pth"
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
    echo Base granularity checkpoint not found under:
    echo %PROJECT_DIR%\experiments\granularity_refine
    pause
    exit /b 1
)

if "%RECOVERY_CHECKPOINT%"=="" (
    echo Granularity recovery checkpoint not found under:
    echo %PROJECT_DIR%\experiments\granularity_recovery
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%\%RECOVERY_CHECKPOINT%" (
    echo Granularity recovery checkpoint not found under:
    echo %PROJECT_DIR%\%RECOVERY_CHECKPOINT%
    pause
    exit /b 1
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%PATH%"
set "PYTHONUTF8=1"
set "PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%"

if /i "%~1"=="--check" (
    echo Granularity Recovery GUI launcher is ready.
    echo Using base checkpoint: %BASE_GRANULARITY_CHECKPOINT%
    echo Using recovery checkpoint: %RECOVERY_CHECKPOINT%
    exit /b 0
)

pushd "%PROJECT_DIR%"
"%PYTHON%" -m granularity_recovery.demo_granularity_recovery ^
  --checkpoint="%CHECKPOINT%" ^
  --base-granularity-checkpoint="%BASE_GRANULARITY_CHECKPOINT%" ^
  --recovery-checkpoint="%RECOVERY_CHECKPOINT%" ^
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
