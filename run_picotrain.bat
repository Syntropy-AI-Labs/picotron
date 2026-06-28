@echo off
:: Batch utility to automate dependencies, preprocessing, and training in Picotron.

if "%~1"=="" (
    echo Usage: run_picotrain.bat ^<path_to_config.yaml^>
    exit /b 1
)

set CONFIG_PATH=%~1

echo [Picotron Boot] Verifying python installation and registering console scripts...
py -m pip install -q -e .
if %ERRORLEVEL% neq 0 (
    echo [Picotron Error] Failed to install package in editable mode.
    exit /b %ERRORLEVEL%
)

echo.
echo [Picotron Preprocess] Running dataset tokenization pipeline...
py -m picotron.data.preprocess "%CONFIG_PATH%"
if %ERRORLEVEL% neq 0 (
    echo [Picotron Error] Preprocessing failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [Picotron Train] Launching model training...
:: Automatically parse dp_size from the config using a small Python helper
py -c "import yaml; cfg=yaml.safe_load(open('%CONFIG_PATH%')); print(cfg.get('parallel', {}).get('dp_size', 1))" > temp_dp.txt
set /p DP_SIZE=<temp_dp.txt
del temp_dp.txt

if "%DP_SIZE%"=="1" (
    echo [Picotron Train] Starting single-GPU/CPU training run...
    py train.py "%CONFIG_PATH%"
) else (
    echo [Picotron Train] Starting multi-GPU distributed DDP training (GPUS: %DP_SIZE%)...
    torchrun --nproc_per_node=%DP_SIZE% train.py "%CONFIG_PATH%"
)
