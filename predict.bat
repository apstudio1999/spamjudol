@echo off
REM Predict single text - Shortcut untuk Windows
REM Usage: predict.bat "your text here"

if "%1"=="" (
    echo Usage: predict.bat "your text"
    echo Example: predict.bat "subscribe my channel"
    pause
    exit /b 1
)

python -m inference_local --text "%1"

pause
