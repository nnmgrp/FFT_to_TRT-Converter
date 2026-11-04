@echo off
setlocal enabledelayedexpansion
title Universal FFT-to-TensorRT 1-Click Converter

echo ===============================================================================
echo                FFT to TensorRT 1-Click Engine Converter
echo ===============================================================================
echo.

:: 1. Check input model argument
set "INPUT_MODEL=%~1"
if "%INPUT_MODEL%"=="" (
    echo [PROMPT] Please drag and drop your model file (.safetensors, .pth, .pt) onto this bat,
    echo          or enter the full path to the model below:
    echo.
    set /p "INPUT_MODEL=Model path: "
)

:: Strip surrounding quotes if present
set "INPUT_MODEL=%INPUT_MODEL:"=%"

if not exist "%INPUT_MODEL%" (
    echo.
    echo [ERROR] Specified model file was not found:
    echo         "%INPUT_MODEL%"
    echo.
    pause
    exit /b 1
)

:: 2. Find Python environment with PyTorch & TensorRT
set "PYTHON_EXE="

:: Check local virtual environments first
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0..\venv_trt\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\venv_trt\Scripts\python.exe"
if not defined PYTHON_EXE if exist "G:\TRT_converter\venv_trt\Scripts\python.exe" set "PYTHON_EXE=G:\TRT_converter\venv_trt\Scripts\python.exe"

:: Check chaiNNer embedded python
if not defined PYTHON_EXE if exist "%~dp0..\chaiNNer-windows-latest-nightly.2026-02-14\python\python\python.exe" (
    set "PYTHON_EXE=%~dp0..\chaiNNer-windows-latest-nightly.2026-02-14\python\python\python.exe"
)
if not defined PYTHON_EXE if exist "F:\Downloads\chaiNNer-windows-latest-nightly.2026-02-14\python\python\python.exe" (
    set "PYTHON_EXE=F:\Downloads\chaiNNer-windows-latest-nightly.2026-02-14\python\python\python.exe"
)

:: Fallback to system python
if not defined PYTHON_EXE (
    where python.exe >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PYTHON_EXE=python.exe"
    )
)

if not defined PYTHON_EXE (
    echo.
    echo [ERROR] Could not find a suitable Python environment.
    echo         Please install requirements from requirements.txt or create a venv.
    echo.
    pause
    exit /b 1
)

echo [ENV] Using Python: %PYTHON_EXE%
echo [INPUT] Target Model: %INPUT_MODEL%
echo.

:: 3. Prepare paths
for %%F in ("%INPUT_MODEL%") do (
    set "MODEL_DIR=%%~dpF"
    set "MODEL_NAME=%%~nF"
)
set "TEMP_ONNX=%TEMP%\%MODEL_NAME%_temp_%RANDOM%.onnx"
set "FINAL_ENGINE=%MODEL_DIR%%MODEL_NAME%.engine"

echo [STEP 1/2] Converting PyTorch checkpoint to optimized ONNX...
echo.

"%PYTHON_EXE%" "%~dp0converter\torch2onnx.py" --model "%INPUT_MODEL%" --output "%TEMP_ONNX%" --height 256 --width 256
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] ONNX export failed.
    if exist "%TEMP_ONNX%" del /f /q "%TEMP_ONNX%" >nul 2>&1
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===============================================================================
echo [STEP 2/2] Compiling TensorRT Engine (this takes 1-3 minutes for kernel autotuning)...
echo ===============================================================================
echo.

"%PYTHON_EXE%" "%~dp0converter\build_engine.py" --onnx "%TEMP_ONNX%" --engine "%FINAL_ENGINE%" --workspace 2048
set "BUILD_RESULT=%ERRORLEVEL%"

:: 4. Clean up temporary ONNX file
if exist "%TEMP_ONNX%" (
    del /f /q "%TEMP_ONNX%" >nul 2>&1
    echo [CLEANUP] Temporary ONNX file removed.
)

if %BUILD_RESULT% neq 0 (
    echo.
    echo [ERROR] TensorRT Engine compilation failed.
    pause
    exit /b %BUILD_RESULT%
)

echo.
echo ===============================================================================
echo                          CONVERSION SUCCESSFUL!
echo ===============================================================================
echo.
echo Output Engine: "%FINAL_ENGINE%"
echo You can now load this engine directly in chaiNNer without any manual steps.
echo.
pause
