@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo Building TensorRT DFT Plugin for Windows
echo ========================================================

if not defined TENSORRT_DIR (
    if exist "G:\TensorRT" (
        set "TENSORRT_DIR=G:\TensorRT"
    ) else (
        echo [ERROR] TENSORRT_DIR is not set. Please set it to your TensorRT directory.
        exit /b 1
    )
)

echo Using TensorRT: %TENSORRT_DIR%

if exist "build" rmdir /s /q build
mkdir build
cd build

cmake -G "Visual Studio 17 2022" -A x64 ..
if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed.
    exit /b %ERRORLEVEL%
)

cmake --build . --config Release
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed.
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo Build SUCCESSFUL!
echo DLL output: build\src\dft_plugins\Release\trt_dft_plugins.dll
echo ========================================================
cd ..
