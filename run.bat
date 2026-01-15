@echo off
setlocal enabledelayedexpansion

:: Find CUDA 12.x installation
set CUDA_PATH=
for %%v in (v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0) do (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%v\bin" (
        set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%v
        goto :found_cuda
    )
)
:found_cuda

:: Set CUDA environment if found
if defined CUDA_PATH (
    set PATH=!CUDA_PATH!\bin;!PATH!
)

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

:: Launch the GUI
start "" "venv\Scripts\pythonw.exe" run_gui.py
