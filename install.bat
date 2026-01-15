@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Audio Visualizer - Installation Script
echo ============================================
echo.

:: Find a compatible Python version (3.10-3.12)
:: Try: py -3.12, py -3.11, py -3.10, python3.12, python3.11, python3.10, python
set PYTHON_CMD=
set PYVER=

:: Try Python Launcher first (most reliable on Windows)
for %%v in (3.12 3.11 3.10) do (
    if not defined PYTHON_CMD (
        py -%%v --version >nul 2>&1
        if not errorlevel 1 (
            set PYTHON_CMD=py -%%v
            for /f "tokens=2" %%i in ('py -%%v --version 2^>^&1') do set PYVER=%%i
        )
    )
)

:: Try pythonX.XX commands
if not defined PYTHON_CMD (
    for %%v in (python3.12 python3.11 python3.10) do (
        if not defined PYTHON_CMD (
            %%v --version >nul 2>&1
            if not errorlevel 1 (
                set PYTHON_CMD=%%v
                for /f "tokens=2" %%i in ('%%v --version 2^>^&1') do set PYVER=%%i
            )
        )
    )
)

:: Try default python command
if not defined PYTHON_CMD (
    python --version >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
        :: Extract major.minor version
        for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
            set PYMAJOR=%%a
            set PYMINOR=%%b
        )
        :: Check if it's compatible
        if !PYMAJOR!==3 if !PYMINOR! GEQ 10 if !PYMINOR! LEQ 12 (
            set PYTHON_CMD=python
        )
    )
)

:: No compatible Python found
if not defined PYTHON_CMD (
    echo ERROR: No compatible Python found.
    echo.
    echo CuPy ^(GPU acceleration^) requires Python 3.10, 3.11, or 3.12.
    echo Python 3.13+ is not yet supported by CuPy.
    echo.
    if defined PYVER (
        echo Found Python !PYVER! but it's not compatible.
    ) else (
        echo No Python installation detected.
    )
    echo.
    echo Would you like to open the Python 3.12 download page?
    set /p OPENPY="Enter Y to open, or N to exit [Y/N]: "
    if /i "!OPENPY!"=="Y" (
        echo Opening Python 3.12 download page...
        start https://www.python.org/downloads/release/python-3129/
    )
    echo.
    echo Please install Python 3.12 and make sure to check
    echo "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Found compatible Python: %PYVER% ^(using: %PYTHON_CMD%^)

:: Check for FFmpeg
echo.
echo Checking for FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo WARNING: FFmpeg is not installed or not in PATH.
    echo FFmpeg is required for video encoding.
    echo.
    echo Would you like to open the FFmpeg download page?
    set /p OPENFFMPEG="Enter Y to open, or N to continue [Y/N]: "
    if /i "!OPENFFMPEG!"=="Y" (
        echo Opening FFmpeg download page...
        start https://www.gyan.dev/ffmpeg/builds/
        echo.
        echo Download "ffmpeg-release-essentials.zip", extract it,
        echo and add the "bin" folder to your system PATH.
        echo.
        echo Alternatively, install via winget:
        echo   winget install ffmpeg
        echo.
        pause
    )
) else (
    for /f "tokens=3" %%i in ('ffmpeg -version 2^>^&1 ^| findstr /i "version"') do (
        echo Found FFmpeg %%i
        goto :ffmpeg_done
    )
    echo Found FFmpeg.
)
:ffmpeg_done

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

:: Activate venv and install requirements
echo.
echo Installing base requirements...
call venv\Scripts\activate.bat

:: Ensure pip uses the venv's pip explicitly
set PIP=venv\Scripts\pip.exe

:: Upgrade pip first
venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1

:: Install base requirements
%PIP% install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install base requirements.
    pause
    exit /b 1
)

echo.
echo Base requirements installed successfully.

:: Check for NVIDIA GPU
echo.
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected. Skipping CUDA installation.
    echo The visualizer will run in Standard Mode ^(CPU rendering^).
    goto :finish
)

echo NVIDIA GPU detected!

:: Check for CUDA Toolkit
echo.
echo Checking for CUDA Toolkit...
set CUDA_FOUND=0
set CUDA_PATH_FOUND=

:: Check common CUDA installation paths
for %%v in (v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0) do (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%v\bin\nvcc.exe" (
        set CUDA_FOUND=1
        set CUDA_PATH_FOUND=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%v
        echo Found CUDA Toolkit at: !CUDA_PATH_FOUND!
        goto :cuda_found
    )
)

:cuda_found
if %CUDA_FOUND%==0 (
    echo.
    echo CUDA Toolkit 12.x not found.
    echo.
    echo Would you like to open the CUDA Toolkit download page?
    set /p OPENCUDA="Enter Y to open, or N to skip [Y/N]: "
    if /i "!OPENCUDA!"=="Y" (
        echo Opening CUDA Toolkit download page...
        start https://developer.nvidia.com/cuda-12-6-0-download-archive
        echo.
        echo After installing CUDA, run this script again to install CuPy.
    )
    echo.
    echo The visualizer will run in Standard Mode for now.
    goto :finish
)

:: Install CuPy for GPU acceleration
echo.
echo Installing CuPy for GPU acceleration...
echo This may take a few minutes...
%PIP% install cupy-cuda12x
if errorlevel 1 (
    echo.
    echo WARNING: CuPy installation failed.
    echo The visualizer will run in Standard Mode ^(CPU rendering^).
    echo.
    echo You can try installing CuPy manually later:
    echo   %PIP% install cupy-cuda12x
    goto :finish
)

echo.
echo CuPy installed successfully! GPU acceleration is enabled.

:finish
echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo Requirements summary:
echo   [Required]  Python 3.10-3.12 - OK
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo   [Required]  FFmpeg          - NOT FOUND ^(install from https://www.gyan.dev/ffmpeg/builds/^)
) else (
    echo   [Required]  FFmpeg          - OK
)
if %CUDA_FOUND%==1 (
    echo   [Optional]  CUDA Toolkit    - OK ^(GPU acceleration enabled^)
    echo   [Optional]  CuPy            - OK
) else (
    echo   [Optional]  CUDA Toolkit    - Not installed ^(Standard Mode^)
    echo   [Optional]  CuPy            - Not installed ^(Standard Mode^)
)
echo.
echo To run the visualizer:
echo   1. Double-click "run.bat"
echo   OR
echo   2. Run: venv\Scripts\python.exe run_gui.py
echo.
pause
