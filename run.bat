@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: IndexTTS For Chaos - Standalone Desktop Application Launcher
:: ============================================================

title IndexTTS For Chaos

set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"
set "LAUNCHER=%APP_DIR%\desktop_launcher.py"

echo.
echo ============================================================
echo   IndexTTS For Chaos - Standalone Desktop Application
echo ============================================================
echo.

:: ============================================================
:: Step 1: Find Python
:: ============================================================
set "PYTHON="

for %%P in (python3.12 python3.11 python3.10 python3 python) do (
    where %%P >nul 2>nul
    if !errorlevel! equ 0 (
        for /f "delims=" %%i in ('%%P -c "import sys; print(sys.executable)"') do set "PYTHON=%%i"
        goto :found_python
    )
)

for %%D in (
    "C:\Python312" "C:\Python311" "C:\Python310"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%LOCALAPPDATA%\Programs\Python\Python310"
) do (
    if exist "%%~D\python.exe" (
        set "PYTHON=%%~D\python.exe"
        goto :found_python
    )
)

echo   [ERROR] Python 3.10+ not found
echo.
echo   Install Python 3.10 or later:
echo   https://www.python.org/downloads/
echo.
pause
exit /b 1

:found_python
for /f "tokens=*" %%i in ('"%PYTHON%" --version') do set PYVER=%%i
echo   [OK] Found Python: %PYTHON%
echo       Version: %PYVER%

"%PYTHON%" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if !errorlevel! neq 0 (
    echo   [ERROR] Python 3.10+ required, found %PYVER%
    pause
    exit /b 1
)

:: ============================================================
:: Step 2: Install dependencies (skip if already present)
:: ============================================================
echo.
echo   [Step 2/3] Checking dependencies

set "DO_INSTALL=0"

"%PYTHON%" -c "import torch, PyQt6" >nul 2>nul
if !errorlevel! neq 0 (
    set "DO_INSTALL=1"
)

if "!DO_INSTALL!"=="1" (
    echo   Installing dependencies, this may take a while

    "%PYTHON%" -c "import torch" >nul 2>nul
    if !errorlevel! neq 0 (
        echo   [1/2] Installing PyTorch with CUDA
        "%PYTHON%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
        if !errorlevel! neq 0 (
            echo   [WARN] CUDA PyTorch failed, trying CPU version
            "%PYTHON%" -m pip install torch torchaudio
        )
    )

    echo   [2/2] Installing project dependencies
    if exist "%APP_DIR%\requirements.txt" (
        "%PYTHON%" -m pip install -r "%APP_DIR%\requirements.txt"
    )
) else (
    echo   [OK] All dependencies already installed
)

:: ============================================================
:: Step 3: Launch Application
:: ============================================================
echo.
echo   [Step 3/3] Launching application
echo.

set "INDEXTTS_MODEL_DIR=%APP_DIR%\checkpoints"
set "HF_HUB_CACHE=%APP_DIR%\checkpoints\hf_cache"
set "QT_AUTO_SCREEN_SCALE_FACTOR=1"

"%PYTHON%" "%LAUNCHER%" %*

if !errorlevel! neq 0 (
    echo.
    echo   [ERROR] Application exited with code !errorlevel!
    pause
)
