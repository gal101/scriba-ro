@echo off
title Scriba Ro Launcher
color 0b
cls

echo ==========================================
echo       Scriba Ro - Asistent Dictare
echo ==========================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [EROARE] Python nu este instalat sau adaugat in PATH!
    echo Te rugam sa instalezi Python 3.12+ de pe site-ul oficial.
    echo Asigura-te ca bifezi "Add Python to PATH" in timpul instalarii.
    echo.
    pause
    exit /b
)

:: 2. Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo [1/3] Se creeaza mediul virtual Python - .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [EROARE] Nu s-a putut crea mediul virtual!
        pause
        exit /b
    )
    echo Mediul virtual a fost creat cu succes.
    echo.
)

:: 3. Activate virtual environment
echo [2/3] Se activeaza mediul virtual...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [EROARE] Nu s-a putut activa mediul virtual!
    pause
    exit /b
)
echo Activare reusita.
echo.

:: 4. Install requirements if not already done
echo [3/3] Se verifica si se instaleaza dependintele...
python -m pip install --upgrade pip

:: Dynamically detect NVIDIA GPU hardware using modern PowerShell (since wmic is deprecated/removed on Windows 11)
powershell -Command "if ((Get-CimInstance Win32_VideoController).Name -match 'NVIDIA') { exit 0 } else { exit 1 }" >nul 2>&1
set HAS_NVIDIA=%errorlevel%

:: Check if PyTorch is already installed and has CUDA support
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
set TORCH_CUDA_OK=%errorlevel%

if %HAS_NVIDIA% equ 0 (
    if %TORCH_CUDA_OK% neq 0 (
        echo [DETECTIE] S-a detectat placa video NVIDIA, dar PyTorch nu are suport CUDA activat!
        echo Se instaleaza versiunea oficiala PyTorch CUDA - GPU - pentru viteza maxima...
        pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
    ) else (
        echo [DETECTIE] PyTorch cu suport GPU - CUDA - este deja instalat corect.
    )
) else (
    python -c "import torch" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [DETECTIE] Nu s-a detectat placa video NVIDIA. Se instaleaza PyTorch standard - CPU -...
        pip install torch
    ) else (
        echo [DETECTIE] PyTorch - CPU - este deja instalat corect.
    )
)

:: Install remaining requirements from requirements.txt
echo Se verifica restul dependintelor (sounddevice, faster-whisper, etc.)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [EROARE] Erori intampinate in timpul instalarii pachetelor pip!
    pause
    exit /b
)
echo Dependinte instalate sau verificate cu succes.
echo.

:: 5. Launch Setup / Main App
echo Lansare Scriba Ro...
start "" python setup.py
echo.
echo ==========================================
echo   Scriba Ro ruleaza ca aplicatie interactiva!
echo ==========================================
exit
