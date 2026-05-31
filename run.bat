@echo off
title Scriba Ro Launcher

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [EROARE] Python nu este instalat sau adaugat in PATH!
    echo Te rugam sa instalezi Python 3.12+ de pe site-ul oficial.
    echo Asigura-te ca bifezi "Add Python to PATH" in timpul instalarii.
    pause
    exit /b
)

:: 2. Lansăm interfața grafică de încărcare (launcher.py)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
python launcher.py
exit
