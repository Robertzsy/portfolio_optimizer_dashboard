@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Could not create venv with py launcher. Trying python...
        python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Failed to create virtual environment. Please install Python 3.11 or newer.
    pause
    exit /b 1
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

echo Environment is ready.
pause

