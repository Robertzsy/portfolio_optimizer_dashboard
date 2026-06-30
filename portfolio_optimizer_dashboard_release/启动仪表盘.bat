@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Running setup...
    call "%~dp0setup_env.bat"
    if errorlevel 1 (
        echo Setup failed.
        pause
        exit /b 1
    )
)

echo Starting Portfolio Optimizer dashboard...
echo Local URL: http://localhost:8501
start "Portfolio Optimizer Dashboard" cmd /k ".venv\Scripts\python.exe -m streamlit run app.py --server.port 8501"
timeout /t 3 /nobreak >nul
start "" "http://localhost:8501"

