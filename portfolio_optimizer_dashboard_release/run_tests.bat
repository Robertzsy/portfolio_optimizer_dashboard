@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call "%~dp0setup_env.bat"
)

".venv\Scripts\python.exe" -m pytest tests
pause

