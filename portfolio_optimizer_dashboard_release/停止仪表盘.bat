@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=(Resolve-Path '.').Path; Get-Process python,streamlit -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase) } | Stop-Process"

echo Dashboard processes under this project have been stopped.
pause

