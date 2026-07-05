@echo off
REM SLR-RAG: start the server (Windows / Command Prompt)
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run setup first:  scripts\setup.bat
  exit /b 1
)
echo Starting server: http://127.0.0.1:8000   (Ctrl+C to stop)
.venv\Scripts\python.exe app.py
