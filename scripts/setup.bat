@echo off
REM SLR-RAG setup (Windows / Command Prompt)
REM Usage:  scripts\setup.bat
cd /d "%~dp0.."

echo [1/4] Python version:
python --version

echo [2/4] Creating virtual environment (.venv)...
python -m venv .venv

echo [3/4] Installing dependencies (includes torch; first install may take a few minutes)...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo [4/4] Configuration...
if not exist .env ( copy .env.example .env >nul & echo     .env created. Add an OPENROUTER_API_KEY for the LLM stages. )

echo.
echo Setup complete.
echo Run the app:  scripts\run.bat    ->  http://127.0.0.1:8000
