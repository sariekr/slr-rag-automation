# SLR-RAG setup (Windows / PowerShell)
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "[1/4] Python version: $(python --version 2>&1)  (3.11+ required)"
Write-Host "[2/4] Creating virtual environment (.venv)..."
python -m venv .venv

Write-Host "[3/4] Installing dependencies (includes torch; first install may take a few minutes)..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "[4/4] Configuration..."
if (-not (Test-Path .env)) {
  Copy-Item .env.example .env
  Write-Host "    .env created. Add an OPENROUTER_API_KEY for the LLM stages (optional for indexing)."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run the app:  .\scripts\run.ps1    ->  http://127.0.0.1:8000"
