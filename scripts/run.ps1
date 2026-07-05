# SLR-RAG: start the server (Windows / PowerShell)
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  Write-Host "Virtual environment not found. Run setup first:  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1"
  exit 1
}
Write-Host "Starting server: http://127.0.0.1:8000   (Ctrl+C to stop)"
.\.venv\Scripts\python.exe app.py
