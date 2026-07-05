#!/usr/bin/env bash
# SLR-RAG setup (macOS / Linux)
# Usage:  bash scripts/setup.sh   (or:  chmod +x scripts/setup.sh && ./scripts/setup.sh)
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

echo "[1/4] Python version: $("$PY" --version 2>&1)  (3.11+ required)"
echo "[2/4] Creating virtual environment (.venv)..."
"$PY" -m venv .venv

echo "[3/4] Installing dependencies (includes torch; first install may take a few minutes)..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -r requirements-dev.txt   # pytest + test-only deps

echo "[4/4] Configuration..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    .env created. Add an OPENROUTER_API_KEY for the LLM stages (optional for indexing)."
fi

echo ""
echo "Setup complete."
echo "Run the app:  bash scripts/run.sh    ->  http://127.0.0.1:8000"
