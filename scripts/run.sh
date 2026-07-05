#!/usr/bin/env bash
# SLR-RAG: start the server (macOS / Linux)
cd "$(dirname "$0")/.."
if [ ! -x ./.venv/bin/python ]; then
  echo "Virtual environment not found. Run setup first:  bash scripts/setup.sh"
  exit 1
fi
echo "Starting server: http://127.0.0.1:8000   (Ctrl+C to stop)"
exec ./.venv/bin/python app.py
