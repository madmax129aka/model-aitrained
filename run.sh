#!/usr/bin/env bash
#
# PixelTruth -- single run command.
#
# 1. Installs backend (Python) + frontend (Node) dependencies.
# 2. Builds the React frontend (Vite production build -> frontend/dist).
# 3. Starts the FastAPI backend, which serves BOTH:
#      - the JSON API under /api/*
#      - the built frontend static files (frontend/dist) at /
#    all from a single process bound to 0.0.0.0 (required for Replit).
#
# Usage:
#   ./run.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
PORT="${PORT:-8000}"

echo "=================================================================="
echo " PixelTruth -- AI Image Detector"
echo " Custom-trained CNN model (no external AI API)"
echo "=================================================================="

echo ""
echo "[1/4] Installing backend (Python) dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "$BACKEND_DIR/requirements.txt"

echo ""
echo "[2/4] Installing frontend (Node) dependencies..."
cd "$FRONTEND_DIR"
npm install --silent

echo ""
echo "[3/4] Building frontend for production..."
npm run build

echo ""
echo "[4/4] Starting PixelTruth backend on 0.0.0.0:${PORT}"
echo "      (serves the API under /api/* and the built frontend at /)"
cd "$BACKEND_DIR"
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
