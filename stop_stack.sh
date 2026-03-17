#!/usr/bin/env bash

set -euo pipefail

ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME="$(eval echo "~$ACTUAL_USER")"
MONITOR_DIR="$ACTUAL_HOME/Documents/Krish-tesh-scripts/vllm-benchmarking"


echo "Stopping Prometheus + Grafana..."
cd "$MONITOR_DIR"

if docker compose version >/dev/null 2>&1; then
  docker compose down
elif command -v docker-compose >/dev/null 2>&1; then
  docker-compose down
else
  echo "No docker compose command found."
fi

echo "Stopping Streamlit chat app if running..."
pkill -f "streamlit run .*llmforge_nemotron.py" || true
pkill -f "python3 .*llmforge_nemotron.py" || true

echo "Stopping vLLM recipe if running..."
pkill -f "run-recipe.sh" || true
pkill -f "vllm" || true

echo "Done."
