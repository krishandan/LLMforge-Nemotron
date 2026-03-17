#!/usr/bin/env bash

set -euo pipefail

# =========================
# Config
# =========================
VLLM_DIR="$HOME/Documents/Projects/eugr-vllm/spark-vllm-docker"
CHAT_APP_DIR="$HOME/Documents/Projects/Chatting_app"
MONITOR_DIR="$HOME/Documents/Krish-tesh-scripts/vllm-benchmarking"

# Default chat app file
CHAT_APP_FILE="${2:-nemotron_chatapp.py}"

# Pass model name as first arg
MODEL_NAME="${1:-}"

# vLLM health endpoint
VLLM_HEALTH_URL="http://127.0.0.1:8000/v1/models"

# Monitoring URLs
PROM_URL="http://127.0.0.1:9091/-/healthy"
GRAFANA_URL="http://127.0.0.1:3001/login"

# Wait settings
MAX_WAIT_SECONDS=300
SLEEP_SECONDS=5

# =========================
# Validate input
# =========================
if [[ -z "$MODEL_NAME" ]]; then
  echo "Usage: $0 <model-name> [chat-app-file]"
  echo "Example: $0 nemotron-3-super-nvfp4"
  echo "Example: $0 nemotron-3-super-nvfp4 nemotron_chatapp.py"
  exit 1
fi

# =========================
# Validate paths
# =========================
if [[ ! -d "$VLLM_DIR" ]]; then
  echo "Error: vLLM directory not found: $VLLM_DIR"
  exit 1
fi

if [[ ! -d "$CHAT_APP_DIR" ]]; then
  echo "Error: Chat app directory not found: $CHAT_APP_DIR"
  exit 1
fi

if [[ ! -f "$CHAT_APP_DIR/$CHAT_APP_FILE" ]]; then
  echo "Error: Chat app file not found: $CHAT_APP_DIR/$CHAT_APP_FILE"
  exit 1
fi

if [[ ! -d "$MONITOR_DIR" ]]; then
  echo "Error: Monitoring directory not found: $MONITOR_DIR"
  exit 1
fi

if [[ ! -f "$MONITOR_DIR/docker-compose.yaml" && ! -f "$MONITOR_DIR/docker-compose.yml" ]]; then
  echo "Error: No docker-compose.yaml or docker-compose.yml found in: $MONITOR_DIR"
  exit 1
fi

# =========================
# Helper functions
# =========================
check_vllm() {
  curl -sf "$VLLM_HEALTH_URL" >/dev/null 2>&1
}

check_prometheus() {
  curl -sf "$PROM_URL" >/dev/null 2>&1
}

check_grafana() {
  curl -s "$GRAFANA_URL" >/dev/null 2>&1
}

wait_for_service() {
  local service_name="$1"
  local check_function="$2"
  local elapsed=0

  echo "Waiting for $service_name..."

  until "$check_function"; do
    if (( elapsed >= MAX_WAIT_SECONDS )); then
      echo "Error: $service_name did not become ready within $MAX_WAIT_SECONDS seconds."
      return 1
    fi

    echo "$service_name not ready yet... waiting ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
    elapsed=$((elapsed + SLEEP_SECONDS))
  done

  echo "$service_name is ready."
  return 0
}

# =========================
# Step 1: Start monitoring stack
# =========================
echo "========================================="
echo "Step 1: Starting Prometheus + Grafana"
echo "========================================="

cd "$MONITOR_DIR"

if command -v docker >/dev/null 2>&1; then
  if docker compose version >/dev/null 2>&1; then
    docker compose up -d
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose up -d
  else
    echo "Error: Neither 'docker compose' nor 'docker-compose' is available."
    exit 1
  fi
else
  echo "Error: Docker is not installed or not in PATH."
  exit 1
fi

# Wait for monitoring stack
wait_for_service "Prometheus" check_prometheus || true
wait_for_service "Grafana" check_grafana || true

# =========================
# Step 2: Start vLLM if needed
# =========================
echo "========================================="
echo "Step 2: Starting vLLM"
echo "========================================="

if check_vllm; then
  echo "vLLM is already running."
else
  echo "Starting vLLM for model: $MODEL_NAME"
  cd "$VLLM_DIR"

  nohup sudo ./run-recipe.sh "$MODEL_NAME" > "$HOME/vllm_${MODEL_NAME}.log" 2>&1 &

  echo "vLLM launch started in background."
  echo "Log file: $HOME/vllm_${MODEL_NAME}.log"
fi

wait_for_service "vLLM" check_vllm

# =========================
# Step 3: Start chat app
# =========================
echo "========================================="
echo "Step 3: Starting chat app"
echo "========================================="

cd "$CHAT_APP_DIR"

if command -v streamlit >/dev/null 2>&1; then
  echo "Launching chat app with Streamlit..."
  exec streamlit run "$CHAT_APP_FILE"
else
  echo "Streamlit not found. Launching with python3..."
  exec python3 "$CHAT_APP_FILE"
fi
