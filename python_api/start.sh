#!/usr/bin/env bash
# Launcher for the Fly.io container: runs `ollama serve` in the background
# on a single GPU, then uvicorn as PID 1's main foreground process. tini
# (entrypoint) reaps zombies and forwards SIGTERM cleanly.

set -euo pipefail

: "${PORT:=8080}"
: "${OLLAMA_HOST:=127.0.0.1:11434}"
export OLLAMA_HOST

mkdir -p "${AGENTIC_DATA_DIR:-/data/python_api}" \
         "${AGENTIC_UPLOADS_DIR:-/data/uploads_python}" \
         "${OLLAMA_MODELS:-/data/ollama}"

echo "[start] launching ollama serve on ${OLLAMA_HOST}"
ollama serve >/tmp/ollama.log 2>&1 &
ollama_pid=$!

shutdown() {
  echo "[start] shutdown requested, stopping ollama (pid=${ollama_pid})"
  kill -TERM "${ollama_pid}" 2>/dev/null || true
  wait "${ollama_pid}" 2>/dev/null || true
}
trap shutdown SIGTERM SIGINT

for attempt in $(seq 1 30); do
  if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "[start] ollama is healthy"
    break
  fi
  if ! kill -0 "${ollama_pid}" 2>/dev/null; then
    echo "[start] ollama exited during startup; dumping log"
    cat /tmp/ollama.log
    exit 1
  fi
  sleep 1
done

echo "[start] launching uvicorn on :${PORT}"
exec python -m uvicorn python_api.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --proxy-headers
