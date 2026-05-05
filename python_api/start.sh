#!/usr/bin/env bash
# Launcher for the GPU/local-training container. Runs `ollama serve` in the
# background, waits for it to become healthy, then starts uvicorn in the
# foreground. `OLLAMA_HOST` is derived from `OLLAMA_BASE_URL` when only the
# latter is configured so the launcher and the Python app stay in sync.

set -euo pipefail

: "${PORT:=8080}"
if [[ -z "${OLLAMA_HOST:-}" ]]; then
  base_url="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
  base_url="${base_url#http://}"
  base_url="${base_url#https://}"
  base_url="${base_url%%/*}"
  export OLLAMA_HOST="${base_url:-127.0.0.1:11434}"
else
  export OLLAMA_HOST
fi

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

ollama_ready=0
for attempt in $(seq 1 30); do
  if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "[start] ollama is healthy"
    ollama_ready=1
    break
  fi
  if ! kill -0 "${ollama_pid}" 2>/dev/null; then
    echo "[start] ollama exited during startup; dumping log"
    cat /tmp/ollama.log
    exit 1
  fi
  sleep 1
done

if [[ "${ollama_ready}" != "1" ]]; then
  echo "[start] ollama did not become healthy at ${OLLAMA_HOST}; dumping log"
  cat /tmp/ollama.log
  exit 1
fi

echo "[start] launching uvicorn on :${PORT}"
exec python -m uvicorn python_api.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --proxy-headers
