#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/docker-build"

log() { printf '\n[build-log] %s\n' "$*"; }
fail() { printf '\n[build-log:error] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  ./scripts/build-docker-log.sh <target> [--no-cache]

Targets:
  api          Build Dockerfile.api
  web-prod     Build docker/web-prod.dockerfile
  web-dev      Build docker/web-dev.dockerfile
  compose-web  Build the web service from docker-compose.yml
  compose-api  Build the python-api service from docker-compose.yml
  compose-all  Build every compose service

Examples:
  ./scripts/build-docker-log.sh api
  ./scripts/build-docker-log.sh web-prod --no-cache
  ./scripts/build-docker-log.sh compose-web
EOF
}

target="${1:-}"
if [[ -z "${target}" ]]; then
  usage
  exit 1
fi
shift

no_cache=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cache)
      no_cache=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
  shift
done

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

timestamp="$(date +%Y%m%d-%H%M%S)"
safe_target="${target//[^a-zA-Z0-9._-]/-}"
log_file="${LOG_DIR}/${timestamp}-${safe_target}.log"

cmd=()

case "${target}" in
  api)
    cmd=(docker build --progress=plain)
    [[ "${no_cache}" == true ]] && cmd+=(--no-cache)
    cmd+=(-f Dockerfile.api -t agentic-api:debug .)
    ;;
  web-prod)
    cmd=(docker build --progress=plain)
    [[ "${no_cache}" == true ]] && cmd+=(--no-cache)
    cmd+=(-f docker/web-prod.dockerfile -t agentic-web-prod:debug .)
    ;;
  web-dev)
    cmd=(docker build --progress=plain)
    [[ "${no_cache}" == true ]] && cmd+=(--no-cache)
    cmd+=(-f docker/web-dev.dockerfile -t agentic-web-dev:debug .)
    ;;
  compose-web)
    cmd=(docker compose build --progress=plain)
    [[ "${no_cache}" == true ]] && cmd+=(--no-cache)
    cmd+=(web)
    ;;
  compose-api)
    cmd=(docker compose build --progress=plain)
    [[ "${no_cache}" == true ]] && cmd+=(--no-cache)
    cmd+=(python-api)
    ;;
  compose-all)
    cmd=(docker compose build --progress=plain)
    [[ "${no_cache}" == true ]] && cmd+=(--no-cache)
    ;;
  *)
    usage
    fail "Unknown target: ${target}"
    ;;
esac

log "Repository: ${REPO_ROOT}"
log "Target: ${target}"
log "Log file: ${log_file}"
log "Command: ${cmd[*]}"

set +e
"${cmd[@]}" 2>&1 | tee "${log_file}"
build_status=${PIPESTATUS[0]}
set -e

if [[ ${build_status} -ne 0 ]]; then
  fail "Build failed with exit code ${build_status}. Full log: ${log_file}"
fi

log "Build finished successfully. Full log: ${log_file}"
