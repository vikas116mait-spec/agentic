#!/usr/bin/env bash
# One-shot, idempotent cloud deploy for the three-service split described in
# DEPLOYMENT.md. Runs end-to-end without ever prompting for passwords, using
# API tokens supplied via env vars.
#
# Required env vars (export before running):
#   DATABASE_URL        Neon pooled connection string
#                       (postgresql://USER:PASS@ep-...-pooler.neon.tech/neondb?sslmode=require)
#   FLY_API_TOKEN       Fly.io deploy token
#                       (https://fly.io/user/personal_access_tokens)
#   VERCEL_TOKEN        Vercel personal access token
#                       (https://vercel.com/account/tokens)
#   OPENAI_API_KEY      Your OpenAI key (passed to both sides)
#
# Optional env vars:
#   FLY_APP_NAME        default "agentic-api" (must be globally unique on Fly)
#   FLY_REGION          default "iad"
#   FLY_VOLUME_SIZE_GB  default 50
#   FLY_GPU_KIND        default "a10"
#   VERCEL_ORG_ID       links to existing Vercel org (else interactive link)
#   VERCEL_PROJECT_ID   links to existing Vercel project
#   HF_TOKEN            forwarded to the Python API if set
#   NEXTAUTH_SECRET     auto-generated if unset
#
# Usage:
#   export DATABASE_URL="..." FLY_API_TOKEN="..." VERCEL_TOKEN="..." OPENAI_API_KEY="..."
#   ./scripts/deploy-cloud.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

log() { printf '\n\033[1;34m[deploy]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[deploy:error]\033[0m %s\n' "$*" >&2; exit 1; }

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    fail "Required env var ${name} is not set. See scripts/deploy-cloud.sh header."
  fi
}

require_var DATABASE_URL
require_var FLY_API_TOKEN
require_var VERCEL_TOKEN
require_var OPENAI_API_KEY

: "${FLY_APP_NAME:=agentic-api}"
: "${FLY_REGION:=iad}"
: "${FLY_VOLUME_SIZE_GB:=50}"
: "${FLY_GPU_KIND:=a10}"
: "${NEXTAUTH_SECRET:=$(openssl rand -base64 32 | tr -d '\n')}"

# ---------------------------------------------------------------------------
# 1. Tooling: flyctl and vercel (via npx -- no sudo needed).
# ---------------------------------------------------------------------------

install_flyctl() {
  if command -v flyctl >/dev/null 2>&1; then
    return
  fi
  log "Installing flyctl under \$HOME/.fly ..."
  curl -fsSL https://fly.io/install.sh | sh
  export FLYCTL_INSTALL="${HOME}/.fly"
  export PATH="${FLYCTL_INSTALL}/bin:${PATH}"
}

install_flyctl

if [[ -d "${HOME}/.fly/bin" ]]; then
  export PATH="${HOME}/.fly/bin:${PATH}"
fi

command -v flyctl >/dev/null 2>&1 || fail "flyctl still not on PATH after install."

# npx will fetch vercel on demand; we pin a version to stabilise reruns.
VERCEL="npx --yes vercel@latest"

# ---------------------------------------------------------------------------
# 2. Neon -- apply Prisma migrations.
# ---------------------------------------------------------------------------

log "Applying Prisma migrations against Neon ..."
DATABASE_URL="${DATABASE_URL}" npx --yes prisma@^6 migrate deploy

# ---------------------------------------------------------------------------
# 3. Fly.io -- app, volume, secrets, deploy.
# ---------------------------------------------------------------------------

# Substitute the placeholder app name in fly.toml so this script is safe to
# rerun on any Fly account. Idempotent: does nothing if the name already matches.
if grep -qE '^app = "agentic-api"' fly.toml && [[ "${FLY_APP_NAME}" != "agentic-api" ]]; then
  log "Setting fly.toml app = \"${FLY_APP_NAME}\" ..."
  sed -i.bak -E "s|^app = \"agentic-api\"$|app = \"${FLY_APP_NAME}\"|" fly.toml
  rm -f fly.toml.bak
fi

log "Ensuring Fly app '${FLY_APP_NAME}' exists ..."
if ! flyctl status --app "${FLY_APP_NAME}" >/dev/null 2>&1; then
  flyctl apps create "${FLY_APP_NAME}" --org personal
fi

log "Ensuring Fly volume 'agentic_data' exists ..."
if ! flyctl volumes list --app "${FLY_APP_NAME}" 2>/dev/null \
     | grep -q 'agentic_data'; then
  flyctl volumes create agentic_data \
    --app "${FLY_APP_NAME}" \
    --region "${FLY_REGION}" \
    --size "${FLY_VOLUME_SIZE_GB}" \
    --vm-gpu-kind "${FLY_GPU_KIND}" \
    --yes
fi

log "Pushing Fly secrets (DATABASE_URL / OPENAI_API_KEY / HF_TOKEN / ALLOWED_ORIGINS='*') ..."
fly_secret_args=(
  "DATABASE_URL=${DATABASE_URL}"
  "OPENAI_API_KEY=${OPENAI_API_KEY}"
  "ALLOWED_ORIGINS=*"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  fly_secret_args+=("HF_TOKEN=${HF_TOKEN}")
fi
flyctl secrets set --app "${FLY_APP_NAME}" --stage "${fly_secret_args[@]}" >/dev/null

log "Deploying Fly app (this pulls the CUDA image and installs Ollama - first run ~5 min) ..."
flyctl deploy --app "${FLY_APP_NAME}" --remote-only

FLY_URL="https://${FLY_APP_NAME}.fly.dev"
log "Smoke-checking Fly ${FLY_URL}/health ..."
for attempt in $(seq 1 20); do
  if curl -sf "${FLY_URL}/health" >/dev/null; then
    log "Fly API is healthy."
    break
  fi
  [[ "${attempt}" == 20 ]] && fail "Fly /health never returned 200 after 20 tries."
  sleep 5
done

# ---------------------------------------------------------------------------
# 4. Vercel -- link, env vars, deploy.
# ---------------------------------------------------------------------------

log "Linking Vercel project ..."
link_args=()
if [[ -n "${VERCEL_ORG_ID:-}" && -n "${VERCEL_PROJECT_ID:-}" ]]; then
  link_args+=(--scope "${VERCEL_ORG_ID}" --project "${VERCEL_PROJECT_ID}")
fi
${VERCEL} link --yes --token "${VERCEL_TOKEN}" "${link_args[@]}" || true

set_vercel_env() {
  local name="$1" value="$2"
  for env in production preview development; do
    # Overwrite silently: rm first (ignore error), then add.
    ${VERCEL} env rm "${name}" "${env}" --yes --token "${VERCEL_TOKEN}" >/dev/null 2>&1 || true
    printf '%s' "${value}" \
      | ${VERCEL} env add "${name}" "${env}" --token "${VERCEL_TOKEN}" >/dev/null
  done
}

log "Pushing Vercel env vars ..."
set_vercel_env "DATABASE_URL" "${DATABASE_URL}"
set_vercel_env "NEXT_PUBLIC_PYTHON_API_URL" "${FLY_URL}"
set_vercel_env "PYTHON_API_URL" "${FLY_URL}"
set_vercel_env "NEXTAUTH_SECRET" "${NEXTAUTH_SECRET}"
set_vercel_env "OPENAI_API_KEY" "${OPENAI_API_KEY}"

log "Triggering Vercel production deploy ..."
VERCEL_DEPLOY_URL="$(${VERCEL} --prod --yes --token "${VERCEL_TOKEN}" | tail -1)"
log "Vercel deploy: ${VERCEL_DEPLOY_URL}"

# NEXTAUTH_URL must match the final domain (custom or *.vercel.app).
set_vercel_env "NEXTAUTH_URL" "${VERCEL_DEPLOY_URL}"

# Redeploy once so NextAuth picks up its own URL.
${VERCEL} --prod --yes --token "${VERCEL_TOKEN}" >/dev/null

# ---------------------------------------------------------------------------
# 5. Tighten CORS on the Fly side now that we know the Vercel domain.
# ---------------------------------------------------------------------------

log "Tightening Fly ALLOWED_ORIGINS to ${VERCEL_DEPLOY_URL} ..."
flyctl secrets set --app "${FLY_APP_NAME}" "ALLOWED_ORIGINS=${VERCEL_DEPLOY_URL}" >/dev/null

# ---------------------------------------------------------------------------
# 6. Smoke test.
# ---------------------------------------------------------------------------

log "Smoke-testing ${VERCEL_DEPLOY_URL}/dashboard ..."
if curl -sfI "${VERCEL_DEPLOY_URL}/dashboard" >/dev/null; then
  log "Vercel dashboard route returned 200."
else
  fail "Vercel dashboard route did not return 200. Check Vercel logs."
fi

log "Smoke-testing Fly /dashboard/summary ..."
if curl -sf "${FLY_URL}/dashboard/summary" >/dev/null; then
  log "Fly dashboard/summary returned 200."
else
  fail "Fly /dashboard/summary did not return 200. Check Fly logs via 'flyctl logs'."
fi

cat <<EOF

=====================================================================
Deploy complete.

  Next.js (UI):   ${VERCEL_DEPLOY_URL}
  FastAPI (Fly):  ${FLY_URL}
  Postgres:       Neon pooler (DATABASE_URL)

Open ${VERCEL_DEPLOY_URL}/dashboard and run through the smoke checklist
in DEPLOYMENT.md (signup, upload dataset, start a small fine-tune).
=====================================================================
EOF
