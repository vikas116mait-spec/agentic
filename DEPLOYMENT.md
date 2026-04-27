# Deploying to Vercel + Fly.io + Neon

The repo splits cleanly into three services:

| Piece | Host | Why |
| --- | --- | --- |
| Next.js UI + `app/api/*` (Prisma + NextAuth) | **Vercel** | stateless, fits serverless |
| FastAPI + Ollama + local QLoRA + Temporal worker | **Fly.io GPU machine** | needs GPU, persistent FS, long-running process |
| Postgres | **Neon** | reachable from both Vercel and Fly |

Everything the codebase needs for this split is already committed:

- [python_api/Dockerfile](python_api/Dockerfile) - CUDA + Python + Ollama image.
- [python_api/start.sh](python_api/start.sh) - launches `ollama serve` then uvicorn.
- [fly.toml](fly.toml) - Fly manifest (app name and secrets are filled in at deploy time).
- [.dockerignore](.dockerignore) - keeps `uploads_python/`, `.next/`, `node_modules/`, secrets out of the image.
- [python_api/main.py](python_api/main.py) reads `ALLOWED_ORIGINS` for CORS.
- [python_api/store.py](python_api/store.py) reads `AGENTIC_DATA_DIR` and `AGENTIC_UPLOADS_DIR` so state + uploads land on the Fly volume at `/data`.
- [package.json](package.json) `build` script runs `prisma generate && next build`, so Vercel's default build works.

### One-shot automated deploy

If you prefer running the whole sequence unattended (no browser OAuth), export API tokens and run [scripts/deploy-cloud.sh](scripts/deploy-cloud.sh):

```bash
export DATABASE_URL="postgresql://...@ep-xyz-pooler.neon.tech/neondb?sslmode=require"
export FLY_API_TOKEN="fo1_..."           # fly.io/user/personal_access_tokens
export VERCEL_TOKEN="..."                # vercel.com/account/tokens
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."                 # optional
# optional tuning: FLY_APP_NAME, FLY_REGION, FLY_VOLUME_SIZE_GB, FLY_GPU_KIND

./scripts/deploy-cloud.sh
```

It is idempotent: safe to rerun after fixing a secret or bumping the Docker image. The script performs every step below (Neon migrate, Fly launch/secrets/deploy, Vercel link/env/deploy, CORS tighten, smoke test). The manual steps in sections 1-5 below are still the source of truth if you prefer to click through each piece.

## 1. Provision Postgres on Neon

```bash
# 1. Sign in at https://neon.tech and create a project.
# 2. Copy the "Pooled connection" string (ends with -pooler.neon.tech).
# 3. Apply Prisma migrations once from your laptop:
export DATABASE_URL="postgresql://USER:PASS@ep-xyz-pooler.neon.tech/neondb?sslmode=require"
npx prisma migrate deploy
```

The FastAPI side creates its own `agentic_app` schema lazily on first request.

## 2. Deploy the Python API on Fly.io

```bash
# Install flyctl from https://fly.io/docs/hands-on/install-flyctl/
fly auth login

# Edit fly.toml -> `app = "your-unique-name"` before the first deploy.

fly volumes create agentic_data --region iad --size 50 --vm-gpu-kind a10

fly secrets set \
  DATABASE_URL="<neon pooled url>" \
  OPENAI_API_KEY="sk-..." \
  HF_TOKEN="hf_..." \
  ALLOWED_ORIGINS="*"   # tighten in step 4

fly deploy

# Sanity check (should return {"status":"ok",...}):
fly ssh console -C "curl -s localhost:8080/health"
```

## 3. Deploy the Next.js app on Vercel

```bash
# Install vercel CLI: npm i -g vercel
vercel link
```

Then in the Vercel dashboard, set these env vars for **Production** (and Preview if you want preview deploys to talk to prod):

| Var | Value |
| --- | --- |
| `DATABASE_URL` | same Neon pooled URL |
| `NEXT_PUBLIC_PYTHON_API_URL` | `https://<your-fly-app>.fly.dev` |
| `PYTHON_API_URL` | `https://<your-fly-app>.fly.dev` |
| `NEXTAUTH_URL` | `https://<your-vercel-domain>` |
| `NEXTAUTH_SECRET` | output of `openssl rand -base64 32` |
| `OPENAI_API_KEY` | `sk-...` |

Trigger a deploy: `vercel --prod` (or just push to the branch Vercel is tracking).

## 4. Lock down CORS

After the Vercel deploy gives you a real domain:

```bash
fly secrets set ALLOWED_ORIGINS="https://<vercel-domain>,https://<preview-domain>"
fly deploy   # or fly apps restart <name> - secrets take effect on next boot
```

## 5. Smoke test

1. Open `https://<vercel-domain>/dashboard` - the server component fetches `${PYTHON_API_URL}/dashboard/summary` and should render.
2. Upload a small JSONL dataset (Datasets -> New).
3. Start a local fine-tune. Watch progress and events stream in on the job detail page.
4. In the Playground, send a prompt to the tuned model; it should run through the Fly-hosted Ollama.

## What does not work in this setup

- **Adaptive Ollama GPU rebalancing** (`POST /settings/gpu/rebalance-ollama`) - Fly containers have no systemd. Leave `OLLAMA_AUTO_MANAGE=0`. Single-GPU machines are fine because Ollama already uses the only device.
- **Multi-GPU training** - Fly GPU machines are single-GPU. Multi-GPU flows should stay on-prem.
- **Multi-tenant concurrent training** - one GPU, one trainer; jobs queue via Temporal.

## Cost notes

- **Vercel Hobby**: free for this UI (no long-running serverless functions).
- **Neon**: free tier covers ~0.5 GB; scale-to-zero is transparent.
- **Fly.io a10**: ~$1.50/hr while running. `auto_stop_machines = "stop"` (already set in fly.toml) lets it idle to zero; cold start adds ~30 s per wake for Ollama.
- **OpenAI**: unchanged - same `OPENAI_API_KEY` from both halves.
