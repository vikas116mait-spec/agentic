# Agentic Fine-Tune App

Production-minded MVP for uploading JSONL datasets, validating them, tracking managed fine-tuning jobs, and comparing outputs in a playground. It now supports a low-cost local-first path with Ollama for inference and agent runs, plus an OpenAI mode you can switch back to later for managed fine-tuning.

The app now runs in direct local-workspace mode:

- no login page is required for normal local use
- the Next.js interface runs on port `3000`
- a Python FastAPI service runs on port `8001`
- Temporal now powers durable agent runs for long-lived orchestration
- UI button clicks call the Python API directly

## What This App Does

- Upload `.jsonl` training datasets
- Validate dataset records line by line
- Run prompts locally with open-source models through Ollama
- Launch agent runs against local or hosted providers
- Create supervised fine-tuning jobs when OpenAI mode is enabled
- Sync job status and cache recent events when OpenAI mode is enabled
- Save fine-tuned model names
- Compare base model vs fine-tuned model output in a playground
- Launch agent runs that can pick tools, wait, resume, and keep the workflow moving

## Tech Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Python FastAPI backend
- Temporal Python SDK for durable agent orchestration
- Local JSON file storage for the Python API
- OpenAI SDKs for Node and Python
- Optional Prisma/PostgreSQL path still present in the repo

## Environment Variables

The project uses the following values in `.env`:

```env
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/agentic"
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_BASE_MODEL="qwen3:8b"
OLLAMA_AGENT_MODEL="qwen3:8b"
OPENAI_API_KEY=""
OPENAI_AGENT_MODEL="gpt-5.4-mini"
NEXTAUTH_SECRET=""
NEXTAUTH_URL="http://localhost:3000"
PYTHON_API_URL="http://127.0.0.1:8001"
TEMPORAL_ADDRESS="127.0.0.1:7233"
TEMPORAL_NAMESPACE="default"
TEMPORAL_TASK_QUEUE="agentic-agent-queue"
TEMPORAL_AUTO_START_DEV_SERVER="1"
TEMPORAL_DEV_SERVER_UI="0"
```

For the current Python-backed and Temporal-backed flow, the most important values are:

- `LLM_PROVIDER`
- `OLLAMA_BASE_URL` plus your local model names for near-free local use
- `OPENAI_API_KEY` only when you want to switch back to managed OpenAI fine-tuning

You can also set:

- `DATABASE_URL` if you want the Prisma/PostgreSQL path available
- `NEXTAUTH_SECRET` if you want to keep the original auth path available
- `OPENAI_AGENT_MODEL` if you want a different OpenAI orchestration model later
- `TEMPORAL_AUTO_START_DEV_SERVER=0` if you want to connect to an already-running Temporal server instead of auto-starting one

Example command to generate a secret:

```powershell
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

## Local Setup Flow

### Cheapest local mode

Install Ollama, then pull a local model before starting the app:

```powershell
ollama pull qwen3:8b
```

Run these commands from the project folder:

```powershell
cd C:\Users\vikas\Desktop\Learning\agentic
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the Python API in one terminal:

```powershell
python -m uvicorn python_api.main:app --host 0.0.0.0 --port 8001
```

On first startup, the API will also bring up a local Temporal dev server and worker in the background when `TEMPORAL_AUTO_START_DEV_SERVER="1"`.

Start Next.js in another terminal:

```powershell
cd C:\Users\vikas\Desktop\Learning\agentic
npm install
npm run dev
```

Then open:

```text
http://localhost:3000/dashboard
```

Open the agentic interface here:

```text
http://localhost:3000/agent
```

If you are opening the app from another device on your network, use your machine IP:

```text
http://<your-ip>:3000/dashboard
```

## Useful Commands

```powershell
ollama serve
ollama pull qwen3:8b
npm run dev
npm run build
npm run start
npm test
.\.venv\Scripts\python.exe -m uvicorn python_api.main:app --host 0.0.0.0 --port 8001
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/agent/runtime
```

## Docker Compose

There are now two Compose paths:

- `docker-compose.yml`: the more scalable default stack
- `docker-compose.dev.yaml`: the hot-reload developer stack

### Scalable stack

This is the default `docker compose up` path. It uses:

- production Next.js build instead of `next dev`
- external Temporal service instead of the embedded dev server
- PostgreSQL as a real backing service
- persistent named volumes for Python uploads/state
- Temporal UI on port `8080`

Start it:

```powershell
docker compose up --build
```

Open:

```text
http://localhost:3000/dashboard
http://localhost:3000/agent
http://localhost:8080
```

### Dev stack

Use this when you want bind mounts and hot reload:

```powershell
docker compose -f docker-compose.dev.yaml up --build
```

Useful Compose commands:

```powershell
docker compose up --build
docker compose down
docker compose logs -f web
docker compose logs -f python-api
docker compose logs -f temporal
```

Compose notes:

- `web` runs Next.js on port `3000`
- `python-api` runs FastAPI on port `8001`
- `temporal` runs on port `7233`
- `temporal-ui` runs on port `8080`
- your `.env` file is loaded into the app containers, so set `LLM_PROVIDER`, Ollama settings, or `OPENAI_API_KEY` there before starting agent runs
- the current Python API still stores workspace state in local files, so the infrastructure is more scalable now than the app-state layer itself

If you want the Prisma/PostgreSQL path too:

```powershell
npx prisma migrate dev --name init
npx prisma generate
```

## Project Notes

- The browser UI calls the Python API on port `8001` using the current hostname.
- Uploaded files for the Python flow are stored in `uploads_python/`.
- The Python API stores local state in `python_api/data/state.json`.
- Temporal dev-server data is stored locally under `python_api/data/` when embedded mode is enabled.
- Dataset validation checks JSONL structure, `messages`, roles, and assistant output presence.
- The playground runs the same prompt against the base model and optional fine-tuned model.
- Agent runs use provider-aware tool calling, with Temporal handling wait-and-resume behavior.
- `LLM_PROVIDER=ollama` is the easiest low-cost mode. It supports local playground and agent inference through Ollama's OpenAI-compatible endpoint.
- Managed dataset upload and remote fine-tuning jobs stay OpenAI-only for now. In local Ollama mode those actions return clear errors instead of failing mysteriously.

## Current Scope

This repository currently includes:

- app shell in direct local mode
- agent page with Temporal-backed run orchestration
- dashboard page
- dataset upload, list, and detail pages
- job list, create, and detail pages
- playground page
- Python FastAPI service with dataset, job, and playground endpoints
- Python FastAPI agent endpoints for runtime status, run creation, run inspection, and cancellation
- JSONL validator in both TypeScript and Python
- optional Prisma schema and Next API routes still present in the repo

## Next Steps

After setup works locally, the normal flow is:

1. Start the Python API on port `8001`
2. Start Next.js on port `3000`
3. Open `/dashboard`
4. Upload a `.jsonl` dataset
5. Review validation results
6. In local Ollama mode, use `/playground` and `/agent` to iterate cheaply with open-source models
7. When you are ready for managed fine-tuning, switch to `LLM_PROVIDER=openai`
8. Upload the validated dataset to OpenAI
9. Create a fine-tuning job
10. Sync the job until it finishes
11. Compare outputs side by side in the playground
