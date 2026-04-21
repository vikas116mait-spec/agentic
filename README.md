# Agentic Fine-Tune App

Production-minded MVP for uploading JSONL datasets, validating them, tracking fine-tuning jobs, and comparing outputs in a playground. It supports a low-cost local-first path with Ollama for inference and agent runs, OpenAI for managed fine-tuning, Hugging Face Jobs for open-source supervised fine-tuning on cloud GPUs, and Local GPU QLoRA for training on your own machine.

The app now runs in direct local-workspace mode:

- no login page is required for normal local use
- the Next.js interface runs on port `3000`
- a Python FastAPI service runs on port `8001`
- Temporal now powers durable agent runs for long-lived orchestration
- UI button clicks call the Python API directly

## Quick Start

Use this path if you want the cheapest and simplest local setup.

### 1. Install prerequisites

Install these tools first:

- Node.js `22`
- Python `3.12`
- Ollama

### 2. Create or update `.env`

If you do not already have a `.env` file, copy `.env.example` to `.env`.

```powershell
Copy-Item .env.example .env
```

Then make sure your `.env` contains these values:

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

If you already have a `NEXTAUTH_SECRET`, you can keep your current value.

### 3. Install the local model

Pull the default Ollama model:

```powershell
ollama pull qwen3:8b
```

If Ollama is not already running, start it:

```powershell
ollama serve
```

### 4. Start the Python API

Open terminal 1:

```powershell
cd C:\Users\vikas\Desktop\Learning\agentic
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn python_api.main:app --host 0.0.0.0 --port 8001
```

If PowerShell blocks script execution, use the virtual environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn python_api.main:app --host 0.0.0.0 --port 8001
```

When the API starts, it will automatically start a local Temporal dev server and worker when `TEMPORAL_AUTO_START_DEV_SERVER="1"`.

### 5. Start the Next.js app

Open terminal 2:

```powershell
cd C:\Users\vikas\Desktop\Learning\agentic
npm install
npm run dev
```

### 6. Open the app

Open these pages in your browser:

```text
http://localhost:3000/dashboard
http://localhost:3000/agent
http://localhost:3000/playground
```

### 7. Check that services are healthy

You can verify the backend with:

```text
http://127.0.0.1:8001/health
http://127.0.0.1:8001/agent/runtime
```

### 8. Know what works in local Ollama mode

These features work in the cheap local setup:

- dashboard
- dataset upload and validation
- playground runs
- agent runs through the Python API and Temporal

These features still require `LLM_PROVIDER=openai` and a real `OPENAI_API_KEY`:

- uploading datasets to OpenAI
- managed fine-tuning jobs
- syncing hosted OpenAI fine-tuning job state

## What This App Does

- Upload `.jsonl` training datasets
- Validate dataset records line by line
- Run prompts locally with open-source models through Ollama
- Launch agent runs against local or hosted providers
- Create managed fine-tuning jobs with OpenAI
- Create open-source SFT jobs with Hugging Face Jobs
- Create Local GPU QLoRA jobs and save LoRA adapters on disk
- Sync job status and cache recent events for both providers
- Save fine-tuned model names
- Compare base model vs fine-tuned model output in a playground
- Launch agent runs that can pick tools, wait, resume, and keep the workflow moving

## Tech Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Python FastAPI backend
- Temporal Python SDK for durable agent orchestration
- PostgreSQL-backed Python API storage with local JSON fallback
- OpenAI SDKs for Node and Python
- Hugging Face Hub Python SDK for Jobs submission and monitoring
- local Transformers, PEFT, TRL, Accelerate, and bitsandbytes for QLoRA training
- Optional Prisma/PostgreSQL path still present in the repo

## Runtime Architecture

The current app is designed around these runtime pieces:

- `Next.js` web app for dashboard, datasets, jobs, playground, settings, and agent UI
- `FastAPI` Python API for dataset validation, job orchestration, playground runs, agent runs, and model profile settings
- `PostgreSQL` for Python API state when `DATABASE_URL` is configured and the Python environment has `psycopg2`
- `Temporal` for durable agent workflows and wait/resume orchestration
- `Ollama` for local open-source model inference
- `OpenAI` for managed fine-tuning and hosted model runs when enabled
- `Hugging Face Jobs` for open-source model training and Hub persistence
- `Local GPU QLoRA` for on-machine adapter training and local artifact storage

Request flow:

1. Browser loads the Next.js app
2. Next.js client calls the Python API at `PYTHON_API_URL`
3. Python API reads or writes state in PostgreSQL under the configured schema
4. Agent runs are executed through Temporal workers
5. Model inference goes to Ollama or OpenAI depending on the selected profile/provider

## PostgreSQL Tables

For the current Python-backed app, PostgreSQL tables are created in the schema from `DATABASE_SCHEMA`.

Recommended schema:

- `agentic_app`

Current table layout:

- `agentic_app.agentic_datasets`
- `agentic_app.agentic_jobs`
- `agentic_app.agentic_job_events`
- `agentic_app.agentic_playground_runs`
- `agentic_app.agentic_agent_runs`
- `agentic_app.agentic_model_profiles`
- `agentic_app.agentic_workspace_settings`

Main purpose of each table:

- `agentic_datasets`: uploaded dataset records and validation metadata
- `agentic_jobs`: fine-tuning job records for OpenAI and Hugging Face Jobs
- `agentic_job_events`: job timeline and sync events
- `agentic_playground_runs`: prompt comparison history
- `agentic_agent_runs`: Temporal-backed agent run snapshots
- `agentic_model_profiles`: user-editable model registry such as small, medium, large, and thinking
- `agentic_workspace_settings`: workspace defaults such as selected profile ids

For the first six tables above, the common structure is:

- primary key column such as `dataset_id`, `job_id`, or `run_id`
- a few searchable text columns for key fields
- `payload JSONB`
- `created_at TIMESTAMPTZ`
- `updated_at TIMESTAMPTZ`

Important note:

- the repo still contains an older Prisma schema and older Next API routes, but the active local app flow is the Next.js UI calling the Python API directly

## Environment Variables

The project uses the following values in `.env`:

```env
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/agentic"
DATABASE_SCHEMA="agentic_app"
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_BASE_MODEL="qwen3:8b"
OLLAMA_AGENT_MODEL="qwen3:8b"
OPENAI_API_KEY=""
OPENAI_AGENT_MODEL="gpt-5.4-mini"
HF_TOKEN=""
HF_NAMESPACE=""
HF_BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
HF_DATASET_REPO=""
HF_MODEL_REPO_ID=""
HF_JOBS_FLAVOR="a10g-large"
HF_JOBS_TIMEOUT="3h"
HF_JOBS_IMAGE="huggingface/trl"
HF_TRACKIO_PROJECT="agentic-training"
HF_TRACKIO_SPACE_ID=""
LOCAL_TRAINING_ENABLED="1"
LOCAL_TRAINING_BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
LOCAL_TRAINING_PYTHON="/path/to/agentic/.venv-train/bin/python"
LOCAL_TRAINING_ALLOW_CPU_FALLBACK="0"
LOCAL_TRAINING_MULTI_GPU="0"
LOCAL_TRAINING_EVAL_RATIO="0.1"
LOCAL_TRAINING_SEED="42"
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
- `OPENAI_API_KEY` when you want managed OpenAI fine-tuning
- `HF_TOKEN` when you want paid cloud fine-tuning through Hugging Face Jobs
- `LOCAL_TRAINING_ENABLED` plus `LOCAL_TRAINING_PYTHON` when you want the recommended free fine-tuning path on your own GPU

You can also set:

- `DATABASE_URL` if you want PostgreSQL-backed Python API storage
- `DATABASE_SCHEMA` to keep this app's tables grouped in their own schema, such as `agentic_app`
- `HF_NAMESPACE` if you want to force the Hugging Face username or organization used for Jobs
- `HF_DATASET_REPO` if you want to reuse a specific dataset repository for uploaded training files
- `HF_MODEL_REPO_ID` if you want every training run to push to a specific model repository instead of generating one automatically
- `HF_JOBS_FLAVOR`, `HF_JOBS_TIMEOUT`, and `HF_JOBS_IMAGE` to control Hugging Face training hardware and runtime
- `LOCAL_TRAINING_BASE_MODEL`, `LOCAL_TRAINING_PYTHON`, `LOCAL_TRAINING_MULTI_GPU`, `LOCAL_TRAINING_EVAL_RATIO`, and `LOCAL_TRAINING_ALLOW_CPU_FALLBACK` to control the local QLoRA path
  Keep `LOCAL_TRAINING_MULTI_GPU=0` for the stable default path unless you are actively working on distributed training support.
- `NEXTAUTH_SECRET` if you want to keep the original auth path available
- `OPENAI_AGENT_MODEL` if you want a different OpenAI orchestration model later
- `TEMPORAL_AUTO_START_DEV_SERVER=0` if you want to connect to an already-running Temporal server instead of auto-starting one

Example command to generate a secret:

```powershell
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

## Local Setup Flow

### Cheapest local mode
The full step-by-step path is in `Quick Start` above.

<<<<<<< HEAD
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

If you want to use the recommended free Local GPU QLoRA backend, install the trainer stack into the Python interpreter referenced by `LOCAL_TRAINING_PYTHON`:

```powershell
pip install -r requirements-local-training.txt
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
=======
If you are opening the app from another device on your network, use your machine IP instead of `localhost`:
>>>>>>> 93ab86cdaa10e62776f18237495ed8b6f274ad23

```text
http://<your-ip>:3000/dashboard
http://<your-ip>:3000/agent
http://<your-ip>:3000/playground
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

Docker is optional. For most local development, use the direct `Quick Start` flow above first.

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
- for Hugging Face Jobs training, also set `HF_TOKEN` and the optional `HF_*` training settings in `.env`
- when `DATABASE_URL` and `DATABASE_SCHEMA` are configured and `psycopg2` is installed, the Python API stores workspace state in PostgreSQL
- if PostgreSQL is not available, the Python API falls back to `python_api/data/state.json`

If you want the Prisma/PostgreSQL path too:

```powershell
npx prisma migrate dev --name init
npx prisma generate
```

## Project Notes

- The browser UI calls the Python API on port `8001` using the current hostname.
- Uploaded files for the Python flow are stored in `uploads_python/`.
- The Python API stores state in PostgreSQL when configured, otherwise it falls back to `python_api/data/state.json`.
- Temporal dev-server data is stored locally under `python_api/data/` when embedded mode is enabled.
- Dataset validation checks JSONL structure and accepts either chat-style `messages` records or `instruction`/`input`/`output` records.
- The playground runs the same prompt against the base model and optional fine-tuned model.
- Agent runs use provider-aware tool calling, with Temporal handling wait-and-resume behavior.
- `LLM_PROVIDER=ollama` is the easiest low-cost mode. It supports local playground and agent inference through Ollama's OpenAI-compatible endpoint.
- OpenAI fine-tuning remains a managed hosted path.
- Hugging Face Jobs fine-tuning submits an SFT/LoRA-style cloud training job and saves the tuned model to the Hub.
- Local GPU QLoRA fine-tuning runs a background trainer process, uses your own CUDA-visible GPU, and saves adapter artifacts under `uploads_python/jobs/`.
- Ollama profiles are inference-only in this app.

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
7. When you are ready for OpenAI managed fine-tuning, set `OPENAI_API_KEY`
8. When you are ready for open-source cloud fine-tuning, set `HF_TOKEN`
9. If you want to train on your own machine, keep `LOCAL_TRAINING_ENABLED=1` and make sure your Python runtime can see the GPU
10. Create a fine-tuning job from `/jobs/new`
11. Sync the job until it finishes
12. Compare outputs side by side in the playground
