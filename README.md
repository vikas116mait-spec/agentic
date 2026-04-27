# Agentic Fine-Tune App

Full-stack app for dataset upload, validation, model comparison, fine-tuning orchestration, and durable agent workflows.

This project is optimized around a local-first workflow:

- `Next.js` frontend on port `3000`
- `FastAPI` backend on port `8001`
- `Temporal` for durable agent runs
- `Ollama` for free local inference
- `Local GPU QLoRA` for free on-device fine-tuning
- optional `OpenAI` and `Hugging Face Jobs` for paid cloud fine-tuning

## What It Does

- Upload `.jsonl` training datasets
- Validate records line by line
- Preview dataset examples in the UI
- Start local or hosted fine-tuning jobs
- Track job status, metrics, warnings, and events
- Compare base vs tuned model outputs in the playground
- Run long-lived agent workflows through Temporal
- Download the uploaded dataset
- Download trained local model artifacts after a run completes
- Export local runs to `GGUF` and optionally register them in `Ollama`

## Current Architecture

```text
Browser / Next.js app (3000)
  -> lib/python-api.ts
  -> FastAPI backend (8001)
     -> services.py business logic
     -> store.py persistence (Postgres or JSON fallback)
     -> local_qlora/ local training runtime
     -> Temporal runtime for agent workflows
```

Main runtime pieces:

- `app/` - Next.js App Router pages
- `components/` - UI components for datasets, jobs, playground, settings, agent
- `lib/` - shared TypeScript helpers, types, and Python API client
- `python_api/main.py` - REST API entry point
- `python_api/services.py` - orchestration and provider logic
- `python_api/store.py` - Postgres plus local JSON storage
- `python_api/local_qlora/` - local GPU training, progress, export, runtime checks

## Quick Start

This is the recommended cheapest path.

### 1. Prerequisites

Install:

- Node.js `20+` (`22` recommended)
- Python `3.11+` (`3.12` recommended)
- Ollama
- Git
- NVIDIA GPU for the free local fine-tuning path

### 2. Create `.env`

```bash
cp .env.example .env
```

At minimum, set:

```env
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_BASE_MODEL="qwen3:8b"
PYTHON_API_URL="http://127.0.0.1:8001"
TEMPORAL_ADDRESS="127.0.0.1:7233"
TEMPORAL_NAMESPACE="default"
TEMPORAL_TASK_QUEUE="agentic-agent-queue"
TEMPORAL_AUTO_START_DEV_SERVER="1"
TEMPORAL_DEV_SERVER_UI="0"
```

For the recommended free local fine-tuning path, also set:

```env
LOCAL_TRAINING_ENABLED="1"
LOCAL_TRAINING_BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
LOCAL_TRAINING_PYTHON="/absolute/path/to/agentic/.venv-train/bin/python"
LOCAL_TRAINING_ALLOW_CPU_FALLBACK="0"
LOCAL_TRAINING_MULTI_GPU="0"
LOCAL_TRAINING_EVAL_RATIO="0.1"
LOCAL_TRAINING_SEED="42"
```

`LOCAL_TRAINING_PYTHON` is required for local QLoRA.

### 3. Set up the API environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Set up the local training environment

Use a separate training venv to avoid dependency conflicts:

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -r requirements-local-training.txt
```

Point `LOCAL_TRAINING_PYTHON` at this interpreter.

### 5. Start Ollama

Pull a base model and start Ollama if needed:

```bash
ollama pull qwen3:8b
ollama serve
```

### 6. Start the Python API

```bash
source .venv/bin/activate
python -m uvicorn python_api.main:app --host 0.0.0.0 --port 8001
```

If `TEMPORAL_AUTO_START_DEV_SERVER="1"`, the backend will auto-start a local Temporal dev server and worker.

### 7. Start the Next.js app

```bash
npm install
npm run dev
```

### 8. Open the app

- `http://localhost:3000/dashboard`
- `http://localhost:3000/datasets`
- `http://localhost:3000/jobs`
- `http://localhost:3000/playground`
- `http://localhost:3000/agent`
- `http://localhost:3000/settings`

### 9. Health checks

- `http://127.0.0.1:8001/health`
- `http://127.0.0.1:8001/agent/runtime`
- `http://127.0.0.1:8001/settings/local-training/runtime`

## Core User Flows

### Dataset flow

1. Upload a `.jsonl` file from `Datasets -> New`
2. The backend validates it and stores it under `uploads_python/datasets/`
3. Open the dataset detail page to:
   - review validation status
   - preview examples
   - download the original dataset
   - start a fine-tuning job

### Local fine-tuning flow

1. Open `Jobs -> New`
2. Choose a valid dataset
3. Choose a local fine-tuning profile
4. Review local runtime readiness
5. Choose a training speed profile:
   - `Fast smoke test`
   - `Balanced`
   - `Best quality`
6. Optionally enable:
   - `GGUF` export
   - `Push to Ollama`
7. Start the run
8. Monitor:
   - progress
   - loss chart
   - GPU metrics
   - warnings
   - training events
9. After success:
   - download model files
   - download GGUF
   - download the training dataset
   - test the tuned model from the job page

### Playground flow

Use `Playground` to compare:

- base model vs fine-tuned model
- local models or hosted providers, depending on configuration

### Agent flow

Use `Agent` to launch durable runs backed by Temporal. The agent can create, wait, resume, and track longer workflows using the configured model providers.

## Local Training Details

### Free local fine-tuning models

Curated local QLoRA profiles include:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### One-click Ollama inference models

Curated Ollama options include:

- `qwen3:4b`
- `qwen3:8b`
- `llama3.2:3b`
- `gemma3:4b`
- `llama3.1:8b`
- `qwen2.5:7b`
- `deepseek-r1:8b`

### Training speed presets

Current presets are:

| Preset | Purpose | Main defaults |
|---|---|---|
| `fast` | quick pipeline validation | `1` epoch, `2e-4` LR, `2` batch, `2` gradient accumulation, `768` context |
| `balanced` | recommended default | `3` epochs, `1e-4` LR, `2` batch, `4` gradient accumulation, `1024` context |
| `quality` | slower, more thorough runs | `4` epochs, `8e-5` LR, `1` batch, `8` gradient accumulation, `1536` context |

### Runtime acceleration

When `Unsloth` is installed in `LOCAL_TRAINING_PYTHON`, the local training path prefers the accelerated runtime:

- faster training
- lower VRAM usage
- direct `GGUF` export support

If `Unsloth` is not installed, the app falls back to the standard `Transformers + PEFT` path.

## Adaptive Ollama GPU

Ollama runs as a single daemon and picks GPU 0 by default. On shared multi-GPU
boxes that GPU is often saturated by another process, which surfaces as a
`CUDA error: out of memory` 500 the moment you run a prompt.

When `OLLAMA_AUTO_MANAGE="1"` the backend:

1. Queries `nvidia-smi` before every Ollama-backed inference call.
2. If the GPU Ollama is currently pinned to has less than
   `OLLAMA_MIN_FREE_GPU_MB` free, rewrites `OLLAMA_DROPIN_PATH` with
   `CUDA_VISIBLE_DEVICES=<freest GPU>` and restarts the service.
3. Waits for `OLLAMA_HEALTH_URL` to come back before forwarding the prompt.
4. Retries exactly once if a request still returns a CUDA OOM.

The Playground shows a small pill with the current pinned GPU and a
`Move to freest GPU` button that forces a rebalance on demand.

### One-time admin setup

Create a writable drop-in file and allow the app user to `daemon-reload` and
`restart` the service without a password. Replace `vs95259v` with the user the
backend runs as.

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo touch    /etc/systemd/system/ollama.service.d/cuda.conf
sudo chgrp    vs95259v /etc/systemd/system/ollama.service.d/cuda.conf
sudo chmod 664 /etc/systemd/system/ollama.service.d/cuda.conf

sudo tee /etc/sudoers.d/agentic-ollama <<'EOF'
vs95259v ALL=(root) NOPASSWD: /bin/systemctl daemon-reload, /bin/systemctl restart ollama.service
EOF
sudo chmod 440 /etc/sudoers.d/agentic-ollama
```

If the drop-in is not writable or sudo is not configured, the backend logs a
one-line warning and falls back to plain Ollama behaviour - no crash, no retry.

## Training job folder layout

Every local fine-tuning run writes to `uploads_python/jobs/<jobId>/`. The
layout is intentionally flat so the useful artefacts are one click away:

```text
uploads_python/jobs/<jobId>/
├── README.md                    auto-generated index of this folder
├── local_train_config.json      frozen job configuration
├── local_train_status.json      live status (what the UI polls)
├── local_train_events.jsonl     append-only event log
├── local_train_metrics.json     final train + eval metrics
├── local_train.log              stdout / stderr of the training subprocess
├── adapter/                     LoRA adapter + tokenizer (the fine-tuned model)
└── gguf/                        optional GGUF export + Modelfile
```

Intermediate HuggingFace Trainer checkpoints (`checkpoint-N/` with
`optimizer.pt`, `scheduler.pt`, `rng_state.pth`) are deleted automatically
once the final adapter is saved. Download zip bundles are built on demand
and streamed; they are not persisted inside the job folder.

## Downloads

### Dataset downloads

Each dataset detail page has a `Download dataset` action.

API route:

- `GET /datasets/{dataset_id}/download`

### Job artifact downloads

Completed local jobs expose:

- adapter bundle download
- optional `GGUF` download
- dataset download shortcut

API route:

- `GET /jobs/{job_id}/download`
- `GET /jobs/{job_id}/download?type=gguf`

The adapter bundle contains:

- LoRA adapter files
- tokenizer files
- metrics
- metadata for the run

## Environment Variables

### Core runtime

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | Default provider selection |
| `PYTHON_API_URL` | FastAPI base URL |
| `NEXT_PUBLIC_PYTHON_API_URL` | Browser-visible FastAPI URL |
| `DATABASE_URL` | Postgres storage, optional |
| `DATABASE_SCHEMA` | Postgres schema name |
| `DATABASE_MIN_POOL_SIZE` | minimum pooled Postgres connections (default 1) |
| `DATABASE_MAX_POOL_SIZE` | maximum pooled Postgres connections (default 5) |
| `DATABASE_STATE_CACHE_TTL_MS` | in-process state snapshot TTL in ms (default 1000, 0 disables). Set to 0 when running multiple uvicorn workers. |
| `AGENTIC_DATA_DIR` | override the Python API data dir (default `python_api/data`). Point at `/data/python_api` in Docker/Fly. |
| `AGENTIC_UPLOADS_DIR` | override the uploads dir (default `uploads_python`). Point at `/data/uploads_python` in Docker/Fly. |
| `ALLOWED_ORIGINS` | comma-separated CORS allow-list for the FastAPI service. Defaults to `*` for dev; set to your Vercel domain in production. |

### Ollama

| Variable | Purpose |
|---|---|
| `OLLAMA_BASE_URL` | Ollama server URL |
| `OLLAMA_SMALL_MODEL` | small Ollama profile |
| `OLLAMA_BASE_MODEL` | medium Ollama profile |
| `OLLAMA_LARGE_MODEL` | large Ollama profile |
| `OLLAMA_THINKING_MODEL` | reasoning Ollama profile |
| `OLLAMA_AGENT_MODEL` | default Ollama agent model |
| `OLLAMA_AUTO_MANAGE` | enable the adaptive GPU feature (see below) |
| `OLLAMA_DROPIN_PATH` | systemd drop-in file the backend rewrites |
| `OLLAMA_SERVICE_NAME` | service name passed to `systemctl` |
| `OLLAMA_MIN_FREE_GPU_MB` | free-VRAM threshold that triggers a GPU switch |
| `OLLAMA_HEALTH_URL` | endpoint polled after a restart |
| `OLLAMA_RESTART_TIMEOUT_S` | how long to wait for Ollama to come back |

### Local GPU QLoRA

| Variable | Purpose |
|---|---|
| `LOCAL_TRAINING_ENABLED` | enable local training |
| `LOCAL_TRAINING_BASE_MODEL` | default local training base model |
| `LOCAL_TRAINING_PYTHON` | required interpreter for training venv |
| `LOCAL_TRAINING_ALLOW_CPU_FALLBACK` | allow CPU training, very slow |
| `LOCAL_TRAINING_MULTI_GPU` | multi-GPU accelerate launch |
| `LOCAL_TRAINING_GPU_INDEX` | pin a GPU for single-GPU runs |
| `LOCAL_TRAINING_EVAL_RATIO` | evaluation split |
| `LOCAL_TRAINING_SEED` | reproducibility seed |

### OpenAI / Hugging Face

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | managed OpenAI fine-tuning and inference |
| `OPENAI_AGENT_MODEL` | hosted model for the agent |
| `HF_TOKEN` | paid Hugging Face Jobs runs |
| `HF_NAMESPACE` | user or org namespace |
| `HF_BASE_MODEL` | default HF fine-tuning base model |
| `HF_DATASET_REPO` | dataset repo override |
| `HF_MODEL_REPO_ID` | model repo override |
| `HF_JOBS_FLAVOR` | hardware flavor |
| `HF_JOBS_TIMEOUT` | timeout |
| `HF_JOBS_IMAGE` | jobs image |

### Temporal

| Variable | Purpose |
|---|---|
| `TEMPORAL_ADDRESS` | Temporal server address |
| `TEMPORAL_NAMESPACE` | namespace |
| `TEMPORAL_TASK_QUEUE` | worker queue |
| `TEMPORAL_AUTO_START_DEV_SERVER` | auto-start embedded dev server |
| `TEMPORAL_DEV_SERVER_UI` | enable Temporal UI |

## Main Pages

- `Dashboard` - summary cards and activity
- `Datasets` - list uploaded datasets
- `Datasets -> New` - upload a new dataset
- `Datasets -> Detail` - validation, download, start training
- `Jobs` - list fine-tuning jobs
- `Jobs -> New` - create a run with presets and export options
- `Jobs -> Detail` - progress, metrics, GPU stats, downloads, test panel
- `Playground` - compare model responses
- `Settings` - model profiles, defaults, provider status
- `Agent` - Temporal-backed agent runs

## Python API Routes

### Settings

- `GET /settings/model-profiles`
- `GET /settings/local-training/runtime`
- `POST /settings/model-profiles`
- `PATCH /settings/model-profiles/defaults`
- `PATCH /settings/model-profiles/{profile_id}`
- `DELETE /settings/model-profiles/{profile_id}`
- `GET /settings/gpu/inventory`
- `POST /settings/gpu/rebalance-ollama`

### Datasets

- `GET /datasets`
- `POST /datasets`
- `GET /datasets/{dataset_id}`
- `GET /datasets/{dataset_id}/download`
- `POST /datasets/{dataset_id}/upload-to-openai`

### Jobs

- `GET /jobs`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/sync`
- `GET /jobs/{job_id}/events`
- `POST /jobs/{job_id}/cancel`
- `GET /jobs/{job_id}/download`

### Playground

- `POST /playground/run`

### Agent

- `GET /agent/runtime`
- `GET /agent/runs`
- `POST /agent/runs`
- `GET /agent/runs/{run_id}`
- `POST /agent/runs/{run_id}/cancel`

## Storage

If `DATABASE_URL` is configured and `psycopg2` is available, the backend stores state in Postgres under `DATABASE_SCHEMA`.

Otherwise it falls back to local files:

- `python_api/data/state.json`
- `uploads_python/datasets/`
- `uploads_python/jobs/`

## Testing

Run frontend and TypeScript tests:

```bash
npm test
```

Run Python tests:

```bash
python -m unittest python_api.test_local_fine_tuning python_api.test_dataset_download
```

## Important Notes

- The active app flow is `Next.js UI -> Python API`, not the older Prisma/Next API path.
- The repo still contains legacy auth and Prisma pieces, but they are not the main local workflow.
- Local QLoRA is the recommended free fine-tuning path.
- `Push to Ollama` requires `GGUF` export.

## Troubleshooting

### Local training says setup needed

Make sure:

- `LOCAL_TRAINING_ENABLED="1"`
- `LOCAL_TRAINING_PYTHON` points to `.venv-train/bin/python`
- the training venv has `requirements-local-training.txt` installed

### No GPU detected

- check CUDA visibility
- verify PyTorch in the training venv can see the GPU
- keep `LOCAL_TRAINING_ALLOW_CPU_FALLBACK=0` unless you intentionally want a very slow CPU run

### Ollama push fails

- confirm `ollama` CLI is installed
- confirm `ollama serve` is running
- keep `exportGguf` enabled when `pushToOllama` is enabled

### Postgres returns `FATAL: sorry, too many clients already`

The backend pools Postgres connections (default min 1 / max 5). If the
database is shared with other tenants and is still saturated, lower
`DATABASE_MAX_POOL_SIZE` in `.env` and restart the Python API. Restart also
drains any leaked connections that were opened before the pool was
introduced.

### Stale reads or multi-worker deployments

The backend caches the full state snapshot in-process for
`DATABASE_STATE_CACHE_TTL_MS` milliseconds (default 1000) to reduce
Postgres round-trips on polling GET endpoints. Writers invalidate the
cache immediately, so the TTL only affects readers that are in-flight
during the window. If you run uvicorn with `--workers N > 1`, set
`DATABASE_STATE_CACHE_TTL_MS=0` so one worker's write isn't hidden by
another worker's stale cache. Live hit/miss counters are exposed at
`GET /health` under `stateCache`.

### Playground returns `CUDA error: out of memory`

Ollama picked a GPU that is saturated by another process. Either:

- enable adaptive GPU selection (`OLLAMA_AUTO_MANAGE="1"` plus the one-time
  admin setup in [Adaptive Ollama GPU](#adaptive-ollama-gpu)), then click
  `Move to freest GPU` on the Playground, or
- set `CUDA_VISIBLE_DEVICES` manually in the systemd unit for
  `ollama.service` and run `sudo systemctl restart ollama.service`.

### OpenAI or Hugging Face actions are unavailable

- set `OPENAI_API_KEY` for OpenAI managed fine-tuning
- set `HF_TOKEN` for paid Hugging Face Jobs runs

## More Detail

For the exhaustive reference, see `MANUAL.md`.
