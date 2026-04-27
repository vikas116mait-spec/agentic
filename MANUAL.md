# Agentic Fine-Tune App — Complete Project Manual

This manual is the exhaustive reference for the current codebase.

Use `README.md` for the fastest path to run the app.

Use this file when you want the detailed explanation of:

- the app architecture
- every major page
- the current fine-tuning flows
- provider behavior
- Python API endpoints
- local training behavior
- downloads and artifacts
- persistence
- testing and troubleshooting

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Repository Structure](#3-repository-structure)
4. [Setup and Installation](#4-setup-and-installation)
5. [Environment Variables](#5-environment-variables)
6. [Running the App](#6-running-the-app)
7. [Pages and User Experience](#7-pages-and-user-experience)
8. [Core Workflows](#8-core-workflows)
9. [Model Providers](#9-model-providers)
10. [Local GPU QLoRA](#10-local-gpu-qlora)
11. [Python API Endpoints](#11-python-api-endpoints)
12. [Persistence and Artifacts](#12-persistence-and-artifacts)
13. [TypeScript and Frontend Helpers](#13-typescript-and-frontend-helpers)
14. [Testing](#14-testing)
15. [Docker and Deployment](#15-docker-and-deployment)
16. [Troubleshooting](#16-troubleshooting)
17. [Code Map](#17-code-map)

---

## 1. Project Summary

The **Agentic Fine-Tune App** is a full-stack application for managing the lifecycle of LLM fine-tuning and evaluation.

The current product supports:

- uploading `.jsonl` datasets
- validating records line by line
- previewing examples
- launching fine-tuning jobs
- monitoring progress, events, warnings, and metrics
- downloading datasets and trained model artifacts
- comparing base model vs tuned model output
- running durable agent workflows through Temporal

The app is built around a **local-first workflow**:

- `Ollama` for free local inference
- `Local GPU QLoRA` for free local fine-tuning
- `OpenAI` for paid managed fine-tuning
- `Hugging Face Jobs` for paid cloud open-source fine-tuning

The active request flow is:

```text
Next.js frontend -> FastAPI backend -> provider/runtime/storage
```

The repository still contains some older Prisma and auth code, but the main app path is the Python-backed flow.

---

## 2. Architecture Overview

### High-level diagram

```text
Browser / Next.js (3000)
  -> lib/python-api.ts
  -> FastAPI (8001)
     -> services.py orchestration
     -> store.py persistence
     -> local_qlora/ local GPU runtime
     -> Temporal worker/runtime
     -> Ollama / OpenAI / Hugging Face Jobs / other providers
```

### Runtime pieces

#### Frontend

The frontend is responsible for:

- page routing
- form state
- model/profile selection
- dataset preview
- job progress UIs
- post-training download UX

Main folders:

- `app/`
- `components/`
- `lib/`

#### Backend

The backend is responsible for:

- dataset validation
- job creation and sync
- provider selection
- training runtime checks
- local artifact packaging
- agent orchestration
- persistence

Main files:

- `python_api/main.py`
- `python_api/services.py`
- `python_api/store.py`
- `python_api/local_qlora/*`
- `python_api/temporal_runtime.py`
- `python_api/agentic_workflow.py`

#### Persistence

The app supports two storage modes:

1. PostgreSQL when `DATABASE_URL` is set and `psycopg2` is available
2. Local JSON fallback when Postgres is not configured

Artifacts such as datasets, status files, logs, and trained adapters are stored in `uploads_python/`.

---

## 3. Repository Structure

```text
agentic/
├── app/
│   ├── dashboard/
│   ├── datasets/
│   │   ├── new/
│   │   └── [id]/
│   ├── jobs/
│   │   ├── new/
│   │   └── [id]/
│   ├── playground/
│   ├── settings/
│   ├── agent/
│   └── (auth)/
├── components/
│   ├── dashboard/
│   ├── dataset/
│   ├── jobs/
│   ├── playground/
│   ├── settings/
│   ├── agent/
│   └── ui/
├── lib/
├── python_api/
│   ├── main.py
│   ├── services.py
│   ├── store.py
│   ├── temporal_runtime.py
│   ├── agentic_workflow.py
│   ├── agentic_activities.py
│   ├── huggingface_jobs.py
│   ├── errors.py
│   ├── env.py
│   ├── local_qlora/
│   └── data/
├── uploads_python/
├── tests/
├── samples/
├── docker/
├── temporal/
├── README.md
├── MANUAL.md
├── package.json
├── requirements.txt
└── requirements-local-training.txt
```

### Most important files

If you are new to the codebase, start here:

1. `README.md`
2. `python_api/main.py`
3. `python_api/services.py`
4. `python_api/store.py`
5. `python_api/local_qlora/config.py`
6. `python_api/local_qlora/train.py`
7. `lib/python-api.ts`
8. `components/jobs/create-job-form.tsx`
9. `components/jobs/job-detail-client.tsx`
10. `components/settings/model-profiles-settings.tsx`

---

## 4. Setup and Installation

### Prerequisites

- Node.js `20+` (`22` recommended)
- Python `3.11+` (`3.12` recommended)
- Ollama
- Git
- NVIDIA GPU for the recommended free local fine-tuning path

### Clone and install frontend dependencies

```bash
git clone <repo-url>
cd agentic
npm install
```

### Create `.env`

```bash
cp .env.example .env
```

At minimum for local inference:

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

At minimum for local GPU fine-tuning:

```env
LOCAL_TRAINING_ENABLED="1"
LOCAL_TRAINING_BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
LOCAL_TRAINING_PYTHON="/absolute/path/to/agentic/.venv-train/bin/python"
LOCAL_TRAINING_ALLOW_CPU_FALLBACK="0"
LOCAL_TRAINING_MULTI_GPU="0"
LOCAL_TRAINING_EVAL_RATIO="0.1"
LOCAL_TRAINING_SEED="42"
```

### Create API venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Create training venv

Use a separate venv for training:

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -r requirements-local-training.txt
```

This venv is the interpreter that `LOCAL_TRAINING_PYTHON` must point to.

### Pull a local model in Ollama

```bash
ollama pull qwen3:8b
```

### Optional PostgreSQL setup

```bash
docker run -d \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=agentic \
  -p 5432:5432 \
  postgres:16-alpine
```

Then set:

```env
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/agentic"
DATABASE_SCHEMA="agentic_app"
```

If you do not configure PostgreSQL, the app will use local JSON state.

---

## 5. Environment Variables

### Core runtime

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | default provider selection |
| `PYTHON_API_URL` | Python API base URL |
| `NEXT_PUBLIC_PYTHON_API_URL` | browser-visible Python API URL |
| `DATABASE_URL` | Postgres connection string |
| `DATABASE_SCHEMA` | Postgres schema |

### Ollama

| Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | Ollama host |
| `OLLAMA_SMALL_MODEL` | small Ollama profile |
| `OLLAMA_BASE_MODEL` | medium Ollama profile |
| `OLLAMA_LARGE_MODEL` | large Ollama profile |
| `OLLAMA_THINKING_MODEL` | reasoning Ollama profile |
| `OLLAMA_AGENT_MODEL` | Ollama model for agent usage |

### Local GPU QLoRA

| Variable | Description |
|---|---|
| `LOCAL_TRAINING_ENABLED` | enable local fine-tuning |
| `LOCAL_TRAINING_BASE_MODEL` | default local training model |
| `LOCAL_TRAINING_PYTHON` | required path to training interpreter |
| `LOCAL_TRAINING_ALLOW_CPU_FALLBACK` | allow CPU fallback, extremely slow |
| `LOCAL_TRAINING_MULTI_GPU` | enable accelerate multi-GPU launch |
| `LOCAL_TRAINING_GPU_INDEX` | pin a single GPU |
| `LOCAL_TRAINING_EVAL_RATIO` | evaluation split |
| `LOCAL_TRAINING_SEED` | reproducibility seed |

### OpenAI

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | required for OpenAI fine-tuning and hosted inference |
| `OPENAI_AGENT_MODEL` | model used for agent orchestration |
| `OPENAI_BASE_URL` | optional override |

### Hugging Face Jobs

| Variable | Description |
|---|---|
| `HF_TOKEN` | required for paid HF Jobs runs |
| `HF_NAMESPACE` | user or org namespace |
| `HF_BASE_MODEL` | default HF base model |
| `HF_DATASET_REPO` | dataset repo override |
| `HF_MODEL_REPO_ID` | model repo override |
| `HF_JOBS_FLAVOR` | hardware flavor |
| `HF_JOBS_TIMEOUT` | runtime timeout |
| `HF_JOBS_IMAGE` | container image |
| `HF_TRACKIO_PROJECT` | training tracking project |
| `HF_TRACKIO_SPACE_ID` | optional trackio space |

### Inference-only providers

| Variable | Provider |
|---|---|
| `GROQ_API_KEY` | Groq |
| `GOOGLE_API_KEY` | Gemini |
| `CEREBRAS_API_KEY` | Cerebras |
| `TOGETHER_API_KEY` | Together |

### Temporal

| Variable | Description |
|---|---|
| `TEMPORAL_ADDRESS` | Temporal host |
| `TEMPORAL_NAMESPACE` | namespace |
| `TEMPORAL_TASK_QUEUE` | worker queue |
| `TEMPORAL_AUTO_START_DEV_SERVER` | auto-start local Temporal dev server |
| `TEMPORAL_DEV_SERVER_UI` | enable local Temporal UI |

### Legacy frontend auth

| Variable | Description |
|---|---|
| `NEXTAUTH_SECRET` | legacy auth secret |
| `NEXTAUTH_URL` | legacy auth URL |

---

## 6. Running the App

### Local development

Use three terminals.

#### Terminal 1 — Python API

```bash
cd agentic
source .venv/bin/activate
python -m uvicorn python_api.main:app --host 0.0.0.0 --port 8001 --reload
```

#### Terminal 2 — Next.js frontend

```bash
cd agentic
npm run dev
```

#### Terminal 3 — Ollama

```bash
ollama serve
```

### Health endpoints

- `GET /health`
- `GET /agent/runtime`
- `GET /settings/local-training/runtime`

### Helpful URLs

- `http://localhost:3000/dashboard`
- `http://localhost:3000/datasets`
- `http://localhost:3000/jobs`
- `http://localhost:3000/playground`
- `http://localhost:3000/settings`
- `http://localhost:3000/agent`

---

## 7. Pages and User Experience

### `/dashboard`

Main landing page for the app.

Shows:

- summary cards
- recent activity
- high-level state of datasets, jobs, and models

### `/datasets`

Dataset list page.

Shows:

- uploaded datasets
- validation status
- record counts

### `/datasets/new`

Dataset upload page.

Flow:

1. choose a JSONL file
2. upload to backend
3. backend validates each record
4. open dataset detail page

### `/datasets/[id]`

Dataset detail page.

Shows:

- file size
- validation status
- validation summary
- example previews
- dataset download button
- start fine-tuning shortcut
- optional OpenAI pre-upload action

### `/jobs`

Job list page.

Shows:

- job status
- provider
- dataset name
- timestamps

### `/jobs/new`

Fine-tuning job creation page.

Important features:

- valid dataset selector
- profile selector
- local runtime readiness panel
- training speed presets
- export options
- advanced hyperparameters

Current local training presets:

| Preset | Use case |
|---|---|
| `Fast smoke test` | shortest run for validation |
| `Balanced` | recommended default |
| `Best quality` | slower but stronger adaptation |

The page also explains:

- low VRAM vs balanced vs stronger-quality model tiers
- whether the local runtime is ready
- whether `Unsloth` is available
- whether `Ollama` is reachable

### `/jobs/[id]`

Job detail page.

This is the main monitoring and post-training page.

It shows:

- live progress
- loss chart
- GPU metrics
- runtime path summary
- warnings
- job events
- model artifact info
- dataset shortcut

For completed local jobs it also shows a dedicated **ready to download** section with:

- `Download model files`
- `Download GGUF model`
- `Download training dataset`

It also includes a model test panel after successful completion.

### `/playground`

Prompt comparison UI.

Lets the user compare:

- base model output
- tuned model output

### `/settings`

Model settings and provider configuration UI.

Used for:

- viewing model profiles
- creating custom profiles
- setting defaults
- seeing provider availability

Curated Ollama options currently include:

- `qwen3:4b`
- `qwen3:8b`
- `llama3.2:3b`
- `gemma3:4b`
- `llama3.1:8b`
- `qwen2.5:7b`
- `deepseek-r1:8b`

Curated local training models currently include:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### `/agent`

Temporal-backed agent run interface.

Used for:

- launching agent workflows
- tracking agent steps
- observing long-running orchestration

---

## 8. Core Workflows

### Workflow A — Upload and validate a dataset

1. Open `Datasets -> New`
2. Upload a `.jsonl` file
3. Backend stores the file under `uploads_python/datasets/{dataset_id}/`
4. Backend validates each line
5. UI shows summary, examples, and validation status
6. User can download the uploaded dataset from the detail page

### Workflow B — Start a local fine-tuning run

1. Open `Jobs -> New`
2. Choose a valid dataset
3. Choose a local QLoRA profile
4. Confirm the local runtime is ready
5. Pick a speed preset
6. Optionally enable:
   - `GGUF` export
   - `Push to Ollama`
7. Start the run
8. Backend builds a local training config and spawns a process
9. Job detail page polls and shows progress

### Workflow C — Download the results

After a successful local job:

1. open the job detail page
2. download the adapter bundle
3. optionally download `GGUF`
4. optionally download the training dataset
5. optionally test the tuned model directly from the page

### Workflow D — Playground comparison

1. pick base model
2. pick tuned model
3. enter a prompt
4. compare outputs side by side

### Workflow E — Agent run

1. start an agent run from `/agent`
2. FastAPI ensures required providers are configured
3. Temporal workflow is started
4. workflow snapshots are persisted and queried

---

## 9. Model Providers

### Local Ollama

Purpose:

- free local inference
- free local agent usage
- optionally host exported local models after training

Strengths:

- cheapest path
- easy local testing
- offline-friendly

### Local GPU QLoRA

Purpose:

- free local fine-tuning on your own GPU

Strengths:

- no cloud training bill
- artifacts stay local
- adapter and `GGUF` downloads

### OpenAI

Purpose:

- managed hosted fine-tuning
- hosted inference

Strengths:

- fully managed
- simple hosted workflow

### Hugging Face Jobs

Purpose:

- paid cloud fine-tuning for open-source models

Strengths:

- hardware managed in the cloud
- model and dataset repo support

### Groq / Gemini / Cerebras / Together

Purpose:

- inference-only usage
- no fine-tuning in this app today

---

## 10. Local GPU QLoRA

### Overview

Local QLoRA is the recommended free fine-tuning path.

The runtime lives in:

- `python_api/local_qlora/__init__.py`
- `python_api/local_qlora/config.py`
- `python_api/local_qlora/train.py`
- `python_api/local_qlora/model.py`
- `python_api/local_qlora/export.py`
- `python_api/local_qlora/state.py`

### Speed presets

Defined in:

- frontend helper: `lib/local-training.ts`
- backend defaults: `python_api/local_qlora/config.py`

Current defaults:

| Preset | Epochs | LR | Batch | Grad Acc | Max Seq |
|---|---:|---:|---:|---:|---:|
| `fast` | 1 | `2e-4` | 2 | 2 | 768 |
| `balanced` | 3 | `1e-4` | 2 | 4 | 1024 |
| `quality` | 4 | `8e-5` | 1 | 8 | 1536 |

### Runtime readiness endpoint

The frontend uses:

- `GET /settings/local-training/runtime`

This returns a snapshot of:

- whether local training is enabled
- whether `LOCAL_TRAINING_PYTHON` is configured
- whether the interpreter exists
- GPU count
- dependency availability
- `Unsloth` availability
- `Ollama` reachability

### Unsloth path

If `Unsloth` is installed in the training venv, the runtime prefers:

- a faster training path
- lower VRAM use
- direct `GGUF` export support

If `Unsloth` is missing, the code falls back to:

- `Transformers + PEFT + bitsandbytes`

### Export behavior

Local runs can:

- save adapters
- save tokenizer files
- write metrics
- export `GGUF`
- register the model in `Ollama`

Important rule:

- `Push to Ollama` requires `GGUF` export

The job creation UI now defensively resets `pushToOllama` to `false` when `exportGguf` is disabled.

### Artifacts created during local runs

Inside `uploads_python/jobs/{job_id}/`:

- `README.md` - auto-generated index of this folder
- `local_train_config.json`
- `local_train_status.json`
- `local_train_events.jsonl`
- `local_train.log`
- `local_train_metrics.json`
- `adapter/` - LoRA adapter + tokenizer
- optional `gguf/*.gguf` + `gguf/Modelfile`

The HuggingFace Trainer scratch directory (`artifacts/` with intermediate
`checkpoint-N/` folders) is deleted automatically once the final adapter is
saved, so only the files above are persisted. Download bundles are built on
demand and never kept inside the job folder.

Older jobs trained before this cleanup was introduced continue to work; they
keep their `artifacts/adapter/` layout and all download + inference paths
resolve correctly for both layouts.

---

## 11. Python API Endpoints

### Health and dashboard

- `GET /health`
- `GET /dashboard/summary`

### Settings

- `GET /settings/model-profiles`
- `GET /settings/local-training/runtime`
- `POST /settings/model-profiles`
- `PATCH /settings/model-profiles/defaults`
- `PATCH /settings/model-profiles/{profile_id}`
- `DELETE /settings/model-profiles/{profile_id}`

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

### Notes about payloads

Local job creation supports:

- `trainingPreset`
- `exportGguf`
- `ggufQuantization`
- `pushToOllama`
- `ollamaModelName`
- `numEpochs`
- `learningRate`
- `perDeviceBatchSize`

Artifact downloads:

- dataset download route returns the original dataset file
- job download route returns the adapter bundle by default
- `?type=gguf` returns the exported GGUF file when available

---

## 12. Persistence and Artifacts

### Postgres mode

When Postgres is enabled, state is stored in tables under `DATABASE_SCHEMA`.

Main collections:

- datasets
- jobs
- job events
- playground runs
- agent runs
- model profiles
- workspace settings

### JSON fallback mode

When Postgres is not enabled, state falls back to:

- `python_api/data/state.json`

### Dataset storage

Uploaded datasets are stored under:

- `uploads_python/datasets/{dataset_id}/`

### Job storage

Local job artifacts are stored under:

- `uploads_python/jobs/{job_id}/`

### Download safety

The backend restricts local dataset downloads to paths under `uploads_python/datasets`.

The backend restricts local job artifact downloads to paths under `uploads_python/jobs`.

---

## 13. TypeScript and Frontend Helpers

### `lib/python-api.ts`

Responsible for:

- resolving the Python API base URL
- running fetches to the FastAPI backend
- normalizing error handling

### `lib/model-profiles.ts`

Responsible for:

- profile selection helpers
- sorting profiles
- keeping local fine-tuning profiles visible even when local setup is incomplete

### `lib/local-training.ts`

Responsible for:

- local training preset definitions
- preset lookup
- model-tier explanations such as:
  - `Low VRAM`
  - `Balanced`
  - `Stronger quality`
  - `Custom`

---

## 14. Testing

### Frontend / TypeScript tests

```bash
npm test
```

Current Vitest coverage includes:

- JSONL validation helpers
- jobs helpers
- model profile sorting
- local training preset helpers

### Python tests

```bash
python -m unittest python_api.test_local_fine_tuning python_api.test_dataset_download
```

Current Python test coverage includes:

- local training preflight
- local training preset defaults
- GGUF and Ollama warning cases
- dataset download path safety

---

## 15. Docker and Deployment

Main files:

- `docker-compose.yml`
- `docker-compose.dev.yaml`
- `Dockerfile.api`
- `docker/web-prod.dockerfile`

The Compose stack can include:

- frontend
- Python API
- PostgreSQL
- Temporal

For local feature development, the simplest path is still:

- run the API directly
- run Next.js directly
- run Ollama locally

---

## 16. Troubleshooting

### Local training says “needs setup”

Check:

- `LOCAL_TRAINING_ENABLED="1"`
- `LOCAL_TRAINING_PYTHON` is set
- the file pointed to by `LOCAL_TRAINING_PYTHON` exists
- `requirements-local-training.txt` was installed into that interpreter

### No CUDA GPU detected

Check:

- NVIDIA drivers
- CUDA visibility
- `torch.cuda.is_available()` from the training venv
- whether you accidentally configured the wrong training interpreter

### `Unsloth` is missing

This is not fatal.

The app will still train using the slower fallback path, but:

- training may be slower
- VRAM usage may be higher
- GGUF export support may be reduced

### Ollama export or registration fails

Check:

- `ollama` CLI is installed
- `ollama serve` is running
- `OLLAMA_BASE_URL` is correct
- `exportGguf` is enabled

### Playground prompt returns `CUDA error: out of memory`

Ollama is pinned to a GPU that is full. Enable adaptive GPU selection by
setting `OLLAMA_AUTO_MANAGE="1"` and completing the one-time admin setup
documented in `README.md` under "Adaptive Ollama GPU". The backend will then
rebalance Ollama onto the freest GPU on demand, and the Playground exposes a
`Move to freest GPU` button that forces a rebalance.

### OpenAI or HF features unavailable

Check:

- `OPENAI_API_KEY` for OpenAI managed fine-tuning
- `HF_TOKEN` for paid Hugging Face Jobs runs

### Dataset download or job download fails

Check:

- the dataset or job still exists on disk
- the requested path is inside the expected uploads directory
- the local job actually completed successfully before trying to download the model

---

## 17. Code Map

### Request path for dataset upload

```text
datasets/new UI
  -> pythonApiFetch("/datasets")
  -> python_api/main.py create_dataset
  -> python_api/services.py create_dataset_record
  -> python_api/store.py update_state
```

### Request path for local job creation

```text
jobs/new UI
  -> pythonApiFetch("/jobs")
  -> python_api/main.py create_job
  -> python_api/services.py create_job_record
  -> python_api/local_qlora/__init__.py spawn_local_training_job
  -> python_api/local_qlora/runner.py
  -> python_api/local_qlora/train.py run_local_qlora_training
```

### Request path for dataset download

```text
dataset detail or job detail UI
  -> GET /datasets/{dataset_id}/download
  -> python_api/main.py download_dataset
  -> python_api/services.py build_dataset_download_package
```

### Request path for model artifact download

```text
job detail UI
  -> GET /jobs/{job_id}/download
  -> python_api/main.py download_job
  -> python_api/services.py build_job_download_package
```

### Request path for job monitoring

```text
job detail UI
  -> GET /jobs/{job_id}
  -> POST /jobs/{job_id}/sync
  -> python_api/services.py sync_job_record
  -> local status, events, metrics, and artifact files
```

### Request path for agent runs

```text
agent page
  -> POST /agent/runs
  -> Temporal workflow start
  -> GET /agent/runs/{run_id}
  -> live workflow snapshot query
```

---

## Final Notes

- The active product flow is `Next.js -> Python API -> provider/runtime/storage`.
- Local GPU QLoRA is the recommended free fine-tuning path.
- The docs in this file are intended to match the current code, not the older Prisma-first architecture.
- If you update providers, endpoints, or training defaults, update both `README.md` and this manual together.
