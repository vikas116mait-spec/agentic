# Agentic Fine-Tune App — Complete Project Manual

> Everything you need to know about how this project works, how to set it up, how to use every feature, and how every piece of code connects together.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [Setup & Installation](#4-setup--installation)
5. [Environment Variables](#5-environment-variables)
6. [Running the App](#6-running-the-app)
7. [Pages & User Interface](#7-pages--user-interface)
8. [Core Workflows](#8-core-workflows)
9. [Model Providers](#9-model-providers)
10. [Python API — All Endpoints](#10-python-api--all-endpoints)
11. [Backend Services — Key Functions](#11-backend-services--key-functions)
12. [Agent Orchestration (Temporal)](#12-agent-orchestration-temporal)
13. [Local QLoRA Training](#13-local-qlora-training)
14. [Data Persistence](#14-data-persistence)
15. [TypeScript Types & API Client](#15-typescript-types--api-client)
16. [Frontend Components](#16-frontend-components)
17. [Docker Deployment](#17-docker-deployment)
18. [Using a Fine-Tuned Model After Training](#18-using-a-fine-tuned-model-after-training)
19. [Troubleshooting](#19-troubleshooting)
20. [Code Map — Who Calls What](#20-code-map--who-calls-what)

---

## 1. What This Project Does

The **Agentic Fine-Tune App** is a full-stack platform for managing the complete lifecycle of fine-tuning large language models (LLMs). It is designed to work with multiple providers and supports both local (free) and cloud (paid) training.

### Core Features

| Feature | What it does |
|---------|-------------|
| **Dataset Management** | Upload JSONL training data, validate line-by-line, preview examples |
| **Fine-Tuning Jobs** | Submit training jobs to OpenAI, Hugging Face, or local GPU |
| **Local QLoRA Training** | Train LoRA adapters on your own GPU — free, private, fast |
| **Playground** | Compare base model vs. fine-tuned model responses side-by-side |
| **Agent Orchestration** | Give the AI a goal and let it manage fine-tuning autonomously |
| **Model Profiles** | Save named configurations for different models and providers |
| **GGUF Export** | Convert trained models to GGUF format and push to Ollama |

### Who Is It For

- ML engineers who want a unified UI for fine-tuning workflows
- Researchers who want to compare base vs. fine-tuned model outputs
- Teams who want both local (free) and cloud training options in one tool

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (User)                        │
│              Next.js Frontend — port 3000                │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP fetch
┌──────────────────────────▼──────────────────────────────┐
│              Python API (FastAPI) — port 8001            │
│                    python_api/main.py                    │
│                   python_api/services.py                 │
└──────┬────────────┬───────────┬───────────┬─────────────┘
       │            │           │           │
   PostgreSQL    Ollama      OpenAI    Hugging Face
   (state DB)  (local LLM)  (cloud)    (cloud jobs)
       │
  Local QLoRA
  (GPU trainer)
       │
   Temporal
  (workflow engine)
```

### Technology Stack

| Layer | Technology | Port |
|-------|-----------|------|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS | 3000 |
| Backend | Python FastAPI, Uvicorn | 8001 |
| Database | PostgreSQL 16 (with JSON file fallback) | 5432 |
| Workflow Engine | Temporal | 7233 |
| Local Inference | Ollama | 11434 |
| Training | PyTorch, PEFT, TRL, Transformers | — |

---

## 3. Directory Structure

```
agentic/
│
├── app/                          ← Next.js pages (App Router)
│   ├── dashboard/                ← Home dashboard
│   ├── datasets/                 ← Dataset list, upload, detail
│   │   ├── [id]/                 ← Dataset detail page
│   │   └── new/                  ← Upload new dataset
│   ├── jobs/                     ← Fine-tuning jobs
│   │   ├── [id]/                 ← Job detail + live monitoring
│   │   └── new/                  ← Create training job
│   ├── playground/               ← Side-by-side model comparison
│   ├── settings/                 ← Model profiles configuration
│   ├── agent/                    ← Autonomous agent orchestration
│   └── (auth)/                   ← Login/signup (legacy stubs)
│
├── components/                   ← React UI components
│   ├── agent/                    ← Agent page components
│   ├── dashboard/                ← Dashboard stats, activity
│   ├── dataset/                  ← Upload, validation, preview
│   ├── jobs/                     ← Job form, detail, loss chart
│   ├── playground/               ← Comparison display
│   ├── settings/                 ← Model profiles CRUD
│   └── ui/                       ← Button, Card, Input, etc.
│
├── lib/                          ← Shared utilities
│   ├── types.ts                  ← All TypeScript type definitions
│   ├── python-api.ts             ← HTTP client for Python API
│   ├── utils.ts                  ← Date formatting, helpers
│   ├── model-profiles.ts         ← Profile filter utilities
│   └── jsonl.ts                  ← JSONL validation helpers
│
├── python_api/                   ← FastAPI backend (all Python)
│   ├── main.py                   ← All API routes (endpoints)
│   ├── services.py               ← All business logic (~1800 lines)
│   ├── agentic_workflow.py       ← Temporal workflow definition
│   ├── agentic_activities.py     ← Agent tool implementations
│   ├── store.py                  ← PostgreSQL + JSON persistence
│   ├── huggingface_jobs.py       ← HF Jobs API integration
│   ├── temporal_runtime.py       ← Temporal setup/management
│   ├── errors.py                 ← ApiError exception class
│   ├── env.py                    ← .env file loader
│   ├── local_qlora/              ← Local GPU training module
│   │   ├── __init__.py           ← Job spawn/monitor (520 lines)
│   │   ├── config.py             ← Training configuration dataclass
│   │   ├── train.py              ← QLoRA training script (436 lines)
│   │   ├── model.py              ← Model loading
│   │   ├── data.py               ← Dataset preparation
│   │   ├── evaluation.py         ← Eval metrics
│   │   ├── export.py             ← GGUF conversion
│   │   └── state.py              ← Status JSON tracking
│   └── data/                     ← Local JSON state (fallback)
│
├── uploads_python/               ← Uploaded datasets & job artifacts
│   ├── datasets/                 ← JSONL training files
│   └── jobs/                     ← Per-job directories
│       └── {job_id}/
│           ├── artifacts/
│           │   ├── adapter/      ← LoRA weights + tokenizer
│           │   └── checkpoint-N/ ← Intermediate checkpoints
│           ├── local_train_config.json
│           ├── local_train_status.json
│           ├── local_train_events.jsonl
│           ├── local_train.log
│           └── local_train_metrics.json
│
├── tests/                        ← Vitest test files
├── samples/                      ← Sample JSONL datasets
├── docker/                       ← Docker build files
├── temporal/                     ← Temporal dynamic config
├── .env.example                  ← Environment variable template
├── .env                          ← Your local config (gitignored)
├── docker-compose.yml            ← Production Docker stack
├── docker-compose.dev.yaml       ← Dev Docker stack (hot reload)
├── Dockerfile.api                ← Python API container
├── package.json                  ← Node.js deps and scripts
├── requirements.txt              ← Python runtime deps
└── requirements-local-training.txt  ← Python training deps
```

---

## 4. Setup & Installation

### Prerequisites

- Node.js 20+ and npm
- Python 3.11+
- Ollama (for local inference) — https://ollama.com
- Git

### Step 1 — Clone and install Node dependencies

```bash
git clone <repo-url>
cd agentic
npm install
```

### Step 2 — Create your .env file

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:
```env
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_BASE_MODEL="qwen3:8b"
```

### Step 3 — Set up Python environment

```bash
# Create venv for the API server
python3 -m venv .venv
source .venv/bin/activate

# Install API dependencies
pip install -r requirements.txt
```

### Step 4 — Set up training environment (for local GPU training)

```bash
# Separate venv for training to avoid dependency conflicts
python3 -m venv .venv-train
source .venv-train/bin/activate

pip install -r requirements-local-training.txt

# Point to this Python in .env (required for free local fine-tuning):
# LOCAL_TRAINING_PYTHON="/path/to/agentic/.venv-train/bin/python"
```

### Step 5 — Pull a model in Ollama

```bash
ollama pull qwen3:8b
```

### Step 6 — (Optional) Set up PostgreSQL

If you want persistent database storage (recommended for production):
```bash
# Start PostgreSQL (with Docker or locally)
docker run -d -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=agentic -p 5432:5432 postgres:16-alpine

# Set in .env:
# DATABASE_URL="postgresql://postgres:postgres@localhost:5432/agentic"
# DATABASE_SCHEMA="agentic_app"
```

If `DATABASE_URL` is not set, the app falls back to a local JSON file at `python_api/data/state.json`.

---

## 5. Environment Variables

### Database
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | _(none)_ | PostgreSQL connection string. If absent, uses JSON fallback |
| `DATABASE_SCHEMA` | `agentic_app` | PostgreSQL schema name |

### Model Provider Selection
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | Active provider: `ollama`, `openai`, `huggingface`, `local`, `groq`, `gemini`, `cerebras`, `together` |

### Ollama (Local, Free)
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server address |
| `OLLAMA_BASE_MODEL` | `qwen3:8b` | Default model for inference |
| `OLLAMA_AGENT_MODEL` | same as base | Model for agent orchestration |

### OpenAI (Paid, Managed Fine-tuning)
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(empty)_ | Required for OpenAI fine-tuning and inference |
| `OPENAI_AGENT_MODEL` | `gpt-5.4-mini` | Agent orchestration model |
| `OPENAI_BASE_URL` | _(OpenAI default)_ | Override for custom endpoints |

### Free Inference Providers (API keys only, no fine-tuning)
| Variable | Provider | Free Tier |
|----------|---------|-----------|
| `GROQ_API_KEY` | Groq | Generous free limits |
| `GOOGLE_API_KEY` | Google Gemini | 15 RPM, 1M tokens/day |
| `CEREBRAS_API_KEY` | Cerebras | Free, ultra-fast hardware |
| `TOGETHER_API_KEY` | Together AI | $25 free credits |

### Hugging Face (Cloud Fine-tuning)
| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | _(empty)_ | HF access token (required for paid HF Jobs runs) |
| `HF_NAMESPACE` | _(your username)_ | HF user/org for repos |
| `HF_BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Default base model |
| `HF_DATASET_REPO` | _(empty)_ | Dataset repo ID on HF Hub |
| `HF_MODEL_REPO_ID` | _(empty)_ | Model repo ID on HF Hub |
| `HF_JOBS_FLAVOR` | `a10g-large` | GPU instance type |
| `HF_JOBS_TIMEOUT` | `3h` | Max job duration |
| `HF_JOBS_IMAGE` | `huggingface/trl` | Training Docker image |

### Local GPU Training (QLoRA)
| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_TRAINING_ENABLED` | `1` | Set to `1` to enable local training |
| `LOCAL_TRAINING_BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Default base model |
| `LOCAL_TRAINING_PYTHON` | _(required)_ | Python path for the separate training venv |
| `LOCAL_TRAINING_ALLOW_CPU_FALLBACK` | `0` | Set `1` to allow CPU training (very slow) |
| `LOCAL_TRAINING_MULTI_GPU` | `0` | Set `1` for multi-GPU training |
| `LOCAL_TRAINING_GPU_INDEX` | _(all)_ | Specific GPU index (e.g., `2`) |
| `LOCAL_TRAINING_EVAL_RATIO` | `0.1` | Fraction of data used for evaluation |
| `LOCAL_TRAINING_SEED` | `42` | Random seed for reproducibility |

### Temporal (Agent Workflow Engine)
| Variable | Default | Description |
|----------|---------|-------------|
| `TEMPORAL_ADDRESS` | `127.0.0.1:7233` | Temporal server address |
| `TEMPORAL_NAMESPACE` | `default` | Temporal namespace |
| `TEMPORAL_TASK_QUEUE` | `agentic-agent-queue` | Worker task queue name |
| `TEMPORAL_AUTO_START_DEV_SERVER` | `1` | Auto-start embedded Temporal dev server |
| `TEMPORAL_DEV_SERVER_UI` | `0` | Enable Temporal UI (port 8233) |

### Frontend
| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_PYTHON_API_URL` | `http://127.0.0.1:8001` | Python API URL visible to browser |
| `PYTHON_API_URL` | `http://127.0.0.1:8001` | Python API URL for server-side Next.js |
| `NEXTAUTH_SECRET` | _(empty)_ | NextAuth secret (legacy, can be any string) |
| `NEXTAUTH_URL` | `http://localhost:10087` | NextAuth base URL (legacy) |

---

## 6. Running the App

### Local Development (No Docker)

Open **3 terminal windows**:

**Terminal 1 — Python API**
```bash
cd agentic
source .venv/bin/activate
uvicorn python_api.main:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 — Next.js Frontend**
```bash
cd agentic
npm run dev
# Open http://localhost:3000
```

**Terminal 3 — Ollama (if not running as service)**
```bash
ollama serve
```

### With Docker (Production)

```bash
# Build and start everything (PostgreSQL + Temporal + API + Frontend)
docker compose up --build

# Access:
# App:          http://localhost:3000
# API:          http://localhost:8001
# Temporal UI:  http://localhost:8080
```

### With Docker (Development / Hot Reload)

```bash
docker compose -f docker-compose.dev.yaml up --build
```

---

## 7. Pages & User Interface

### `/dashboard` — Home

What you see:
- **Stats cards**: Total datasets, training runs, running jobs, completed models
- **3-step onboarding guide**: Pick model → Upload dataset → Train & export
- **Recent activity**: Latest jobs and their status

What it connects to:
- `GET /dashboard/summary` Python API endpoint

---

### `/datasets` — Dataset List

What you see:
- All uploaded datasets with name, record count, validation status, date
- **Breadcrumb**: Datasets
- Filter by validation status (VALID / INVALID)
- Click any dataset → detail page

What it connects to:
- `GET /datasets` Python API endpoint

---

### `/datasets/new` — Upload Dataset

Two tabs:
1. **Upload file** — drag-drop or click to upload a `.jsonl` file
2. **Enter manually** — type records in a table interface

**Supported JSONL formats:**
```jsonl
// Chat format (preferred)
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

// Instruction format
{"instruction": "...", "input": "...", "output": "..."}
```

What it connects to:
- `POST /datasets` Python API endpoint (multipart upload)

---

### `/datasets/[id]` — Dataset Detail

What you see:
- **Breadcrumb**: Datasets → Dataset Name
- File size, creation date, validation badge
- **Validation Summary**: Total records, valid count, invalid count
- **Sample preview**: First 3 rows shown in a readable format (not raw JSON)
- **Error list**: Lines that failed validation and why
- **Warning list**: Lines that passed but have issues
- Action buttons: Start fine-tuning job, Pre-upload to OpenAI

What it connects to:
- `GET /datasets/{id}` Python API endpoint

---

### `/jobs` — Training Jobs List

What you see:
- **Filter pills**: All / Running / Failed / Completed (with counts)
- Each job card: dataset name, provider, base model, status badge, mini loss chart
- Color-coded left bar: green=success, blue=running, red=failed

What it connects to:
- `GET /jobs` Python API endpoint

---

### `/jobs/new` — Create Training Job

What you see:
1. **Dataset selector** — only shows VALID datasets
2. **Base model selector** — shows fine-tuning capable profiles
3. Selected run preview
4. **Export options** (local GPU only):
   - Export to GGUF checkbox
   - Quantization format (Q4_K_M recommended)
   - Push to Ollama toggle + model name field
5. **Training hyperparameters** (collapsible, local GPU only):
   - Epochs (default: 3)
   - Learning rate (default: 0.0002)
   - Batch size per device (default: 2)
6. Start fine-tuning button

What it connects to:
- `GET /datasets?validationStatus=VALID`
- `GET /settings/model-profiles`
- `POST /jobs`

---

### `/jobs/[id]` — Job Detail & Live Monitoring

What you see:
- **Breadcrumb**: Training runs → Dataset Name
- Status badge, provider, base model, creation date
- **Progress bar** (for local training): percent complete
- **Current stage**: queued → loading_model → training → evaluating → saving → succeeded
- **Loss chart**: Live training/eval loss over epochs
- **GPU metrics**: Memory usage, utilization, temperature per GPU
- **Event log**: Timestamped messages from the trainer
- **Download buttons** (when succeeded): Download adapter ZIP, Download GGUF
- **Test Model panel**: Chat interface to test the model via Ollama

Auto-refreshes every 5 seconds while running.

What it connects to:
- `GET /jobs/{id}`
- `POST /jobs/{id}/sync`
- `GET /jobs/{id}/events`
- `GET /jobs/{id}/download`
- `POST /playground/run` (for the test panel)

---

### `/playground` — Side-by-Side Comparison

What you see:
- **Base model selector** — any inference-capable profile
- **Comparison model selector** — optional second model
- Prompt text area
- **Run comparison** button
- Two columns of output (base vs. fine-tuned)

Use case: After training, compare "Qwen 0.5B base" vs. "Qwen 0.5B fine-tuned" on the same prompt.

What it connects to:
- `GET /settings/model-profiles`
- `POST /playground/run`

---

### `/settings` — Model Configuration

Three tabs:

**Model Profiles tab** (default):
- Create new profiles: name, provider, model, category, description
- Edit or delete existing profiles
- Status badge: "Ready to use" (green) or "Needs provider setup" (amber)

**Workspace Defaults tab**:
- Set which profile is used by default for:
  - Playground base model
  - Playground comparison model
  - Agent base model
  - Agent reasoning model
  - Default training profile
- Save defaults button

**Providers tab**:
- Shows all 8 providers with configured/needs-setup status
- Shows base URL and capabilities (inference / fine-tuning)

What it connects to:
- `GET /settings/model-profiles`
- `POST /settings/model-profiles`
- `PATCH /settings/model-profiles/{id}`
- `PATCH /settings/model-profiles/defaults`
- `DELETE /settings/model-profiles/{id}`

---

### `/agent` — Autonomous Agent

What you see:
- Runtime status (Temporal connected / not)
- **Create agent run form**:
  - Goal (text description of what you want)
  - Dataset to work with (optional)
  - Base model for training
  - Agent model for reasoning
  - Evaluation prompt (optional)
- List of past agent runs
- Click a run → see step-by-step execution log

What the agent does:
1. Lists available datasets
2. Validates and uploads dataset if needed
3. Creates fine-tuning job
4. Waits for job to complete (polls every 30–60 seconds)
5. Runs playground evaluation
6. Reports final model and metrics

What it connects to:
- `GET /agent/runtime`
- `GET /agent/runs`
- `POST /agent/runs`
- `GET /agent/runs/{id}`

---

## 8. Core Workflows

### Workflow A: Fine-tune a Model (Manual)

```
1. Upload dataset
   → POST /datasets (multipart)
   → Backend validates each JSONL line
   → Status: VALID or INVALID

2. Create training job
   → POST /jobs { datasetId, baseModel, modelProvider, ... }
   → Backend checks dataset validity
   → For local: spawns Python subprocess (train.py)
   → For OpenAI: uploads file, calls fine_tuning.jobs.create()
   → For HF: submits Hugging Face Job

3. Monitor job
   → GET /jobs/{id} (every 5 seconds auto-refresh)
   → POST /jobs/{id}/sync (forces status refresh from provider)
   → Status progresses: queued → loading_model → training → succeeded

4. Download or use
   → GET /jobs/{id}/download → ZIP with adapter weights
   → Use in Ollama (if GGUF exported)
   → Use in playground for comparison
```

### Workflow B: Fine-tune a Model (Autonomous Agent)

```
1. Open /agent
2. Enter goal: "Fine-tune a coding model using my Python dataset"
3. Select dataset, base model, agent model
4. Click "Start agent run"

Agent automatically:
→ Calls list_datasets tool
→ Calls get_dataset tool to validate
→ Calls upload_dataset_to_openai if needed
→ Calls create_job tool
→ Calls wait_for_seconds (60s) in a loop
→ Calls sync_job until status = "succeeded"
→ Calls run_playground with your evaluation prompt
→ Returns summary with model name and metrics
```

### Workflow C: Compare Models in Playground

```
1. Open /playground
2. Select base model profile (e.g., "Ollama Qwen 0.5B")
3. Select comparison profile (e.g., "My Fine-tuned Model in Ollama")
4. Type a prompt
5. Click "Run comparison"
6. See both responses side-by-side
```

---

## 9. Model Providers

### Provider Capabilities Matrix

| Provider | Inference | Fine-tuning | Setup Required |
|----------|-----------|-------------|----------------|
| **Ollama** | ✅ Yes | ❌ No | Install Ollama + pull model |
| **OpenAI** | ✅ Yes | ✅ Yes (managed) | `OPENAI_API_KEY` |
| **Hugging Face** | ❌ No | ✅ Yes (cloud SFT) | `HF_TOKEN` + hub repos |
| **Local QLoRA** | ❌ No | ✅ Yes (on-device) | Training Python venv + GPU |
| **Groq** | ✅ Yes | ❌ No | `GROQ_API_KEY` (free tier) |
| **Google Gemini** | ✅ Yes | ❌ No | `GOOGLE_API_KEY` (free tier) |
| **Cerebras** | ✅ Yes | ❌ No | `CEREBRAS_API_KEY` (free tier) |
| **Together AI** | ✅ Yes | ❌ No | `TOGETHER_API_KEY` ($25 free) |

### How Provider Selection Works

```python
# python_api/services.py — get_model_provider()

1. If provider explicitly passed → validate and use it
2. Else read LLM_PROVIDER env var
3. Else if OPENAI_API_KEY set → use "openai"
4. Else → use "ollama" (default)
```

### How Inference Works (All Providers)

All 6 inference providers use the **OpenAI Python SDK** with different `base_url` values:

```python
# python_api/services.py — get_model_client()

Provider URLs:
  ollama    → http://127.0.0.1:11434/v1/
  openai    → https://api.openai.com/v1 (default)
  groq      → https://api.groq.com/openai/v1
  gemini    → https://generativelanguage.googleapis.com/v1beta/openai/
  cerebras  → https://api.cerebras.ai/v1
  together  → https://api.together.xyz/v1
```

All use `client.chat.completions.create(model=model, messages=[...])`.

---

## 10. Python API — All Endpoints

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health check |
| GET | `/agent/runtime` | Temporal runtime status + provider info |

### Dashboard

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard/summary` | Stats: dataset count, job count, recent activity |

### Model Profiles

| Method | Path | Description |
|--------|------|-------------|
| GET | `/settings/model-profiles` | List all profiles, defaults, provider statuses |
| POST | `/settings/model-profiles` | Create new profile |
| PATCH | `/settings/model-profiles/defaults` | Update workspace defaults |
| PATCH | `/settings/model-profiles/{id}` | Update a profile |
| DELETE | `/settings/model-profiles/{id}` | Delete a profile |

### Datasets

| Method | Path | Description |
|--------|------|-------------|
| GET | `/datasets` | List datasets (filter: `?validationStatus=VALID`) |
| POST | `/datasets` | Upload JSONL file (multipart form) |
| GET | `/datasets/{id}` | Full dataset detail with validation |
| POST | `/datasets/{id}/upload-to-openai` | Upload to OpenAI Files API |

### Jobs

| Method | Path | Description |
|--------|------|-------------|
| GET | `/jobs` | List all training jobs |
| POST | `/jobs` | Create new training job |
| GET | `/jobs/{id}` | Full job detail with events |
| POST | `/jobs/{id}/sync` | Force status refresh from provider |
| GET | `/jobs/{id}/events` | Job event log |
| POST | `/jobs/{id}/cancel` | Cancel active job |
| GET | `/jobs/{id}/download` | Download adapter/GGUF as ZIP |

### Playground

| Method | Path | Description |
|--------|------|-------------|
| POST | `/playground/run` | Run prompt on 1 or 2 models |

### Agent Runs

| Method | Path | Description |
|--------|------|-------------|
| GET | `/agent/runs` | List all agent runs |
| POST | `/agent/runs` | Start new agent run |
| GET | `/agent/runs/{id}` | Full run detail with steps |
| POST | `/agent/runs/{id}/cancel` | Cancel active run |

### Request/Response Formats

**POST /jobs — CreateJobRequest**
```json
{
  "datasetId": "string (required)",
  "baseModel": "string (e.g. Qwen/Qwen2.5-0.5B-Instruct)",
  "modelProvider": "local | openai | huggingface",
  "hyperparameters": {},
  "exportGguf": true,
  "ggufQuantization": "q4_k_m",
  "pushToOllama": false,
  "ollamaModelName": "",
  "numEpochs": 3,
  "learningRate": 0.0002,
  "perDeviceBatchSize": 2
}
```

**POST /playground/run — PlaygroundRunRequest**
```json
{
  "prompt": "string (required)",
  "baseModel": "string",
  "baseModelProvider": "ollama | openai | groq ...",
  "fineTunedModel": "string (optional second model)",
  "fineTunedModelProvider": "string"
}
```

**POST /agent/runs — CreateAgentRunRequest**
```json
{
  "goal": "string (required, describe what you want)",
  "datasetId": "string (optional)",
  "baseModel": "string",
  "baseModelProvider": "string",
  "evaluationPrompt": "string (optional)",
  "agentModel": "string",
  "agentModelProvider": "string"
}
```

---

## 11. Backend Services — Key Functions

**File**: `python_api/services.py` (~1800 lines)

All business logic lives here. Grouped by category:

### Provider Functions
```
get_model_provider(provider?)       → resolves which provider to use
get_model_client(provider, model)   → returns OpenAI SDK client for that provider
provider_supports_inference(p)      → True for ollama, openai, groq, gemini, cerebras, together
provider_supports_fine_tuning(p)    → True for openai, huggingface, local
provider_is_configured(p)           → True if required API key is set
get_provider_base_url(p)            → base URL string for that provider
```

### Model Profile Functions
```
_default_model_profiles()           → generates 22 seeded profiles at startup
list_model_profiles()               → returns profiles + defaults + provider statuses
create_model_profile(...)           → creates new profile, stores in DB/JSON
update_model_profile(id, ...)       → patches existing profile
delete_model_profile(id)            → removes profile
update_model_profile_defaults(...)  → sets default profile IDs
```

### Dataset Functions
```
create_dataset_record(filename, payload, name)  → saves JSONL + runs validation
list_datasets(validation_status?)               → list with optional filter
retrieve_dataset_detail(dataset_id)             → full record with validation
upload_dataset_record_to_openai(dataset_id)     → uploads file to OpenAI API
upload_dataset_record_to_huggingface(dataset_id)→ uploads to HF Hub
```

### Job Functions
```
create_job_record(dataset_id, base_model, ...)  → creates job, starts training process
list_jobs()                                      → all jobs with provider labels
retrieve_job_detail(job_id)                      → full job + events + results
sync_job_record(job_id)                          → refreshes status from provider API
list_job_events(job_id)                          → event log for a job
cancel_job_record(job_id)                        → cancels active job
build_job_download_package(job_id, type)         → creates ZIP for download
```

### Inference Functions
```
run_model(prompt, model, provider)              → single inference call
run_playground_prompt(prompt, base_model, ...)  → runs 1 or 2 models, returns both outputs
```

### Agent Functions
```
create_agent_run_record(...)        → creates Temporal workflow + DB record
list_agent_runs()                   → all agent runs
get_agent_run_detail(run_id)        → full snapshot with steps
save_agent_run_snapshot(snapshot)   → persists current workflow state
mark_agent_run_cancelled(run_id)    → cancels workflow + updates DB
```

### Dashboard
```
dashboard_summary()                 → aggregated stats for home page
```

---

## 12. Agent Orchestration (Temporal)

### What Temporal Does

Temporal is a durable workflow engine. When you start an agent run:
1. A `AgentRunWorkflow` is submitted to the Temporal server
2. The Temporal server persists the workflow state
3. Even if the Python API crashes and restarts, the workflow continues
4. The frontend can query workflow state at any time

### Workflow State Machine

```
        ┌──────────────────────┐
        │         queued       │
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │        running       │◄──────────────┐
        └──────────┬───────────┘               │
                   │ agent turn completes       │
        ┌──────────▼───────────┐               │
        │  tools executed?     │               │
        └──────────┬───────────┘               │
          ┌────────┴──────────┐                │
   need_wait_seconds?      done?               │
          │                   │                │
    ┌─────▼──────┐    ┌───────▼──────┐         │
    │  waiting   │    │ succeeded or │         │
    └─────┬──────┘    │   failed     │         │
          │           └──────────────┘         │
          └───────────────────────────────────►┘
              (after sleepSeconds)
```

### Agent Tools Available

The agent has 8 tools it can call:

| Tool | What it does |
|------|-------------|
| `list_datasets` | List all available datasets |
| `get_dataset` | Get full dataset details and validation |
| `upload_dataset_to_openai` | Upload dataset to OpenAI Files API |
| `list_jobs` | List all fine-tuning jobs |
| `create_job` | Create a new fine-tuning job |
| `sync_job` | Get latest status of a job |
| `run_playground` | Compare base vs. fine-tuned model |
| `wait_for_seconds` | Pause 10–300 seconds before next action |

### How a Single Agent Turn Works

```
1. Load snapshot (goal, current state, past steps)
2. Build messages list:
   - System prompt with instructions + tool definitions
   - History of steps taken so far
3. Call agent model (OpenAI or Ollama) with tools enabled
4. Parse tool_calls from response
5. Execute each tool in sequence
6. Add tool results to messages
7. Call model again with results
8. Repeat up to 8 loops
9. Return updated snapshot
```

---

## 13. Local QLoRA Training

### What QLoRA Is

**QLoRA** = Quantized Low-Rank Adaptation

Instead of fine-tuning all billions of parameters (expensive), it:
1. Loads the base model in 4-bit precision (uses ~4x less VRAM)
2. Adds small "LoRA adapter" matrices to specific layers
3. Only trains those small adapters (~17MB for a 0.5B model)
4. The result: a small adapter file you apply on top of the base model

### Training Process Step-by-Step

```
1. API receives POST /jobs with local provider

2. build_local_training_job_config() creates LocalQLoraJobConfig with:
   - paths for config, status, events, logs, output
   - hyperparameters (epochs, LR, batch size, lora_r, etc.)
   - export settings (GGUF, Ollama push)

3. spawn_local_training_job() runs:
   - Writes config JSON to disk
   - Spawns subprocess: LOCAL_TRAINING_PYTHON train.py config.json
   - Returns process PID

4. train.py runs (in .venv-train):
   a. load_model(base_model, lora_config) — downloads from HF if needed
   b. prepare_dataset(jsonl_path, tokenizer, eval_ratio)
   c. SFTTrainer.train() — LoRA training loop
   d. model.save_pretrained(output_dir/adapter) — saves weights
   e. Updates status.json at each step

5. Status stages (visible in UI):
   queued → loading_model → training → evaluating → saving → succeeded

6. If export_gguf=True (requires Unsloth):
   - Merges adapter into base model
   - Exports to GGUF format with quantization
   - If push_to_ollama=True: creates Modelfile + runs "ollama create"

7. resultFilesJson contains paths to:
   - local_adapter (adapter/ directory)
   - local_gguf (if exported)
   - local_metrics (metrics JSON)
   - local_log (training log)
```

### Your Completed Job: What You Have

Your job `045ce2858f844539a1d5bea764f2fe3a` produced:

```
Base model:  Qwen/Qwen2.5-0.5B-Instruct
Adapter:     17MB LoRA (rank 16, alpha 32)
Training:    3 epochs, eval_loss=1.99, perplexity=7.33
GPU used:    GPU index 2
Duration:    ~9 seconds (12 training records)
GGUF:        NOT exported (was not enabled)

Files produced:
  artifacts/adapter/adapter_model.safetensors  (17MB — the weights)
  artifacts/adapter/adapter_config.json        (LoRA config)
  artifacts/adapter/tokenizer.json             (tokenizer)
  artifacts/adapter/chat_template.jinja        (chat format)
  artifacts/checkpoint-6/                      (mid-training checkpoint)
  local_train_metrics.json                     (loss, perplexity)
  local_train_status.json                      (full status)
  local_train.log                              (stdout from trainer)
```

### Training Hyperparameters Explained

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_train_epochs` | 3 | More epochs = fits training data better, risk of overfitting |
| `learning_rate` | 2e-4 | Higher = faster but less stable. Lower = slower but more stable |
| `per_device_train_batch_size` | 2 | Higher = faster but needs more VRAM |
| `gradient_accumulation_steps` | 4 | Simulates larger batch without extra VRAM |
| `lora_r` | 16 | LoRA rank — higher means more parameters, more capacity |
| `lora_alpha` | 32 | LoRA scaling factor (usually 2x lora_r) |
| `lora_dropout` | 0.05 | Regularization to prevent overfitting |
| `max_seq_length` | 1024 | Max tokens per training sample |
| `warmup_ratio` | 0.1 | % of steps for learning rate warmup |

---

## 14. Data Persistence

### Storage Backend Selection

At startup, `store.py` checks for `DATABASE_URL`:
- **Found** → uses PostgreSQL with schema `agentic_app`
- **Not found** → uses `python_api/data/state.json`

### PostgreSQL Schema (7 Tables)

```sql
-- All tables are in the DATABASE_SCHEMA (default: agentic_app)

agentic_datasets
  dataset_id TEXT PRIMARY KEY
  name TEXT
  original_filename TEXT
  validation_status TEXT
  payload JSONB        ← full dataset record including validation
  created_at TIMESTAMPTZ
  updated_at TIMESTAMPTZ

agentic_jobs
  job_id TEXT PRIMARY KEY
  dataset_id TEXT
  status TEXT
  base_model TEXT
  model_provider TEXT
  payload JSONB        ← progress, events, result files, metrics
  created_at TIMESTAMPTZ
  updated_at TIMESTAMPTZ

agentic_job_events
  event_id TEXT PRIMARY KEY
  job_id TEXT
  level TEXT           ← info | warning | error
  event_type TEXT      ← started | progress | completed | failed
  payload JSONB
  created_at TIMESTAMPTZ

agentic_playground_runs
  run_id TEXT PRIMARY KEY
  base_model TEXT
  base_model_provider TEXT
  fine_tuned_model TEXT
  payload JSONB        ← base_output, tuned_output
  created_at TIMESTAMPTZ

agentic_agent_runs
  run_id TEXT PRIMARY KEY
  workflow_id TEXT
  status TEXT
  base_model TEXT
  agent_model TEXT
  payload JSONB        ← steps, snapshot, summary
  created_at TIMESTAMPTZ
  updated_at TIMESTAMPTZ

agentic_model_profiles
  profile_id TEXT PRIMARY KEY
  name TEXT
  provider TEXT
  model TEXT
  category TEXT
  payload JSONB
  created_at TIMESTAMPTZ
  updated_at TIMESTAMPTZ

agentic_workspace_settings
  setting_key TEXT PRIMARY KEY
  setting_value JSONB  ← arbitrary settings by key
```

### JSON Fallback Structure

When using `python_api/data/state.json`:

```json
{
  "datasets": [...],
  "jobs": [...],
  "job_events": [...],
  "playground_runs": [...],
  "agent_runs": [...],
  "model_profiles": [...],
  "model_profile_defaults": {
    "playgroundBaseProfileId": null,
    "playgroundCompareProfileId": null,
    "agentBaseProfileId": null,
    "agentModelProfileId": null,
    "jobBaseProfileId": null
  },
  "model_profiles_initialized": true
}
```

### Thread Safety

The `update_state(mutator)` function uses `STATE_LOCK` (threading.Lock) to prevent concurrent writes from corrupting state. All writes go through this function.

---

## 15. TypeScript Types & API Client

### Key Types (`lib/types.ts`)

```typescript
// Model provider — all supported providers
type ModelProvider =
  | "ollama" | "openai" | "huggingface" | "local"
  | "groq" | "gemini" | "cerebras" | "together";

// Profile categories
type ModelProfileCategory = "small" | "medium" | "large" | "thinking" | "custom";

// A single model profile (e.g., "Groq Llama 70B")
type ModelProfile = {
  id: string;
  name: string;
  provider: ModelProvider;
  providerLabel: string;        // Human-readable (e.g., "Groq")
  providerConfigured: boolean;  // Is the API key set?
  supportsInference: boolean;
  supportsFineTuning: boolean;
  model: string;                // Exact model ID
  category: ModelProfileCategory;
  description: string | null;
  createdAt: string;
  updatedAt: string;
};

// Status of a fine-tuning job
type FineTuneJobStatus =
  | "validating_files" | "queued" | "running"
  | "succeeded" | "failed" | "cancelled" | "paused" | "unknown";

// Validation result for a dataset
type DatasetValidationSummary = {
  totalRecords: number;
  validRecords: number;
  invalidRecords: number;
  errors: { line: number; message: string }[];
  warnings: { line?: number; message: string }[];
  examples: { line: number; preview: unknown }[];
};
```

### API Client (`lib/python-api.ts`)

```typescript
// Get the Python API base URL (from env or default)
getPythonApiBaseUrl(): string
// → reads NEXT_PUBLIC_PYTHON_API_URL or PYTHON_API_URL
// → defaults to http://127.0.0.1:8001

// Generic fetch wrapper — use this for all API calls
pythonApiFetch<T>(path: string, init?: RequestInit): Promise<T>
// → automatically handles error responses
// → throws Error with message from API error code
// → returns parsed JSON as type T

// Usage examples:
const datasets = await pythonApiFetch<Dataset[]>("/datasets");
const job = await pythonApiFetch<Job>(`/jobs/${id}`);
await pythonApiFetch("/jobs", { method: "POST", body: JSON.stringify({...}) });
```

---

## 16. Frontend Components

### Reusable UI Components (`components/ui/`)

| Component | File | Usage |
|-----------|------|-------|
| `<Button>` | button.tsx | Primary action button (variant: default, ghost, danger) |
| `<Card>` | card.tsx | Rounded container with padding |
| `<Input>` | input.tsx | Text input field |
| `<Textarea>` | textarea.tsx | Multi-line text input |
| `<ErrorAlert>` | error-alert.tsx | Red error message with title + description |
| `<LoadingState>` | loading-state.tsx | Skeleton loading (variant: text, list, detail) |
| `<EmptyState>` | empty-state.tsx | Empty list message with optional action button |
| `<Breadcrumb>` | breadcrumb.tsx | Navigation trail (e.g., Datasets → My Dataset) |
| `<JsonPreview>` | json-preview.tsx | Formatted JSON display |

### Layout Components

| Component | File | What it renders |
|-----------|------|----------------|
| `<AppShell>` | app-shell.tsx | Main grid: sidebar + header + content |
| `<Sidebar>` | sidebar.tsx | Left navigation with links and user info |
| `<Header>` | header.tsx | Top bar with "Upload dataset" and "New run" buttons |
| `<NavLinks>` | nav-links.tsx | Dashboard, Datasets, Jobs, Models, Playground, Agent |

### Page Client Components

Each page has a client component that handles data fetching and state:

| Component | File | Key State |
|-----------|------|-----------|
| `DashboardClient` | dashboard/dashboard-client.tsx | stats, recentActivity |
| `DatasetsPageClient` | dataset/datasets-page-client.tsx | datasets[] |
| `DatasetDetailClient` | dataset/dataset-detail-client.tsx | dataset, runtime |
| `CreateJobForm` | jobs/create-job-form.tsx | datasets[], profiles, form fields |
| `JobsPageClient` | jobs/jobs-page-client.tsx | jobs[], filter |
| `JobDetailClient` | jobs/job-detail-client.tsx | job, syncing, polling |
| `PlaygroundComparison` | playground/playground-comparison.tsx | result |
| `ModelProfilesSettings` | settings/model-profiles-settings.tsx | data, drafts, tab |
| `AgentPageClient` | agent/agent-page-client.tsx | runs[], form, runtime |

---

## 17. Docker Deployment

### Production Stack (`docker-compose.yml`)

```
Services started:
1. postgres      — PostgreSQL 16 database
2. temporal      — Temporal workflow server (uses postgres)
3. temporal-ui   — Temporal web UI at http://localhost:8080
4. python-api    — FastAPI backend at http://localhost:8001
5. web           — Next.js frontend at http://localhost:3000

Start command:
  docker compose up --build

Health checks ensure services start in correct order:
  postgres → temporal → python-api → web
```

### Dev Stack (`docker-compose.dev.yaml`)

Same services but:
- Source code mounted as volumes (changes without rebuild)
- Uses `next dev` for hot reload
- Slower cold start, instant code updates

```bash
docker compose -f docker-compose.dev.yaml up --build
```

### Important: Ollama with Docker

The containers cannot access your host's Ollama directly using `localhost`. Use:
```env
OLLAMA_BASE_URL="http://host.docker.internal:11434"
```
On Linux, you may need:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

---

## 18. Using a Fine-Tuned Model After Training

### What You Have After Training

The adapter (17MB LoRA) is **not a standalone model**. It is a small patch that modifies the base model's behavior. You have two ways to use it:

### Option A: Direct Python Inference (Works Right Now)

```python
# inference.py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

ADAPTER_PATH = "/mnt/nvme_disk2/User_data/vs95259v/Vikas/project/agentic/uploads_python/jobs/045ce2858f844539a1d5bea764f2fe3a/artifacts/adapter"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def ask(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(ask("Explain machine learning in simple terms"))
```

Run with: `source .venv-train/bin/activate && python inference.py`

Note: First run downloads ~1GB base model from Hugging Face.

### Option B: Convert to GGUF → Ollama → Use in App

**Step 1: Merge adapter into full model**
```python
# merge.py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

ADAPTER = ".../artifacts/adapter"
MERGED  = ".../artifacts/merged"

model = AutoPeftModelForCausalLM.from_pretrained(ADAPTER, torch_dtype=torch.float16)
merged = model.merge_and_unload()
merged.save_pretrained(MERGED)
AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(MERGED)
```

**Step 2: Convert to GGUF**
```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py .../artifacts/merged \
  --outfile ~/my-model.gguf --outtype q4_k_m
```

**Step 3: Register with Ollama**
```bash
cat > ~/Modelfile << 'EOF'
FROM /root/my-model.gguf
PARAMETER temperature 0.7
PARAMETER stop "<|im_end|>"
EOF

ollama create my-finetuned-qwen -f ~/Modelfile
ollama run my-finetuned-qwen "Your prompt here"
```

**Step 4: Add to app and use in Playground**
- Go to `/settings` → Model Profiles tab → Create profile
- Provider: Ollama, Model: `my-finetuned-qwen`, Category: custom
- Now available in Playground for side-by-side comparison

### Option C: Enable GGUF Export on Next Training Job

When creating the next job, check "Export to GGUF" and "Push to Ollama". The app handles everything automatically.

---

## 19. Troubleshooting

### Python API won't start

```bash
# Check Python venv is activated
source .venv/bin/activate
which python  # should point to .venv/bin/python

# Check required packages
pip install -r requirements.txt

# Check port is not in use
lsof -i :8001
```

### "Could not connect to Ollama"

```bash
# Check Ollama is running
ollama list

# Start Ollama if not running
ollama serve &

# Check the base URL matches
echo $OLLAMA_BASE_URL  # should be http://127.0.0.1:11434
```

### Local training fails immediately

```bash
# Check training Python is set correctly
echo $LOCAL_TRAINING_PYTHON

# Check training deps are installed
source .venv-train/bin/activate
python -c "import peft, transformers, trl; print('OK')"

# Check GPU is available
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"

# Allow CPU fallback for testing (very slow):
# LOCAL_TRAINING_ALLOW_CPU_FALLBACK="1"
```

### "Temporal not connected"

```bash
# Check TEMPORAL_AUTO_START_DEV_SERVER=1 in .env
# The Python API auto-starts Temporal when it launches

# Or start Temporal manually:
temporal server start-dev

# Check Temporal is accessible
temporal operator namespace list --address 127.0.0.1:7233
```

### Dataset validation fails

JSONL format must be one of:
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"instruction": "...", "output": "..."}
{"instruction": "...", "input": "...", "output": "..."}
```

Rules:
- One JSON object per line
- No blank lines between records
- UTF-8 encoding
- Minimum 10 records recommended

### State got corrupted (JSON mode)

```bash
# Reset to fresh state (loses all data)
rm python_api/data/state.json
# Restart API — it recreates automatically
```

---

## 20. Code Map — Who Calls What

This shows the chain of calls for each major action:

### User uploads a dataset

```
Browser → POST /datasets (multipart)
  python_api/main.py: upload_dataset()
    → create_dataset_record(filename, file_data, name)
      python_api/services.py: validate_jsonl_file(path)
        → checks each line: messages format OR instruction format
      → update_state(mutator)
        python_api/store.py: write to PostgreSQL or state.json
  → returns DatasetDetail JSON
```

### User starts a local training job

```
Browser → POST /jobs { datasetId, modelProvider: "local", numEpochs: 5 }
  python_api/main.py: create_job()
    → create_job_record(dataset_id, base_model, ..., num_epochs=5)
      python_api/services.py:
        → get_model_provider("local")
        → ensure_fine_tuning_available("local")
        → retrieve_dataset_detail(dataset_id)
        → _build_local_training_job_config(...)
          python_api/local_qlora/config.py: LocalQLoraJobConfig(...)
        → ensure_local_training_ready()
          python_api/local_qlora/__init__.py: check GPU, Python venv
        → spawn_local_training_job(config)
          python_api/local_qlora/__init__.py:
            → config.write() → local_train_config.json
            → subprocess.Popen([python, train.py, config_path])
        → update_state(mutator)  → saves job record
  → returns job JSON with status "queued"

  [In subprocess]:
  python_api/local_qlora/train.py:
    → load_model(base_model, lora_config)
    → prepare_dataset(jsonl_path, tokenizer, eval_ratio)
    → SFTTrainer.train()  [writes progress to status.json]
    → model.save_pretrained(output/adapter)
    → [if export_gguf]: export.py → convert to GGUF
    → [if push_to_ollama]: create Modelfile, run ollama create
```

### User runs playground comparison

```
Browser → POST /playground/run { prompt, baseModel, fineTunedModel }
  python_api/main.py: run_playground()
    → run_playground_prompt(prompt, base_model, fine_tuned_model, ...)
      python_api/services.py:
        → get_model_client("ollama", base_model)
          → OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1/")
        → client.chat.completions.create(model=base_model, messages=[...])
        → [if fineTunedModel]: same for second model
        → update_state(mutator)  → saves playground run
  → returns { baseOutput: "...", tunedOutput: "..." }
```

### Agent starts autonomous run

```
Browser → POST /agent/runs { goal: "Fine-tune a coding model" }
  python_api/main.py: create_agent_run()
    → create_agent_run_record(...)
      python_api/services.py:
        → temporal_runtime.client.start_workflow(AgentRunWorkflow, ...)
          python_api/agentic_workflow.py: AgentRunWorkflow.run(initial_snapshot)
            → loop:
                execute_activity(run_agent_turn_activity, snapshot)
                  python_api/agentic_activities.py: run_agent_turn_activity()
                    → call agent model with tools
                    → execute tools (list_datasets, create_job, sync_job, ...)
                    → return updated snapshot
                → check if terminal (succeeded/failed/cancelled)
                → if sleepSeconds: await asyncio.sleep(seconds)
                → repeat
        → update_state(mutator)  → saves agent run record

  [Browser polls]:
  GET /agent/runs/{id}
    → get_agent_run_detail(run_id)
      → temporal_runtime.client.get_workflow_handle().query("snapshot")
      → returns current snapshot with steps
```

---

## Quick Reference Card

```
Start dev servers:
  Terminal 1: source .venv/bin/activate && uvicorn python_api.main:app --port 8001 --reload
  Terminal 2: npm run dev
  Terminal 3: ollama serve

Key URLs:
  App:          http://localhost:3000
  Python API:   http://localhost:8001
  API Docs:     http://localhost:8001/docs  (FastAPI Swagger)
  Temporal UI:  http://localhost:8080 (Docker only)

Most important files:
  python_api/services.py     ← ALL business logic
  python_api/main.py         ← ALL API endpoints
  lib/types.ts               ← ALL TypeScript types
  lib/python-api.ts          ← Frontend HTTP client

Check if API is running:
  curl http://localhost:8001/health

Test a model from command line:
  curl -X POST http://localhost:8001/playground/run \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Hello!","baseModel":"qwen3:8b","baseModelProvider":"ollama"}'

View logs for a job:
  cat uploads_python/jobs/{job_id}/local_train.log

View job status:
  cat uploads_python/jobs/{job_id}/local_train_status.json | python3 -m json.tool
```

---

*Manual generated for Agentic Fine-Tune App — covers all code, workflows, and operational details.*
