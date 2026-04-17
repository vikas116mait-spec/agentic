from __future__ import annotations

import os
import re
from hashlib import sha1
from pathlib import Path
from typing import Any
from urllib.parse import quote

from python_api.env import ensure_env_loaded
from python_api.errors import ApiError
from python_api.store import ROOT, utc_now_iso

try:
    from huggingface_hub import (
        HfApi,
        Volume,
        cancel_job,
        fetch_job_logs,
        get_token,
        inspect_job,
        run_uv_job,
        whoami,
    )
except ImportError:  # pragma: no cover
    HfApi = None
    Volume = None
    cancel_job = None
    fetch_job_logs = None
    get_token = None
    inspect_job = None
    run_uv_job = None
    whoami = None

ensure_env_loaded()


HF_HUB_URL = "https://huggingface.co"
HF_DATASET_REPO_DEFAULT = "agentic-datasets"
HF_JOBS_FLAVOR_DEFAULT = "a10g-large"
HF_JOBS_TIMEOUT_DEFAULT = "3h"
HF_JOBS_IMAGE_DEFAULT = "huggingface/trl"
HF_TRACKIO_SPACE_DEFAULT = "trackio"
HF_TRAINING_DEPENDENCIES = [
    "datasets>=2.18.0",
    "trl>=0.12.0",
    "peft>=0.7.0",
    "transformers>=4.36.0",
    "accelerate>=0.24.0",
    "trackio",
]


def huggingface_jobs_available() -> bool:
    return all(
        dependency is not None
        for dependency in (HfApi, Volume, cancel_job, fetch_job_logs, inspect_job, run_uv_job, whoami)
    )


def huggingface_token() -> str | None:
    explicit_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if explicit_token:
        return explicit_token
    if get_token is None:
        return None
    return get_token()


def huggingface_provider_is_configured() -> bool:
    return huggingface_jobs_available() and bool(huggingface_token())


def ensure_huggingface_provider() -> str:
    if not huggingface_jobs_available():
        raise ApiError(
            "MODEL_PROVIDER_NOT_CONFIGURED",
            "Install `huggingface_hub` in the Python environment to enable Hugging Face Jobs fine-tuning.",
            503,
        )

    token = huggingface_token()
    if not token:
        raise ApiError(
            "MODEL_PROVIDER_NOT_CONFIGURED",
            "Set `HF_TOKEN` in `.env` or log in with `huggingface_hub` to enable Hugging Face Jobs fine-tuning.",
            503,
        )
    return token


def huggingface_namespace(token: str) -> str:
    configured_namespace = (os.environ.get("HF_NAMESPACE") or os.environ.get("HF_JOBS_NAMESPACE") or "").strip()
    if configured_namespace:
        return configured_namespace
    if whoami is None:  # pragma: no cover
        raise ApiError("MODEL_PROVIDER_NOT_CONFIGURED", "Hugging Face account lookup is unavailable.", 503)
    identity = whoami(token=token, cache=False)
    return identity["name"]


def huggingface_dataset_repo_id(namespace: str) -> str:
    configured_repo = (os.environ.get("HF_DATASET_REPO") or "").strip()
    if configured_repo:
        return configured_repo
    return f"{namespace}/{HF_DATASET_REPO_DEFAULT}"


def huggingface_trackio_space_id(namespace: str) -> str:
    configured_space = (os.environ.get("HF_TRACKIO_SPACE_ID") or "").strip()
    if configured_space:
        return configured_space
    return f"{namespace}/{HF_TRACKIO_SPACE_DEFAULT}"


def huggingface_job_flavor() -> str:
    return (os.environ.get("HF_JOBS_FLAVOR") or HF_JOBS_FLAVOR_DEFAULT).strip()


def huggingface_job_timeout() -> str:
    return (os.environ.get("HF_JOBS_TIMEOUT") or HF_JOBS_TIMEOUT_DEFAULT).strip()


def huggingface_job_image() -> str:
    return (os.environ.get("HF_JOBS_IMAGE") or HF_JOBS_IMAGE_DEFAULT).strip()


def _safe_slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or fallback


def build_model_repo_id(namespace: str, dataset_name: str, base_model: str, job_id: str) -> str:
    configured_repo = (os.environ.get("HF_MODEL_REPO_ID") or "").strip()
    if configured_repo:
        return configured_repo

    dataset_slug = _safe_slug(dataset_name, "dataset")
    model_slug = _safe_slug(base_model.split("/")[-1], "model")
    return f"{namespace}/agentic-{dataset_slug}-{model_slug}-{job_id[:8]}"


def _repo_url(repo_id: str, *, repo_type: str = "model", path: str | None = None) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    if path:
        return f"{HF_HUB_URL}/{prefix}{repo_id}/tree/main/{quote(path.lstrip('/'))}"
    return f"{HF_HUB_URL}/{prefix}{repo_id}"


def upload_dataset_to_huggingface(dataset: dict[str, Any]) -> dict[str, str]:
    token = ensure_huggingface_provider()
    namespace = huggingface_namespace(token)
    api = HfApi(token=token)

    repo_id = dataset.get("huggingFaceDatasetRepoId") or huggingface_dataset_repo_id(namespace)
    repo_path = dataset.get("huggingFaceDatasetPath") or f"datasets/{dataset['id']}/train.jsonl"

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
        api.upload_file(
            path_or_fileobj=dataset["storagePath"],
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Upload dataset {dataset['id']} from Agentic",
        )
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("JOB_CREATION_FAILED", f"Could not upload the dataset to Hugging Face: {error}", 500) from error

    return {
        "namespace": namespace,
        "datasetRepoId": repo_id,
        "datasetPath": repo_path,
        "datasetUrl": _repo_url(repo_id, repo_type="dataset", path=repo_path.rsplit("/", 1)[0]),
        "uploadedAt": utc_now_iso(),
    }


def _training_script_path(job_id: str) -> Path:
    jobs_dir = ROOT / "uploads_python" / "jobs" / job_id
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir / "hf_train_sft.py"


def _resolve_hyperparameters(hyperparameters: dict[str, Any] | None) -> dict[str, str]:
    source = hyperparameters or {}
    defaults: dict[str, Any] = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "save_steps": 50,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }
    resolved = {key: source.get(key, default_value) for key, default_value in defaults.items()}
    return {key.upper(): str(value) for key, value in resolved.items()}


def write_training_script(job_id: str) -> str:
    script_path = _training_script_path(job_id)
    script_path.write_text(
        """
from __future__ import annotations

import os

import trackio
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


dataset = load_dataset("json", data_files="/training-data/train.jsonl", split="train")
dataset_size = len(dataset)

if dataset_size >= 20:
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    eval_strategy = "steps"
    eval_steps = max(10, min(100, dataset_size // 4))
else:
    train_dataset = dataset
    eval_dataset = None
    eval_strategy = "no"
    eval_steps = None

trackio_ready = False
try:
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", "agentic-training"),
        space_id=os.environ.get("TRACKIO_SPACE_ID"),
    )
    trackio_ready = True
except Exception as error:
    print(f"Trackio init skipped: {error}")

config_kwargs = {
    "output_dir": "training-output",
    "push_to_hub": True,
    "hub_model_id": os.environ["HF_MODEL_REPO_ID"],
    "hub_strategy": "every_save",
    "num_train_epochs": float(os.environ.get("NUM_TRAIN_EPOCHS", "3")),
    "per_device_train_batch_size": int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "2")),
    "gradient_accumulation_steps": int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")),
    "learning_rate": float(os.environ.get("LEARNING_RATE", "2e-5")),
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": int(os.environ.get("SAVE_STEPS", "50")),
    "save_total_limit": 2,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": True,
    "report_to": "trackio" if trackio_ready else "none",
    "run_name": os.environ.get("TRACKIO_RUN_NAME", "agentic-training-run"),
}

if eval_strategy == "steps":
    config_kwargs["eval_strategy"] = "steps"
    config_kwargs["eval_steps"] = eval_steps
else:
    config_kwargs["eval_strategy"] = "no"

eos_token = os.environ.get("EOS_TOKEN")
if eos_token:
    config_kwargs["eos_token"] = eos_token

training_args = SFTConfig(**config_kwargs)

peft_config = LoraConfig(
    r=int(os.environ.get("LORA_R", "16")),
    lora_alpha=int(os.environ.get("LORA_ALPHA", "32")),
    lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

trainer = SFTTrainer(
    model=os.environ["BASE_MODEL_ID"],
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()
trainer.push_to_hub()

if trackio_ready:
    trackio.finish()

print(f"Model saved to: https://huggingface.co/{os.environ['HF_MODEL_REPO_ID']}")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return str(script_path)


def submit_training_job(
    *,
    job_id: str,
    dataset: dict[str, Any],
    base_model: str,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token = ensure_huggingface_provider()
    upload_metadata = upload_dataset_to_huggingface(dataset)
    namespace = upload_metadata["namespace"]
    dataset_repo_id = upload_metadata["datasetRepoId"]
    dataset_path = upload_metadata["datasetPath"]

    model_repo_id = build_model_repo_id(namespace, dataset["name"], base_model, job_id)
    trackio_space_id = huggingface_trackio_space_id(namespace)
    script_path = write_training_script(job_id)
    hyperparameter_env = _resolve_hyperparameters(hyperparameters)

    env = {
        "BASE_MODEL_ID": base_model,
        "HF_MODEL_REPO_ID": model_repo_id,
        "HF_DATASET_REPO_ID": dataset_repo_id,
        "TRACKIO_PROJECT": os.environ.get("HF_TRACKIO_PROJECT", "agentic-training"),
        "TRACKIO_RUN_NAME": f"{_safe_slug(dataset['name'], 'dataset')}-{job_id[:8]}",
        "TRACKIO_SPACE_ID": trackio_space_id,
        **hyperparameter_env,
    }
    if "qwen" in base_model.lower():
        env["EOS_TOKEN"] = "<|im_end|>"

    dataset_mount_path = dataset_path.rsplit("/", 1)[0]
    try:
        job = run_uv_job(
            script=script_path,
            dependencies=HF_TRAINING_DEPENDENCIES,
            image=huggingface_job_image(),
            env=env,
            secrets={"HF_TOKEN": token},
            flavor=huggingface_job_flavor(),
            timeout=huggingface_job_timeout(),
            volumes=[Volume(type="dataset", source=dataset_repo_id, path=dataset_mount_path, mount_path="/training-data")],
            namespace=namespace,
            token=token,
        )
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("JOB_CREATION_FAILED", f"Could not submit the Hugging Face training job: {error}", 500) from error

    return {
        "status": normalize_huggingface_stage(job.status.stage if job.status else None),
        "providerJobId": job.id,
        "providerJobUrl": job.url,
        "providerNamespace": namespace,
        "fineTunedModel": model_repo_id,
        "modelRepoId": model_repo_id,
        "modelRepoUrl": _repo_url(model_repo_id),
        "datasetRepoId": dataset_repo_id,
        "datasetRepoPath": dataset_path,
        "datasetRepoUrl": upload_metadata["datasetUrl"],
        "trackioUrl": f"{HF_HUB_URL}/spaces/{trackio_space_id}",
        "resultFilesJson": [
            {"type": "model_repo", "repoId": model_repo_id, "url": _repo_url(model_repo_id)},
            {"type": "dataset_repo", "repoId": dataset_repo_id, "path": dataset_path, "url": upload_metadata["datasetUrl"]},
        ],
        "trainingBackend": "huggingface_jobs",
        "statusMessage": job.status.message if job.status else None,
        "uploadedToHuggingFaceAt": upload_metadata["uploadedAt"],
    }


def normalize_huggingface_stage(stage: str | None) -> str:
    if not stage:
        return "unknown"

    normalized = stage.upper()
    if normalized == "RUNNING":
        return "running"
    if normalized == "COMPLETED":
        return "succeeded"
    if normalized == "ERROR":
        return "failed"
    if normalized in {"CANCELED", "DELETED"}:
        return "cancelled"
    return "queued"


def inspect_training_job(provider_job_id: str, provider_namespace: str | None = None) -> dict[str, Any]:
    token = ensure_huggingface_provider()
    namespace = provider_namespace or huggingface_namespace(token)
    try:
        job = inspect_job(job_id=provider_job_id, namespace=namespace, token=token)
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("JOB_SYNC_FAILED", f"Could not inspect the Hugging Face job: {error}", 500) from error

    return {
        "status": normalize_huggingface_stage(job.status.stage if job.status else None),
        "providerJobId": job.id,
        "providerJobUrl": job.url,
        "providerNamespace": namespace,
        "statusMessage": job.status.message if job.status else None,
    }


def collect_training_job_events(
    *,
    workspace_job_id: str,
    provider_job_id: str,
    provider_namespace: str | None,
    existing_event_ids: set[str],
) -> list[dict[str, Any]]:
    token = ensure_huggingface_provider()
    namespace = provider_namespace or huggingface_namespace(token)
    try:
        logs = list(fetch_job_logs(job_id=provider_job_id, namespace=namespace, follow=False, token=token))
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("JOB_SYNC_FAILED", f"Could not fetch Hugging Face job logs: {error}", 500) from error

    new_events: list[dict[str, Any]] = []
    for index, raw_line in enumerate(logs):
        line = raw_line.strip()
        if not line:
            continue
        digest = sha1(line.encode("utf-8")).hexdigest()[:16]
        event_id = f"{workspace_job_id}-hf-{index}-{digest}"
        if event_id in existing_event_ids:
            continue
        new_events.append(
            {
                "id": event_id,
                "jobId": workspace_job_id,
                "level": "info",
                "message": line,
                "eventType": "provider_log",
                "createdAt": utc_now_iso(),
            }
        )
    return new_events


def cancel_training_job(provider_job_id: str, provider_namespace: str | None = None) -> dict[str, Any]:
    token = ensure_huggingface_provider()
    namespace = provider_namespace or huggingface_namespace(token)
    try:
        cancel_job(job_id=provider_job_id, namespace=namespace, token=token)
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("JOB_SYNC_FAILED", f"Could not cancel the Hugging Face job: {error}", 500) from error

    return inspect_training_job(provider_job_id, namespace)
