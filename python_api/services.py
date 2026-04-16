from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from python_api.errors import ApiError
from python_api.store import (
    UPLOADS_DIR,
    get_agent_run,
    get_dataset,
    get_job,
    load_state,
    sort_desc,
    update_state,
    utc_now_iso,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


SUPPORTED_MODEL_PROVIDERS = {"ollama", "openai"}
PROVIDER_LABELS = {"ollama": "Ollama", "openai": "OpenAI"}
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OPENAI_BASE_MODEL = "gpt-4.1-mini-2025-04-14"


def _default_model_provider() -> str:
    configured = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if configured in SUPPORTED_MODEL_PROVIDERS:
        return configured
    return "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"


def get_model_provider() -> str:
    provider = _default_model_provider()
    configured = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if configured and configured not in SUPPORTED_MODEL_PROVIDERS:
        raise ApiError(
            "MODEL_PROVIDER_INVALID",
            f"Unsupported LLM_PROVIDER `{configured}`. Use `ollama` or `openai`.",
            500,
        )
    return provider


def get_provider_display_name(provider: str | None = None) -> str:
    resolved = provider or get_model_provider()
    return PROVIDER_LABELS.get(resolved, resolved.title())


def provider_supports_fine_tuning(provider: str | None = None) -> bool:
    return (provider or get_model_provider()) == "openai"


def _normalize_ollama_base_url(raw: str | None) -> str:
    normalized = (raw or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/"
    return f"{normalized}/v1/"


def get_provider_base_url(provider: str | None = None) -> str | None:
    resolved = provider or get_model_provider()
    if resolved == "ollama":
        return _normalize_ollama_base_url(os.environ.get("OLLAMA_BASE_URL"))
    return os.environ.get("OPENAI_BASE_URL")


def provider_is_configured(provider: str | None = None) -> bool:
    resolved = provider or get_model_provider()
    if resolved == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    return True


def provider_status_payload() -> dict[str, Any]:
    provider = get_model_provider()
    return {
        "modelProvider": provider,
        "modelProviderLabel": get_provider_display_name(provider),
        "providerBaseUrl": get_provider_base_url(provider),
        "providerConfigured": provider_is_configured(provider),
        "supportsFineTuning": provider_supports_fine_tuning(provider),
        "defaultBaseModel": DEFAULT_BASE_MODEL,
        "defaultAgentModel": DEFAULT_AGENT_MODEL,
    }


def ensure_fine_tuning_available() -> None:
    provider = get_model_provider()
    if provider_supports_fine_tuning(provider):
        return
    raise ApiError(
        "FINE_TUNING_UNSUPPORTED",
        (
            f"Managed fine-tuning is not available while `LLM_PROVIDER={provider}`. "
            f"{get_provider_display_name(provider)} mode works for local playground and agent inference. "
            "Switch to `LLM_PROVIDER=openai` and set `OPENAI_API_KEY` when you want remote fine-tuning jobs."
        ),
        400,
    )


if _default_model_provider() == "ollama":
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or os.environ.get("OLLAMA_BASE_MODEL") or "qwen3:8b"
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("OLLAMA_AGENT_MODEL") or DEFAULT_BASE_MODEL
else:
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or DEFAULT_OPENAI_BASE_MODEL
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("OPENAI_AGENT_MODEL") or "gpt-5.4-mini"


def validate_jsonl_file(file_path: Path) -> dict[str, Any]:
    if file_path.suffix.lower() != ".jsonl":
        return {
            "totalRecords": 0,
            "validRecords": 0,
            "invalidRecords": 1,
            "errors": [{"line": 0, "message": "File must use the .jsonl extension."}],
            "warnings": [],
            "examples": [],
        }

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    valid_records = 0

    lines = [line for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            errors.append({"line": index, "message": "Line is not valid JSON."})
            continue

        messages = record.get("messages")
        if not isinstance(messages, list):
            errors.append({"line": index, "message": "`messages` must be an array."})
            continue

        if not messages:
            errors.append({"line": index, "message": "`messages` must not be empty."})
            continue

        malformed_message = False
        assistant_messages: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                errors.append({"line": index, "message": "Each message must be an object."})
                malformed_message = True
                break
            if "role" not in message:
                errors.append({"line": index, "message": "Each message must contain `role`."})
                malformed_message = True
                break
            if "content" not in message:
                errors.append({"line": index, "message": "Each message must contain `content`."})
                malformed_message = True
                break
            if message["role"] == "assistant":
                assistant_messages.append(message)
            if message["role"] not in {"system", "user", "assistant", "developer", "tool"}:
                warnings.append({"line": index, "message": "Record contains a non-standard role value."})

        if malformed_message:
            continue

        if not assistant_messages:
            errors.append({"line": index, "message": "At least one assistant message is required."})
            continue

        empty_assistant = False
        for message in assistant_messages:
            content = message.get("content")
            if isinstance(content, str) and not content.strip():
                empty_assistant = True
            elif isinstance(content, list) and len(content) == 0:
                empty_assistant = True
            elif content is None:
                empty_assistant = True

        if empty_assistant:
            errors.append({"line": index, "message": "Assistant content should not be empty."})
            continue

        valid_records += 1
        if len(examples) < 3:
            examples.append({"line": index, "preview": record})

    return {
        "totalRecords": len(lines),
        "validRecords": valid_records,
        "invalidRecords": len(lines) - valid_records,
        "errors": errors,
        "warnings": warnings,
        "examples": examples,
    }


def get_model_client() -> OpenAI:
    if OpenAI is None:
        raise ApiError("MODEL_CLIENT_PACKAGE_MISSING", "The OpenAI-compatible Python package is not installed.", 500)

    provider = get_model_provider()
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ApiError("MODEL_PROVIDER_NOT_CONFIGURED", "OPENAI_API_KEY is not configured.", 503)
        base_url = get_provider_base_url(provider)
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    return OpenAI(
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        base_url=get_provider_base_url(provider),
    )


def get_openai_client() -> OpenAI:
    return get_model_client()


def upload_training_file_to_openai(file_path: str) -> Any:
    ensure_fine_tuning_available()
    client = get_model_client()
    with open(file_path, "rb") as file_handle:
        return client.files.create(file=file_handle, purpose="fine-tune")


def create_fine_tuning_job(training_file_id: str, base_model: str, method_config: dict[str, Any] | None = None) -> Any:
    ensure_fine_tuning_available()
    client = get_model_client()
    method: dict[str, Any] = {"type": "supervised"}
    if method_config:
        method["supervised"] = {"hyperparameters": method_config}
    return client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=base_model,
        method=method,
    )


def retrieve_fine_tuning_job(openai_job_id: str) -> Any:
    ensure_fine_tuning_available()
    return get_model_client().fine_tuning.jobs.retrieve(openai_job_id)


def list_fine_tuning_events(openai_job_id: str) -> Any:
    ensure_fine_tuning_available()
    return get_model_client().fine_tuning.jobs.list_events(fine_tuning_job_id=openai_job_id)


def cancel_fine_tuning_job(openai_job_id: str) -> Any:
    ensure_fine_tuning_available()
    return get_model_client().fine_tuning.jobs.cancel(openai_job_id)


def run_model(prompt: str, model: str) -> str:
    response = get_model_client().responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text or ""


def normalize_job_status(status: str | None) -> str:
    known = {"validating_files", "queued", "running", "succeeded", "failed", "cancelled", "paused"}
    if not status:
        return "unknown"
    return status if status in known else "unknown"


def map_remote_job(job: Any) -> dict[str, Any]:
    return {
        "status": normalize_job_status(getattr(job, "status", None)),
        "fineTunedModel": getattr(job, "fine_tuned_model", None),
        "trainedTokens": getattr(job, "trained_tokens", None),
        "estimatedFinishAt": (
            datetime.fromtimestamp(job.estimated_finish, tz=timezone.utc).isoformat()
            if getattr(job, "estimated_finish", None)
            else None
        ),
        "resultFilesJson": getattr(job, "result_files", None),
        "hyperparametersJson": getattr(job, "method", None),
        "finishedAt": (
            datetime.fromtimestamp(job.finished_at, tz=timezone.utc).isoformat()
            if getattr(job, "finished_at", None)
            else None
        ),
    }


def list_datasets(validation_status: str | None = None) -> list[dict[str, Any]]:
    state = load_state()
    datasets = sort_desc(state["datasets"])
    if validation_status:
        datasets = [dataset for dataset in datasets if dataset["validationStatus"] == validation_status]
    return datasets


def create_dataset_record(original_filename: str, payload: bytes, name: str | None = None) -> dict[str, Any]:
    dataset_id = uuid4().hex
    dataset_dir = UPLOADS_DIR / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    storage_path = dataset_dir / original_filename
    storage_path.write_bytes(payload)

    summary = validate_jsonl_file(storage_path)
    now = utc_now_iso()
    dataset = {
        "id": dataset_id,
        "name": name or original_filename.removesuffix(".jsonl"),
        "originalFilename": original_filename,
        "storagePath": str(storage_path),
        "fileSizeBytes": len(payload),
        "recordCount": summary["totalRecords"],
        "validationStatus": "INVALID" if summary["invalidRecords"] > 0 else "VALID",
        "validationSummary": summary,
        "openaiFileId": None,
        "uploadedToOpenAIAt": None,
        "createdAt": now,
        "updatedAt": now,
    }

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        state["datasets"].append(dataset)
        return deepcopy(dataset)

    return update_state(mutator)


def retrieve_dataset_detail(dataset_id: str) -> dict[str, Any]:
    state = load_state()
    dataset = get_dataset(state, dataset_id)
    if not dataset:
        raise ApiError("DATASET_NOT_FOUND", "Dataset not found.", 404)
    return deepcopy(dataset)


def upload_dataset_record_to_openai(dataset_id: str) -> dict[str, Any]:
    dataset = retrieve_dataset_detail(dataset_id)
    if dataset["validationStatus"] != "VALID":
        raise ApiError("DATASET_INVALID", "Dataset must validate successfully before upload.", 400)

    try:
        uploaded = upload_training_file_to_openai(dataset["storagePath"])
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("OPENAI_UPLOAD_FAILED", str(error), 500) from error

    now = utc_now_iso()

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        dataset_record = get_dataset(state, dataset_id)
        if not dataset_record:
            raise ApiError("DATASET_NOT_FOUND", "Dataset not found.", 404)
        dataset_record["openaiFileId"] = uploaded.id
        dataset_record["uploadedToOpenAIAt"] = now
        dataset_record["updatedAt"] = now
        return deepcopy(dataset_record)

    return update_state(mutator)


def list_jobs() -> list[dict[str, Any]]:
    state = load_state()
    jobs = []
    for job in sort_desc(state["jobs"]):
        dataset = get_dataset(state, job["datasetId"])
        jobs.append({**deepcopy(job), "datasetName": dataset["name"] if dataset else "Unknown dataset"})
    return jobs


def create_job_record(dataset_id: str, base_model: str, hyperparameters: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset = retrieve_dataset_detail(dataset_id)
    if dataset["validationStatus"] != "VALID":
        raise ApiError("DATASET_INVALID", "Only valid datasets can be used for fine-tuning.", 400)

    openai_file_id = dataset.get("openaiFileId")
    uploaded_to_openai_at = dataset.get("uploadedToOpenAIAt")

    if not openai_file_id:
        try:
            uploaded = upload_training_file_to_openai(dataset["storagePath"])
        except ApiError:
            raise
        except Exception as error:  # pragma: no cover
            raise ApiError("OPENAI_UPLOAD_FAILED", str(error), 500) from error
        openai_file_id = uploaded.id
        uploaded_to_openai_at = utc_now_iso()

    try:
        remote_job = create_fine_tuning_job(openai_file_id, base_model, hyperparameters)
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("JOB_CREATION_FAILED", str(error), 500) from error

    mapped = map_remote_job(remote_job)
    now = utc_now_iso()
    job = {
        "id": uuid4().hex,
        "datasetId": dataset_id,
        "openaiJobId": remote_job.id,
        "baseModel": base_model,
        "methodType": "supervised",
        "status": mapped["status"],
        "fineTunedModel": mapped["fineTunedModel"],
        "trainedTokens": mapped["trainedTokens"],
        "estimatedFinishAt": mapped["estimatedFinishAt"],
        "resultFilesJson": mapped["resultFilesJson"],
        "hyperparametersJson": hyperparameters,
        "lastSyncedAt": now,
        "createdAt": now,
        "updatedAt": now,
        "finishedAt": mapped["finishedAt"],
    }

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        dataset_record = get_dataset(state, dataset_id)
        if not dataset_record:
            raise ApiError("DATASET_NOT_FOUND", "Dataset not found.", 404)
        dataset_record["openaiFileId"] = openai_file_id
        dataset_record["uploadedToOpenAIAt"] = uploaded_to_openai_at
        dataset_record["updatedAt"] = now
        state["jobs"].append(job)
        state["job_events"].append(
            {
                "id": uuid4().hex,
                "jobId": job["id"],
                "level": "info",
                "message": "Fine-tuning job created.",
                "eventType": "job_created",
                "createdAt": now,
            }
        )
        return deepcopy(job)

    return update_state(mutator)


def retrieve_job_detail(job_id: str) -> dict[str, Any]:
    state = load_state()
    job = get_job(state, job_id)
    if not job:
        raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)
    dataset = get_dataset(state, job["datasetId"])
    events = sort_desc([deepcopy(event) for event in state["job_events"] if event["jobId"] == job_id])
    return {
        **deepcopy(job),
        "dataset": deepcopy(dataset) if dataset else None,
        "events": events,
    }


def sync_job_record(job_id: str) -> dict[str, Any]:
    job = retrieve_job_detail(job_id)

    try:
        remote_job = retrieve_fine_tuning_job(job["openaiJobId"])
        remote_events = list_fine_tuning_events(job["openaiJobId"])
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("JOB_SYNC_FAILED", str(error), 500) from error

    mapped = map_remote_job(remote_job)
    now = utc_now_iso()

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        job_record = get_job(state, job_id)
        if not job_record:
            raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)
        job_record.update(mapped)
        job_record["lastSyncedAt"] = now
        job_record["updatedAt"] = now

        existing_event_ids = {event["id"] for event in state["job_events"]}
        for remote_event in getattr(remote_events, "data", []):
            event_id = f"{job_id}-{getattr(remote_event, 'id', uuid4().hex)}"
            if event_id in existing_event_ids:
                continue
            created_timestamp = getattr(remote_event, "created_at", None)
            created_at = (
                datetime.fromtimestamp(created_timestamp, tz=timezone.utc).isoformat()
                if created_timestamp
                else utc_now_iso()
            )
            state["job_events"].append(
                {
                    "id": event_id,
                    "jobId": job_id,
                    "level": getattr(remote_event, "level", "info"),
                    "message": getattr(remote_event, "message", ""),
                    "eventType": getattr(remote_event, "type", None),
                    "createdAt": created_at,
                }
            )

        dataset = get_dataset(state, job_record["datasetId"])
        events = sort_desc([deepcopy(event) for event in state["job_events"] if event["jobId"] == job_id])
        return {
            **deepcopy(job_record),
            "dataset": deepcopy(dataset) if dataset else None,
            "events": events,
        }

    return update_state(mutator)


def list_job_events(job_id: str) -> list[dict[str, Any]]:
    state = load_state()
    job = get_job(state, job_id)
    if not job:
        raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)
    return sort_desc([deepcopy(event) for event in state["job_events"] if event["jobId"] == job_id])


def cancel_job_record(job_id: str) -> dict[str, Any]:
    job = retrieve_job_detail(job_id)

    try:
        remote_job = cancel_fine_tuning_job(job["openaiJobId"])
        mapped = map_remote_job(remote_job)
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("JOB_SYNC_FAILED", str(error), 500) from error

    now = utc_now_iso()

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        job_record = get_job(state, job_id)
        if not job_record:
            raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)
        job_record.update(mapped)
        job_record["lastSyncedAt"] = now
        job_record["updatedAt"] = now
        state["job_events"].append(
            {
                "id": uuid4().hex,
                "jobId": job_id,
                "level": "warning",
                "message": "Cancel requested.",
                "eventType": "job_cancelled",
                "createdAt": now,
            }
        )
        dataset = get_dataset(state, job_record["datasetId"])
        events = sort_desc([deepcopy(event) for event in state["job_events"] if event["jobId"] == job_id])
        return {
            **deepcopy(job_record),
            "dataset": deepcopy(dataset) if dataset else None,
            "events": events,
        }

    return update_state(mutator)


def run_playground_prompt(prompt: str, base_model: str, fine_tuned_model: str | None = None) -> dict[str, Any]:
    if not prompt.strip():
        raise ApiError("PLAYGROUND_RUN_FAILED", "Prompt is required.", 400)

    try:
        base_output = run_model(prompt, base_model)
        tuned_output = run_model(prompt, fine_tuned_model) if fine_tuned_model else None
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("PLAYGROUND_RUN_FAILED", str(error), 500) from error

    run = {
        "id": uuid4().hex,
        "baseModel": base_model,
        "fineTunedModel": fine_tuned_model,
        "prompt": prompt,
        "baseOutput": base_output,
        "tunedOutput": tuned_output,
        "createdAt": utc_now_iso(),
    }

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        state["playground_runs"].append(run)
        return {
            "id": run["id"],
            "baseOutput": base_output,
            "tunedOutput": tuned_output,
        }

    return update_state(mutator)


def dashboard_summary() -> dict[str, Any]:
    state = load_state()
    jobs = sort_desc(state["jobs"])
    completed_models = len([job for job in jobs if job.get("fineTunedModel")])
    running_jobs = len([job for job in jobs if job.get("status") == "running"])

    latest_activity = []
    for job in jobs[:5]:
        dataset = get_dataset(state, job["datasetId"])
        latest_activity.append(
            {
                "id": job["id"],
                "datasetName": dataset["name"] if dataset else "Unknown dataset",
                "baseModel": job["baseModel"],
                "status": job["status"],
                "createdAt": job["createdAt"],
            }
        )

    return {
        "totalDatasets": len(state["datasets"]),
        "totalJobs": len(state["jobs"]),
        "runningJobs": running_jobs,
        "completedModels": completed_models,
        "latestActivity": latest_activity,
    }


def create_agent_run_record(
    run_id: str,
    workflow_id: str,
    goal: str,
    base_model: str,
    dataset_id: str | None = None,
    evaluation_prompt: str | None = None,
    agent_model: str | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    snapshot = {
        "id": run_id,
        "workflowId": workflow_id,
        "status": "queued",
        "goal": goal,
        "baseModel": base_model,
        "datasetId": dataset_id,
        "evaluationPrompt": evaluation_prompt,
        "agentModel": agent_model or DEFAULT_AGENT_MODEL,
        "summary": None,
        "latestJobId": None,
        "fineTunedModel": None,
        "lastError": None,
        "latestResponseId": None,
        "sleepSeconds": None,
        "steps": [],
        "createdAt": now,
        "updatedAt": now,
        "finishedAt": None,
    }

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        state["agent_runs"].append(snapshot)
        return deepcopy(snapshot)

    return update_state(mutator)


def list_agent_runs() -> list[dict[str, Any]]:
    state = load_state()
    return sort_desc([deepcopy(run) for run in state["agent_runs"]], key="updatedAt")


def get_agent_run_detail(run_id: str) -> dict[str, Any]:
    state = load_state()
    agent_run = get_agent_run(state, run_id)
    if not agent_run:
        raise ApiError("AGENT_RUN_NOT_FOUND", "Agent run not found.", 404)
    return deepcopy(agent_run)


def save_agent_run_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(snapshot)
    snapshot["updatedAt"] = snapshot.get("updatedAt") or utc_now_iso()

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        existing = get_agent_run(state, snapshot["id"])
        if existing:
            created_at = existing.get("createdAt")
            existing.clear()
            existing.update(snapshot)
            if created_at:
                existing["createdAt"] = created_at
            target = existing
        else:
            state["agent_runs"].append(snapshot)
            target = snapshot
        return deepcopy(target)

    return update_state(mutator)


def mark_agent_run_cancelled(run_id: str) -> dict[str, Any]:
    now = utc_now_iso()

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        agent_run = get_agent_run(state, run_id)
        if not agent_run:
            raise ApiError("AGENT_RUN_NOT_FOUND", "Agent run not found.", 404)
        agent_run["status"] = "cancelled"
        agent_run["finishedAt"] = agent_run.get("finishedAt") or now
        agent_run["updatedAt"] = now
        return deepcopy(agent_run)

    return update_state(mutator)
