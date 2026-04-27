from __future__ import annotations

import json
import logging
import os
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from python_api import ollama_manager
from python_api.env import ensure_env_loaded
from python_api.errors import ApiError
from python_api.huggingface_jobs import (
    cancel_training_job,
    collect_training_job_events,
    huggingface_provider_is_configured,
    inspect_training_job,
    submit_training_job,
    upload_dataset_to_huggingface,
)
from python_api.local_qlora import (
    cancel_local_training_job,
    collect_local_training_events,
    ensure_local_training_ready,
    inspect_local_training_runtime,
    inspect_local_training_job,
    local_training_provider_is_configured,
    spawn_local_training_job,
)
from python_api.local_qlora.config import LocalQLoraJobConfig
from python_api.store import (
    DATASETS_DIR,
    JOBS_DIR,
    ROOT,
    UPLOADS_DIR,
    get_agent_run,
    get_dataset,
    get_job,
    get_model_profile,
    load_state,
    sort_desc,
    update_state,
    utc_now_iso,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

ensure_env_loaded()


logger = logging.getLogger(__name__)


SUPPORTED_MODEL_PROVIDERS = {"ollama", "openai", "huggingface", "local", "groq", "gemini", "cerebras", "together"}
PROVIDER_LABELS = {
    "ollama": "Ollama",
    "openai": "OpenAI",
    "huggingface": "Hugging Face Jobs (Paid Cloud)",
    "local": "Local GPU QLoRA",
    "groq": "Groq",
    "gemini": "Google Gemini",
    "cerebras": "Cerebras",
    "together": "Together AI",
}
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_SMALL_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_MEDIUM_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_LARGE_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_THINKING_MODEL = "deepseek-r1:8b"
DEFAULT_OPENAI_BASE_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_HUGGINGFACE_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_LOCAL_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_PROFILE_CATEGORIES = {"small", "medium", "large", "thinking", "custom"}
MODEL_PROFILE_DEFAULT_KEYS = {
    "playgroundBaseProfileId",
    "playgroundCompareProfileId",
    "agentBaseProfileId",
    "agentModelProfileId",
    "jobBaseProfileId",
}
GATED_MODEL_REPLACEMENTS = {
    "google/gemma-2-2b-it": [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
    ],
    "meta-llama/Llama-3.2-3B-Instruct": [
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
}
SEEDED_GATED_PROFILE_MODELS = {
    "profile-local-gemma-2b": "google/gemma-2-2b-it",
    "profile-local-llama-32-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "profile-hf-gemma-2b": "google/gemma-2-2b-it",
}
SEEDED_DEFAULT_PROFILE_IDS = {"profile-open-source-sft", "profile-local-qlora"}
LEGACY_FREE_TUNING_DEFAULT_PROFILE_IDS = {"profile-large", "profile-open-source-sft"}
LEGACY_PAID_SEEDED_PROFILES = {
    "profile-large": ("openai", DEFAULT_OPENAI_BASE_MODEL),
    "profile-thinking": ("openai", "gpt-5.4-mini"),
}
_READY_OLLAMA_MODELS: set[str] = set()


def _slugify_archive_name(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "model"


def _ollama_model_family_slug(model_id: str) -> str:
    family = (model_id or "").split("/")[-1]
    for suffix in ("-Instruct", "-instruct", "-IT", "-it", "-Chat", "-chat"):
        if family.endswith(suffix):
            family = family[: -len(suffix)]
            break
    return _slugify_archive_name(family)[:24]


def _default_local_ollama_model_name(dataset_name: str, base_model: str, job_id: str) -> str:
    dataset_slug = _slugify_archive_name(dataset_name)[:32]
    model_slug = _ollama_model_family_slug(base_model)
    return "-".join(part for part in (dataset_slug, model_slug, job_id[:8]) if part)


def _latest_path_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    latest = path.stat().st_mtime
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                latest = max(latest, child.stat().st_mtime)
    return latest


def _resolve_local_job_path(raw_path: str | None) -> Path:
    if not raw_path:
        raise ApiError("NOT_FOUND", "Local training artifact path is missing.", 404)

    resolved = Path(raw_path).resolve()
    jobs_root = JOBS_DIR.resolve()
    try:
        resolved.relative_to(jobs_root)
    except ValueError as error:
        raise ApiError("FORBIDDEN", "Requested artifact path is outside the local jobs directory.", 403) from error

    if not resolved.exists():
        raise ApiError("NOT_FOUND", "Requested local training artifact was not found on disk.", 404)

    return resolved


def _resolve_dataset_path(raw_path: str | None) -> Path:
    if not raw_path:
        raise ApiError("NOT_FOUND", "Dataset file path is missing.", 404)

    resolved = Path(raw_path).resolve()
    datasets_root = DATASETS_DIR.resolve()
    try:
        resolved.relative_to(datasets_root)
    except ValueError as error:
        raise ApiError("FORBIDDEN", "Requested dataset path is outside the datasets directory.", 403) from error

    if not resolved.exists() or not resolved.is_file():
        raise ApiError("NOT_FOUND", "Requested dataset file was not found on disk.", 404)

    return resolved


def _is_seeded_gated_profile(profile: dict[str, Any]) -> bool:
    expected_model = SEEDED_GATED_PROFILE_MODELS.get(profile.get("id"))
    return bool(expected_model and profile.get("model") == expected_model)


def _preferred_accessible_model(model: str) -> str:
    alternatives = GATED_MODEL_REPLACEMENTS.get(model)
    return alternatives[0] if alternatives else model


def _is_legacy_paid_seeded_profile(profile: dict[str, Any]) -> bool:
    legacy = LEGACY_PAID_SEEDED_PROFILES.get(profile.get("id"))
    return bool(legacy and profile.get("provider") == legacy[0] and profile.get("model") == legacy[1])


def _sanitize_seeded_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if profile.get("id") not in SEEDED_DEFAULT_PROFILE_IDS:
        return profile

    safe_model = _preferred_accessible_model(profile.get("model", ""))
    if safe_model and safe_model != profile.get("model"):
        profile["model"] = safe_model
        profile["updatedAt"] = utc_now_iso()
    return profile


def _profile_uses_gated_fine_tuning_model(profile: dict[str, Any]) -> bool:
    provider = profile.get("provider")
    model = profile.get("model")
    return provider in {"huggingface", "local"} and model in GATED_MODEL_REPLACEMENTS


def _ensure_accessible_profile_model(model: str, provider: str) -> None:
    if provider not in {"huggingface", "local"}:
        return

    alternatives = GATED_MODEL_REPLACEMENTS.get(model)
    if not alternatives:
        return

    raise ApiError(
        "MODEL_ACCESS_RESTRICTED",
        (
            f"`{model}` requires separate Hugging Face approval and cannot be saved as a supported fine-tuning profile here. "
            f"Use one of: {', '.join(alternatives)}."
        ),
        400,
    )


def _ensure_accessible_fine_tuning_model(base_model: str, model_provider: str) -> None:
    if model_provider not in {"huggingface", "local"}:
        return

    alternatives = GATED_MODEL_REPLACEMENTS.get(base_model)
    if not alternatives:
        return

    raise ApiError(
        "MODEL_ACCESS_RESTRICTED",
        (
            f"`{base_model}` requires separate Hugging Face approval and is not available in this workspace. "
            f"Use one of: {', '.join(alternatives)}."
        ),
        400,
    )


def get_model_provider(provider: str | None = None) -> str:
    if provider is not None:
        normalized = provider.strip().lower()
        if normalized not in SUPPORTED_MODEL_PROVIDERS:
            raise ApiError(
                "MODEL_PROVIDER_INVALID",
                f"Unsupported provider `{provider}`. Supported: ollama, openai, huggingface, local, groq, gemini, cerebras, together.",
                400,
            )
        return normalized

    configured = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if configured in SUPPORTED_MODEL_PROVIDERS:
        return configured
    if configured:
        raise ApiError(
            "MODEL_PROVIDER_INVALID",
            f"Unsupported LLM_PROVIDER `{configured}`. Supported: ollama, openai, huggingface, local, groq, gemini, cerebras, together.",
            500,
        )
    return "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"


def get_provider_display_name(provider: str | None = None) -> str:
    resolved = get_model_provider(provider)
    return PROVIDER_LABELS.get(resolved, resolved.title())


def provider_supports_inference(provider: str | None = None) -> bool:
    return get_model_provider(provider) in {"ollama", "openai", "groq", "gemini", "cerebras", "together"}


def provider_supports_fine_tuning(provider: str | None = None) -> bool:
    return get_model_provider(provider) in {"openai", "huggingface", "local"}


def _normalize_ollama_base_url(raw: str | None) -> str:
    normalized = (raw or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/"
    return f"{normalized}/v1/"


def _ollama_cli_env() -> dict[str, str]:
    env = os.environ.copy()
    parsed = urlparse(_normalize_ollama_base_url(os.environ.get("OLLAMA_BASE_URL")))
    if parsed.scheme and parsed.netloc:
        env["OLLAMA_HOST"] = f"{parsed.scheme}://{parsed.netloc}"
    return env


def _run_ollama_cli(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["ollama", *args],
            cwd=str(ROOT),
            env=_ollama_cli_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise ApiError(
            "MODEL_PROVIDER_NOT_CONFIGURED",
            "The `ollama` CLI is not installed or not on PATH, so local models cannot be prepared automatically.",
            503,
        ) from error


def _find_local_tuned_job_for_ollama_name(normalized_model: str) -> dict[str, Any] | None:
    """Return the most recent local training job whose registered Ollama name matches.

    Locally fine-tuned models are registered with the user's Ollama server via
    `ollama create`; they are never published to the Ollama registry, so a
    registry pull can never succeed for them. We use this lookup to distinguish
    "unknown registry model" from "known local tuned model that isn't registered
    with the Ollama instance the backend is talking to".
    """

    if not normalized_model:
        return None
    try:
        state = load_state()
    except Exception:  # pragma: no cover - defensive: never break pull path on state errors
        return None

    matches = [
        job
        for job in state.get("jobs", [])
        if (job.get("ollamaModelName") or "").strip() == normalized_model
    ]
    if not matches:
        return None
    matches.sort(key=lambda job: job.get("updatedAt") or job.get("createdAt") or "", reverse=True)
    return matches[0]


def _extract_local_artifact_path(job: dict[str, Any], artifact_type: str) -> str | None:
    for artifact in job.get("resultFilesJson") or []:
        if isinstance(artifact, dict) and artifact.get("type") == artifact_type:
            path = artifact.get("path")
            if isinstance(path, str) and path.strip():
                return path.strip()
    return None


def _extract_local_modelfile_path(job: dict[str, Any]) -> str | None:
    return _extract_local_artifact_path(job, "local_modelfile")


def _candidate_job_dirs(job: dict[str, Any]) -> list[Path]:
    """Return plausible on-disk roots for a job's artifacts.

    Covers environments where JOBS_DIR was relocated via `AGENTIC_UPLOADS_DIR`
    between training and now, by consulting any recorded absolute paths first.
    """

    candidates: list[Path] = []
    for field in ("localArtifactsPath", "localStatusPath", "localConfigPath", "localLogPath"):
        raw = job.get(field)
        if isinstance(raw, str) and raw.strip():
            try:
                parent = Path(raw).expanduser().resolve().parent
            except OSError:
                continue
            if parent and parent not in candidates:
                candidates.append(parent)
    job_id = (job.get("id") or "").strip()
    if job_id:
        try:
            derived = (JOBS_DIR / job_id).resolve()
            if derived not in candidates:
                candidates.append(derived)
        except OSError:
            pass
    return candidates


def _find_on_disk_modelfile(job: dict[str, Any]) -> str | None:
    """Locate a Modelfile on disk for this job, even if `resultFilesJson` is stale."""

    recorded = _extract_local_modelfile_path(job)
    if recorded and Path(recorded).is_file():
        return recorded

    for root in _candidate_job_dirs(job):
        if not root.exists() or not root.is_dir():
            continue
        try:
            for candidate in root.rglob("Modelfile"):
                if candidate.is_file():
                    return str(candidate)
        except OSError:
            continue
    return None


def _extract_training_warning_messages(job: dict[str, Any]) -> list[str]:
    progress = job.get("progressJson")
    if not isinstance(progress, dict):
        return []
    warnings = progress.get("warnings") or []
    messages: list[str] = []
    for entry in warnings:
        if isinstance(entry, str) and entry.strip():
            messages.append(entry.strip())
        elif isinstance(entry, dict):
            text = entry.get("message") or entry.get("warning") or entry.get("detail")
            if isinstance(text, str) and text.strip():
                messages.append(text.strip())
    return messages


def _ollama_host_hint() -> str:
    parsed = urlparse(_normalize_ollama_base_url(os.environ.get("OLLAMA_BASE_URL")))
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return DEFAULT_OLLAMA_BASE_URL


def ensure_ollama_model_available(model: str) -> None:
    normalized_model = (model or "").strip()
    if not normalized_model or normalized_model in _READY_OLLAMA_MODELS:
        return

    show_result = _run_ollama_cli("show", normalized_model)
    if show_result.returncode == 0:
        _READY_OLLAMA_MODELS.add(normalized_model)
        return

    # If this name corresponds to a locally fine-tuned job, skip the registry
    # pull (which can never succeed) and try to re-register it from an on-disk
    # Modelfile. Fall back to an actionable error if that isn't possible.
    local_job = _find_local_tuned_job_for_ollama_name(normalized_model)
    if local_job is not None:
        host_hint = _ollama_host_hint()
        modelfile_path = _find_on_disk_modelfile(local_job)

        if modelfile_path:
            logger.info(
                "Ollama model %s not found on %s; auto-registering from local Modelfile %s.",
                normalized_model,
                host_hint,
                modelfile_path,
            )
            create_result = _run_ollama_cli("create", normalized_model, "-f", modelfile_path)
            if create_result.returncode == 0:
                _READY_OLLAMA_MODELS.add(normalized_model)
                return
            create_message = (
                create_result.stderr.strip()
                or create_result.stdout.strip()
                or "unknown error"
            )
            raise ApiError(
                "MODEL_PROVIDER_NOT_CONFIGURED",
                (
                    f"Fine-tuned model `{normalized_model}` is not registered with the Ollama server at "
                    f"{host_hint} and auto-registration from `{modelfile_path}` failed: {create_message}. "
                    f"Register it manually with: `ollama create {normalized_model} -f {modelfile_path}`."
                ),
                503,
            )

        # No Modelfile anywhere. Build the most informative error we can so the
        # user understands *why* the model cannot be loaded.
        progress = local_job.get("progressJson") if isinstance(local_job.get("progressJson"), dict) else {}
        ollama_registered = bool(progress.get("ollamaRegistered"))
        training_warnings = _extract_training_warning_messages(local_job)
        adapter_path = _extract_local_artifact_path(local_job, "local_adapter")
        gguf_path = _extract_local_artifact_path(local_job, "local_gguf")
        status = (local_job.get("status") or "").strip() or "unknown"

        if ollama_registered:
            # Was registered at some point but the Modelfile / Ollama blob has
            # since been deleted and we have nothing to re-register from.
            reason = (
                f"It was previously registered with Ollama, but the Modelfile artifact is "
                f"no longer on disk, so auto-re-registration is not possible."
            )
        else:
            failure_detail = "; ".join(training_warnings[:3]) if training_warnings else (
                "GGUF export or Ollama registration did not complete."
            )
            reason = (
                f"Ollama registration was never completed during training ({failure_detail}), "
                f"so there is nothing to load."
            )

        recovery_hints: list[str] = []
        if gguf_path:
            recovery_hints.append(
                f"Run `ollama create {normalized_model} -f <Modelfile>` where `<Modelfile>` points "
                f"at (or is generated for) the existing GGUF at `{gguf_path}`."
            )
        elif adapter_path:
            recovery_hints.append(
                f"The LoRA adapter is still saved at `{adapter_path}`. Re-run the job with GGUF "
                f"export enabled (or manually export the adapter to GGUF) and then register it with "
                f"`ollama create {normalized_model} -f <path-to-new-Modelfile>`."
            )
        else:
            recovery_hints.append(
                "Retrain the model with `pushToOllama=true` and ensure Unsloth is installed in "
                "LOCAL_TRAINING_PYTHON so the GGUF export step can complete."
            )
        recovery_hints.append(
            f"Or verify OLLAMA_BASE_URL ({host_hint}) points at the Ollama server where the model "
            f"was originally created."
        )

        raise ApiError(
            "MODEL_PROVIDER_NOT_CONFIGURED",
            (
                f"Fine-tuned model `{normalized_model}` cannot be loaded from the Ollama server at "
                f"{host_hint} (training job status: {status}, ollamaRegistered: {ollama_registered}). "
                f"{reason} Next steps: " + " ".join(recovery_hints)
            ),
            503,
        )

    pull_result = _run_ollama_cli("pull", normalized_model)
    if pull_result.returncode != 0:
        message = pull_result.stderr.strip() or pull_result.stdout.strip() or show_result.stderr.strip() or show_result.stdout.strip()
        host_hint = _ollama_host_hint()
        raise ApiError(
            "MODEL_PROVIDER_NOT_CONFIGURED",
            (
                f"Could not pull Ollama model `{normalized_model}` from the registry via the Ollama "
                f"server at {host_hint}. {message} "
                f"If this is a locally fine-tuned model, register it with "
                f"`ollama create {normalized_model} -f <path>/Modelfile` first."
            ).strip(),
            503,
        )

    _READY_OLLAMA_MODELS.add(normalized_model)


def get_provider_base_url(provider: str | None = None) -> str | None:
    resolved = get_model_provider(provider)
    if resolved == "ollama":
        return _normalize_ollama_base_url(os.environ.get("OLLAMA_BASE_URL"))
    if resolved == "huggingface":
        return "https://huggingface.co"
    if resolved == "local":
        return None
    if resolved == "groq":
        return GROQ_BASE_URL
    if resolved == "gemini":
        return GEMINI_BASE_URL
    if resolved == "cerebras":
        return CEREBRAS_BASE_URL
    if resolved == "together":
        return TOGETHER_BASE_URL
    return os.environ.get("OPENAI_BASE_URL")


def provider_is_configured(provider: str | None = None) -> bool:
    resolved = get_model_provider(provider)
    if resolved == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if resolved == "huggingface":
        return huggingface_provider_is_configured()
    if resolved == "local":
        return local_training_provider_is_configured()
    if resolved == "groq":
        return bool(os.environ.get("GROQ_API_KEY"))
    if resolved == "gemini":
        return bool(os.environ.get("GOOGLE_API_KEY"))
    if resolved == "cerebras":
        return bool(os.environ.get("CEREBRAS_API_KEY"))
    if resolved == "together":
        return bool(os.environ.get("TOGETHER_API_KEY"))
    return True


def provider_payload(provider: str) -> dict[str, Any]:
    resolved = get_model_provider(provider)
    return {
        "provider": resolved,
        "label": get_provider_display_name(resolved),
        "baseUrl": get_provider_base_url(resolved),
        "configured": provider_is_configured(resolved),
        "supportsInference": provider_supports_inference(resolved),
        "supportsFineTuning": provider_supports_fine_tuning(resolved),
    }


def provider_status_payload(provider: str | None = None) -> dict[str, Any]:
    resolved = get_model_provider(provider)
    return {
        "modelProvider": resolved,
        "modelProviderLabel": get_provider_display_name(resolved),
        "providerBaseUrl": get_provider_base_url(resolved),
        "providerConfigured": provider_is_configured(resolved),
        "supportsInference": provider_supports_inference(resolved),
        "supportsFineTuning": provider_supports_fine_tuning(resolved),
        "managedFineTuningAvailable": provider_is_configured("openai"),
        "providers": [
            provider_payload("local"), provider_payload("ollama"), provider_payload("openai"), provider_payload("huggingface"),
            provider_payload("groq"), provider_payload("gemini"), provider_payload("cerebras"), provider_payload("together"),
        ],
        "defaultBaseModel": DEFAULT_BASE_MODEL,
        "defaultAgentModel": DEFAULT_AGENT_MODEL,
    }


def local_training_runtime_payload() -> dict[str, Any]:
    return inspect_local_training_runtime()


def ensure_fine_tuning_available(provider: str | None = None) -> None:
    resolved = get_model_provider(provider)
    if provider_supports_fine_tuning(resolved):
        return
    raise ApiError(
        "FINE_TUNING_UNSUPPORTED",
        (
            f"Fine-tuning is not available for {get_provider_display_name(resolved)} profiles yet. "
            "Use Local GPU QLoRA for the free on-device path, OpenAI for managed fine-tuning, or Hugging Face Jobs for paid cloud SFT training. "
            "Groq, Gemini, Cerebras, and Together AI are inference-only."
        ),
        400,
    )


if get_model_provider() == "ollama":
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or os.environ.get("OLLAMA_BASE_MODEL") or "qwen3:8b"
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("OLLAMA_AGENT_MODEL") or DEFAULT_BASE_MODEL
elif get_model_provider() == "huggingface":
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or os.environ.get("HF_BASE_MODEL") or DEFAULT_HUGGINGFACE_BASE_MODEL
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("OLLAMA_AGENT_MODEL") or "qwen3:8b"
elif get_model_provider() == "local":
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or os.environ.get("LOCAL_TRAINING_BASE_MODEL") or DEFAULT_LOCAL_BASE_MODEL
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("OLLAMA_AGENT_MODEL") or "qwen3:8b"
elif get_model_provider() == "groq":
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or "llama-3.3-70b-versatile"
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or DEFAULT_BASE_MODEL
elif get_model_provider() == "gemini":
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or "gemini-2.0-flash"
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or DEFAULT_BASE_MODEL
elif get_model_provider() == "cerebras":
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or "llama-4-scout-17b-16e-instruct"
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or DEFAULT_BASE_MODEL
elif get_model_provider() == "together":
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or "meta-llama/Meta-Llama-3.1-70B-Instruct"
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or DEFAULT_BASE_MODEL
else:
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or DEFAULT_OPENAI_BASE_MODEL
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("OPENAI_AGENT_MODEL") or "gpt-5.4-mini"


def _default_model_profiles() -> list[dict[str, Any]]:
    now = utc_now_iso()
    huggingface_base_model = _preferred_accessible_model(os.environ.get("HF_BASE_MODEL") or DEFAULT_HUGGINGFACE_BASE_MODEL)
    local_base_model = _preferred_accessible_model(os.environ.get("LOCAL_TRAINING_BASE_MODEL") or DEFAULT_LOCAL_BASE_MODEL)
    ollama_small_model = os.environ.get("OLLAMA_SMALL_MODEL") or DEFAULT_OLLAMA_SMALL_MODEL
    ollama_medium_model = os.environ.get("OLLAMA_BASE_MODEL") or DEFAULT_OLLAMA_MEDIUM_MODEL
    ollama_large_model = os.environ.get("OLLAMA_LARGE_MODEL") or DEFAULT_OLLAMA_LARGE_MODEL
    ollama_thinking_model = os.environ.get("OLLAMA_THINKING_MODEL") or DEFAULT_OLLAMA_THINKING_MODEL

    profiles = [
        {
            "id": "profile-small",
            "name": "Small",
            "provider": "ollama",
            "model": ollama_small_model,
            "category": "small",
            "description": "Fast local iteration and low-cost prompt tests.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-medium",
            "name": "Medium",
            "provider": "ollama",
            "model": ollama_medium_model,
            "category": "medium",
            "description": "Balanced everyday workspace model.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-large",
            "name": "Large",
            "provider": "ollama",
            "model": ollama_large_model,
            "category": "large",
            "description": "Stronger free local model for side-by-side comparisons.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-thinking",
            "name": "Thinking",
            "provider": "ollama",
            "model": ollama_thinking_model,
            "category": "thinking",
            "description": "Reasoning-heavy free local model that is auto-pulled on first use.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-ollama-llama32-3b",
            "name": "Llama 3.2 3B",
            "provider": "ollama",
            "model": "llama3.2:3b",
            "category": "small",
            "description": "Very fast free Ollama model for laptops and quick local evaluation loops.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-ollama-gemma3-4b",
            "name": "Gemma 3 4B",
            "provider": "ollama",
            "model": "gemma3:4b",
            "category": "medium",
            "description": "Compact Google model with strong quality-per-size for free local inference.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-ollama-qwen25-7b",
            "name": "Qwen 2.5 7B",
            "provider": "ollama",
            "model": "qwen2.5:7b",
            "category": "large",
            "description": "Stronger free local fallback when you want a different model family than Qwen 3 or Llama.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-open-source-sft",
            "name": "Open Source SFT",
            "provider": "huggingface",
            "model": huggingface_base_model,
            "category": "large",
            "description": "Paid cloud fine-tuning for open-source models through Hugging Face Jobs.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-qlora",
            "name": "Local GPU QLoRA",
            "provider": "local",
            "model": local_base_model,
            "category": "large",
            "description": "Recommended free fine-tuning path: train on your own GPU, save adapters locally, and export to GGUF for Ollama.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-qwen-05b",
            "name": "Qwen 0.5B Local",
            "provider": "local",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "category": "small",
            "description": "Low-VRAM fallback for pipeline validation when 3B is too heavy.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-qwen-15b",
            "name": "Qwen 1.5B Local",
            "provider": "local",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "category": "medium",
            "description": "Mid-range local fallback when you want lower VRAM than the recommended 3B profile.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-smollm2-17b",
            "name": "SmolLM2 1.7B Local",
            "provider": "local",
            "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "category": "medium",
            "description": "Very small free instruct model for low-VRAM local QLoRA experiments and quick iterations.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-granite-2b",
            "name": "Granite 3.1 2B Local",
            "provider": "local",
            "model": "ibm-granite/granite-3.1-2b-instruct",
            "category": "medium",
            "description": "Compact IBM Granite profile that stays free to fine-tune on your own GPU.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-falcon3-3b",
            "name": "Falcon 3 3B Local",
            "provider": "local",
            "model": "tiiuae/Falcon3-3B-Instruct",
            "category": "medium",
            "description": "Alternative free 3B instruct family for local QLoRA runs when you want more variety than Qwen alone.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-phi-35-mini",
            "name": "Phi 3.5 Mini Local",
            "provider": "local",
            "model": "microsoft/Phi-3.5-mini-instruct",
            "category": "medium",
            "description": "Compact Microsoft instruct model for local QLoRA experiments.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-mistral-7b",
            "name": "Mistral 7B Local",
            "provider": "local",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "category": "large",
            "description": "Stronger open instruct model for local QLoRA runs when you want another family.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-qwen-7b",
            "name": "Qwen 7B Local",
            "provider": "local",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "category": "large",
            "description": "Larger free local QLoRA option when you want stronger quality and have the VRAM budget.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-hf-qwen-15b",
            "name": "Qwen 1.5B SFT",
            "provider": "huggingface",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "category": "medium",
            "description": "Paid cloud fine-tune on Hugging Face Jobs when you do not want to use your own GPU.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-hf-phi-35-mini",
            "name": "Phi 3.5 Mini SFT",
            "provider": "huggingface",
            "model": "microsoft/Phi-3.5-mini-instruct",
            "category": "medium",
            "description": "Paid cloud fine-tune with Phi 3.5 Mini through Hugging Face Jobs.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-hf-mistral-7b",
            "name": "Mistral 7B SFT",
            "provider": "huggingface",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "category": "large",
            "description": "Paid cloud Mistral fine-tune when you want a stronger open-source base model.",
            "createdAt": now,
            "updatedAt": now,
        },
        # Groq — free tier, inference-only
        {
            "id": "profile-groq-gemma2-9b",
            "name": "Gemma 2 9B (Groq)",
            "provider": "groq",
            "model": "gemma2-9b-it",
            "category": "small",
            "description": "Fast free Groq inference with Google's Gemma 2 9B instruct model.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-groq-mixtral-8x7b",
            "name": "Mixtral 8x7B (Groq)",
            "provider": "groq",
            "model": "mixtral-8x7b-32768",
            "category": "medium",
            "description": "Mistral mixture-of-experts model served at ultra-low latency on Groq.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-groq-llama33-70b",
            "name": "Llama 3.3 70B (Groq)",
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "category": "large",
            "description": "Fastest free 70B inference available — best Groq model for quality.",
            "createdAt": now,
            "updatedAt": now,
        },
        # Google Gemini — free tier (15 RPM, 1M tokens/day), inference-only
        {
            "id": "profile-gemini-flash-8b",
            "name": "Gemini 1.5 Flash 8B",
            "provider": "gemini",
            "model": "gemini-1.5-flash-8b",
            "category": "small",
            "description": "Smallest Gemini model — free tier, fast for quick iteration.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-gemini-flash-2",
            "name": "Gemini 2.0 Flash",
            "provider": "gemini",
            "model": "gemini-2.0-flash",
            "category": "large",
            "description": "Latest Gemini model on the free tier — 1M tokens/day.",
            "createdAt": now,
            "updatedAt": now,
        },
        # Cerebras — free tier, ultra-fast hardware, inference-only
        {
            "id": "profile-cerebras-llama31-70b",
            "name": "Llama 3.1 70B (Cerebras)",
            "provider": "cerebras",
            "model": "llama3.1-70b",
            "category": "large",
            "description": "Ultra-fast 70B inference on Cerebras wafer-scale hardware.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-cerebras-llama4-scout",
            "name": "Llama 4 Scout (Cerebras)",
            "provider": "cerebras",
            "model": "llama-4-scout-17b-16e-instruct",
            "category": "medium",
            "description": "Meta's Llama 4 Scout served on Cerebras — free and extremely fast.",
            "createdAt": now,
            "updatedAt": now,
        },
        # Together AI — $25 free credits, inference-only
        {
            "id": "profile-together-qwen25-72b",
            "name": "Qwen 2.5 72B (Together)",
            "provider": "together",
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "category": "large",
            "description": "Strong open-source 72B model served via Together AI free credits.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-together-mixtral-8x7b",
            "name": "Mixtral 8x7B (Together)",
            "provider": "together",
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "category": "medium",
            "description": "Mistral MoE inference via Together AI — $25 free credits included.",
            "createdAt": now,
            "updatedAt": now,
        },
    ]

    return [profile for profile in profiles if not _is_seeded_gated_profile(profile)]


def _first_profile_id(
    profiles: list[dict[str, Any]],
    *,
    category: str | None = None,
    provider: str | None = None,
) -> str | None:
    for profile in profiles:
        if category and profile.get("category") != category:
            continue
        if provider and profile.get("provider") != provider:
            continue
        return profile["id"]
    return profiles[0]["id"] if profiles else None


def _default_model_profile_defaults(profiles: list[dict[str, Any]]) -> dict[str, str | None]:
    medium_id = _first_profile_id(profiles, category="medium")
    large_id = _first_profile_id(profiles, category="large")
    thinking_id = _first_profile_id(profiles, category="thinking")
    openai_id = _first_profile_id(profiles, provider="openai")
    huggingface_id = _first_profile_id(profiles, provider="huggingface")
    local_id = _first_profile_id(profiles, provider="local")

    compare_id = large_id if large_id and large_id != medium_id else None

    return {
        "playgroundBaseProfileId": medium_id,
        "playgroundCompareProfileId": compare_id,
        "agentBaseProfileId": medium_id,
        "agentModelProfileId": thinking_id or medium_id,
        "jobBaseProfileId": local_id or huggingface_id or openai_id or large_id,
    }


def _default_profile_id_is_compatible(default_key: str, profile: dict[str, Any]) -> bool:
    provider = profile.get("provider")
    if default_key == "jobBaseProfileId":
        return provider_supports_fine_tuning(provider)
    return provider_supports_inference(provider)


def _serialize_model_profile(profile: dict[str, Any]) -> dict[str, Any]:
    provider = get_model_provider(profile["provider"])
    serialized = deepcopy(profile)
    serialized["provider"] = provider
    serialized["providerLabel"] = get_provider_display_name(provider)
    serialized["providerConfigured"] = provider_is_configured(provider)
    serialized["supportsInference"] = provider_supports_inference(provider)
    serialized["supportsFineTuning"] = provider_supports_fine_tuning(provider)
    return serialized


def _model_profiles_payload(state: dict[str, Any]) -> dict[str, Any]:
    profiles = [_serialize_model_profile(profile) for profile in state["model_profiles"]]
    defaults = deepcopy(state["model_profile_defaults"])
    return {
        "profiles": sort_desc(profiles),
        "defaults": defaults,
        "providers": [
            provider_payload("local"), provider_payload("ollama"), provider_payload("openai"), provider_payload("huggingface"),
            provider_payload("groq"), provider_payload("gemini"), provider_payload("cerebras"), provider_payload("together"),
        ],
    }


def _ensure_model_profiles_initialized(state: dict[str, Any]) -> None:
    default_profiles = _default_model_profiles()
    default_profiles_by_id = {profile["id"]: profile for profile in default_profiles}

    if not state.get("model_profiles_initialized"):
        state["model_profiles"] = default_profiles
        state["model_profile_defaults"] = _default_model_profile_defaults(state["model_profiles"])
        state["model_profiles_initialized"] = True
    else:
        sanitized_profiles: list[dict[str, Any]] = []
        for profile in state["model_profiles"]:
            if _is_seeded_gated_profile(profile):
                continue
            profile = _sanitize_seeded_profile(profile)
            if _is_legacy_paid_seeded_profile(profile):
                replacement = deepcopy(default_profiles_by_id.get(profile["id"], profile))
                replacement["createdAt"] = profile.get("createdAt", replacement["createdAt"])
                replacement["updatedAt"] = utc_now_iso()
                profile = replacement
            if _profile_uses_gated_fine_tuning_model(profile):
                continue
            sanitized_profiles.append(profile)
        state["model_profiles"] = sanitized_profiles
        existing_ids = {profile["id"] for profile in state["model_profiles"]}
        missing_profiles = [deepcopy(profile) for profile in default_profiles if profile["id"] not in existing_ids]
        if missing_profiles:
            state["model_profiles"].extend(missing_profiles)

    valid_ids = {profile["id"] for profile in state["model_profiles"]}
    fallback_defaults = _default_model_profile_defaults(state["model_profiles"])
    current_defaults = state.get("model_profile_defaults") or {}

    profiles_by_id = {profile["id"]: profile for profile in state["model_profiles"]}
    sanitized_defaults = {}
    for key, value in current_defaults.items():
        if key not in MODEL_PROFILE_DEFAULT_KEYS:
            continue
        if value is None:
            sanitized_defaults[key] = value
            continue
        if key == "jobBaseProfileId" and value in LEGACY_FREE_TUNING_DEFAULT_PROFILE_IDS and fallback_defaults.get("jobBaseProfileId"):
            continue
        if value not in valid_ids:
            continue
        profile = profiles_by_id.get(value)
        if profile and _default_profile_id_is_compatible(key, profile):
            sanitized_defaults[key] = value
    state["model_profile_defaults"] = {**fallback_defaults, **sanitized_defaults}


def list_model_profiles() -> dict[str, Any]:
    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        _ensure_model_profiles_initialized(state)
        return _model_profiles_payload(state)

    return update_state(mutator)


def create_model_profile(
    name: str,
    provider: str,
    model: str,
    category: str = "custom",
    description: str | None = None,
) -> dict[str, Any]:
    resolved_provider = get_model_provider(provider)
    normalized_category = (category or "custom").strip().lower()
    if normalized_category not in MODEL_PROFILE_CATEGORIES:
        raise ApiError("VALIDATION_ERROR", "Choose a valid profile category.", 400)

    if not name.strip():
        raise ApiError("VALIDATION_ERROR", "Profile name is required.", 400)
    if not model.strip():
        raise ApiError("VALIDATION_ERROR", "Model name is required.", 400)
    _ensure_accessible_profile_model(model.strip(), resolved_provider)

    now = utc_now_iso()
    profile = {
        "id": uuid4().hex,
        "name": name.strip(),
        "provider": resolved_provider,
        "model": model.strip(),
        "category": normalized_category,
        "description": description.strip() if description and description.strip() else None,
        "createdAt": now,
        "updatedAt": now,
    }

    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        _ensure_model_profiles_initialized(state)
        state["model_profiles"].append(profile)
        _ensure_model_profiles_initialized(state)
        return _serialize_model_profile(profile)

    return update_state(mutator)


def update_model_profile(
    profile_id: str,
    name: str,
    provider: str,
    model: str,
    category: str = "custom",
    description: str | None = None,
) -> dict[str, Any]:
    resolved_provider = get_model_provider(provider)
    normalized_category = (category or "custom").strip().lower()
    if normalized_category not in MODEL_PROFILE_CATEGORIES:
        raise ApiError("VALIDATION_ERROR", "Choose a valid profile category.", 400)
    if not name.strip():
        raise ApiError("VALIDATION_ERROR", "Profile name is required.", 400)
    if not model.strip():
        raise ApiError("VALIDATION_ERROR", "Model name is required.", 400)
    _ensure_accessible_profile_model(model.strip(), resolved_provider)

    now = utc_now_iso()

    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        _ensure_model_profiles_initialized(state)
        profile = get_model_profile(state, profile_id)
        if not profile:
            raise ApiError("NOT_FOUND", "Model profile not found.", 404)
        profile.update(
            {
                "name": name.strip(),
                "provider": resolved_provider,
                "model": model.strip(),
                "category": normalized_category,
                "description": description.strip() if description and description.strip() else None,
                "updatedAt": now,
            }
        )
        _ensure_model_profiles_initialized(state)
        return _serialize_model_profile(profile)

    return update_state(mutator)


def delete_model_profile(profile_id: str) -> dict[str, Any]:
    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        _ensure_model_profiles_initialized(state)
        profile = get_model_profile(state, profile_id)
        if not profile:
            raise ApiError("NOT_FOUND", "Model profile not found.", 404)
        state["model_profiles"] = [item for item in state["model_profiles"] if item["id"] != profile_id]
        _ensure_model_profiles_initialized(state)
        return _model_profiles_payload(state)

    return update_state(mutator)


def update_model_profile_defaults(defaults: dict[str, str | None]) -> dict[str, Any]:
    unknown_keys = [key for key in defaults if key not in MODEL_PROFILE_DEFAULT_KEYS]
    if unknown_keys:
        raise ApiError("VALIDATION_ERROR", f"Unknown default keys: {', '.join(sorted(unknown_keys))}.", 400)

    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        _ensure_model_profiles_initialized(state)
        valid_ids = {profile["id"] for profile in state["model_profiles"]}
        for key, value in defaults.items():
            if value is not None and value not in valid_ids:
                raise ApiError("VALIDATION_ERROR", f"Default profile `{value}` does not exist.", 400)
            state["model_profile_defaults"][key] = value
        _ensure_model_profiles_initialized(state)
        return _model_profiles_payload(state)

    return update_state(mutator)


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
        if isinstance(messages, list):
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
            continue

        instruction = record.get("instruction")
        output = record.get("output")
        if isinstance(instruction, str) and instruction.strip() and isinstance(output, str) and output.strip():
            valid_records += 1
            if len(examples) < 3:
                examples.append({"line": index, "preview": record})
            continue

        errors.append(
            {
                "line": index,
                "message": "Each line must contain either a `messages` array or `instruction`/`output` fields.",
            }
        )

    return {
        "totalRecords": len(lines),
        "validRecords": valid_records,
        "invalidRecords": len(lines) - valid_records,
        "errors": errors,
        "warnings": warnings,
        "examples": examples,
    }


_INFERENCE_PROVIDER_KEYS: dict[str, tuple[str, str]] = {
    "groq":     ("GROQ_API_KEY",     GROQ_BASE_URL),
    "gemini":   ("GOOGLE_API_KEY",   GEMINI_BASE_URL),
    "cerebras": ("CEREBRAS_API_KEY", CEREBRAS_BASE_URL),
    "together": ("TOGETHER_API_KEY", TOGETHER_BASE_URL),
}


def get_model_client(provider: str | None = None, model: str | None = None) -> OpenAI:
    resolved = get_model_provider(provider)
    if resolved in {"huggingface", "local"}:
        raise ApiError(
            "PLAYGROUND_RUN_FAILED",
            f"{get_provider_display_name(resolved)} profiles are training-only right now. Use Ollama, OpenAI, Groq, Gemini, Cerebras, or Together AI profiles for inference.",
            400,
        )

    if OpenAI is None:
        raise ApiError("MODEL_CLIENT_PACKAGE_MISSING", "The OpenAI-compatible Python package is not installed.", 500)

    if resolved == "ollama" and model:
        ensure_ollama_model_available(model)

    if resolved == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ApiError("MODEL_PROVIDER_NOT_CONFIGURED", "OPENAI_API_KEY is not configured.", 503)
        base_url = get_provider_base_url(resolved)
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    if resolved in _INFERENCE_PROVIDER_KEYS:
        env_var, base_url = _INFERENCE_PROVIDER_KEYS[resolved]
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ApiError("MODEL_PROVIDER_NOT_CONFIGURED", f"{env_var} is not configured.", 503)
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        base_url=get_provider_base_url(resolved),
    )


def get_openai_client() -> OpenAI:
    return get_model_client("openai")


def upload_training_file_to_openai(file_path: str) -> Any:
    ensure_fine_tuning_available("openai")
    client = get_model_client("openai")
    with open(file_path, "rb") as file_handle:
        return client.files.create(file=file_handle, purpose="fine-tune")


def create_fine_tuning_job(training_file_id: str, base_model: str, method_config: dict[str, Any] | None = None) -> Any:
    ensure_fine_tuning_available("openai")
    client = get_model_client("openai")
    method: dict[str, Any] = {"type": "supervised"}
    if method_config:
        method["supervised"] = {"hyperparameters": method_config}
    return client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=base_model,
        method=method,
    )


def retrieve_fine_tuning_job(openai_job_id: str) -> Any:
    ensure_fine_tuning_available("openai")
    return get_model_client("openai").fine_tuning.jobs.retrieve(openai_job_id)


def list_fine_tuning_events(openai_job_id: str) -> Any:
    ensure_fine_tuning_available("openai")
    return get_model_client("openai").fine_tuning.jobs.list_events(fine_tuning_job_id=openai_job_id)


def cancel_fine_tuning_job(openai_job_id: str) -> Any:
    ensure_fine_tuning_available("openai")
    return get_model_client("openai").fine_tuning.jobs.cancel(openai_job_id)


def _preflight_ollama_gpu() -> None:
    """Best-effort Ollama GPU preflight. Never raises to the caller."""
    if not ollama_manager.auto_manage_enabled():
        return
    try:
        ollama_manager.ensure_ollama_on_free_gpu()
    except ollama_manager.OllamaManageUnavailable as exc:
        logger.info("Skipping Ollama auto-manage: %s", exc)
    except Exception:  # pragma: no cover - defensive, never break the prompt
        logger.exception("Unexpected Ollama auto-manage failure")


def run_model(prompt: str, model: str, provider: str | None = None) -> str:
    resolved_provider = get_model_provider(provider)
    is_ollama = resolved_provider == "ollama"

    if is_ollama:
        _preflight_ollama_gpu()

    def _invoke() -> str:
        response = get_model_client(provider, model=model).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    try:
        return _invoke()
    except Exception as exc:
        if not is_ollama or not ollama_manager.is_cuda_oom_error(exc):
            raise
        if not ollama_manager.auto_manage_enabled():
            raise
        logger.warning("Ollama returned CUDA OOM; forcing a GPU rebalance and retrying once.")
        try:
            ollama_manager.ensure_ollama_on_free_gpu(force=True)
        except ollama_manager.OllamaManageUnavailable as manage_exc:
            logger.info("Ollama auto-manage retry unavailable: %s", manage_exc)
            raise exc from manage_exc
        return _invoke()


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


def build_dataset_download_package(dataset_id: str) -> dict[str, str]:
    dataset = retrieve_dataset_detail(dataset_id)
    dataset_path = _resolve_dataset_path(dataset.get("storagePath"))
    filename = dataset.get("originalFilename") or dataset_path.name or f"{dataset_id}.jsonl"
    if not filename.endswith(".jsonl"):
        filename = f"{filename}.jsonl"
    return {
        "path": str(dataset_path),
        "filename": filename,
        "mediaType": "application/x-ndjson",
    }


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


def upload_dataset_record_to_huggingface(dataset_id: str) -> dict[str, Any]:
    dataset = retrieve_dataset_detail(dataset_id)
    if dataset["validationStatus"] != "VALID":
        raise ApiError("DATASET_INVALID", "Dataset must validate successfully before upload.", 400)

    upload_metadata = upload_dataset_to_huggingface(dataset)

    def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        dataset_record = get_dataset(state, dataset_id)
        if not dataset_record:
            raise ApiError("DATASET_NOT_FOUND", "Dataset not found.", 404)
        dataset_record["huggingFaceDatasetRepoId"] = upload_metadata["datasetRepoId"]
        dataset_record["huggingFaceDatasetPath"] = upload_metadata["datasetPath"]
        dataset_record["uploadedToHuggingFaceAt"] = upload_metadata["uploadedAt"]
        dataset_record["updatedAt"] = upload_metadata["uploadedAt"]
        return deepcopy(dataset_record)

    return update_state(mutator)


_JOB_README_TEMPLATE = """# Training job `{job_id}`

This folder is auto-created by the local QLoRA trainer. Everything you need
to re-use the fine-tuned model lives right here; you do not need to dig
into any subfolder unless you want training internals.

## Layout

- `adapter/` - the LoRA adapter + tokenizer. Load this on top of the base
  model to use the fine-tuned weights.
- `gguf/` - GGUF export + `Modelfile` (only present when GGUF export was
  enabled at job creation).
- `local_train_config.json` - the frozen job configuration.
- `local_train_status.json` - the most recent run status shown in the UI.
- `local_train_events.jsonl` - append-only event log.
- `local_train_metrics.json` - final train + eval metrics.
- `local_train.log` - stdout / stderr of the training subprocess.

## What is **not** kept

- Intermediate HuggingFace Trainer checkpoints (`checkpoint-N/` with
  optimizer/scheduler/rng state) are deleted once the final adapter is
  saved. They are not needed for inference.
- Download zip bundles are built on demand and never persisted here.
"""


def _write_job_readme(working_dir: Path) -> None:
    try:
        working_dir.mkdir(parents=True, exist_ok=True)
        readme_path = working_dir / "README.md"
        readme_path.write_text(
            _JOB_README_TEMPLATE.format(job_id=working_dir.name),
            encoding="utf-8",
        )
    except OSError:
        pass


def _build_local_training_job_config(
    *,
    job_id: str,
    dataset: dict[str, Any],
    base_model: str,
    hyperparameters: dict[str, Any] | None = None,
    training_preset: str | None = "balanced",
    export_gguf: bool = True,
    gguf_quantization: str = "q4_k_m",
    push_to_ollama: bool = False,
    ollama_model_name: str = "",
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    per_device_batch_size: int | None = None,
) -> LocalQLoraJobConfig:
    working_dir = JOBS_DIR / job_id
    # output_dir holds the HuggingFace Trainer scratch (intermediate checkpoints,
    # auto README, etc). It is deleted after training succeeds so the final job
    # folder only contains the adapter + logs.
    output_dir = working_dir / "artifacts"
    merged_hyperparameters: dict[str, Any] = dict(hyperparameters or {})
    if num_epochs is not None:
        merged_hyperparameters.setdefault("num_train_epochs", num_epochs)
    if learning_rate is not None:
        merged_hyperparameters.setdefault("learning_rate", learning_rate)
    if per_device_batch_size is not None:
        merged_hyperparameters.setdefault("per_device_train_batch_size", per_device_batch_size)
    _write_job_readme(working_dir)
    return LocalQLoraJobConfig(
        job_id=job_id,
        dataset_id=dataset["id"],
        dataset_name=dataset["name"],
        dataset_path=dataset["storagePath"],
        base_model=base_model,
        working_dir=str(working_dir),
        output_dir=str(output_dir),
        model_output_path=str(working_dir / "adapter"),
        config_path=str(working_dir / "local_train_config.json"),
        status_path=str(working_dir / "local_train_status.json"),
        events_path=str(working_dir / "local_train_events.jsonl"),
        log_path=str(working_dir / "local_train.log"),
        metrics_path=str(working_dir / "local_train_metrics.json"),
        hyperparameters=merged_hyperparameters,
        allow_cpu_fallback=os.environ.get("LOCAL_TRAINING_ALLOW_CPU_FALLBACK", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        seed=int(os.environ.get("LOCAL_TRAINING_SEED", "42")),
        eval_ratio=float(os.environ.get("LOCAL_TRAINING_EVAL_RATIO", "0.1")),
        export_gguf=export_gguf,
        gguf_quantization=gguf_quantization,
        push_to_ollama=push_to_ollama,
        ollama_model_name=ollama_model_name,
        training_preset=training_preset or "balanced",
    )


def list_jobs() -> list[dict[str, Any]]:
    state = load_state()
    jobs = []
    for job in sort_desc(state["jobs"]):
        dataset = get_dataset(state, job["datasetId"])
        jobs.append(
            {
                **deepcopy(job),
                "datasetName": dataset["name"] if dataset else "Unknown dataset",
                "modelProviderLabel": get_provider_display_name(job.get("modelProvider")),
            }
        )
    return jobs


def _job_detail_payload(state: dict[str, Any], job_record: dict[str, Any]) -> dict[str, Any]:
    dataset = get_dataset(state, job_record["datasetId"])
    events = sort_desc([deepcopy(event) for event in state["job_events"] if event["jobId"] == job_record["id"]])
    return {
        **deepcopy(job_record),
        "modelProviderLabel": get_provider_display_name(job_record.get("modelProvider")),
        "dataset": deepcopy(dataset) if dataset else None,
        "events": events,
    }


def create_job_record(
    dataset_id: str,
    base_model: str,
    hyperparameters: dict[str, Any] | None = None,
    provider: str | None = None,
    training_preset: str | None = "balanced",
    export_gguf: bool = True,
    gguf_quantization: str = "q4_k_m",
    push_to_ollama: bool = False,
    ollama_model_name: str = "",
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    per_device_batch_size: int | None = None,
) -> dict[str, Any]:
    model_provider = get_model_provider(provider or "openai")
    ensure_fine_tuning_available(model_provider)
    dataset = retrieve_dataset_detail(dataset_id)
    if dataset["validationStatus"] != "VALID":
        raise ApiError("DATASET_INVALID", "Only valid datasets can be used for fine-tuning.", 400)
    _ensure_accessible_fine_tuning_model(base_model, model_provider)

    if model_provider == "local":
        job_id = uuid4().hex
        resolved_ollama_model_name = (ollama_model_name or "").strip()
        if push_to_ollama and not resolved_ollama_model_name:
            resolved_ollama_model_name = _default_local_ollama_model_name(dataset["name"], base_model, job_id)
        config = _build_local_training_job_config(
            job_id=job_id,
            dataset=dataset,
            base_model=base_model,
            hyperparameters=hyperparameters,
            training_preset=training_preset,
            export_gguf=export_gguf,
            gguf_quantization=gguf_quantization,
            push_to_ollama=push_to_ollama,
            ollama_model_name=resolved_ollama_model_name,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_batch_size=per_device_batch_size,
        )
        try:
            runtime_summary = ensure_local_training_ready(
                allow_cpu_fallback=config.allow_cpu_fallback,
                export_gguf=config.export_gguf,
                push_to_ollama=config.push_to_ollama,
            )
        except RuntimeError as error:
            raise ApiError("MODEL_PROVIDER_NOT_CONFIGURED", str(error), 400) from error
        submission = spawn_local_training_job(config, runtime_summary)
        now = utc_now_iso()
        job = {
            "id": job_id,
            "datasetId": dataset_id,
            "modelProvider": model_provider,
            "providerJobId": submission["providerJobId"],
            "providerJobUrl": submission["providerJobUrl"],
            "providerNamespace": submission["providerNamespace"],
            "openaiJobId": None,
            "baseModel": base_model,
            "methodType": "qlora",
            "status": submission["status"],
            "statusMessage": submission["statusMessage"],
            "fineTunedModel": submission["fineTunedModel"],
            "trainedTokens": None,
            "estimatedFinishAt": None,
            "resultFilesJson": submission["resultFilesJson"],
            "hyperparametersJson": hyperparameters,
            "lastSyncedAt": now,
            "createdAt": now,
            "updatedAt": now,
            "finishedAt": None,
            "modelRepoId": submission["modelRepoId"],
            "modelRepoUrl": None,
            "datasetRepoId": submission["datasetRepoId"],
            "datasetRepoPath": submission["datasetRepoPath"],
            "datasetRepoUrl": None,
            "trackioUrl": None,
            "trainingBackend": submission["trainingBackend"],
            "trainingPreset": config.training_preset,
            "localConfigPath": config.config_path,
            "localStatusPath": config.status_path,
            "localEventsPath": config.events_path,
            "localLogPath": config.log_path,
            "localMetricsPath": config.metrics_path,
            "localArtifactsPath": config.model_output_path,
            "progressJson": submission["progressJson"],
            "ollamaModelName": resolved_ollama_model_name or None,
        }

        def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
            dataset_record = get_dataset(state, dataset_id)
            if not dataset_record:
                raise ApiError("DATASET_NOT_FOUND", "Dataset not found.", 404)
            dataset_record["updatedAt"] = now
            state["jobs"].append(job)
            state["job_events"].append(
                {
                    "id": uuid4().hex,
                    "jobId": job_id,
                    "level": "info",
                    "message": "Local GPU QLoRA job created.",
                    "eventType": "job_created",
                    "createdAt": now,
                }
            )
            return deepcopy(job)

        return update_state(mutator)

    if model_provider == "huggingface":
        job_id = uuid4().hex
        submission = submit_training_job(
            job_id=job_id,
            dataset=dataset,
            base_model=base_model,
            hyperparameters=hyperparameters,
        )
        now = utc_now_iso()
        job = {
            "id": job_id,
            "datasetId": dataset_id,
            "modelProvider": model_provider,
            "providerJobId": submission["providerJobId"],
            "providerJobUrl": submission["providerJobUrl"],
            "providerNamespace": submission["providerNamespace"],
            "openaiJobId": None,
            "baseModel": base_model,
            "methodType": "sft_lora",
            "status": submission["status"],
            "statusMessage": submission["statusMessage"],
            "fineTunedModel": submission["fineTunedModel"],
            "trainedTokens": None,
            "estimatedFinishAt": None,
            "resultFilesJson": submission["resultFilesJson"],
            "hyperparametersJson": hyperparameters,
            "lastSyncedAt": now,
            "createdAt": now,
            "updatedAt": now,
            "finishedAt": now if submission["status"] in {"succeeded", "failed", "cancelled"} else None,
            "modelRepoId": submission["modelRepoId"],
            "modelRepoUrl": submission["modelRepoUrl"],
            "datasetRepoId": submission["datasetRepoId"],
            "datasetRepoPath": submission["datasetRepoPath"],
            "datasetRepoUrl": submission["datasetRepoUrl"],
            "trackioUrl": submission["trackioUrl"],
            "trainingBackend": submission["trainingBackend"],
            "trainingPreset": None,
            "localConfigPath": None,
            "localStatusPath": None,
            "localEventsPath": None,
            "localLogPath": None,
            "localMetricsPath": None,
            "localArtifactsPath": None,
            "progressJson": None,
        }

        def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
            dataset_record = get_dataset(state, dataset_id)
            if not dataset_record:
                raise ApiError("DATASET_NOT_FOUND", "Dataset not found.", 404)
            dataset_record["huggingFaceDatasetRepoId"] = submission["datasetRepoId"]
            dataset_record["huggingFaceDatasetPath"] = submission["datasetRepoPath"]
            dataset_record["uploadedToHuggingFaceAt"] = submission["uploadedToHuggingFaceAt"]
            dataset_record["updatedAt"] = now
            state["jobs"].append(job)
            state["job_events"].append(
                {
                    "id": uuid4().hex,
                    "jobId": job_id,
                    "level": "info",
                    "message": "Hugging Face training job submitted.",
                    "eventType": "job_created",
                    "createdAt": now,
                }
            )
            return deepcopy(job)

        return update_state(mutator)

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
        "modelProvider": model_provider,
        "providerJobId": remote_job.id,
        "providerJobUrl": None,
        "providerNamespace": None,
        "openaiJobId": remote_job.id,
        "baseModel": base_model,
        "methodType": "supervised",
        "status": mapped["status"],
        "statusMessage": None,
        "fineTunedModel": mapped["fineTunedModel"],
        "trainedTokens": mapped["trainedTokens"],
        "estimatedFinishAt": mapped["estimatedFinishAt"],
        "resultFilesJson": mapped["resultFilesJson"],
        "hyperparametersJson": hyperparameters,
        "lastSyncedAt": now,
        "createdAt": now,
        "updatedAt": now,
        "finishedAt": mapped["finishedAt"],
        "modelRepoId": None,
        "modelRepoUrl": None,
        "datasetRepoId": None,
        "datasetRepoPath": None,
        "datasetRepoUrl": None,
        "trackioUrl": None,
        "trainingBackend": "openai_api",
        "trainingPreset": None,
        "localConfigPath": None,
        "localStatusPath": None,
        "localEventsPath": None,
        "localLogPath": None,
        "localMetricsPath": None,
        "localArtifactsPath": None,
        "progressJson": None,
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
    return _job_detail_payload(state, job)


def _find_local_result_file(job: dict[str, Any], result_type: str) -> dict[str, Any] | None:
    for result in job.get("resultFilesJson") or []:
        if result.get("type") == result_type:
            return result
    return None


def build_job_download_package(job_id: str, artifact_type: str = "adapter") -> dict[str, str]:
    job = retrieve_job_detail(job_id)

    if job["modelProvider"] != "local":
        raise ApiError(
            "FINE_TUNING_UNSUPPORTED",
            "Direct model download is currently available only for Local GPU QLoRA jobs.",
            400,
        )

    if job["status"] != "succeeded":
        raise ApiError("JOB_SYNC_FAILED", "The local training job must finish successfully before download is available.", 400)

    if artifact_type == "gguf":
        gguf_result = _find_local_result_file(job, "local_gguf")
        if not gguf_result:
            raise ApiError(
                "NOT_FOUND",
                "This job does not have a GGUF export. Enable GGUF export before starting the local fine-tuning run.",
                404,
            )
        gguf_path = _resolve_local_job_path(gguf_result.get("path"))
        return {
            "path": str(gguf_path),
            "filename": gguf_path.name,
            "mediaType": "application/octet-stream",
        }

    if artifact_type != "adapter":
        raise ApiError("VALIDATION_ERROR", f"Unsupported download type `{artifact_type}`.", 400)

    adapter_path = _resolve_local_job_path(job.get("fineTunedModel") or job.get("localArtifactsPath"))
    metrics_path = Path(job["localMetricsPath"]).resolve() if job.get("localMetricsPath") else None
    status_path = Path(job["localStatusPath"]).resolve() if job.get("localStatusPath") else None
    working_dir = _resolve_local_job_path(job.get("localConfigPath")).parent
    downloads_dir = working_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = job.get("dataset", {}).get("name") if isinstance(job.get("dataset"), dict) else None
    filename = (
        f"{_slugify_archive_name(dataset_name or job_id)}-"
        f"{_slugify_archive_name(job['baseModel'].split('/')[-1])}-adapter.zip"
    )
    archive_path = downloads_dir / filename

    source_mtime = _latest_path_mtime(adapter_path)
    if metrics_path and metrics_path.exists():
        source_mtime = max(source_mtime, _latest_path_mtime(metrics_path))
    if status_path and status_path.exists():
        source_mtime = max(source_mtime, _latest_path_mtime(status_path))

    if not archive_path.exists() or archive_path.stat().st_mtime < source_mtime:
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            metadata = {
                "jobId": job["id"],
                "datasetName": dataset_name,
                "baseModel": job["baseModel"],
                "provider": job["modelProvider"],
                "providerLabel": job["modelProviderLabel"],
                "status": job["status"],
                "createdAt": job["createdAt"],
                "finishedAt": job.get("finishedAt"),
                "downloadType": "local_qlora_adapter_bundle",
                "notes": "This bundle contains the LoRA adapter, tokenizer files, and metrics. Load it on top of the base model.",
            }
            archive.writestr("agentic-metadata.json", json.dumps(metadata, indent=2))

            if adapter_path.is_file():
                archive.write(adapter_path, arcname=str(Path("adapter") / adapter_path.name))
            else:
                for file_path in sorted(adapter_path.rglob("*")):
                    if file_path.is_file():
                        archive.write(file_path, arcname=str(Path("adapter") / file_path.relative_to(adapter_path)))

            if metrics_path and metrics_path.exists():
                archive.write(metrics_path, arcname="metrics.json")

            if status_path and status_path.exists():
                archive.write(status_path, arcname="status.json")

    return {
        "path": str(archive_path),
        "filename": filename,
        "mediaType": "application/zip",
    }


def sync_job_record(job_id: str) -> dict[str, Any]:
    job = retrieve_job_detail(job_id)

    if job["modelProvider"] == "local":
        inspection = inspect_local_training_job(job)
        now = utc_now_iso()

        def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
            job_record = get_job(state, job_id)
            if not job_record:
                raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)

            job_record["status"] = inspection["status"]
            job_record["statusMessage"] = inspection["statusMessage"]
            job_record["providerJobId"] = inspection["providerJobId"]
            job_record["providerJobUrl"] = inspection["providerJobUrl"]
            job_record["providerNamespace"] = inspection["providerNamespace"]
            job_record["fineTunedModel"] = inspection["fineTunedModel"]
            job_record["trainedTokens"] = inspection["trainedTokens"]
            job_record["resultFilesJson"] = inspection["resultFilesJson"]
            job_record["modelRepoId"] = inspection["modelRepoId"]
            job_record["modelRepoUrl"] = inspection["modelRepoUrl"]
            job_record["datasetRepoId"] = inspection["datasetRepoId"]
            job_record["datasetRepoPath"] = inspection["datasetRepoPath"]
            job_record["datasetRepoUrl"] = inspection["datasetRepoUrl"]
            job_record["trackioUrl"] = inspection["trackioUrl"]
            job_record["progressJson"] = inspection["progressJson"]
            job_record["lastSyncedAt"] = now
            job_record["updatedAt"] = now
            if inspection["status"] in {"succeeded", "failed", "cancelled"}:
                job_record["finishedAt"] = inspection["finishedAt"] or job_record.get("finishedAt") or now

            existing_event_ids = {event["id"] for event in state["job_events"]}
            state["job_events"].extend(
                collect_local_training_events(
                    events_path=job_record["localEventsPath"],
                    existing_event_ids=existing_event_ids,
                )
            )

            return _job_detail_payload(state, job_record)

        return update_state(mutator)

    if job["modelProvider"] == "huggingface":
        inspection = inspect_training_job(job["providerJobId"], job.get("providerNamespace"))
        now = utc_now_iso()

        def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
            job_record = get_job(state, job_id)
            if not job_record:
                raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)

            job_record["status"] = inspection["status"]
            job_record["statusMessage"] = inspection["statusMessage"]
            job_record["providerJobId"] = inspection["providerJobId"]
            job_record["providerJobUrl"] = inspection["providerJobUrl"]
            job_record["providerNamespace"] = inspection["providerNamespace"]
            job_record["progressJson"] = None
            job_record["lastSyncedAt"] = now
            job_record["updatedAt"] = now
            if inspection["status"] in {"succeeded", "failed", "cancelled"}:
                job_record["finishedAt"] = job_record.get("finishedAt") or now

            existing_event_ids = {event["id"] for event in state["job_events"]}
            state["job_events"].extend(
                collect_training_job_events(
                    workspace_job_id=job_id,
                    provider_job_id=inspection["providerJobId"],
                    provider_namespace=inspection["providerNamespace"],
                    existing_event_ids=existing_event_ids,
                )
            )

            return _job_detail_payload(state, job_record)

        return update_state(mutator)

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
        job_record["progressJson"] = None
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

        return _job_detail_payload(state, job_record)

    return update_state(mutator)


def list_job_events(job_id: str) -> list[dict[str, Any]]:
    state = load_state()
    job = get_job(state, job_id)
    if not job:
        raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)
    return sort_desc([deepcopy(event) for event in state["job_events"] if event["jobId"] == job_id])


def cancel_job_record(job_id: str) -> dict[str, Any]:
    job = retrieve_job_detail(job_id)

    if job["modelProvider"] == "local":
        inspection = cancel_local_training_job(job)
        now = utc_now_iso()

        def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
            job_record = get_job(state, job_id)
            if not job_record:
                raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)
            job_record["status"] = inspection["status"]
            job_record["statusMessage"] = inspection["statusMessage"]
            job_record["providerJobId"] = inspection["providerJobId"]
            job_record["providerJobUrl"] = inspection["providerJobUrl"]
            job_record["providerNamespace"] = inspection["providerNamespace"]
            job_record["fineTunedModel"] = inspection["fineTunedModel"]
            job_record["trainedTokens"] = inspection["trainedTokens"]
            job_record["resultFilesJson"] = inspection["resultFilesJson"]
            job_record["modelRepoId"] = inspection["modelRepoId"]
            job_record["modelRepoUrl"] = inspection["modelRepoUrl"]
            job_record["progressJson"] = inspection["progressJson"]
            job_record["lastSyncedAt"] = now
            job_record["updatedAt"] = now
            if inspection["status"] in {"succeeded", "failed", "cancelled"}:
                job_record["finishedAt"] = inspection["finishedAt"] or job_record.get("finishedAt") or now
            state["job_events"].append(
                {
                    "id": uuid4().hex,
                    "jobId": job_id,
                    "level": "warning",
                    "message": "Cancel requested for the local GPU QLoRA job.",
                    "eventType": "job_cancelled",
                    "createdAt": now,
                }
            )
            return _job_detail_payload(state, job_record)

        return update_state(mutator)

    if job["modelProvider"] == "huggingface":
        inspection = cancel_training_job(job["providerJobId"], job.get("providerNamespace"))
        now = utc_now_iso()

        def mutator(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
            job_record = get_job(state, job_id)
            if not job_record:
                raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)
            job_record["status"] = inspection["status"]
            job_record["statusMessage"] = inspection["statusMessage"]
            job_record["providerJobId"] = inspection["providerJobId"]
            job_record["providerJobUrl"] = inspection["providerJobUrl"]
            job_record["providerNamespace"] = inspection["providerNamespace"]
            job_record["progressJson"] = None
            job_record["lastSyncedAt"] = now
            job_record["updatedAt"] = now
            if inspection["status"] in {"succeeded", "failed", "cancelled"}:
                job_record["finishedAt"] = job_record.get("finishedAt") or now
            state["job_events"].append(
                {
                    "id": uuid4().hex,
                    "jobId": job_id,
                    "level": "warning",
                    "message": "Cancel requested from Hugging Face Jobs.",
                    "eventType": "job_cancelled",
                    "createdAt": now,
                }
            )
            return _job_detail_payload(state, job_record)

        return update_state(mutator)

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
        return _job_detail_payload(state, job_record)

    return update_state(mutator)


def run_playground_prompt(
    prompt: str,
    base_model: str,
    fine_tuned_model: str | None = None,
    base_provider: str | None = None,
    fine_tuned_provider: str | None = None,
) -> dict[str, Any]:
    if not prompt.strip():
        raise ApiError("PLAYGROUND_RUN_FAILED", "Prompt is required.", 400)

    try:
        base_output = run_model(prompt, base_model, base_provider)
        tuned_output = run_model(prompt, fine_tuned_model, fine_tuned_provider) if fine_tuned_model else None
    except ApiError:
        raise
    except Exception as error:  # pragma: no cover
        raise ApiError("PLAYGROUND_RUN_FAILED", str(error), 500) from error

    run = {
        "id": uuid4().hex,
        "baseModel": base_model,
        "baseModelProvider": get_model_provider(base_provider) if base_provider else get_model_provider(),
        "fineTunedModel": fine_tuned_model,
        "fineTunedModelProvider": get_model_provider(fine_tuned_provider) if fine_tuned_provider else None,
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
    completed_models = len([job for job in jobs if job.get("status") == "succeeded" and job.get("fineTunedModel")])
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
    base_model_provider: str | None = None,
    dataset_id: str | None = None,
    evaluation_prompt: str | None = None,
    agent_model: str | None = None,
    agent_model_provider: str | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    snapshot = {
        "id": run_id,
        "workflowId": workflow_id,
        "status": "queued",
        "goal": goal,
        "baseModel": base_model,
        "baseModelProvider": get_model_provider(base_model_provider) if base_model_provider else get_model_provider(),
        "datasetId": dataset_id,
        "evaluationPrompt": evaluation_prompt,
        "agentModel": agent_model or DEFAULT_AGENT_MODEL,
        "agentModelProvider": get_model_provider(agent_model_provider)
        if agent_model_provider
        else get_model_provider(),
        "summary": None,
        "latestJobId": None,
        "fineTunedModel": None,
        "fineTunedModelProvider": None,
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
