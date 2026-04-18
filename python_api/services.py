from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

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
    inspect_local_training_job,
    local_training_provider_is_configured,
    spawn_local_training_job,
)
from python_api.local_qlora.config import LocalQLoraJobConfig
from python_api.store import (
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


SUPPORTED_MODEL_PROVIDERS = {"ollama", "openai", "huggingface", "local"}
PROVIDER_LABELS = {
    "ollama": "Ollama",
    "openai": "OpenAI",
    "huggingface": "Hugging Face Jobs",
    "local": "Local GPU QLoRA",
}
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
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


def _slugify_archive_name(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "model"


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
    jobs_root = (ROOT / "uploads_python" / "jobs").resolve()
    try:
        resolved.relative_to(jobs_root)
    except ValueError as error:
        raise ApiError("FORBIDDEN", "Requested artifact path is outside the local jobs directory.", 403) from error

    if not resolved.exists():
        raise ApiError("NOT_FOUND", "Requested local training artifact was not found on disk.", 404)

    return resolved


def _is_seeded_gated_profile(profile: dict[str, Any]) -> bool:
    expected_model = SEEDED_GATED_PROFILE_MODELS.get(profile.get("id"))
    return bool(expected_model and profile.get("model") == expected_model)


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
                f"Unsupported provider `{provider}`. Use `ollama`, `openai`, `huggingface`, or `local`.",
                400,
            )
        return normalized

    configured = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if configured in SUPPORTED_MODEL_PROVIDERS:
        return configured
    if configured:
        raise ApiError(
            "MODEL_PROVIDER_INVALID",
            f"Unsupported LLM_PROVIDER `{configured}`. Use `ollama`, `openai`, `huggingface`, or `local`.",
            500,
        )
    return "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"


def get_provider_display_name(provider: str | None = None) -> str:
    resolved = get_model_provider(provider)
    return PROVIDER_LABELS.get(resolved, resolved.title())


def provider_supports_inference(provider: str | None = None) -> bool:
    return get_model_provider(provider) in {"ollama", "openai"}


def provider_supports_fine_tuning(provider: str | None = None) -> bool:
    return get_model_provider(provider) in {"openai", "huggingface", "local"}


def _normalize_ollama_base_url(raw: str | None) -> str:
    normalized = (raw or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/"
    return f"{normalized}/v1/"


def get_provider_base_url(provider: str | None = None) -> str | None:
    resolved = get_model_provider(provider)
    if resolved == "ollama":
        return _normalize_ollama_base_url(os.environ.get("OLLAMA_BASE_URL"))
    if resolved == "huggingface":
        return "https://huggingface.co"
    if resolved == "local":
        return None
    return os.environ.get("OPENAI_BASE_URL")


def provider_is_configured(provider: str | None = None) -> bool:
    resolved = get_model_provider(provider)
    if resolved == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if resolved == "huggingface":
        return huggingface_provider_is_configured()
    if resolved == "local":
        return local_training_provider_is_configured()
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
        "providers": [provider_payload("ollama"), provider_payload("openai"), provider_payload("huggingface"), provider_payload("local")],
        "defaultBaseModel": DEFAULT_BASE_MODEL,
        "defaultAgentModel": DEFAULT_AGENT_MODEL,
    }


def ensure_fine_tuning_available(provider: str | None = None) -> None:
    resolved = get_model_provider(provider)
    if provider_supports_fine_tuning(resolved):
        return
    raise ApiError(
        "FINE_TUNING_UNSUPPORTED",
        (
            f"Fine-tuning is not available for {get_provider_display_name(resolved)} profiles yet. "
            "Use OpenAI for managed fine-tuning, Hugging Face Jobs for cloud SFT training, or Local GPU QLoRA to use your own hardware."
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
else:
    DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL") or DEFAULT_OPENAI_BASE_MODEL
    DEFAULT_AGENT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("OPENAI_AGENT_MODEL") or "gpt-5.4-mini"


def _default_model_profiles() -> list[dict[str, Any]]:
    now = utc_now_iso()
    openai_base_model = os.environ.get("OPENAI_BASE_MODEL") or DEFAULT_OPENAI_BASE_MODEL
    openai_agent_model = os.environ.get("OPENAI_AGENT_MODEL") or "gpt-5.4-mini"
    huggingface_base_model = os.environ.get("HF_BASE_MODEL") or DEFAULT_HUGGINGFACE_BASE_MODEL
    local_base_model = os.environ.get("LOCAL_TRAINING_BASE_MODEL") or DEFAULT_LOCAL_BASE_MODEL
    ollama_medium_model = os.environ.get("OLLAMA_BASE_MODEL") or "qwen3:8b"
    ollama_small_model = os.environ.get("OLLAMA_SMALL_MODEL") or "qwen3:4b"

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
            "provider": "openai",
            "model": openai_base_model,
            "category": "large",
            "description": "Higher quality responses and a good base for managed fine-tuning.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-thinking",
            "name": "Thinking",
            "provider": "openai",
            "model": openai_agent_model,
            "category": "thinking",
            "description": "Reasoning-heavy profile for harder agent tasks.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-open-source-sft",
            "name": "Open Source SFT",
            "provider": "huggingface",
            "model": huggingface_base_model,
            "category": "large",
            "description": "Cloud fine-tuning for open-source models through Hugging Face Jobs.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-qlora",
            "name": "Local GPU QLoRA",
            "provider": "local",
            "model": local_base_model,
            "category": "large",
            "description": "Fine-tune open-source models on your own GPU and save LoRA adapters locally.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-qwen-05b",
            "name": "Qwen 0.5B Local",
            "provider": "local",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "category": "small",
            "description": "Tiny local smoke-test fine-tune that is great for pipeline validation.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-local-qwen-15b",
            "name": "Qwen 1.5B Local",
            "provider": "local",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "category": "medium",
            "description": "Fast local fine-tune with a little more quality than the 0.5B starter.",
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
            "id": "profile-hf-qwen-15b",
            "name": "Qwen 1.5B SFT",
            "provider": "huggingface",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "category": "medium",
            "description": "Small open-source cloud fine-tune through Hugging Face Jobs.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-hf-phi-35-mini",
            "name": "Phi 3.5 Mini SFT",
            "provider": "huggingface",
            "model": "microsoft/Phi-3.5-mini-instruct",
            "category": "medium",
            "description": "Cloud fine-tune with Phi 3.5 Mini through Hugging Face Jobs.",
            "createdAt": now,
            "updatedAt": now,
        },
        {
            "id": "profile-hf-mistral-7b",
            "name": "Mistral 7B SFT",
            "provider": "huggingface",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "category": "large",
            "description": "Mistral cloud fine-tune when you want a stronger open-source base model.",
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
        "jobBaseProfileId": local_id or openai_id or huggingface_id or large_id,
    }


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
        "providers": [provider_payload("ollama"), provider_payload("openai"), provider_payload("huggingface"), provider_payload("local")],
    }


def _ensure_model_profiles_initialized(state: dict[str, Any]) -> None:
    default_profiles = _default_model_profiles()

    if not state.get("model_profiles_initialized"):
        state["model_profiles"] = default_profiles
        state["model_profile_defaults"] = _default_model_profile_defaults(state["model_profiles"])
        state["model_profiles_initialized"] = True
    else:
        state["model_profiles"] = [profile for profile in state["model_profiles"] if not _is_seeded_gated_profile(profile)]
        existing_ids = {profile["id"] for profile in state["model_profiles"]}
        missing_profiles = [deepcopy(profile) for profile in default_profiles if profile["id"] not in existing_ids]
        if missing_profiles:
            state["model_profiles"].extend(missing_profiles)

    valid_ids = {profile["id"] for profile in state["model_profiles"]}
    fallback_defaults = _default_model_profile_defaults(state["model_profiles"])
    current_defaults = state.get("model_profile_defaults") or {}

    sanitized_defaults = {
        key: value
        for key, value in current_defaults.items()
        if key in MODEL_PROFILE_DEFAULT_KEYS and (value is None or value in valid_ids)
    }
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


def get_model_client(provider: str | None = None) -> OpenAI:
    resolved = get_model_provider(provider)
    if resolved in {"huggingface", "local"}:
        raise ApiError(
            "PLAYGROUND_RUN_FAILED",
            f"{get_provider_display_name(resolved)} profiles are training-only right now. Use Ollama or OpenAI profiles for inference.",
            400,
        )

    if OpenAI is None:
        raise ApiError("MODEL_CLIENT_PACKAGE_MISSING", "The OpenAI-compatible Python package is not installed.", 500)

    if resolved == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ApiError("MODEL_PROVIDER_NOT_CONFIGURED", "OPENAI_API_KEY is not configured.", 503)
        base_url = get_provider_base_url(resolved)
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

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


def run_model(prompt: str, model: str, provider: str | None = None) -> str:
    response = get_model_client(provider).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


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


def _build_local_training_job_config(
    *,
    job_id: str,
    dataset: dict[str, Any],
    base_model: str,
    hyperparameters: dict[str, Any] | None = None,
    export_gguf: bool = False,
    gguf_quantization: str = "q4_k_m",
    push_to_ollama: bool = False,
    ollama_model_name: str = "",
) -> LocalQLoraJobConfig:
    working_dir = ROOT / "uploads_python" / "jobs" / job_id
    output_dir = working_dir / "artifacts"
    return LocalQLoraJobConfig(
        job_id=job_id,
        dataset_id=dataset["id"],
        dataset_name=dataset["name"],
        dataset_path=dataset["storagePath"],
        base_model=base_model,
        working_dir=str(working_dir),
        output_dir=str(output_dir),
        model_output_path=str(output_dir / "adapter"),
        config_path=str(working_dir / "local_train_config.json"),
        status_path=str(working_dir / "local_train_status.json"),
        events_path=str(working_dir / "local_train_events.jsonl"),
        log_path=str(working_dir / "local_train.log"),
        metrics_path=str(working_dir / "local_train_metrics.json"),
        hyperparameters=hyperparameters or {},
        allow_cpu_fallback=os.environ.get("LOCAL_TRAINING_ALLOW_CPU_FALLBACK", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        seed=int(os.environ.get("LOCAL_TRAINING_SEED", "42")),
        eval_ratio=float(os.environ.get("LOCAL_TRAINING_EVAL_RATIO", "0.1")),
        export_gguf=export_gguf,
        gguf_quantization=gguf_quantization,
        push_to_ollama=push_to_ollama,
        ollama_model_name=ollama_model_name,
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
    export_gguf: bool = False,
    gguf_quantization: str = "q4_k_m",
    push_to_ollama: bool = False,
    ollama_model_name: str = "",
) -> dict[str, Any]:
    model_provider = get_model_provider(provider or "openai")
    ensure_fine_tuning_available(model_provider)
    dataset = retrieve_dataset_detail(dataset_id)
    if dataset["validationStatus"] != "VALID":
        raise ApiError("DATASET_INVALID", "Only valid datasets can be used for fine-tuning.", 400)
    _ensure_accessible_fine_tuning_model(base_model, model_provider)

    if model_provider == "local":
        job_id = uuid4().hex
        config = _build_local_training_job_config(
            job_id=job_id,
            dataset=dataset,
            base_model=base_model,
            hyperparameters=hyperparameters,
            export_gguf=export_gguf,
            gguf_quantization=gguf_quantization,
            push_to_ollama=push_to_ollama,
            ollama_model_name=ollama_model_name,
        )
        try:
            runtime_summary = ensure_local_training_ready(allow_cpu_fallback=config.allow_cpu_fallback)
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
            "localConfigPath": config.config_path,
            "localStatusPath": config.status_path,
            "localEventsPath": config.events_path,
            "localLogPath": config.log_path,
            "localMetricsPath": config.metrics_path,
            "localArtifactsPath": config.model_output_path,
            "progressJson": submission["progressJson"],
            "ollamaModelName": ollama_model_name or None,
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


def build_job_download_package(job_id: str) -> dict[str, str]:
    job = retrieve_job_detail(job_id)

    if job["modelProvider"] != "local":
        raise ApiError(
            "FINE_TUNING_UNSUPPORTED",
            "Direct model download is currently available only for Local GPU QLoRA jobs.",
            400,
        )

    if job["status"] != "succeeded":
        raise ApiError("JOB_SYNC_FAILED", "The local training job must finish successfully before download is available.", 400)

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
