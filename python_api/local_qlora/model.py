from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from python_api.debug_log import write_debug_log
from python_api.local_qlora.config import LocalQLoraJobConfig
from python_api.store import ROOT

_ATTENTION_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query_key_value",
    "c_attn",
    "c_proj",
    "out_proj",
    "Wqkv",
]

_MLP_TARGET_MODULES = [
    "gate_proj",
    "up_proj",
    "down_proj",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
    "fc1",
    "fc2",
    "w1",
    "w2",
    "w3",
]

_TARGET_MODULE_STRATEGIES = {"auto", "attention_only", "attention_mlp", "expanded"}


def local_training_enabled() -> bool:
    return os.environ.get("LOCAL_TRAINING_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def local_gpu_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def local_training_runtime_summary() -> dict[str, Any]:
    return {
        "enabled": local_training_enabled(),
        "gpuCount": local_gpu_count(),
        "unslothAvailable": _module_available("unsloth"),
        "dependencies": {
            "torch": _module_available("torch"),
            "transformers": _module_available("transformers"),
            "datasets": _module_available("datasets"),
            "peft": _module_available("peft"),
            "trl": _module_available("trl"),
            "accelerate": _module_available("accelerate"),
            "bitsandbytes": _module_available("bitsandbytes"),
        },
    }


def _prefer_unsloth(model_id: str) -> bool:
    lowered = model_id.strip().lower()
    if lowered.startswith("unsloth/"):
        return True
    if any(family in lowered for family in ("falcon", "granite")):
        return False
    return any(family in lowered for family in ("llama", "qwen", "mistral", "phi", "gemma", "smollm"))


def _unsloth_error_supports_fallback(error: Exception) -> bool:
    lowered = str(error).strip().lower()
    return any(
        token in lowered
        for token in (
            "not supported",
            "unsupported",
            "unknown model",
            "unrecognized configuration class",
            "model type",
            "architecture",
            "trust_remote_code",
        )
    )


def _normalize_target_module_strategy(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized in _TARGET_MODULE_STRATEGIES:
        return normalized
    return "auto"


def _family_default_target_modules(model_id: str) -> list[str]:
    lowered = model_id.lower()
    if "phi" in lowered:
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    if "falcon" in lowered:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    if any(family in lowered for family in ("gpt2", "gpt-j", "gpt-neox", "starcoder")):
        return ["c_attn", "c_proj", "fc1", "fc2"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _supported_target_modules_for_strategy(model_id: str, strategy: str) -> list[str]:
    family_defaults = _family_default_target_modules(model_id)
    if strategy == "attention_only":
        family_attention = [name for name in family_defaults if name in _ATTENTION_TARGET_MODULES]
        return family_attention or _ATTENTION_TARGET_MODULES
    if strategy == "expanded":
        return list(dict.fromkeys(family_defaults + _ATTENTION_TARGET_MODULES + _MLP_TARGET_MODULES))
    if strategy == "attention_mlp":
        return list(dict.fromkeys(family_defaults + _ATTENTION_TARGET_MODULES + _MLP_TARGET_MODULES))
    return family_defaults


def _discover_model_leaf_names(model: Any) -> set[str]:
    discovered: set[str] = set()
    for module_name, module in getattr(model, "named_modules", lambda: [])():
        if not module_name:
            continue
        try:
            if any(True for _ in module.children()):
                continue
        except Exception:
            continue
        leaf_name = module_name.rsplit(".", 1)[-1]
        discovered.add(leaf_name)
    return discovered


def _resolve_target_modules(
    model_id: str,
    override: list[str] | None = None,
    *,
    model: Any | None = None,
    strategy: str | None = None,
) -> list[str]:
    if override:
        return override

    normalized_strategy = _normalize_target_module_strategy(strategy)
    candidates = _supported_target_modules_for_strategy(model_id, normalized_strategy)
    if model is None:
        return candidates

    available_leaf_names = _discover_model_leaf_names(model)
    matched = [name for name in candidates if name in available_leaf_names]
    if matched:
        return matched
    return candidates


def load_quantized_model(config: LocalQLoraJobConfig) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Install `torch` to enable local GPU training."
        ) from error

    gpu_available = torch.cuda.is_available()
    if not gpu_available and not config.allow_cpu_fallback:
        raise RuntimeError(
            "Local GPU training requires a CUDA-visible GPU. Set LOCAL_TRAINING_ALLOW_CPU_FALLBACK=1 to override."
        )

    hyperparameters = config.resolved_hyperparameters()
    target_module_strategy = _normalize_target_module_strategy(hyperparameters.get("target_module_strategy"))
    workspace_compile_cache = ROOT / "unsloth_compiled_cache"
    configured_compile_location = os.environ.get("UNSLOTH_COMPILE_LOCATION")

    # ── Unsloth path: 2x faster, 70% less VRAM ───────────────────────────────
    if _prefer_unsloth(config.base_model):
        try:
            # region agent log
            write_debug_log(
                location="python_api/local_qlora/model.py:load_quantized_model:before_unsloth_import",
                message="Attempting Unsloth model load",
                data={
                    "jobId": config.job_id,
                    "cwd": os.getcwd(),
                    "workspaceCompileCacheExists": workspace_compile_cache.exists(),
                    "workspaceCompileCachePath": str(workspace_compile_cache),
                    "compileLocationEnv": configured_compile_location,
                    "baseModel": config.base_model,
                },
                run_id=config.job_id,
                hypothesis_id="H1",
            )
            # endregion
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.base_model,
                max_seq_length=int(hyperparameters["max_seq_length"]),
                load_in_4bit=True,
                dtype=None,  # auto-detects bfloat16 / float16
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=int(hyperparameters["lora_r"]),
                lora_alpha=int(hyperparameters["lora_alpha"]),
                lora_dropout=float(hyperparameters["lora_dropout"]),
                target_modules=_resolve_target_modules(
                    config.base_model,
                    model=model,
                    strategy=target_module_strategy,
                ),
                bias="none",
                use_gradient_checkpointing="unsloth",  # saves extra 30% VRAM vs standard
                random_state=config.seed,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            # region agent log
            write_debug_log(
                location="python_api/local_qlora/model.py:load_quantized_model:after_unsloth_load",
                message="Completed Unsloth model load",
                data={
                    "jobId": config.job_id,
                    "workspaceCompileCacheExists": workspace_compile_cache.exists(),
                    "workspaceCompileCacheFiles": sorted(path.name for path in Path(workspace_compile_cache).glob("*.py"))
                    if workspace_compile_cache.exists()
                    else [],
                    "compileLocationEnv": configured_compile_location,
                },
                run_id=config.job_id,
                hypothesis_id="H1",
            )
            # endregion

            # peft_config=None signals to the trainer that LoRA is already applied
            return model, tokenizer, None, torch

        except ImportError:
            # region agent log
            write_debug_log(
                location="python_api/local_qlora/model.py:load_quantized_model:unsloth_missing",
                message="Falling back to standard PEFT path",
                data={
                    "jobId": config.job_id,
                    "workspaceCompileCacheExists": workspace_compile_cache.exists(),
                    "compileLocationEnv": configured_compile_location,
                },
                run_id=config.job_id,
                hypothesis_id="H4",
            )
            # endregion
            pass  # Unsloth not installed — fall through to standard HF path
        except Exception as error:
            if not _unsloth_error_supports_fallback(error):
                raise
            write_debug_log(
                location="python_api/local_qlora/model.py:load_quantized_model:unsloth_fallback",
                message="Unsloth could not prepare this model family, falling back to standard PEFT path",
                data={
                    "jobId": config.job_id,
                    "baseModel": config.base_model,
                    "error": str(error),
                    "compileLocationEnv": configured_compile_location,
                },
                run_id=config.job_id,
                hypothesis_id="H4",
            )

    # ── Fallback: standard HuggingFace + PEFT (original behaviour) ───────────
    try:
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Install `torch`, `transformers`, `peft`, and `bitsandbytes` to enable local GPU QLoRA training."
        ) from error

    compute_dtype = torch.bfloat16 if gpu_available and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        device_map="auto" if gpu_available else None,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=int(hyperparameters["lora_r"]),
        lora_alpha=int(hyperparameters["lora_alpha"]),
        lora_dropout=float(hyperparameters["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_resolve_target_modules(
            config.base_model,
            model=model,
            strategy=target_module_strategy,
        ),
    )
    return model, tokenizer, peft_config, torch
