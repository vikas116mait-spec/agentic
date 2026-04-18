from __future__ import annotations

import importlib.util
import os
from typing import Any

from python_api.local_qlora.config import LocalQLoraJobConfig


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


def _resolve_target_modules(model_id: str, override: list[str] | None = None) -> list[str]:
    if override:
        return override

    lowered = model_id.lower()
    if "phi" in lowered:
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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

    # ── Unsloth path: 2x faster, 70% less VRAM ───────────────────────────────
    try:
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
            target_modules=_resolve_target_modules(config.base_model),
            bias="none",
            use_gradient_checkpointing="unsloth",  # saves extra 30% VRAM vs standard
            random_state=config.seed,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # peft_config=None signals to the trainer that LoRA is already applied
        return model, tokenizer, None, torch

    except ImportError:
        pass  # Unsloth not installed — fall through to standard HF path

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
        target_modules=_resolve_target_modules(config.base_model),
    )
    return model, tokenizer, peft_config, torch
