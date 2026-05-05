from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from python_api.store import (
    resolve_dataset_storage_path,
    resolve_job_storage_path,
    serialize_dataset_storage_path,
    serialize_job_storage_path,
)

LOCAL_TRAINING_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "fast": {
        "num_train_epochs": 1,
        "max_steps": 60,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "save_steps": 100,
        "logging_steps": 1,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "max_seq_length": 1024,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
    },
    "balanced": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "save_steps": 100,
        "logging_steps": 10,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_seq_length": 1024,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
    },
    "quality": {
        "num_train_epochs": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 8e-5,
        "save_steps": 50,
        "logging_steps": 10,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_seq_length": 1536,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
    },
}


def normalize_training_preset(value: str | None) -> str:
    if value in LOCAL_TRAINING_PRESET_DEFAULTS:
        return str(value)
    return "balanced"


@dataclass
class LocalQLoraJobConfig:
    job_id: str
    dataset_id: str
    dataset_name: str
    dataset_path: str
    base_model: str
    working_dir: str
    output_dir: str
    model_output_path: str
    config_path: str
    status_path: str
    events_path: str
    log_path: str
    metrics_path: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    allow_cpu_fallback: bool = False
    seed: int = 42
    eval_ratio: float = 0.1
    export_gguf: bool = True
    gguf_quantization: str = "q4_k_m"
    push_to_ollama: bool = False
    ollama_model_name: str = ""
    training_preset: str = "balanced"

    def resolved_hyperparameters(self) -> dict[str, Any]:
        defaults = LOCAL_TRAINING_PRESET_DEFAULTS[normalize_training_preset(self.training_preset)]
        return {**defaults, **(self.hyperparameters or {})}

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "datasetId": self.dataset_id,
            "datasetName": self.dataset_name,
            "datasetPath": serialize_dataset_storage_path(self.dataset_path) or self.dataset_path,
            "baseModel": self.base_model,
            "workingDir": serialize_job_storage_path(self.working_dir) or self.working_dir,
            "outputDir": serialize_job_storage_path(self.output_dir) or self.output_dir,
            "modelOutputPath": serialize_job_storage_path(self.model_output_path) or self.model_output_path,
            "configPath": serialize_job_storage_path(self.config_path) or self.config_path,
            "statusPath": serialize_job_storage_path(self.status_path) or self.status_path,
            "eventsPath": serialize_job_storage_path(self.events_path) or self.events_path,
            "logPath": serialize_job_storage_path(self.log_path) or self.log_path,
            "metricsPath": serialize_job_storage_path(self.metrics_path) or self.metrics_path,
            "hyperparameters": self.hyperparameters,
            "allowCpuFallback": self.allow_cpu_fallback,
            "seed": self.seed,
            "evalRatio": self.eval_ratio,
            "exportGguf": self.export_gguf,
            "ggufQuantization": self.gguf_quantization,
            "pushToOllama": self.push_to_ollama,
            "ollamaModelName": self.ollama_model_name,
            "trainingPreset": normalize_training_preset(self.training_preset),
        }

    def write(self) -> str:
        path = Path(self.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(path)

    @classmethod
    def from_file(cls, path: str | Path) -> "LocalQLoraJobConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            job_id=payload["jobId"],
            dataset_id=payload["datasetId"],
            dataset_name=payload["datasetName"],
            dataset_path=str(resolve_dataset_storage_path(payload["datasetPath"]) or payload["datasetPath"]),
            base_model=payload["baseModel"],
            working_dir=str(resolve_job_storage_path(payload["workingDir"]) or payload["workingDir"]),
            output_dir=str(resolve_job_storage_path(payload["outputDir"]) or payload["outputDir"]),
            model_output_path=str(resolve_job_storage_path(payload["modelOutputPath"]) or payload["modelOutputPath"]),
            config_path=str(resolve_job_storage_path(payload["configPath"]) or payload["configPath"]),
            status_path=str(resolve_job_storage_path(payload["statusPath"]) or payload["statusPath"]),
            events_path=str(resolve_job_storage_path(payload["eventsPath"]) or payload["eventsPath"]),
            log_path=str(resolve_job_storage_path(payload["logPath"]) or payload["logPath"]),
            metrics_path=str(resolve_job_storage_path(payload["metricsPath"]) or payload["metricsPath"]),
            hyperparameters=payload.get("hyperparameters") or {},
            allow_cpu_fallback=bool(payload.get("allowCpuFallback", False)),
            seed=int(payload.get("seed", 42)),
            eval_ratio=float(payload.get("evalRatio", 0.1)),
            export_gguf=bool(payload.get("exportGguf", True)),
            gguf_quantization=str(payload.get("ggufQuantization", "q4_k_m")),
            push_to_ollama=bool(payload.get("pushToOllama", False)),
            ollama_model_name=str(payload.get("ollamaModelName", "")),
            training_preset=normalize_training_preset(payload.get("trainingPreset")),
        )
