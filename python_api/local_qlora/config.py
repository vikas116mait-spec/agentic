from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

    def resolved_hyperparameters(self) -> dict[str, Any]:
        defaults = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "save_steps": 50,
            "logging_steps": 10,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "max_seq_length": 1024,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
        }
        return {**defaults, **(self.hyperparameters or {})}

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "datasetId": self.dataset_id,
            "datasetName": self.dataset_name,
            "datasetPath": self.dataset_path,
            "baseModel": self.base_model,
            "workingDir": self.working_dir,
            "outputDir": self.output_dir,
            "modelOutputPath": self.model_output_path,
            "configPath": self.config_path,
            "statusPath": self.status_path,
            "eventsPath": self.events_path,
            "logPath": self.log_path,
            "metricsPath": self.metrics_path,
            "hyperparameters": self.hyperparameters,
            "allowCpuFallback": self.allow_cpu_fallback,
            "seed": self.seed,
            "evalRatio": self.eval_ratio,
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
            dataset_path=payload["datasetPath"],
            base_model=payload["baseModel"],
            working_dir=payload["workingDir"],
            output_dir=payload["outputDir"],
            model_output_path=payload["modelOutputPath"],
            config_path=payload["configPath"],
            status_path=payload["statusPath"],
            events_path=payload["eventsPath"],
            log_path=payload["logPath"],
            metrics_path=payload["metricsPath"],
            hyperparameters=payload.get("hyperparameters") or {},
            allow_cpu_fallback=bool(payload.get("allowCpuFallback", False)),
            seed=int(payload.get("seed", 42)),
            eval_ratio=float(payload.get("evalRatio", 0.1)),
        )
