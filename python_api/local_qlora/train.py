from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from python_api.local_qlora.config import LocalQLoraJobConfig
from python_api.local_qlora.data import build_sft_datasets
from python_api.local_qlora.evaluation import summarize_eval_metrics
from python_api.local_qlora.model import load_quantized_model, local_gpu_count
from python_api.local_qlora.state import append_event, read_status_file, write_status_file
from python_api.store import utc_now_iso

class LocalProgressCallback:
    def __init__(self, *, config: LocalQLoraJobConfig, status_path: Path, gpu_count: int) -> None:
        self.config = config
        self.status_path = status_path
        self.gpu_count = gpu_count
        self.started_at = time.monotonic()
        self.last_progress: dict[str, Any] = {}
        self.loss_history: list[dict[str, Any]] = []

    # Trainer integrations may call additional `on_*` hooks depending on the
    # installed transformers/trl versions. We only override the ones we need.
    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def _progress_payload(self, state: Any, logs: dict[str, Any] | None = None) -> dict[str, Any]:
        hyperparameters = self.config.resolved_hyperparameters()
        current_step = int(getattr(state, "global_step", 0) or 0)
        max_steps_raw = getattr(state, "max_steps", None)
        max_steps = int(max_steps_raw) if max_steps_raw else None
        epoch_raw = getattr(state, "epoch", None)
        elapsed_seconds = max(0.0, time.monotonic() - self.started_at)
        eta_seconds = None
        percent = None

        if max_steps and max_steps > 0:
            percent = min(100.0, round((current_step / max_steps) * 100, 1))
            if current_step > 0:
                eta_seconds = max(0.0, (elapsed_seconds / current_step) * max(max_steps - current_step, 0))

        progress = {
            "currentStep": current_step,
            "maxSteps": max_steps,
            "percent": percent,
            "epoch": round(float(epoch_raw), 2) if epoch_raw is not None else None,
            "totalEpochs": float(hyperparameters["num_train_epochs"]),
            "elapsedSeconds": round(elapsed_seconds, 1),
            "etaSeconds": round(eta_seconds, 1) if eta_seconds is not None else None,
            "gpuCount": self.gpu_count,
            "perDeviceTrainBatchSize": int(hyperparameters["per_device_train_batch_size"]),
            "gradientAccumulationSteps": int(hyperparameters["gradient_accumulation_steps"]),
            "learningRate": self.last_progress.get("learningRate"),
            "trainLoss": self.last_progress.get("trainLoss"),
            "evalLoss": self.last_progress.get("evalLoss"),
            "stepsPerSecond": self.last_progress.get("stepsPerSecond"),
            "samplesPerSecond": self.last_progress.get("samplesPerSecond"),
        }

        if logs:
            if logs.get("loss") is not None:
                progress["trainLoss"] = float(logs["loss"])
            if logs.get("eval_loss") is not None:
                progress["evalLoss"] = float(logs["eval_loss"])
            if logs.get("learning_rate") is not None:
                progress["learningRate"] = float(logs["learning_rate"])
            if logs.get("train_steps_per_second") is not None:
                progress["stepsPerSecond"] = float(logs["train_steps_per_second"])
            if logs.get("train_samples_per_second") is not None:
                progress["samplesPerSecond"] = float(logs["train_samples_per_second"])
            if logs.get("eval_steps_per_second") is not None and progress.get("stepsPerSecond") is None:
                progress["stepsPerSecond"] = float(logs["eval_steps_per_second"])
            if logs.get("eval_samples_per_second") is not None and progress.get("samplesPerSecond") is None:
                progress["samplesPerSecond"] = float(logs["eval_samples_per_second"])

        self.last_progress = progress
        return progress

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        write_status_file(
            self.status_path,
            {
                "stage": "training",
                "statusMessage": "Training LoRA adapters on the local runtime.",
                "progress": self._progress_payload(state),
            },
        )

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        progress = self._progress_payload(state, logs)
        if logs and logs.get("loss") is not None:
            self.loss_history.append({
                "step": int(getattr(state, "global_step", 0) or 0),
                "loss": round(float(logs["loss"]), 5),
            })
        write_status_file(
            self.status_path,
            {
                "stage": "training",
                "statusMessage": "Training LoRA adapters on the local runtime.",
                "progress": progress,
                "lossHistory": self.loss_history,
            },
        )

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:
        write_status_file(
            self.status_path,
            {
                "stage": "evaluating",
                "statusMessage": "Running evaluation on the held-out split.",
                "progress": self._progress_payload(state, metrics),
            },
        )

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        write_status_file(
            self.status_path,
            {
                "stage": "saving",
                "statusMessage": "Saving the latest LoRA adapter checkpoint.",
                "progress": self._progress_payload(state),
            },
        )


def _recommended_dataset_num_proc(train_records: int, eval_records: int) -> int:
    cpu_count = os.cpu_count() or 1
    largest_split = max(train_records, eval_records, 1)
    if largest_split < 128:
        return 1
    if largest_split < 512:
        return min(2, cpu_count)
    return min(8, cpu_count, largest_split)


def _recommended_dataloader_workers(train_records: int, eval_records: int) -> int:
    cpu_count = os.cpu_count() or 1
    largest_split = max(train_records, eval_records, 1)
    if largest_split < 256 or cpu_count < 4:
        return 0
    return min(2, max(cpu_count - 1, 0))


def _build_sft_config_kwargs(
    config: LocalQLoraJobConfig,
    torch_module: Any,
    has_eval: bool,
    *,
    train_records: int,
    eval_records: int,
) -> dict[str, Any]:
    hyperparameters = config.resolved_hyperparameters()
    gpu_available = local_gpu_count() > 0
    bf16_enabled = gpu_available and bool(getattr(torch_module.cuda, "is_bf16_supported", lambda: False)())
    dataset_num_proc = _recommended_dataset_num_proc(train_records, eval_records)
    dataloader_workers = _recommended_dataloader_workers(train_records, eval_records)

    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "num_train_epochs": float(hyperparameters["num_train_epochs"]),
        "per_device_train_batch_size": int(hyperparameters["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(hyperparameters["gradient_accumulation_steps"]),
        "learning_rate": float(hyperparameters["learning_rate"]),
        "optim": "adamw_8bit",
        "logging_steps": int(hyperparameters["logging_steps"]),
        "save_strategy": "steps",
        "save_total_limit": 2,
        "warmup_ratio": float(hyperparameters["warmup_ratio"]),
        "weight_decay": float(hyperparameters["weight_decay"]),
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        "report_to": "none",
        "run_name": f"local-qlora-{config.job_id[:8]}",
        "max_length": int(hyperparameters["max_seq_length"]),
        "dataset_num_proc": dataset_num_proc,
        "dataloader_num_workers": dataloader_workers,
    }

    max_steps = int(hyperparameters.get("max_steps") or 0)
    save_steps = int(hyperparameters["save_steps"])
    if max_steps > 0:
        kwargs["max_steps"] = max_steps
        save_steps = max(1, min(save_steps, max_steps))

    kwargs["save_steps"] = save_steps

    if bf16_enabled:
        kwargs["bf16"] = True
    elif gpu_available:
        kwargs["fp16"] = True

    kwargs["eval_strategy"] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = max(1, min(save_steps, max_steps)) if max_steps > 0 else max(10, save_steps)

    return kwargs


def _instantiate_sft_config(kwargs: dict[str, Any]) -> Any:
    try:
        from trl import SFTConfig
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install `trl` to enable local GPU QLoRA training.") from error

    try:
        return SFTConfig(**kwargs)
    except TypeError:
        translated = dict(kwargs)
        if "eval_strategy" in translated:
            translated["evaluation_strategy"] = translated.pop("eval_strategy")
        if "max_length" in translated:
            translated["max_seq_length"] = translated.pop("max_length")
        return SFTConfig(**translated)


def _instantiate_trainer(
    *,
    model: Any,
    tokenizer: Any,
    peft_config: Any,
    train_dataset: Any,
    eval_dataset: Any,
    training_args: Any,
) -> Any:
    try:
        from trl import SFTTrainer
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install `trl` to enable local GPU QLoRA training.") from error

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

    # peft_config is None when Unsloth is active (LoRA already fused into model)
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    signature = inspect.signature(SFTTrainer.__init__)
    parameters = signature.parameters

    if "dataset_text_field" in parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in parameters:
        trainer_kwargs["formatting_func"] = lambda sample: sample["text"]

    if "processing_class" in parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    return SFTTrainer(**trainer_kwargs)


def run_local_qlora_training(config: LocalQLoraJobConfig) -> None:
    status_path = Path(config.status_path)
    events_path = Path(config.events_path)
    metrics_path = Path(config.metrics_path)
    model_output_path = Path(config.model_output_path)
    model_output_path.mkdir(parents=True, exist_ok=True)
    warnings = list(read_status_file(status_path).get("warnings") or [])

    def record_warning(message: str, *, event_type: str) -> None:
        warnings.append(message)
        append_event(events_path, "warning", message, event_type=event_type)
        write_status_file(status_path, {"warnings": warnings})

    append_event(events_path, "info", f"Preparing local QLoRA training for {config.base_model}.", event_type="trainer_prepare")
    write_status_file(
        status_path,
        {
            "status": "running",
            "statusMessage": "Preparing local QLoRA training artifacts.",
            "startedAt": utc_now_iso(),
            "baseModel": config.base_model,
            "modelOutputPath": config.model_output_path,
        },
    )

    try:
        write_status_file(
            status_path,
            {
                "stage": "loading_model",
                "statusMessage": "Loading quantized base model and tokenizer.",
            },
        )

        model, tokenizer, peft_config, torch_module = load_quantized_model(config)
        unsloth_active = peft_config is None
        append_event(
            events_path,
            "info",
            f"Loaded model via {'Unsloth (2x faster, ~70% less VRAM)' if unsloth_active else 'HuggingFace + PEFT (standard mode)'}.",
            event_type="model_ready",
        )
        write_status_file(status_path, {"unslothActive": unsloth_active})

        write_status_file(
            status_path,
            {
                "stage": "loading_model",
                "statusMessage": "Formatting the dataset with the selected model template.",
            },
        )
        train_dataset, eval_dataset, dataset_stats = build_sft_datasets(config, tokenizer=tokenizer)
        append_event(
            events_path,
            "info",
            (
                f"Loaded {dataset_stats['totalRecords']} records "
                f"({dataset_stats['trainRecords']} train / {dataset_stats['evalRecords']} eval)."
            ),
            event_type="dataset_ready",
        )
        write_status_file(
            status_path,
            {
                "stage": "loading_model",
                "statusMessage": "Model and dataset are ready. Preparing the trainer.",
                "datasetStats": dataset_stats,
                "runtimeSummary": {
                    "gpuCount": local_gpu_count(),
                    "runtimePython": sys.executable,
                    "multiGpu": False,
                    "selectedGpu": (os.environ.get("LOCAL_TRAINING_SELECTED_GPU") or None),
                    "speedPreset": config.training_preset,
                    "datasetNumProc": _recommended_dataset_num_proc(
                        int(dataset_stats["trainRecords"]), int(dataset_stats["evalRecords"])
                    ),
                    "dataloaderWorkers": _recommended_dataloader_workers(
                        int(dataset_stats["trainRecords"]), int(dataset_stats["evalRecords"])
                    ),
                    "maxSeqLength": int(config.resolved_hyperparameters()["max_seq_length"]),
                    "gradientAccumulationSteps": int(config.resolved_hyperparameters()["gradient_accumulation_steps"]),
                },
            },
        )

        training_args = _instantiate_sft_config(
            _build_sft_config_kwargs(
                config=config,
                torch_module=torch_module,
                has_eval=eval_dataset is not None,
                train_records=int(dataset_stats["trainRecords"]),
                eval_records=int(dataset_stats["evalRecords"]),
            )
        )
        trainer = _instantiate_trainer(
            model=model,
            tokenizer=tokenizer,
            peft_config=peft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
        )
        progress_cb = LocalProgressCallback(config=config, status_path=status_path, gpu_count=local_gpu_count())
        trainer.add_callback(progress_cb)

        append_event(events_path, "info", "Starting local QLoRA fine-tuning loop.", event_type="training_started")
        write_status_file(
            status_path,
            {
                "stage": "training",
                "statusMessage": "Training LoRA adapters on the local runtime.",
            },
        )
        train_result = trainer.train()
        train_metrics = dict(train_result.metrics)

        write_status_file(
            status_path,
            {
                "stage": "evaluating",
                "statusMessage": "Running evaluation on the held-out split.",
            },
        )
        eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
        evaluation_summary = summarize_eval_metrics(eval_metrics)

        write_status_file(
            status_path,
            {
                "stage": "saving",
                "statusMessage": "Saving the LoRA adapter and tokenizer.",
            },
        )
        trainer.model.save_pretrained(model_output_path)
        tokenizer.save_pretrained(model_output_path)

        # Clean up the HF Trainer scratch directory (intermediate checkpoints,
        # auto-generated README, etc). Safe because the final adapter has
        # already been copied to `model_output_path` above. Older jobs whose
        # `model_output_path` sat inside `output_dir` are left untouched.
        try:
            trainer_scratch = Path(config.output_dir).resolve()
            final_adapter = Path(model_output_path).resolve()
            if trainer_scratch != final_adapter and trainer_scratch not in final_adapter.parents:
                shutil.rmtree(trainer_scratch, ignore_errors=True)
        except Exception:
            pass

        # ── GGUF export (only available via Unsloth) ──────────────────────────
        gguf_path: str | None = None
        modelfile_path: str | None = None
        ollama_registered = False
        if config.export_gguf:
            if not unsloth_active:
                warning_message = (
                    "GGUF export was skipped because Unsloth is not installed in LOCAL_TRAINING_PYTHON."
                    if not config.push_to_ollama
                    else "GGUF export and Ollama registration were skipped because Unsloth is not installed in LOCAL_TRAINING_PYTHON."
                )
                record_warning(warning_message, event_type="gguf_export_skipped")
            else:
                try:
                    from python_api.local_qlora.export import export_gguf, push_to_ollama, write_ollama_modelfile

                    write_status_file(
                        status_path,
                        {
                            "stage": "exporting",
                            "statusMessage": f"Exporting GGUF ({config.gguf_quantization}). This may take a few minutes.",
                        },
                    )
                    append_event(
                        events_path,
                        "info",
                        f"Exporting to GGUF format ({config.gguf_quantization}).",
                        event_type="gguf_export_started",
                    )
                    gguf_export_root = Path(config.working_dir)
                    gguf_path = export_gguf(trainer.model, tokenizer, gguf_export_root, config.gguf_quantization)
                    modelfile_path = write_ollama_modelfile(gguf_path)
                    append_event(events_path, "info", f"GGUF saved to {gguf_path}.", event_type="gguf_export_done")

                    if config.push_to_ollama and config.ollama_model_name:
                        append_event(
                            events_path,
                            "info",
                            f"Pushing to Ollama as '{config.ollama_model_name}'.",
                            event_type="ollama_push_started",
                        )
                        try:
                            push_to_ollama(gguf_path, config.ollama_model_name)
                            ollama_registered = True
                            append_event(
                                events_path,
                                "info",
                                f"Model available in Ollama as '{config.ollama_model_name}'.",
                                event_type="ollama_push_done",
                            )
                        except Exception as ollama_error:
                            recovery_command = f"ollama create {config.ollama_model_name} -f {modelfile_path}"
                            record_warning(
                                (
                                    f"GGUF export succeeded, but Ollama registration failed for '{config.ollama_model_name}': "
                                    f"{ollama_error}. Start Ollama, then run `{recovery_command}`."
                                ),
                                event_type="ollama_push_failed",
                            )
                except Exception as export_error:
                    record_warning(
                        f"GGUF export failed (adapter is still saved): {export_error}",
                        event_type="gguf_export_failed",
                    )

        metrics_payload = {
            "train": train_metrics,
            "eval": eval_metrics,
            "summary": evaluation_summary,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        append_event(
            events_path,
            "info",
            f"Local QLoRA training completed. Adapter saved to {config.model_output_path}.",
            event_type="training_succeeded",
        )
        final_status_message = (
            "Local GPU QLoRA training finished successfully."
            if not warnings
            else f"Local GPU QLoRA training finished with warnings. {warnings[0]}"
        )
        write_status_file(
            status_path,
            {
                "status": "succeeded",
                "stage": "succeeded",
                "statusMessage": final_status_message,
                "finishedAt": utc_now_iso(),
                "fineTunedModel": config.model_output_path,
                "modelOutputPath": config.model_output_path,
                "metricsPath": str(metrics_path),
                "trainedTokens": train_metrics.get("train_num_tokens"),
                "warnings": warnings,
                "ollamaRegistered": ollama_registered,
                "progress": {
                    "percent": 100,
                    "currentStep": train_metrics.get("global_step"),
                    "maxSteps": train_metrics.get("global_step"),
                    "epoch": float(config.resolved_hyperparameters()["num_train_epochs"]),
                    "totalEpochs": float(config.resolved_hyperparameters()["num_train_epochs"]),
                    "elapsedSeconds": train_metrics.get("train_runtime"),
                    "etaSeconds": 0,
                    "trainLoss": train_metrics.get("train_loss"),
                    "evalLoss": eval_metrics.get("eval_loss"),
                    "learningRate": None,
                    "stepsPerSecond": train_metrics.get("train_steps_per_second"),
                    "samplesPerSecond": train_metrics.get("train_samples_per_second"),
                    "gpuCount": local_gpu_count(),
                    "perDeviceTrainBatchSize": int(config.resolved_hyperparameters()["per_device_train_batch_size"]),
                    "gradientAccumulationSteps": int(config.resolved_hyperparameters()["gradient_accumulation_steps"]),
                },
                "resultFilesJson": [
                    {"type": "local_adapter", "path": config.model_output_path},
                    {"type": "local_metrics", "path": str(metrics_path)},
                    {"type": "local_log", "path": config.log_path},
                    *([{"type": "local_gguf", "path": gguf_path}] if gguf_path else []),
                    *([{"type": "local_modelfile", "path": modelfile_path}] if modelfile_path else []),
                ],
                "metrics": metrics_payload,
                "lossHistory": progress_cb.loss_history,
            },
        )
    except Exception as error:  # pragma: no cover
        append_event(events_path, "error", f"Local QLoRA training failed: {error}", event_type="training_failed")
        append_event(events_path, "error", traceback.format_exc(), event_type="training_traceback")
        write_status_file(
            status_path,
            {
                "status": "failed",
                "stage": "failed",
                "statusMessage": str(error),
                "finishedAt": utc_now_iso(),
            },
        )
        raise
