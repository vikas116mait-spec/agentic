from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from python_api.local_qlora.config import LocalQLoraJobConfig
from python_api.local_qlora.export import ensure_ollama_runtime_ready, ollama_runtime_summary
from python_api.local_qlora.model import local_training_enabled
from python_api.local_qlora.state import append_event, read_status_file, write_status_file
from python_api.store import ROOT, utc_now_iso


logger = logging.getLogger("uvicorn.error")


def configured_local_training_python() -> str:
    return (os.environ.get("LOCAL_TRAINING_PYTHON") or "").strip()


def local_training_python() -> str:
    return configured_local_training_python() or sys.executable


def local_training_provider_is_configured() -> bool:
    if not local_training_enabled():
        return False
    python_executable = configured_local_training_python()
    if not python_executable:
        return False
    return Path(python_executable).exists() if os.path.isabs(python_executable) else shutil.which(python_executable) is not None


def _probe_training_runtime() -> dict[str, Any]:
    probe = """
import importlib.util
import json
import sys

def available(name):
    return importlib.util.find_spec(name) is not None

summary = {
    "python": sys.executable,
    "dependencies": {
        "torch": available("torch"),
        "transformers": available("transformers"),
        "datasets": available("datasets"),
        "peft": available("peft"),
        "trl": available("trl"),
        "accelerate": available("accelerate"),
        "bitsandbytes": available("bitsandbytes"),
    },
    "gpuCount": 0,
    "unslothAvailable": available("unsloth"),
}

if summary["dependencies"]["torch"]:
    import torch
    summary["gpuCount"] = torch.cuda.device_count() if torch.cuda.is_available() else 0

print(json.dumps(summary))
""".strip()

    result = subprocess.run(
        [local_training_python(), "-c", probe],
        cwd=str(ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect the local training runtime: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout)


def ensure_local_training_ready(
    *,
    allow_cpu_fallback: bool = False,
    export_gguf: bool = False,
    push_to_ollama: bool = False,
) -> dict[str, Any]:
    if not local_training_enabled():
        raise RuntimeError("Local GPU QLoRA training is disabled. Set LOCAL_TRAINING_ENABLED=1 to use your own GPU.")

    configured_python = configured_local_training_python()
    if not configured_python:
        raise RuntimeError(
            "Set LOCAL_TRAINING_PYTHON to the Python executable inside your training venv before starting a local GPU QLoRA job."
        )

    python_executable = local_training_python()
    python_exists = Path(python_executable).exists() if os.path.isabs(python_executable) else shutil.which(python_executable) is not None
    if not python_exists:
        raise RuntimeError(
            f"LOCAL_TRAINING_PYTHON points to `{python_executable}`, but that executable was not found. Update the path to your training venv Python."
        )

    if push_to_ollama and not export_gguf:
        raise RuntimeError("Push to Ollama requires GGUF export. Enable exportGguf before enabling Push to Ollama.")

    summary = _probe_training_runtime()
    missing = [name for name, available in summary["dependencies"].items() if not available]
    if missing:
        raise RuntimeError(
            "Local GPU QLoRA training dependencies are missing: "
            + ", ".join(sorted(missing))
            + ". Install the local training requirements into the interpreter from LOCAL_TRAINING_PYTHON before starting a local training job."
        )

    if summary["gpuCount"] == 0 and not allow_cpu_fallback:
        raise RuntimeError(
            "No CUDA-visible GPU was detected by the Python runtime. Set LOCAL_TRAINING_ALLOW_CPU_FALLBACK=1 only if you intentionally want CPU fallback."
        )

    warnings: list[str] = []
    if export_gguf and not summary.get("unslothAvailable"):
        warnings.append(
            "Unsloth is not installed in LOCAL_TRAINING_PYTHON, so this run will still save the adapter but skip GGUF export and Ollama registration."
        )

    if push_to_ollama:
        summary["ollama"] = ensure_ollama_runtime_ready()

    if warnings:
        summary["warnings"] = warnings

    return summary


def inspect_local_training_runtime() -> dict[str, Any]:
    configured_python = configured_local_training_python()
    python_executable = local_training_python()
    python_exists = Path(python_executable).exists() if os.path.isabs(python_executable) else shutil.which(python_executable) is not None
    payload: dict[str, Any] = {
        "enabled": local_training_enabled(),
        "configuredPython": configured_python or None,
        "pythonExists": python_exists,
        "providerConfigured": local_training_provider_is_configured(),
        "runtimePython": None,
        "gpuCount": 0,
        "unslothAvailable": False,
        "dependencies": {},
        "missingDependencies": [],
        "warnings": [],
        "ollama": ollama_runtime_summary(),
    }

    if not payload["enabled"]:
        payload["warnings"].append("Local GPU QLoRA is disabled. Set LOCAL_TRAINING_ENABLED=1 to enable the free local path.")
        return payload

    if not configured_python:
        payload["warnings"].append("Set LOCAL_TRAINING_PYTHON to the Python executable inside your training venv.")
        return payload

    if not python_exists:
        payload["warnings"].append(f"LOCAL_TRAINING_PYTHON points to `{python_executable}`, but that executable was not found.")
        return payload

    try:
        runtime = _probe_training_runtime()
    except RuntimeError as error:
        payload["warnings"].append(str(error))
        return payload

    payload["runtimePython"] = runtime.get("python")
    payload["gpuCount"] = int(runtime.get("gpuCount") or 0)
    payload["unslothAvailable"] = bool(runtime.get("unslothAvailable"))
    payload["dependencies"] = runtime.get("dependencies") or {}
    payload["missingDependencies"] = sorted(
        name for name, available in payload["dependencies"].items() if not available
    )

    if payload["gpuCount"] == 0:
        payload["warnings"].append("No CUDA-visible GPU was detected for the training runtime.")
    if payload["missingDependencies"]:
        payload["warnings"].append(
            "Missing local training dependencies: " + ", ".join(payload["missingDependencies"])
        )
    if not payload["unslothAvailable"]:
        payload["warnings"].append("Unsloth is not installed, so local training will use the slower HuggingFace + PEFT path.")

    return payload


def _runtime_summary_payload(
    runtime: dict[str, Any],
    *,
    gpu_count: int,
    use_multi_gpu: bool,
    selected_gpu: dict[str, int | str] | None,
    config: LocalQLoraJobConfig,
) -> dict[str, Any]:
    ollama_summary = runtime.get("ollama") or {}
    return {
        "gpuCount": gpu_count,
        "runtimePython": runtime.get("python"),
        "multiGpu": use_multi_gpu,
        "selectedGpu": selected_gpu["index"] if selected_gpu else None,
        "selectedGpuFreeMb": selected_gpu["memoryFreeMb"] if selected_gpu else None,
        "unslothAvailable": runtime.get("unslothAvailable"),
        "speedPreset": config.training_preset,
        "maxSeqLength": int(config.resolved_hyperparameters()["max_seq_length"]),
        "gradientAccumulationSteps": int(config.resolved_hyperparameters()["gradient_accumulation_steps"]),
        "warnings": runtime.get("warnings") or [],
        "ollamaHost": ollama_summary.get("host"),
        "ollamaReachable": ollama_summary.get("reachable"),
        "ollamaCliAvailable": ollama_summary.get("cliAvailable"),
    }


def _process_is_running(process_id: int | None) -> bool:
    if not process_id or process_id <= 0:
        return False
    proc_stat = Path(f"/proc/{process_id}/stat")
    if proc_stat.exists():
        try:
            fields = proc_stat.read_text(encoding="utf-8").split()
            if len(fields) >= 3 and fields[2] == "Z":
                return False
        except OSError:
            return False
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


def _summarize_failure_from_log(log_path: str | None) -> str | None:
    if not log_path:
        return None

    path = Path(log_path)
    if not path.exists():
        return None

    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except OSError:
        return None

    if not lines:
        return None

    for line in reversed(lines):
        if "ModuleNotFoundError:" in line:
            return line
        if "RuntimeError:" in line:
            return line
        if "ValueError:" in line:
            return line
        if "ImportError:" in line:
            return line

    for line in reversed(lines):
        if "ChildFailedError" in line:
            return "Local QLoRA launcher failed. Open the trainer log for the Python traceback."

    for line in reversed(lines):
        if line.startswith("torch.distributed.elastic.multiprocessing.errors.ChildFailedError"):
            return "Local QLoRA launcher failed during multi-GPU startup."

    return None


def _query_host_gpu_inventory() -> list[dict[str, int | str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=str(ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    inventory: list[dict[str, int | str]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
            total_mb = int(parts[2])
            used_mb = int(parts[3])
        except ValueError:
            continue
        inventory.append(
            {
                "index": index,
                "name": parts[1],
                "memoryTotalMb": total_mb,
                "memoryUsedMb": used_mb,
                "memoryFreeMb": max(total_mb - used_mb, 0),
            }
        )
    return inventory


def _select_single_gpu() -> dict[str, int | str] | None:
    preferred_gpu = (os.environ.get("LOCAL_TRAINING_GPU_INDEX") or "").strip()
    if preferred_gpu:
        try:
            preferred_index = int(preferred_gpu)
        except ValueError:
            preferred_index = None
        else:
            inventory = _query_host_gpu_inventory()
            for gpu in inventory:
                if gpu["index"] == preferred_index:
                    return gpu
            return {
                "index": preferred_index,
                "name": "Pinned GPU",
                "memoryTotalMb": 0,
                "memoryUsedMb": 0,
                "memoryFreeMb": 0,
            }

    inventory = _query_host_gpu_inventory()
    if not inventory:
        return None

    return max(inventory, key=lambda gpu: int(gpu["memoryFreeMb"]))


def _poll_gpu_metrics(status_path: Path, process_id: int, stop_event: threading.Event) -> None:
    """Poll nvidia-smi every 3 seconds and write GPU stats to status.json while training runs."""
    while not stop_event.is_set():
        if not _process_is_running(process_id):
            break
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            gpus: list[dict] = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 5:
                    try:
                        gpus.append(
                            {
                                "index": int(parts[0]),
                                "memUsedMb": int(parts[1]),
                                "memTotalMb": int(parts[2]),
                                "utilizationPct": int(parts[3]),
                                "temperatureC": int(parts[4]),
                            }
                        )
                    except ValueError:
                        pass
            if gpus:
                write_status_file(status_path, {"gpuMetrics": gpus})
        stop_event.wait(timeout=3)


def spawn_local_training_job(config: LocalQLoraJobConfig, runtime_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = config.write()
    log_path = Path(config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    runtime = runtime_summary or _probe_training_runtime()
    gpu_count = int(runtime.get("gpuCount") or 0)
    use_multi_gpu = gpu_count > 1 and os.environ.get("LOCAL_TRAINING_MULTI_GPU", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    selected_gpu = None if use_multi_gpu else _select_single_gpu()

    if use_multi_gpu:
        command = [
            local_training_python(),
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(gpu_count),
            "-m",
            "python_api.local_qlora.runner",
            str(config_path),
        ]
    else:
        command = [local_training_python(), "-m", "python_api.local_qlora.runner", str(config_path)]

    runtime_payload = _runtime_summary_payload(
        runtime,
        gpu_count=gpu_count,
        use_multi_gpu=use_multi_gpu,
        selected_gpu=selected_gpu,
        config=config,
    )

    initial_status = {
        "status": "running",
        "stage": "queued",
        "statusMessage": "Local GPU QLoRA training process started.",
        "createdAt": utc_now_iso(),
        "updatedAt": utc_now_iso(),
        "baseModel": config.base_model,
        "modelOutputPath": config.model_output_path,
        "runtimeSummary": runtime_payload,
        "warnings": runtime_payload["warnings"],
        "ollamaRegistered": False,
    }
    write_status_file(Path(config.status_path), initial_status)
    append_event(Path(config.events_path), "info", "Local GPU QLoRA job queued.", event_type="job_created")
    for warning in runtime_payload["warnings"]:
        append_event(Path(config.events_path), "warning", warning, event_type="preflight_warning")
    if selected_gpu:
        append_event(
            Path(config.events_path),
            "info",
            (
                f"Selected GPU {selected_gpu['index']} "
                f"with {selected_gpu['memoryFreeMb']} MB free for this run."
            ),
            event_type="gpu_selected",
        )

    launch_env = os.environ.copy()
    if selected_gpu and not use_multi_gpu:
        launch_env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])
        launch_env["LOCAL_TRAINING_SELECTED_GPU"] = str(selected_gpu["index"])
    launch_env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=launch_env,
        )

    # Start GPU metrics polling — stops automatically when the process exits
    _gpu_poll_stop = threading.Event()
    threading.Thread(
        target=_poll_gpu_metrics,
        args=(Path(config.status_path), process.pid, _gpu_poll_stop),
        name=f"gpu-metrics-{config.job_id[:8]}",
        daemon=True,
    ).start()

    write_status_file(
        Path(config.status_path),
        {
            **initial_status,
            "processId": process.pid,
            "runtimePython": runtime.get("python"),
            "statusMessage": (
                f"Local GPU QLoRA training is running with PID {process.pid}."
                + (
                    f" Using GPU {selected_gpu['index']}."
                    if selected_gpu and not use_multi_gpu
                    else ""
                )
                + (" Multi-GPU accelerate launch enabled." if use_multi_gpu else "")
            ),
        },
    )
    append_event(
        Path(config.events_path),
        "info",
        f"Spawned local trainer process {process.pid}.",
        event_type="trainer_started",
    )

    return {
        "providerJobId": str(process.pid),
        "providerJobUrl": None,
        "providerNamespace": None,
        "status": "running",
        "statusMessage": f"Local GPU QLoRA training started with PID {process.pid}.",
        "fineTunedModel": config.model_output_path,
        "modelRepoId": config.model_output_path,
        "modelRepoUrl": None,
        "datasetRepoId": config.dataset_path,
        "datasetRepoPath": config.dataset_path,
        "datasetRepoUrl": None,
        "trackioUrl": None,
        "uploadedToHuggingFaceAt": None,
        "resultFilesJson": [
            {"type": "local_config", "path": config_path},
            {"type": "local_log", "path": config.log_path},
            {"type": "local_status", "path": config.status_path},
            {"type": "local_adapter", "path": config.model_output_path},
        ],
        "trainingBackend": "local_qlora",
        "progressJson": {
            "stage": "queued",
            "runtimeSummary": runtime_payload,
            "datasetStats": None,
            "progress": None,
            "metrics": None,
            "warnings": runtime_payload["warnings"],
            "ollamaRegistered": False,
        },
    }


def inspect_local_training_job(job_record: dict[str, Any]) -> dict[str, Any]:
    status_path = Path(job_record["localStatusPath"])
    status_payload = read_status_file(status_path)
    process_id = status_payload.get("processId") or job_record.get("providerJobId")
    try:
        process_id_int = int(process_id) if process_id is not None else None
    except (TypeError, ValueError):
        process_id_int = None

    running = _process_is_running(process_id_int)
    status = status_payload.get("status") or job_record.get("status") or "queued"
    if status in {"queued", "running"} and not running:
        status = "failed" if status_payload.get("status") not in {"succeeded", "cancelled"} else status_payload["status"]

    status_message = status_payload.get("statusMessage")
    failure_summary = _summarize_failure_from_log(job_record.get("localLogPath"))
    if status == "failed":
        if failure_summary:
            status_message = failure_summary
        elif not status_message or "running with PID" in str(status_message):
            status_message = "Local trainer process exited unexpectedly. Check the trainer log for details."

    return {
        "status": status,
        "statusMessage": status_message,
        "providerJobId": str(process_id_int) if process_id_int else job_record.get("providerJobId"),
        "providerJobUrl": None,
        "providerNamespace": None,
        "fineTunedModel": status_payload.get("fineTunedModel") or job_record.get("fineTunedModel"),
        "trainedTokens": status_payload.get("trainedTokens"),
        "resultFilesJson": status_payload.get("resultFilesJson") or job_record.get("resultFilesJson"),
        "modelRepoId": status_payload.get("modelOutputPath") or job_record.get("modelRepoId"),
        "modelRepoUrl": None,
        "datasetRepoId": job_record.get("datasetRepoId"),
        "datasetRepoPath": job_record.get("datasetRepoPath"),
        "datasetRepoUrl": None,
        "trackioUrl": None,
        "updatedAt": status_payload.get("updatedAt") or utc_now_iso(),
        "finishedAt": status_payload.get("finishedAt"),
        "progressJson": {
            "stage": status_payload.get("stage"),
            "runtimeSummary": status_payload.get("runtimeSummary"),
            "datasetStats": status_payload.get("datasetStats"),
            "progress": status_payload.get("progress"),
            "metrics": status_payload.get("metrics"),
            "lossHistory": status_payload.get("lossHistory") or [],
            "gpuMetrics": status_payload.get("gpuMetrics") or [],
            "unslothActive": status_payload.get("unslothActive"),
            "warnings": status_payload.get("warnings") or [],
            "ollamaRegistered": status_payload.get("ollamaRegistered"),
        },
    }


def collect_local_training_events(
    *,
    events_path: str,
    existing_event_ids: set[str],
) -> list[dict[str, Any]]:
    path = Path(events_path)
    if not path.exists():
        return []

    collected: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["id"] in existing_event_ids:
            continue
        collected.append(event)
    return collected


def cancel_local_training_job(job_record: dict[str, Any]) -> dict[str, Any]:
    status_path = Path(job_record["localStatusPath"])
    events_path = Path(job_record["localEventsPath"])
    status_payload = read_status_file(status_path)
    process_id = status_payload.get("processId") or job_record.get("providerJobId")
    try:
        process_id_int = int(process_id) if process_id is not None else None
    except (TypeError, ValueError):
        process_id_int = None

    if process_id_int and _process_is_running(process_id_int):
        try:
            os.killpg(process_id_int, signal.SIGTERM)
        except OSError:
            os.kill(process_id_int, signal.SIGTERM)

    write_status_file(
        status_path,
        {
            **status_payload,
            "status": "cancelled",
            "statusMessage": "Cancellation requested for the local GPU QLoRA job.",
            "updatedAt": utc_now_iso(),
            "finishedAt": utc_now_iso(),
        },
    )
    append_event(events_path, "warning", "Cancellation requested for the local trainer process.", event_type="job_cancelled")

    return inspect_local_training_job(job_record)
