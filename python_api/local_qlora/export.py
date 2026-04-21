from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def normalize_ollama_host(raw: str | None = None) -> str:
    normalized = (raw or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized or DEFAULT_OLLAMA_BASE_URL


def ollama_runtime_summary(raw: str | None = None) -> dict[str, Any]:
    host = normalize_ollama_host(raw)
    summary: dict[str, Any] = {
        "host": host,
        "cliAvailable": shutil.which("ollama") is not None,
        "reachable": False,
        "error": None,
    }

    request = Request(f"{host}/api/tags", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=2) as response:
            summary["reachable"] = 200 <= getattr(response, "status", 200) < 500
    except URLError as error:
        summary["error"] = str(getattr(error, "reason", error))
    except Exception as error:  # pragma: no cover
        summary["error"] = str(error)

    return summary


def ensure_ollama_runtime_ready(raw: str | None = None) -> dict[str, Any]:
    summary = ollama_runtime_summary(raw)
    if not summary["cliAvailable"]:
        raise RuntimeError(
            "The `ollama` CLI is not installed or not on PATH. Install Ollama before enabling Push to Ollama."
        )
    if not summary["reachable"]:
        raise RuntimeError(
            f"Ollama is not reachable at {summary['host']}. Start `ollama serve` or update OLLAMA_BASE_URL before enabling Push to Ollama."
        )
    return summary


def write_ollama_modelfile(gguf_path: str) -> str:
    modelfile_path = Path(gguf_path).parent / "Modelfile"
    modelfile_path.write_text(
        f'FROM "{gguf_path}"\n'
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|eot_id|>"\n',
        encoding="utf-8",
    )
    return str(modelfile_path)


def _ollama_cli_env(host: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = host
    return env


def export_gguf(model: Any, tokenizer: Any, output_dir: str | Path, quantization: str = "q4_k_m") -> str:
    """Export merged model to GGUF via Unsloth. Returns path to the .gguf file."""
    gguf_dir = Path(output_dir) / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=quantization,
    )

    candidates = list(gguf_dir.glob("*.gguf"))
    if not candidates:
        raise RuntimeError(f"GGUF export produced no .gguf file in {gguf_dir}")
    return str(candidates[0])


def push_to_ollama(gguf_path: str, model_name: str) -> None:
    """Register the GGUF with local Ollama under model_name."""
    runtime = ensure_ollama_runtime_ready()
    modelfile_path = write_ollama_modelfile(gguf_path)
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        env=_ollama_cli_env(runtime["host"]),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama create failed for `{model_name}` using `{modelfile_path}`: {result.stderr.strip() or result.stdout.strip()}"
        )
