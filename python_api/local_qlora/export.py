from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
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
    resolved_gguf = Path(gguf_path).expanduser().resolve()
    modelfile_path = resolved_gguf.parent / "Modelfile"
    modelfile_path.write_text(
        f"FROM {resolved_gguf}\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|eot_id|>"\n',
        encoding="utf-8",
    )
    return str(modelfile_path)


def _ollama_cli_env(host: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = host
    return env


def _default_convert_script() -> Path:
    env_path = os.environ.get("LLAMA_CPP_CONVERT_SCRIPT", "").strip()
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    return REPO_ROOT / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"


def resolve_fallback_gguf_outtype(requested_quantization: str) -> tuple[str, str | None]:
    normalized = (requested_quantization or "q4_k_m").strip().lower()
    if normalized in {"f32", "f16", "bf16", "q8_0"}:
        return normalized, None
    if normalized == "q4_k_m":
        return (
            "q8_0",
            "Requested GGUF quantization `q4_k_m` is not supported by the llama.cpp fallback exporter; using `q8_0` instead.",
        )
    return (
        "q8_0",
        f"Requested GGUF quantization `{requested_quantization}` is not supported by the llama.cpp fallback exporter; using `q8_0` instead.",
    )


def _merge_saved_adapter(base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16

    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)

    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").is_file() else base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))

    del tokenizer, merged, model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def export_saved_adapter_to_gguf(
    *,
    base_model: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    quantization: str = "q4_k_m",
    python_executable: str | None = None,
) -> str:
    """Fallback GGUF export path that merges the saved adapter, then runs llama.cpp conversion."""
    adapter_path = Path(adapter_dir)
    if not (adapter_path / "adapter_config.json").is_file():
        raise RuntimeError(f"Expected PEFT adapter at {adapter_path}, but adapter_config.json is missing.")

    convert_script = _default_convert_script()
    if not convert_script.is_file():
        raise RuntimeError(
            f"convert_hf_to_gguf.py not found at {convert_script}. "
            "Set LLAMA_CPP_CONVERT_SCRIPT or clone llama.cpp into third_party/llama.cpp."
        )

    outtype, _ = resolve_fallback_gguf_outtype(quantization)
    export_root = Path(output_dir)
    merged_dir = export_root / "merged"
    gguf_dir = export_root / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_dir / f"model.{outtype}.gguf"

    if gguf_path.is_file():
        gguf_path.unlink()

    try:
        shutil.rmtree(merged_dir, ignore_errors=True)
        _merge_saved_adapter(base_model, adapter_path, merged_dir)
        result = subprocess.run(
            [
                python_executable or sys.executable,
                str(convert_script),
                str(merged_dir),
                "--outtype",
                outtype,
                "--outfile",
                str(gguf_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "llama.cpp GGUF conversion failed: "
                f"{result.stderr.strip() or result.stdout.strip() or f'exit code {result.returncode}'}"
            )
        if not gguf_path.is_file():
            raise RuntimeError(f"GGUF conversion reported success but {gguf_path} is missing.")
        return str(gguf_path)
    finally:
        shutil.rmtree(merged_dir, ignore_errors=True)


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
