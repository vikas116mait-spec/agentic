from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


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
    modelfile_path = Path(gguf_path).parent / "Modelfile"
    modelfile_path.write_text(
        f'FROM "{gguf_path}"\n'
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|eot_id|>"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama create failed: {result.stderr.strip() or result.stdout.strip()}"
        )
