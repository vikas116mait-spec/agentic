#!/usr/bin/env python3
"""Recover an Ollama-registered GGUF for a completed Local QLoRA job.

Background: `python_api/local_qlora/train.py` tries to export the trained
LoRA adapter to GGUF via Unsloth and then register it with Ollama. If
Unsloth's bundled GGUF pipeline breaks (seen on some hosts with
"Unsloth: GGUF conversion failed: EOF when reading a line"), the job still
completes -- it saves the adapter -- but the Ollama side is never done,
leaving the playground unable to run the model.

This script redoes only the post-training steps using plain transformers +
peft + llama.cpp's `convert_hf_to_gguf.py`:

  1. Read `local_train_config.json` from the job directory.
  2. Load the base model, attach the LoRA adapter, and merge it into the
     base weights (`peft.merge_and_unload`).
  3. Save the merged model + tokenizer to `<job>/artifacts/merged/`.
  4. Run `convert_hf_to_gguf.py` to produce `<job>/artifacts/gguf/<name>.gguf`.
  5. Write a `Modelfile` and run `ollama create <name> -f Modelfile`.

Run with the project virtual environment's python, e.g.:

    .venv/bin/python scripts/recover_ollama.py \
        --job-dir uploads_python/jobs/<job_id>

Options:
    --ollama-name       Override the Ollama model name (default: from config).
    --outtype           GGUF output type: f16|bf16|q8_0|f32 (default q8_0).
    --no-register       Produce the GGUF + Modelfile but skip `ollama create`.
    --keep-merged       Keep the merged HF model dir (default: removed).
    --convert-script    Path to convert_hf_to_gguf.py (default: third_party/
                        llama.cpp/convert_hf_to_gguf.py relative to repo).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_convert_script() -> Path:
    """Locate llama.cpp's convert_hf_to_gguf.py.

    Checks, in order:
      1. `LLAMA_CPP_CONVERT_SCRIPT` env var (set by the Fly.io Dockerfile).
      2. `<repo>/third_party/llama.cpp/convert_hf_to_gguf.py` (local clone).
    """

    env_path = os.environ.get("LLAMA_CPP_CONVERT_SCRIPT", "").strip()
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    return REPO_ROOT / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"


DEFAULT_CONVERT_SCRIPT = _default_convert_script()


def _read_config(job_dir: Path) -> dict:
    config_path = job_dir / "local_train_config.json"
    if not config_path.is_file():
        raise SystemExit(f"local_train_config.json not found under {job_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_adapter_dir(config: dict, job_dir: Path) -> Path:
    adapter_dir_raw = config.get("modelOutputPath") or str(job_dir / "artifacts" / "adapter")
    adapter_dir = Path(adapter_dir_raw)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(
            f"Expected PEFT adapter at {adapter_dir}, but adapter_config.json is missing."
        )
    return adapter_dir


def _merge_adapter(base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    """Load base + adapter, merge LoRA, save merged model and tokenizer."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/3] Loading base model `{base_model}` (this may download weights)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print(f"[1/3] Loading adapter from `{adapter_dir}` and merging...", flush=True)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    print(f"[1/3] Saving merged model to `{merged_dir}`...", flush=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)

    # Prefer the tokenizer saved alongside the adapter (it carries any
    # training-time chat-template tweaks); fall back to the base model's.
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").is_file() else base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    tokenizer.save_pretrained(str(merged_dir))

    del merged, model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _convert_to_gguf(
    merged_dir: Path,
    gguf_path: Path,
    convert_script: Path,
    outtype: str,
    python_exe: str,
) -> None:
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    if gguf_path.is_file():
        gguf_path.unlink()

    cmd = [
        python_exe,
        str(convert_script),
        str(merged_dir),
        "--outtype",
        outtype,
        "--outfile",
        str(gguf_path),
    ]
    print(f"[2/3] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"convert_hf_to_gguf.py exited with code {result.returncode}. "
            f"See its stderr above for details."
        )
    if not gguf_path.is_file():
        raise SystemExit(f"Conversion reported success but {gguf_path} is missing.")


def _write_modelfile(gguf_path: Path) -> Path:
    modelfile_path = gguf_path.parent / "Modelfile"
    modelfile_path.write_text(
        f'FROM "{gguf_path}"\n'
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|eot_id|>"\n',
        encoding="utf-8",
    )
    return modelfile_path


def _register_with_ollama(model_name: str, modelfile: Path) -> None:
    cmd = ["ollama", "create", model_name, "-f", str(modelfile)]
    print(f"[3/3] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"`ollama create` failed with exit code {result.returncode}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-dir", required=True, help="Path to the job directory (contains local_train_config.json).")
    parser.add_argument("--ollama-name", default=None, help="Override the target Ollama model name.")
    parser.add_argument("--outtype", default="q8_0", choices=["f32", "f16", "bf16", "q8_0"], help="GGUF output type.")
    parser.add_argument("--no-register", action="store_true", help="Skip `ollama create` at the end.")
    parser.add_argument("--keep-merged", action="store_true", help="Keep the merged HF model dir after GGUF conversion.")
    parser.add_argument(
        "--convert-script",
        default=str(DEFAULT_CONVERT_SCRIPT),
        help="Path to llama.cpp's convert_hf_to_gguf.py.",
    )
    args = parser.parse_args()

    job_dir = Path(args.job_dir).expanduser().resolve()
    if not job_dir.is_dir():
        raise SystemExit(f"Job directory does not exist: {job_dir}")

    convert_script = Path(args.convert_script).expanduser().resolve()
    if not convert_script.is_file():
        raise SystemExit(
            f"convert_hf_to_gguf.py not found at {convert_script}. "
            f"Clone llama.cpp (e.g. `git clone --depth=1 https://github.com/ggerganov/llama.cpp third_party/llama.cpp`)."
        )

    config = _read_config(job_dir)
    base_model = config.get("baseModel")
    if not base_model:
        raise SystemExit("baseModel missing in local_train_config.json.")

    ollama_name = (args.ollama_name or config.get("ollamaModelName") or "").strip()
    if not args.no_register and not ollama_name:
        raise SystemExit("ollamaModelName is not set; pass --ollama-name explicitly or use --no-register.")

    adapter_dir = _resolve_adapter_dir(config, job_dir)
    artifacts_dir = adapter_dir.parent
    merged_dir = artifacts_dir / "merged"
    gguf_dir = artifacts_dir / "gguf"
    gguf_name = ollama_name or f"{job_dir.name}-merged"
    gguf_path = gguf_dir / f"{gguf_name}.{args.outtype}.gguf"

    print(f"Job directory:   {job_dir}")
    print(f"Base model:      {base_model}")
    print(f"Adapter dir:     {adapter_dir}")
    print(f"Merged output:   {merged_dir}")
    print(f"GGUF output:     {gguf_path}")
    print(f"Ollama name:     {ollama_name or '(skipping registration)'}")
    print()

    try:
        _merge_adapter(base_model, adapter_dir, merged_dir)
        _convert_to_gguf(
            merged_dir=merged_dir,
            gguf_path=gguf_path,
            convert_script=convert_script,
            outtype=args.outtype,
            python_exe=sys.executable,
        )
        modelfile = _write_modelfile(gguf_path)
        if args.no_register:
            print(f"Done. Modelfile at `{modelfile}`. Skipping `ollama create` (per --no-register).")
        else:
            _register_with_ollama(ollama_name, modelfile)
            print(f"Done. Model `{ollama_name}` is now registered with Ollama.")
    finally:
        if not args.keep_merged and merged_dir.exists():
            try:
                shutil.rmtree(merged_dir)
                print(f"Removed intermediate merged directory `{merged_dir}`.")
            except OSError as err:
                print(f"Warning: could not remove `{merged_dir}`: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
