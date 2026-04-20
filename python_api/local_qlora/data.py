from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from python_api.local_qlora.config import LocalQLoraJobConfig


def _format_messages_record(record: dict[str, Any]) -> str:
    chunks: list[str] = []
    for message in record["messages"]:
        role = str(message.get("role", "user")).strip().title()
        content = message.get("content", "")
        if isinstance(content, list):
            rendered = json.dumps(content, ensure_ascii=False)
        else:
            rendered = str(content).strip()
        chunks.append(f"### {role}:\n{rendered}")
    return "\n\n".join(chunks).strip()


def _format_instruction_record(record: dict[str, Any]) -> str:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output_text = str(record.get("output", "")).strip()
    parts = [f"### Instruction:\n{instruction}"]
    if input_text:
        parts.append(f"### Input:\n{input_text}")
    parts.append(f"### Response:\n{output_text}")
    return "\n\n".join(parts).strip()


def format_training_record(record: dict[str, Any]) -> dict[str, str]:
    if isinstance(record.get("messages"), list):
        return {"text": _format_messages_record(record)}
    return {"text": _format_instruction_record(record)}


def load_training_records(file_path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in Path(file_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def split_training_records(records: list[dict[str, Any]], *, eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) < 10 or eval_ratio <= 0:
        return records, []

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_records = shuffled[:eval_size]
    train_records = shuffled[eval_size:]
    return train_records, eval_records


def build_sft_datasets(config: LocalQLoraJobConfig) -> tuple[Any, Any, dict[str, int]]:
    try:
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install `datasets` to enable local GPU QLoRA training.") from error

    records = load_training_records(config.dataset_path)
    train_records, eval_records = split_training_records(records, eval_ratio=config.eval_ratio, seed=config.seed)

    train_dataset = Dataset.from_list([format_training_record(record) for record in train_records])
    eval_dataset = Dataset.from_list([format_training_record(record) for record in eval_records]) if eval_records else None

    return train_dataset, eval_dataset, {
        "totalRecords": len(records),
        "trainRecords": len(train_records),
        "evalRecords": len(eval_records),
    }
