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


def _render_message_content(content: Any) -> str:
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def _record_as_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        messages: list[dict[str, str]] = []
        for message in record["messages"]:
            role = str(message.get("role", "user")).strip().lower() or "user"
            if role not in {"system", "user", "assistant"}:
                role = "user"
            messages.append({
                "role": role,
                "content": _render_message_content(message.get("content", "")),
            })
        return messages

    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output_text = str(record.get("output", "")).strip()
    user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]


def _apply_tokenizer_chat_template(messages: list[dict[str, str]], tokenizer: Any | None) -> str | None:
    if tokenizer is None:
        return None

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return None

    try:
        rendered = apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except TypeError:
        try:
            rendered = apply_chat_template(messages, tokenize=False)
        except Exception:
            return None
    except Exception:
        return None

    if isinstance(rendered, list):
        rendered = "".join(str(chunk) for chunk in rendered)

    text = str(rendered).strip()
    return text or None


def _is_llama3_instruct_model(model_id: str | None) -> bool:
    lowered = (model_id or "").strip().lower()
    return "llama-3" in lowered or "llama3" in lowered


def _format_llama3_messages(messages: list[dict[str, str]]) -> str:
    parts = ["<|begin_of_text|>"]
    for message in messages:
        role = message["role"] if message["role"] in {"system", "user", "assistant"} else "user"
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{message['content']}<|eot_id|>"
        )
    return "".join(parts)


def _format_instruction_record(record: dict[str, Any]) -> str:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output_text = str(record.get("output", "")).strip()
    parts = [f"### Instruction:\n{instruction}"]
    if input_text:
        parts.append(f"### Input:\n{input_text}")
    parts.append(f"### Response:\n{output_text}")
    return "\n\n".join(parts).strip()


def format_training_record(
    record: dict[str, Any],
    model_id: str | None = None,
    tokenizer: Any | None = None,
) -> dict[str, str]:
    messages = _record_as_messages(record)
    templated_text = _apply_tokenizer_chat_template(messages, tokenizer)
    if templated_text:
        return {"text": templated_text}
    if _is_llama3_instruct_model(model_id):
        return {"text": _format_llama3_messages(messages)}
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


def build_sft_datasets(config: LocalQLoraJobConfig, *, tokenizer: Any | None = None) -> tuple[Any, Any, dict[str, int]]:
    try:
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install `datasets` to enable local GPU QLoRA training.") from error

    records = load_training_records(config.dataset_path)
    train_records, eval_records = split_training_records(records, eval_ratio=config.eval_ratio, seed=config.seed)

    train_dataset = Dataset.from_list(
        [format_training_record(record, config.base_model, tokenizer=tokenizer) for record in train_records]
    )
    eval_dataset = (
        Dataset.from_list(
            [format_training_record(record, config.base_model, tokenizer=tokenizer) for record in eval_records]
        )
        if eval_records
        else None
    )

    return train_dataset, eval_dataset, {
        "totalRecords": len(records),
        "trainRecords": len(train_records),
        "evalRecords": len(eval_records),
    }
