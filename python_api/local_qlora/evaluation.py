from __future__ import annotations

import math
import re
from typing import Any

from python_api.local_qlora.data import build_evaluation_example


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _tokenize_for_overlap(value: str) -> list[str]:
    return [token for token in re.findall(r"\w+", _normalize_text(value)) if token]


def _exact_match(prediction: str, target: str) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(target) else 0.0


def _token_f1(prediction: str, target: str) -> float:
    prediction_tokens = _tokenize_for_overlap(prediction)
    target_tokens = _tokenize_for_overlap(target)
    if not prediction_tokens and not target_tokens:
        return 1.0
    if not prediction_tokens or not target_tokens:
        return 0.0

    prediction_counts: dict[str, int] = {}
    for token in prediction_tokens:
        prediction_counts[token] = prediction_counts.get(token, 0) + 1

    overlap = 0
    target_counts: dict[str, int] = {}
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1
    for token, count in target_counts.items():
        overlap += min(count, prediction_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(previous[index_b], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l_f1(prediction: str, target: str) -> float:
    prediction_tokens = _tokenize_for_overlap(prediction)
    target_tokens = _tokenize_for_overlap(target)
    if not prediction_tokens and not target_tokens:
        return 1.0
    if not prediction_tokens or not target_tokens:
        return 0.0

    lcs = _lcs_length(prediction_tokens, target_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(prediction_tokens)
    recall = lcs / len(target_tokens)
    return (2 * precision * recall) / (precision + recall)


def run_holdout_generation_evaluation(
    *,
    model: Any,
    tokenizer: Any,
    eval_records: list[dict[str, Any]],
    model_id: str | None = None,
    max_samples: int = 0,
    max_new_tokens: int = 160,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    if max_samples <= 0 or not eval_records:
        return {}

    examples = [
        sample
        for record in eval_records
        if (sample := build_evaluation_example(record, model_id=model_id, tokenizer=tokenizer)) is not None
    ][:max_samples]
    if not examples:
        return {}

    if torch_module is None:
        import torch as torch_module  # pragma: no cover

    if hasattr(model, "eval"):
        model.eval()

    if hasattr(model, "device"):
        device = getattr(model, "device")
    else:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = None

    exact_match_total = 0.0
    token_f1_total = 0.0
    rouge_l_total = 0.0
    generated = 0

    for sample in examples:
        encoded = tokenizer(sample["prompt"], return_tensors="pt")
        if device is not None:
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        generate_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max(32, int(max_new_tokens)),
            "do_sample": False,
        }
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        with torch_module.no_grad():
            output_ids = model.generate(**generate_kwargs)

        prompt_length = int(encoded["input_ids"].shape[-1])
        continuation_ids = output_ids[0][prompt_length:]
        prediction = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
        target = sample["target"]

        exact_match_total += _exact_match(prediction, target)
        token_f1_total += _token_f1(prediction, target)
        rouge_l_total += _rouge_l_f1(prediction, target)
        generated += 1

    if generated == 0:
        return {}

    return {
        "sampleCount": generated,
        "exactMatch": exact_match_total / generated,
        "tokenF1": token_f1_total / generated,
        "rougeL": rouge_l_total / generated,
        "maxNewTokens": max(32, int(max_new_tokens)),
    }


def summarize_eval_metrics(
    metrics: dict[str, Any] | None,
    generation_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if metrics:
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            summary["evalLoss"] = eval_loss
            try:
                summary["perplexity"] = math.exp(eval_loss)
            except OverflowError:
                summary["perplexity"] = None

        for key in ("eval_runtime", "eval_samples_per_second", "eval_steps_per_second"):
            if key in metrics:
                summary[key] = metrics[key]

    generation = generation_metrics or {}
    if generation.get("sampleCount"):
        summary["generationSampleCount"] = generation["sampleCount"]
        summary["generationExactMatch"] = generation.get("exactMatch")
        summary["generationTokenF1"] = generation.get("tokenF1")
        summary["generationRougeL"] = generation.get("rougeL")

    return summary
