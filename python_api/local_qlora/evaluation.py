from __future__ import annotations

import math
from typing import Any


def summarize_eval_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}

    summary: dict[str, Any] = {}
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

    return summary
