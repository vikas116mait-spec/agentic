from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "python_api" / "data"
STATE_FILE = DATA_DIR / "state.json"
UPLOADS_DIR = ROOT / "uploads_python"
TEMPORAL_CACHE_DIR = DATA_DIR / "temporal"
TEMPORAL_DB_FILE = DATA_DIR / "temporal-dev.db"
STATE_LOCK = threading.Lock()

StateMutator = TypeVar("StateMutator")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_state() -> dict[str, list[dict[str, Any]]]:
    return {
        "datasets": [],
        "jobs": [],
        "job_events": [],
        "playground_runs": [],
        "agent_runs": [],
    }


def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    defaults = empty_state()
    for key, default_value in defaults.items():
        state.setdefault(key, default_value.copy())
    return state


def ensure_paths() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TEMPORAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps(empty_state(), indent=2), encoding="utf-8")


def load_state() -> dict[str, list[dict[str, Any]]]:
    ensure_paths()
    with STATE_LOCK:
        raw_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return _normalize_state(raw_state)


def update_state(mutator: Callable[[dict[str, list[dict[str, Any]]]], StateMutator]) -> StateMutator:
    ensure_paths()
    with STATE_LOCK:
        raw_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state = _normalize_state(raw_state)
        result = mutator(state)
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return result


def sort_desc(items: list[dict[str, Any]], key: str = "createdAt") -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item.get(key, ""), reverse=True)


def get_dataset(state: dict[str, list[dict[str, Any]]], dataset_id: str) -> dict[str, Any] | None:
    return next((dataset for dataset in state["datasets"] if dataset["id"] == dataset_id), None)


def get_job(state: dict[str, list[dict[str, Any]]], job_id: str) -> dict[str, Any] | None:
    return next((job for job in state["jobs"] if job["id"] == job_id), None)


def get_agent_run(state: dict[str, list[dict[str, Any]]], run_id: str) -> dict[str, Any] | None:
    return next((run for run in state["agent_runs"] if run["id"] == run_id), None)
