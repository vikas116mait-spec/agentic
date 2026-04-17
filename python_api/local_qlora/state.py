from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from uuid import uuid4

from python_api.store import utc_now_iso


def write_status_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_status_file(path)
    merged = {**existing, **payload, "updatedAt": payload.get("updatedAt") or utc_now_iso()}
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    temp_path.replace(path)


def read_status_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        return {}


def append_event(path: Path, level: str, message: str, *, event_type: str | None = None) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "id": uuid4().hex,
        "jobId": path.parent.name,
        "level": level,
        "message": message,
        "eventType": event_type,
        "createdAt": utc_now_iso(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    return event
