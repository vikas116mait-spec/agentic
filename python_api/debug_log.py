from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


DEBUG_LOG_PATH = Path("/mnt/nvme_disk2/User_data/vs95259v/Vikas/project/agentic/.cursor/debug-f30c87.log")
DEBUG_SESSION_ID = "f30c87"


def write_debug_log(*, location: str, message: str, data: dict[str, Any], run_id: str, hypothesis_id: str) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except OSError:
        pass
