from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def ensure_env_loaded() -> None:
    load_dotenv(ENV_PATH, override=False)


ensure_env_loaded()
