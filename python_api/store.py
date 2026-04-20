from __future__ import annotations

from copy import deepcopy
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from python_api.env import ensure_env_loaded

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:  # pragma: no cover
    psycopg2 = None
    sql = None

ensure_env_loaded()


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "python_api" / "data"
STATE_FILE = DATA_DIR / "state.json"
UPLOADS_DIR = ROOT / "uploads_python"
TEMPORAL_CACHE_DIR = DATA_DIR / "temporal"
TEMPORAL_DB_FILE = DATA_DIR / "temporal-dev.db"
STATE_LOCK = threading.Lock()

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DATABASE_SCHEMA = (os.environ.get("DATABASE_SCHEMA") or "agentic_app").strip() or "agentic_app"

StateMutator = TypeVar("StateMutator")
State = dict[str, Any]

POSTGRES_COLLECTION_SPECS = {
    "datasets": {
        "table": "agentic_datasets",
        "id_column": "dataset_id",
        "columns": ["name", "validation_status"],
    },
    "jobs": {
        "table": "agentic_jobs",
        "id_column": "job_id",
        "columns": ["dataset_id", "status", "base_model", "model_provider"],
    },
    "job_events": {
        "table": "agentic_job_events",
        "id_column": "event_id",
        "columns": ["job_id", "level", "event_type"],
    },
    "playground_runs": {
        "table": "agentic_playground_runs",
        "id_column": "run_id",
        "columns": ["base_model", "base_model_provider", "fine_tuned_model"],
    },
    "agent_runs": {
        "table": "agentic_agent_runs",
        "id_column": "run_id",
        "columns": ["workflow_id", "status", "base_model", "agent_model"],
    },
    "model_profiles": {
        "table": "agentic_model_profiles",
        "id_column": "profile_id",
        "columns": ["name", "provider", "model", "category"],
    },
}

POSTGRES_STATE_KEY_SPECS = {
    "datasets": {"id_key": "id", "derived": {"name": "name", "validation_status": "validationStatus"}},
    "jobs": {"id_key": "id", "derived": {"dataset_id": "datasetId", "status": "status", "base_model": "baseModel", "model_provider": "modelProvider"}},
    "job_events": {"id_key": "id", "derived": {"job_id": "jobId", "level": "level", "event_type": "eventType"}},
    "playground_runs": {
        "id_key": "id",
        "derived": {
            "base_model": "baseModel",
            "base_model_provider": "baseModelProvider",
            "fine_tuned_model": "fineTunedModel",
        },
    },
    "agent_runs": {
        "id_key": "id",
        "derived": {"workflow_id": "workflowId", "status": "status", "base_model": "baseModel", "agent_model": "agentModel"},
    },
    "model_profiles": {"id_key": "id", "derived": {"name": "name", "provider": "provider", "model": "model", "category": "category"}},
}

POSTGRES_SETTINGS_TABLE = "agentic_workspace_settings"
POSTGRES_SETTINGS_KEYS = {"model_profile_defaults", "model_profiles_initialized"}

_POSTGRES_READY = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_state() -> State:
    return {
        "datasets": [],
        "jobs": [],
        "job_events": [],
        "playground_runs": [],
        "agent_runs": [],
        "model_profiles": [],
        "model_profile_defaults": {},
        "model_profiles_initialized": False,
    }


def _normalize_state(state: dict[str, Any]) -> State:
    defaults = empty_state()
    for key, default_value in defaults.items():
        state.setdefault(key, deepcopy(default_value))
    return state


def _postgres_enabled() -> bool:
    return bool(DATABASE_URL) and DATABASE_URL.startswith("postgresql") and psycopg2 is not None and sql is not None


def _postgres_state_warning() -> str | None:
    if not DATABASE_URL:
        return None
    if psycopg2 is None or sql is None:
        return "DATABASE_URL is configured, but psycopg2 is not installed in the Python environment. Falling back to local JSON state."
    if not DATABASE_URL.startswith("postgresql"):
        return "DATABASE_URL is configured with an unsupported scheme. Falling back to local JSON state."
    return None


def ensure_paths() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TEMPORAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not _postgres_enabled() and not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps(empty_state(), indent=2), encoding="utf-8")


def _connect_postgres():
    return psycopg2.connect(DATABASE_URL)


def _ensure_postgres_storage() -> None:
    global _POSTGRES_READY
    if _POSTGRES_READY or not _postgres_enabled():
        return

    with _connect_postgres() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(DATABASE_SCHEMA)))
            for spec in POSTGRES_COLLECTION_SPECS.values():
                cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{} (
                            {} TEXT PRIMARY KEY,
                            {},
                            payload JSONB NOT NULL,
                            created_at TIMESTAMPTZ,
                            updated_at TIMESTAMPTZ
                        )
                        """
                    ).format(
                        sql.Identifier(DATABASE_SCHEMA),
                        sql.Identifier(spec["table"]),
                        sql.Identifier(spec["id_column"]),
                        sql.SQL(", ").join(sql.SQL("{} TEXT").format(sql.Identifier(column_name)) for column_name in spec["columns"]),
                    )
                )
                for column_name in spec["columns"]:
                    cursor.execute(
                        sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} TEXT").format(
                            _table_identifier(spec["table"]),
                            sql.Identifier(column_name),
                        )
                    )

            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.{} (
                        setting_key TEXT PRIMARY KEY,
                        setting_value JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                ).format(sql.Identifier(DATABASE_SCHEMA), sql.Identifier(POSTGRES_SETTINGS_TABLE))
            )

    _POSTGRES_READY = True


def _table_identifier(table_name: str):
    return sql.SQL("{}.{}").format(sql.Identifier(DATABASE_SCHEMA), sql.Identifier(table_name))


def _load_state_from_postgres() -> State:
    _ensure_postgres_storage()
    state = empty_state()

    with _connect_postgres() as connection:
        with connection.cursor() as cursor:
            for collection_name, spec in POSTGRES_COLLECTION_SPECS.items():
                cursor.execute(
                    sql.SQL("SELECT payload::text FROM {}").format(_table_identifier(spec["table"]))
                )
                state[collection_name] = [json.loads(row[0]) for row in cursor.fetchall()]

            cursor.execute(
                sql.SQL("SELECT setting_key, setting_value::text FROM {}").format(_table_identifier(POSTGRES_SETTINGS_TABLE))
            )
            settings = {row[0]: json.loads(row[1]) for row in cursor.fetchall() if row[0] in POSTGRES_SETTINGS_KEYS}
            state["model_profile_defaults"] = settings.get("model_profile_defaults", {})
            state["model_profiles_initialized"] = settings.get("model_profiles_initialized", False)

    return _normalize_state(state)


def _collection_insert_values(collection_name: str, item: dict[str, Any]) -> list[Any]:
    state_spec = POSTGRES_STATE_KEY_SPECS[collection_name]
    spec = POSTGRES_COLLECTION_SPECS[collection_name]
    created_at = item.get("createdAt")
    updated_at = item.get("updatedAt", created_at)

    values: list[Any] = [item[state_spec["id_key"]]]
    for column_name in spec["columns"]:
        if column_name == "created_at":
            values.append(created_at)
            continue
        if column_name == "updated_at":
            values.append(updated_at)
            continue
        values.append(item.get(state_spec["derived"][column_name]))
    values.extend([json.dumps(item), created_at, updated_at])
    return values


def _persist_state_to_postgres(state: State) -> None:
    _ensure_postgres_storage()

    with _connect_postgres() as connection:
        with connection.cursor() as cursor:
            for collection_name, spec in POSTGRES_COLLECTION_SPECS.items():
                cursor.execute(sql.SQL("DELETE FROM {}").format(_table_identifier(spec["table"])))
                for item in state[collection_name]:
                    derived_columns = [sql.Identifier(spec["id_column"])]
                    derived_columns.extend(sql.Identifier(column_name) for column_name in spec["columns"])
                    derived_columns.extend([sql.Identifier("payload"), sql.Identifier("created_at"), sql.Identifier("updated_at")])
                    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in range(len(derived_columns)))
                    cursor.execute(
                        sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                            _table_identifier(spec["table"]),
                            sql.SQL(", ").join(derived_columns),
                            placeholders,
                        ),
                        _collection_insert_values(collection_name, item),
                    )

            settings_payload = {
                "model_profile_defaults": state["model_profile_defaults"],
                "model_profiles_initialized": state["model_profiles_initialized"],
            }
            for setting_key, setting_value in settings_payload.items():
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (setting_key, setting_value, updated_at)
                        VALUES (%s, %s::jsonb, NOW())
                        ON CONFLICT (setting_key)
                        DO UPDATE SET setting_value = EXCLUDED.setting_value, updated_at = EXCLUDED.updated_at
                        """
                    ).format(_table_identifier(POSTGRES_SETTINGS_TABLE)),
                    (setting_key, json.dumps(setting_value)),
                )


def load_state() -> State:
    ensure_paths()
    with STATE_LOCK:
        if _postgres_enabled():
            return _load_state_from_postgres()

        raw_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return _normalize_state(raw_state)


def update_state(mutator: Callable[[State], StateMutator]) -> StateMutator:
    ensure_paths()
    with STATE_LOCK:
        if _postgres_enabled():
            state = _load_state_from_postgres()
            result = mutator(state)
            _persist_state_to_postgres(state)
            return result

        raw_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state = _normalize_state(raw_state)
        result = mutator(state)
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return result


def sort_desc(items: list[dict[str, Any]], key: str = "createdAt") -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item.get(key, ""), reverse=True)


def storage_status_payload() -> dict[str, Any]:
    warning = _postgres_state_warning()
    return {
        "backend": "postgresql" if _postgres_enabled() else "json_file",
        "databaseUrlConfigured": bool(DATABASE_URL),
        "databaseSchema": DATABASE_SCHEMA if DATABASE_URL else None,
        "warning": warning,
        "jsonStateFile": str(STATE_FILE),
        "uploadsDir": str(UPLOADS_DIR),
    }


def get_dataset(state: State, dataset_id: str) -> dict[str, Any] | None:
    return next((dataset for dataset in state["datasets"] if dataset["id"] == dataset_id), None)


def get_job(state: State, job_id: str) -> dict[str, Any] | None:
    return next((job for job in state["jobs"] if job["id"] == job_id), None)


def get_agent_run(state: State, run_id: str) -> dict[str, Any] | None:
    return next((run for run in state["agent_runs"] if run["id"] == run_id), None)


def get_model_profile(state: State, profile_id: str) -> dict[str, Any] | None:
    return next((profile for profile in state["model_profiles"] if profile["id"] == profile_id), None)
