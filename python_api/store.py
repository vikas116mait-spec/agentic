from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from python_api.env import ensure_env_loaded

try:
    import psycopg2
    from psycopg2 import pool as psycopg2_pool
    from psycopg2 import sql
except ImportError:  # pragma: no cover
    psycopg2 = None
    psycopg2_pool = None
    sql = None


logger = logging.getLogger(__name__)

ensure_env_loaded()


ROOT = Path(__file__).resolve().parents[1]


def _resolve_root(env_name: str, default: Path) -> Path:
    """Resolve a configurable filesystem root.

    Relative paths are resolved against the repo root; absolute paths are
    honoured as-is. Exists so Docker/Fly deployments can redirect state and
    uploads to a mounted volume (e.g. AGENTIC_DATA_DIR=/data/python_api).
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


DATA_DIR = _resolve_root("AGENTIC_DATA_DIR", ROOT / "python_api" / "data")
STATE_FILE = DATA_DIR / "state.json"
UPLOADS_DIR = _resolve_root("AGENTIC_UPLOADS_DIR", ROOT / "uploads_python")
TEMPORAL_CACHE_DIR = DATA_DIR / "temporal"
TEMPORAL_DB_FILE = DATA_DIR / "temporal-dev.db"
JOBS_DIR = UPLOADS_DIR / "jobs"
DATASETS_DIR = UPLOADS_DIR / "datasets"

_WRITE_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_STATE_CACHE: dict[str, Any] = {"state": None, "expires_at": 0.0}
_STATE_CACHE_STATS = {"hits": 0, "misses": 0}

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
_POSTGRES_POOL: Any = None
_POSTGRES_POOL_LOCK = threading.Lock()


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


def _postgres_pool_bounds() -> tuple[int, int]:
    try:
        min_size = max(int(os.environ.get("DATABASE_MIN_POOL_SIZE") or "1"), 1)
    except ValueError:
        min_size = 1
    try:
        max_size = max(int(os.environ.get("DATABASE_MAX_POOL_SIZE") or "5"), min_size)
    except ValueError:
        max_size = max(min_size, 5)
    return min_size, max_size


def _get_postgres_pool():
    """Lazily create (and memoize) a thread-safe connection pool."""
    global _POSTGRES_POOL
    if _POSTGRES_POOL is not None:
        return _POSTGRES_POOL
    with _POSTGRES_POOL_LOCK:
        if _POSTGRES_POOL is None:
            min_size, max_size = _postgres_pool_bounds()
            _POSTGRES_POOL = psycopg2_pool.ThreadedConnectionPool(
                min_size,
                max_size,
                dsn=DATABASE_URL,
            )
            logger.info(
                "Initialised Postgres connection pool (min=%s, max=%s).",
                min_size,
                max_size,
            )
    return _POSTGRES_POOL


def close_postgres_pool() -> None:
    """Close all pooled connections (use on application shutdown)."""
    global _POSTGRES_POOL
    with _POSTGRES_POOL_LOCK:
        if _POSTGRES_POOL is not None:
            try:
                _POSTGRES_POOL.closeall()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to close Postgres pool cleanly.")
            _POSTGRES_POOL = None


@contextmanager
def _connect_postgres():
    """Check out a pooled Postgres connection.

    On normal exit the transaction is committed and the connection is returned
    to the pool. On exception it is rolled back; broken connections are closed
    rather than returned so a stale handle cannot re-enter the pool.
    """
    pool = _get_postgres_pool()
    connection = pool.getconn()
    broken = False
    try:
        yield connection
        connection.commit()
    except Exception:
        broken = True
        try:
            connection.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            if broken or getattr(connection, "closed", 1):
                pool.putconn(connection, close=True)
            else:
                pool.putconn(connection)
        except Exception:  # pragma: no cover - defensive
            try:
                connection.close()
            except Exception:
                pass


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


def _coerce_jsonb(value: Any) -> Any:
    """Normalise a JSONB column value to a Python object.

    psycopg2's default adapter already returns Python dicts/lists for JSONB
    columns, but some distro builds or connection settings return the raw
    text. Handle both so the caller never has to care.
    """
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def _cache_ttl_s() -> float:
    try:
        raw = int(os.environ.get("DATABASE_STATE_CACHE_TTL_MS") or "1000")
    except ValueError:
        return 1.0
    return max(raw, 0) / 1000.0


def _invalidate_state_cache() -> None:
    with _CACHE_LOCK:
        _STATE_CACHE["state"] = None
        _STATE_CACHE["expires_at"] = 0.0


def state_cache_stats() -> dict[str, int]:
    """Return a snapshot of cache hit/miss counters (for /health)."""
    with _CACHE_LOCK:
        ttl_active = bool(_STATE_CACHE["state"]) and _STATE_CACHE["expires_at"] > time.monotonic()
        return {
            "hits": _STATE_CACHE_STATS["hits"],
            "misses": _STATE_CACHE_STATS["misses"],
            "populated": 1 if ttl_active else 0,
        }


def _load_state_from_postgres() -> State:
    _ensure_postgres_storage()
    state = empty_state()

    with _connect_postgres() as connection:
        with connection.cursor() as cursor:
            for collection_name, spec in POSTGRES_COLLECTION_SPECS.items():
                cursor.execute(
                    sql.SQL("SELECT payload FROM {}").format(_table_identifier(spec["table"]))
                )
                state[collection_name] = [_coerce_jsonb(row[0]) for row in cursor.fetchall()]

            cursor.execute(
                sql.SQL("SELECT setting_key, setting_value FROM {}").format(_table_identifier(POSTGRES_SETTINGS_TABLE))
            )
            settings = {
                row[0]: _coerce_jsonb(row[1])
                for row in cursor.fetchall()
                if row[0] in POSTGRES_SETTINGS_KEYS
            }
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
    if _postgres_enabled():
        ttl = _cache_ttl_s()
        if ttl > 0:
            with _CACHE_LOCK:
                snapshot = _STATE_CACHE["state"]
                if snapshot is not None and _STATE_CACHE["expires_at"] > time.monotonic():
                    _STATE_CACHE_STATS["hits"] += 1
                    return deepcopy(snapshot)
                _STATE_CACHE_STATS["misses"] += 1

        state = _load_state_from_postgres()

        if ttl > 0:
            with _CACHE_LOCK:
                _STATE_CACHE["state"] = deepcopy(state)
                _STATE_CACHE["expires_at"] = time.monotonic() + ttl
        return state

    # JSON-file path: no cache, readers are cheap and already local.
    with _WRITE_LOCK:
        raw_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return _normalize_state(raw_state)


def update_state(mutator: Callable[[State], StateMutator]) -> StateMutator:
    ensure_paths()
    with _WRITE_LOCK:
        if _postgres_enabled():
            # Always refetch from Postgres on the write path so the mutator
            # sees a fresh, uncached view even if another writer updated the
            # DB from outside this process.
            state = _load_state_from_postgres()
            result = mutator(state)
            _persist_state_to_postgres(state)
            _invalidate_state_cache()
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
