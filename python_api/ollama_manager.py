"""Adaptive Ollama GPU manager.

When `OLLAMA_AUTO_MANAGE=1`, the app can atomically rewrite a systemd drop-in
at `OLLAMA_DROPIN_PATH`, `sudo systemctl daemon-reload && restart` the Ollama
service, and wait for `/api/tags` to come back healthy before forwarding a
prompt. All state transitions are guarded by an `fcntl.flock` over
`uploads_python/.ollama-manager.lock` so concurrent playground / agent calls
do not dogpile a restart.

The module fails open: if sudo, the drop-in path, or `nvidia-smi` are
unavailable, it raises `OllamaManageUnavailable` and the caller simply logs
and proceeds with normal Ollama behaviour.
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from python_api import gpu
from python_api.store import UPLOADS_DIR


logger = logging.getLogger(__name__)


DEFAULT_DROPIN_PATH = "/etc/systemd/system/ollama.service.d/cuda.conf"
DEFAULT_SERVICE_NAME = "ollama.service"
DEFAULT_HEALTH_URL = "http://127.0.0.1:11434/api/tags"
DEFAULT_RESTART_TIMEOUT_S = 30
DEFAULT_MIN_FREE_GPU_MB = 8192


_LOCK_PATH = UPLOADS_DIR / ".ollama-manager.lock"
_DROPIN_TEMPLATE = (
    "[Service]\n"
    'Environment="CUDA_VISIBLE_DEVICES={gpu_index}"\n'
)
_CUDA_VISIBLE_PATTERN = re.compile(
    r'Environment\s*=\s*"?CUDA_VISIBLE_DEVICES=(?P<value>[^"\s]+)"?'
)


class OllamaManageUnavailable(RuntimeError):
    """Raised when the adaptive-GPU feature cannot run in the current env."""


def _env_bool(name: str, default: bool = False) -> bool:
    value = (os.environ.get(name) or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def auto_manage_enabled() -> bool:
    return _env_bool("OLLAMA_AUTO_MANAGE", default=False)


def dropin_path() -> Path:
    return Path(os.environ.get("OLLAMA_DROPIN_PATH") or DEFAULT_DROPIN_PATH)


def service_name() -> str:
    return os.environ.get("OLLAMA_SERVICE_NAME") or DEFAULT_SERVICE_NAME


def health_url() -> str:
    return os.environ.get("OLLAMA_HEALTH_URL") or DEFAULT_HEALTH_URL


def min_free_gpu_mb() -> int:
    raw = (os.environ.get("OLLAMA_MIN_FREE_GPU_MB") or "").strip()
    if not raw:
        return DEFAULT_MIN_FREE_GPU_MB
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_MIN_FREE_GPU_MB


def restart_timeout_s() -> int:
    raw = (os.environ.get("OLLAMA_RESTART_TIMEOUT_S") or "").strip()
    if not raw:
        return DEFAULT_RESTART_TIMEOUT_S
    try:
        return max(int(raw), 1)
    except ValueError:
        return DEFAULT_RESTART_TIMEOUT_S


def dropin_writable() -> bool:
    path = dropin_path()
    if path.exists():
        return os.access(path, os.W_OK)
    return os.access(path.parent, os.W_OK)


def sudo_available() -> bool:
    """Passwordless `sudo -n true` works for this process."""
    binary = shutil.which("sudo")
    if not binary:
        return False
    try:
        result = subprocess.run(
            [binary, "-n", "true"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def get_pinned_ollama_gpu() -> int | None:
    """Parse the drop-in for the CUDA_VISIBLE_DEVICES index the app last wrote.

    Returns the first index in the list (e.g. `3` from `3,5`). Returns None
    when the drop-in is missing or does not contain a pin.
    """
    path = dropin_path()
    try:
        content = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None

    match = _CUDA_VISIBLE_PATTERN.search(content)
    if not match:
        return None

    raw = match.group("value").split(",")[0].strip()
    try:
        return int(raw)
    except ValueError:
        return None


def rewrite_dropin(gpu_index: int) -> None:
    """Atomically write the drop-in with `CUDA_VISIBLE_DEVICES=<gpu_index>`."""
    path = dropin_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise OllamaManageUnavailable(
            f"Drop-in directory {path.parent} is not writable: {exc}"
        ) from exc

    payload = _DROPIN_TEMPLATE.format(gpu_index=gpu_index)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
    except PermissionError as exc:
        raise OllamaManageUnavailable(
            f"Drop-in file {path} is not writable by the app user: {exc}"
        ) from exc
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _wait_for_ollama_ready(timeout_s: int) -> bool:
    url = health_url()
    deadline = time.monotonic() + timeout_s
    backoff = 0.5
    while time.monotonic() < deadline:
        try:
            with urlopen(Request(url, method="GET"), timeout=3) as response:
                if 200 <= response.status < 300:
                    return True
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 2.0)
    return False


def restart_ollama() -> None:
    """`sudo -n systemctl daemon-reload` + `restart <service>`; poll health."""
    if not sudo_available():
        raise OllamaManageUnavailable(
            "Passwordless sudo is not configured. See README 'Adaptive Ollama GPU'."
        )

    systemctl = shutil.which("systemctl") or "/bin/systemctl"
    svc = service_name()

    for action in (["daemon-reload"], ["restart", svc]):
        command = ["sudo", "-n", systemctl, *action]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=restart_timeout_s(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OllamaManageUnavailable(
                f"`{' '.join(command)}` failed: {exc}"
            ) from exc
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise OllamaManageUnavailable(
                f"`{' '.join(command)}` exited {result.returncode}: {stderr or 'no stderr'}"
            )

    if not _wait_for_ollama_ready(restart_timeout_s()):
        raise OllamaManageUnavailable(
            f"Ollama did not become ready at {health_url()} within {restart_timeout_s()}s."
        )


def _acquire_lock():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    handle = _LOCK_PATH.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _release_lock(handle) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def gpu_status_payload() -> dict[str, Any]:
    """Snapshot for the UI: inventory + current pin + recommendation + diag flags."""
    inventory = gpu.get_gpu_inventory()
    pinned = get_pinned_ollama_gpu()
    recommended = gpu.pick_free_gpu(min_free_mb=min_free_gpu_mb(), inventory=inventory)
    if recommended is None and inventory:
        recommended = gpu.pick_free_gpu(inventory=inventory)

    pinned_entry = None
    if pinned is not None:
        for entry in inventory:
            if int(entry.get("index", -1)) == pinned:
                pinned_entry = entry
                break

    return {
        "gpus": inventory,
        "pinnedGpu": pinned,
        "pinnedGpuFreeMb": int(pinned_entry["memoryFreeMb"]) if pinned_entry else None,
        "recommendedGpu": int(recommended["index"]) if recommended else None,
        "recommendedGpuFreeMb": int(recommended["memoryFreeMb"]) if recommended else None,
        "autoManageEnabled": auto_manage_enabled(),
        "dropinWritable": dropin_writable(),
        "sudoAvailable": sudo_available(),
        "nvidiaSmiAvailable": gpu.nvidia_smi_available(),
        "minFreeGpuMb": min_free_gpu_mb(),
        "serviceName": service_name(),
    }


def ensure_ollama_on_free_gpu(
    min_free_mb: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Ensure Ollama is running on a GPU with enough free VRAM.

    Behaviour:
    - If `OLLAMA_AUTO_MANAGE` is off, raises `OllamaManageUnavailable`.
    - If the pinned GPU already has enough free memory and `force=False`,
      returns `{changed: False, ...}` without touching the service.
    - Otherwise writes the drop-in for the freest GPU, restarts the service,
      waits for the health endpoint, and returns `{changed: True, ...}`.
    """
    if not auto_manage_enabled():
        raise OllamaManageUnavailable(
            "OLLAMA_AUTO_MANAGE is off. Set it to 1 and complete the README setup."
        )

    inventory = gpu.get_gpu_inventory()
    if not inventory:
        raise OllamaManageUnavailable(
            "nvidia-smi returned no GPUs. Cannot pick a target for Ollama."
        )

    threshold = min_free_mb if min_free_mb is not None else min_free_gpu_mb()

    handle = _acquire_lock()
    try:
        pinned = get_pinned_ollama_gpu()
        pinned_entry = None
        if pinned is not None:
            for entry in inventory:
                if int(entry.get("index", -1)) == pinned:
                    pinned_entry = entry
                    break

        if (
            not force
            and pinned_entry is not None
            and int(pinned_entry["memoryFreeMb"]) >= threshold
        ):
            return {
                "changed": False,
                "gpuIndex": pinned,
                "freeMb": int(pinned_entry["memoryFreeMb"]),
                "reason": "pinned-gpu-has-headroom",
            }

        target = gpu.pick_free_gpu(min_free_mb=threshold, inventory=inventory)
        if target is None:
            target = gpu.pick_free_gpu(inventory=inventory)
        if target is None:
            raise OllamaManageUnavailable(
                "No GPU with enough free VRAM was found."
            )

        target_index = int(target["index"])
        target_free_mb = int(target["memoryFreeMb"])

        if pinned == target_index and not force:
            return {
                "changed": False,
                "gpuIndex": target_index,
                "freeMb": target_free_mb,
                "reason": "already-on-freest-gpu",
            }

        rewrite_dropin(target_index)
        restart_ollama()

        logger.info(
            "Ollama re-pinned to GPU %s (~%s MB free) via %s",
            target_index,
            target_free_mb,
            dropin_path(),
        )

        return {
            "changed": True,
            "gpuIndex": target_index,
            "freeMb": target_free_mb,
            "previousGpu": pinned,
            "reason": "restarted",
        }
    finally:
        _release_lock(handle)


def is_cuda_oom_error(error: BaseException | None) -> bool:
    """Heuristic: does this error message look like a CUDA OOM from Ollama?"""
    if error is None:
        return False
    message = str(error).lower()
    return (
        "out of memory" in message
        or "cudamalloc" in message
        or "cuda error" in message
    )
