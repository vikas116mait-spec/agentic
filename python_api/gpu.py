"""GPU inventory helpers shared across inference and training paths.

Thin wrapper over `nvidia-smi --query-gpu=...` with no external dependencies.
Returns camelCase dicts so the payload can be served straight to the frontend.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any


NvidiaSmiMissing = RuntimeError


def nvidia_smi_available() -> bool:
    """Return True if `nvidia-smi` is on PATH and exits cleanly."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return False
    try:
        result = subprocess.run(
            [binary, "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def get_gpu_inventory() -> list[dict[str, Any]]:
    """Return per-GPU stats.

    Each entry has:
        index, name, memoryTotalMb, memoryUsedMb, memoryFreeMb,
        utilizationPct, temperatureC
    """
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []

    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    inventory: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 6:
            continue
        try:
            index = int(parts[0])
            total_mb = int(parts[2])
            used_mb = int(parts[3])
            utilization = int(parts[4])
            temperature = int(parts[5])
        except ValueError:
            continue
        inventory.append(
            {
                "index": index,
                "name": parts[1],
                "memoryTotalMb": total_mb,
                "memoryUsedMb": used_mb,
                "memoryFreeMb": max(total_mb - used_mb, 0),
                "utilizationPct": utilization,
                "temperatureC": temperature,
            }
        )
    return inventory


def pick_free_gpu(
    min_free_mb: int | None = None,
    inventory: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the GPU with the most free VRAM, optionally filtered by threshold.

    Returns None when `nvidia-smi` is unavailable or when no GPU meets
    `min_free_mb`. Shares the selection heuristic used by the local training
    path in `python_api/local_qlora/__init__.py::_select_single_gpu`.
    """
    gpus = inventory if inventory is not None else get_gpu_inventory()
    if not gpus:
        return None

    if min_free_mb is not None and min_free_mb > 0:
        gpus = [gpu for gpu in gpus if int(gpu.get("memoryFreeMb") or 0) >= min_free_mb]
        if not gpus:
            return None

    return max(gpus, key=lambda gpu: int(gpu.get("memoryFreeMb") or 0))
