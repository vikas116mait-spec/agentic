from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApiError(Exception):
    code: str
    message: str
    status: int = 400
    details: list[Any] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message
