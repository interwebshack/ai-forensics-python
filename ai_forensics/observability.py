# ai_forensics/observability.py
"""
Observability helpers: timers and dataclass → dict conversion.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Timer:
    """Context manager for measuring durations in milliseconds."""

    name: str
    start: float = 0.0
    duration_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.duration_ms = (time.perf_counter() - self.start) * 1000.0


def to_dict(obj: Any) -> Dict[str, Any] | list[Any] | Any:
    """Recursively convert dataclasses to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        return {k: to_dict(v) for k, v in d.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj
