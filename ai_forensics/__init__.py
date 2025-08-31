# ai_forensics/__init__.py
"""
ai_forensics
============

Pure-Python forensic analyzers for GGUF and SafeTensors binaries with zero-copy mmap,
concurrency-ready structure, rich console reporting, and detailed “reason matrix”
explanations for version mismatches.
"""
from __future__ import annotations

from importlib.metadata import version as _pkg_version

__all__ = ["__version__"]

try:
    # Read version dynamically from installed package metadata
    __version__: str = _pkg_version("aiforensics")
except Exception:  # pragma: no cover - fallback for development environments
    __version__ = "0.0.0-dev"
