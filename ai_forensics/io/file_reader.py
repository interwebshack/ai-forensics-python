"""
Zero-copy local file reader using mmap + memoryview.
"""

from __future__ import annotations

import mmap
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LocalFileSource:
    """Local file source with zero-copy memory mapping.

    Attributes:
        path: Path to the local file.
    """

    path: str

    def open(self) -> "MappedFile":
        """Open and memory-map the file read-only."""
        return MappedFile(self.path)


class MappedFile:
    """Context manager that wraps an mmapped file and exposes a memoryview."""

    __slots__ = ("_fd", "_m", "_mv", "size", "path")

    def __init__(self, path: str):
        self.path = path
        self._fd: Optional[int] = None
        self._m: Optional[mmap.mmap] = None
        self._mv: Optional[memoryview] = None
        self.size: int = 0

    def __enter__(self) -> "MappedFile":
        self._fd = os.open(self.path, os.O_RDONLY)
        self.size = os.path.getsize(self.path)
        self._m = mmap.mmap(self._fd, self.size, access=mmap.ACCESS_READ)
        self._mv = memoryview(self._m)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._mv is not None:
            self._mv.release()
        if self._m is not None:
            self._m.close()
        if self._fd is not None:
            os.close(self._fd)

    @property
    def view(self) -> memoryview:
        """Zero-copy memoryview over the file bytes."""
        if self._mv is None:
            raise RuntimeError("MappedFile is not entered")
        return self._mv
