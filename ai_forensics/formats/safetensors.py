"""
Pure-Python SafeTensors parser (v1-style header).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class SafeTensorsParseError(Exception):
    """Raised when a SafeTensors file is malformed."""


@dataclass
class STTensor:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int]


@dataclass
class SafeTensorsModel:
    header_size: int
    header_json: Dict[str, Any]
    tensors: List[STTensor]
    data_start: int
    file_size: int


def parse_safetensors(buf: memoryview, *, file_size: int) -> SafeTensorsModel:
    """Parse header + tensor metadata (no data reads)."""
    if file_size < 8:
        raise SafeTensorsParseError("File too small for safetensors header")
    header_size = struct.unpack_from("<Q", buf, 0)[0]
    header_start = 8
    header_end = header_start + header_size
    if header_end > file_size:
        raise SafeTensorsParseError("Header extends beyond EOF")
    raw = bytes(buf[header_start:header_end]).lstrip()
    if not raw or raw[:1] != b"{":
        raise SafeTensorsParseError("Header does not start with '{'")
    try:
        header = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise SafeTensorsParseError(f"Invalid JSON header: {e}") from e

    tensors: List[STTensor] = []
    for name, meta in header.items():
        if not isinstance(meta, dict):
            raise SafeTensorsParseError(f"Invalid tensor meta for {name}")
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if not (
            isinstance(dtype, str)
            and isinstance(shape, list)
            and isinstance(offsets, (list, tuple))
            and len(offsets) == 2
        ):
            raise SafeTensorsParseError(f"Missing/invalid fields for {name}")
        if not all(isinstance(x, int) and x >= 0 for x in shape):
            raise SafeTensorsParseError(f"Invalid shape for {name}")
        if not all(isinstance(x, int) and x >= 0 for x in offsets):
            raise SafeTensorsParseError(f"Invalid data_offsets for {name}")
        tensors.append(
            STTensor(
                name=name,
                dtype=dtype,
                shape=tuple(int(x) for x in shape),
                data_offsets=(int(offsets[0]), int(offsets[1])),
            )
        )

    data_start = header_end
    return SafeTensorsModel(
        header_size=header_size,
        header_json=header,
        tensors=tensors,
        data_start=data_start,
        file_size=file_size,
    )
