# ai_forensics/model_formats/gguf/gguf.py
"""
GGUF shared structures and exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class GGUFKV:
    key: str
    type: int
    is_array: bool
    value: Any


@dataclass
class GGUFTensorInfo:
    name: str
    n_dims: int
    dims: Tuple[int, ...]
    ggml_type: int
    offset: int  # relative to data section


@dataclass
class GGUFModel:
    version: int
    endian: str  # 'LE' or 'BE'
    alignment: int
    n_kv: int
    n_tensors: int
    kv: Dict[str, GGUFKV]
    tensors: List[GGUFTensorInfo]
    data_offset: int  # absolute offset of data section
    file_size: int


class GGUFParseError(Exception):
    """Raised when a GGUF file is malformed."""
