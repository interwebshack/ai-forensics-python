# ai_forensics/model_formats/gguf/gguf.py
"""
GGUF shared structures and exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ai_forensics.model_formats.gguf.gguf_quantization import GGMLType


@dataclass
class GGUFKV:
    key: str
    type: int
    is_array: bool
    value: Any
    offset_start: int
    offset_end: int


@dataclass
class GGUFTensorInfo:
    name: str
    n_dims: int
    dims: Tuple[int, ...]
    ggml_type: GGMLType  # Changed from int to our Enum
    offset: int  # relative to data section

    @property
    def n_elements(self) -> int:
        """Total number of elements in the tensor."""
        if not self.dims:
            return 0
        p = 1
        for d in self.dims:
            p *= d
        return p


@dataclass
class GGUFModel:
    version: int
    endian: str  # 'LE' or 'BE'
    alignment: int
    n_kv: int
    n_tensors: int
    kv: Dict[str, GGUFKV]
    tensors: List[GGUFTensorInfo]
    header_end_offset: int
    kv_end_offset: int
    tensor_info_end_offset: int
    data_offset: int  # absolute offset of data section
    file_size: int


class GGUFParseError(Exception):
    """Raised when a GGUF file is malformed."""
