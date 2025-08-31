# ai_forensics/model_formats/gguf/gguf_versions.py
"""
Version-aware GGUF parsing with endianness detection (v1/v2/v3).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .gguf import GGUFKV, GGUFModel, GGUFParseError, GGUFTensorInfo

# GGUF type codes
T_UINT8 = 0
T_INT8 = 1
T_UINT16 = 2
T_INT16 = 3
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10
T_INT64 = 11
T_FLOAT64 = 12

NUMERIC_SIZES = {
    T_UINT8: 1,
    T_INT8: 1,
    T_UINT16: 2,
    T_INT16: 2,
    T_UINT32: 4,
    T_INT32: 4,
    T_FLOAT32: 4,
    T_UINT64: 8,
    T_INT64: 8,
    T_FLOAT64: 8,
}


@dataclass
class VersionMismatch:
    """Why a specific GGUF version/parser failed."""

    version: str  # e.g., "v1-LE", "v2-LE", "v3-BE"
    reason: str


@dataclass
class GGUFParsed:
    model: Optional[GGUFModel]
    mismatches: List[VersionMismatch]
    endian: Optional[str]  # 'LE' or 'BE'


def _unpack(buf: memoryview, off: int, fmt: str) -> tuple[tuple[int, ...], int]:
    vals = struct.unpack_from(fmt, buf, off)
    return vals, off + struct.calcsize(fmt)


def _u32(buf: memoryview, off: int, endian: str) -> tuple[int, int]:
    (v,), off = _unpack(buf, off, "<I" if endian == "LE" else ">I")
    return v, off


def _i32(buf: memoryview, off: int, endian: str) -> tuple[int, int]:
    (v,), off = _unpack(buf, off, "<i" if endian == "LE" else ">i")
    return v, off


def _u64(buf: memoryview, off: int, endian: str) -> tuple[int, int]:
    (v,), off = _unpack(buf, off, "<Q" if endian == "LE" else ">Q")
    return v, off


def _bytes(buf: memoryview, off: int, n: int) -> tuple[bytes, int]:
    if off + n > len(buf):
        raise GGUFParseError("Read beyond EOF")
    return bytes(buf[off : off + n]), off + n


def _str(buf: memoryview, off: int, endian: str, len_fmt: str) -> tuple[str, int]:
    ln, off = _u32(buf, off, endian) if len_fmt == "u32" else _u64(buf, off, endian)
    s, off = _bytes(buf, off, ln)
    return s.decode("utf-8", "strict"), off


def _align_up(x: int, a: int) -> int:
    return (x + (a - 1)) & ~(a - 1)


def _parse_kv(buf: memoryview, off: int, endian: str, size_fmt: str) -> tuple[GGUFKV, int]:
    key, off = _str(buf, off, endian, size_fmt)
    type_code, off = _i32(buf, off, endian)
    is_array = False
    elem_type = type_code
    count = 1
    if type_code == T_ARRAY:
        is_array = True
        elem_type, off = _i32(buf, off, endian)
        count, off = _u32(buf, off, endian) if size_fmt == "u32" else _u64(buf, off, endian)
    # value(s)
    if elem_type == T_STRING:
        if is_array:
            vals: list[str] = []
            for _ in range(count):
                s, off = _str(buf, off, endian, size_fmt)
                vals.append(s)
            value = vals
        else:
            s, off = _str(buf, off, endian, size_fmt)
            value = s
    elif elem_type == T_BOOL:
        raw, off = _bytes(buf, off, count if is_array else 1)
        value = [bool(b) for b in raw] if is_array else bool(raw[0])
    elif elem_type in NUMERIC_SIZES:
        size = NUMERIC_SIZES[elem_type]
        total = size * count
        raw, off = _bytes(buf, off, total)
        value = raw
    else:
        raise GGUFParseError(f"Unknown GGUF value type {elem_type}")
    return GGUFKV(key=key, type=elem_type, is_array=is_array, value=value), off


def _parse_tensor_info(
    buf: memoryview, off: int, endian: str, size_fmt: str
) -> tuple[GGUFTensorInfo, int]:
    name, off = _str(buf, off, endian, size_fmt)
    n_dims, off = _u32(buf, off, endian)
    dims: list[int] = []
    for _ in range(n_dims):
        d, off = _u64(buf, off, endian)  # accept 64-bit dims across versions
        dims.append(int(d))
    ggml_type, off = _i32(buf, off, endian)
    rel_off, off = _u64(buf, off, endian)  # offset relative to data section
    return (
        GGUFTensorInfo(
            name=name,
            n_dims=int(n_dims),
            dims=tuple(dims),
            ggml_type=ggml_type,
            offset=int(rel_off),
        ),
        off,
    )


def _parse_by_version(buf: memoryview, *, file_size: int, version: int, endian: str) -> GGUFModel:
    off = 8  # after magic + version
    size_fmt = "u32" if version == 1 else "u64"

    if size_fmt == "u32":
        n_tensors, off = _u32(buf, off, endian)
        n_kv, off = _u32(buf, off, endian)
    else:
        # v2/v3: 64-bit signed counts in practice
        (n_tensors,), off = _unpack(buf, off, "<q" if endian == "LE" else ">q")
        (n_kv,), off = _unpack(buf, off, "<q" if endian == "LE" else ">q")
    if n_tensors < 0 or n_kv < 0:
        raise GGUFParseError("Negative counts in header")

    kv: Dict[str, GGUFKV] = {}
    for _ in range(n_kv):
        item, off = _parse_kv(buf, off, endian, size_fmt)
        kv[item.key] = item

    alignment = 32
    if "general.alignment" in kv:
        v = kv["general.alignment"].value
        if isinstance(v, (bytes, bytearray)) and len(v) in (4, 8):
            alignment = int.from_bytes(v, "little" if endian == "LE" else "big")

    tensors: List[GGUFTensorInfo] = []
    for _ in range(n_tensors):
        ti, off = _parse_tensor_info(buf, off, endian, size_fmt)
        tensors.append(ti)

    data_start = _align_up(off, alignment)
    if data_start > file_size:
        raise GGUFParseError("Data section offset beyond EOF")

    return GGUFModel(
        version=version,
        endian=endian,
        alignment=alignment,
        n_kv=int(n_kv),
        n_tensors=int(n_tensors),
        kv=kv,
        tensors=tensors,
        data_offset=data_start,
        file_size=file_size,
    )


def parse_gguf_versioned(buf: memoryview, *, file_size: int) -> GGUFParsed:
    """Try GGUF v1/v2/v3 (LE/BE) and report mismatches."""
    if len(buf) < 8:
        raise GGUFParseError("File too small for GGUF header")
    if bytes(buf[:4]) != b"GGUF":
        raise GGUFParseError("Invalid magic; not GGUF")

    mismatches: list[VersionMismatch] = []
    version_le = struct.unpack_from("<I", buf, 4)[0]
    version_be = struct.unpack_from(">I", buf, 4)[0]

    def attempt(v: int, e: str) -> Optional[GGUFModel]:
        try:
            return _parse_by_version(buf, file_size=file_size, version=v, endian=e)
        except GGUFParseError as err:
            mismatches.append(VersionMismatch(version=f"v{v}-{e}", reason=str(err)))
            return None

    tried = False
    # Try plausible versions 1..3
    for e, v in (("LE", version_le), ("BE", version_be)):
        if 1 <= v <= 3:
            tried = True
            m = attempt(v, e)
            if m is not None:
                return GGUFParsed(model=m, mismatches=[], endian=e)

    if not tried:
        # We didn't see a plausible version field
        mismatches.append(
            VersionMismatch(
                version=f"LE={version_le}/BE={version_be}",
                reason="Unsupported version field; not in {1,2,3}",
            )
        )
    return GGUFParsed(model=None, mismatches=mismatches, endian=None)
