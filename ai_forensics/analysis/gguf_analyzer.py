# ai_forensics/analysis/gguf_analyzer.py
"""
GGUF analyzer: version-aware structural verification + reason matrix.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Tuple

from loguru import logger

from ai_forensics.analysis.analyzer import Analyzer
from ai_forensics.analysis.base import AnalysisReport
from ai_forensics.model_formats.gguf.gguf import GGUFParseError
from ai_forensics.model_formats.gguf.gguf_quantization import QUANTIZATION_MAP
from ai_forensics.model_formats.gguf.gguf_versions import (
    T_BOOL,
    T_FLOAT32,
    T_FLOAT64,
    T_INT8,
    T_INT16,
    T_INT32,
    T_INT64,
    T_STRING,
    T_UINT8,
    T_UINT16,
    T_UINT32,
    T_UINT64,
    parse_gguf_versioned,
)


class GGUFAnalyzer(Analyzer):
    """Analyzer implementation for GGUF files."""

    def get_format_name(self) -> str:
        return "gguf"

    def _validate_and_format_kv(self, k, v, report: AnalysisReport):
        """Validates KV entries and creates findings."""
        # Type validation check
        ok = True
        value_str = ""
        # The parser stores numerics as raw bytes, but strings and bools are pre-converted.
        is_numeric_bytes = isinstance(v.value, bytes) and not v.is_array

        try:
            if is_numeric_bytes:
                if v.type in (
                    T_UINT8,
                    T_INT8,
                    T_UINT16,
                    T_INT16,
                    T_UINT32,
                    T_INT32,
                    T_UINT64,
                    T_INT64,
                ):
                    # Check if the endianness is LE, as GGUF specifies
                    signed = v.type in (T_INT8, T_INT16, T_INT32, T_INT64)
                    val = int.from_bytes(v.value, "little", signed=signed)
                    value_str = str(val)
                elif v.type in (T_FLOAT32, T_FLOAT64):
                    fmt = "<f" if v.type == T_FLOAT32 else "<d"
                    val = struct.unpack(fmt, v.value)[0]
                    value_str = f"{val:.6f}"
                else:
                    value_str = "Unhandled numeric bytes type"
                    ok = False
            elif v.is_array:
                # Provide a more useful summary for arrays
                count = len(v.value) if isinstance(v.value, list) else 0
                preview = f"[{', '.join(map(str, v.value[:3]))}{', ...' if count > 3 else ''}]"
                value_str = f"Array, Count={count}, Preview={preview}"
            else:
                # Handle pre-converted types (string, bool)
                value_str = str(v.value)

        except (struct.error, UnicodeDecodeError, TypeError) as e:
            ok = False
            value_str = f"[Decode Error: {type(e).__name__}]"

        # Truncate long strings to keep the table clean
        if len(value_str) > 70:
            value_str = value_str[:67] + "..."

        report.add(
            f"kv_store:{k}",
            ok,
            details="",
            **{
                "key": k,
                "type": v.type if isinstance(v.type, str) else v.type.__class__.__name__,
                "value": value_str,
            },
        )

    def _perform_analysis(self, mv: memoryview, report: AnalysisReport) -> None:
        """Core GGUF analysis logic."""
        file_size = report.file_size  # Get file size from the report

        try:
            parsed = parse_gguf_versioned(mv, file_size=file_size)
        except GGUFParseError as e:
            report.add("parse", False, f"GGUF parse error: {e}")
            return

        if parsed.model is None:
            report.add("parse", False, "No GGUF version parser accepted this file")
            for mm in parsed.mismatches:
                report.add_reason(f"gguf {mm.version}", mm.reason)
            return

        model = parsed.model
        # Update the report metadata
        report.metadata.update(
            {k: v for k, v in model.kv.items() if "profile" not in k}
        )  # Keep profile for later

        # Create dedicated findings for Model Metadata ---
        report.add("model_metadata:Version", True, f"v{model.version} ({model.endian})")
        report.add("model_metadata:Alignment", True, str(model.alignment))
        report.add("model_metadata:KV_Count", True, str(model.n_kv))
        report.add("model_metadata:Tensor_Count", True, str(model.n_tensors))
        report.add("model_metadata:Data_Offset", True, str(model.data_offset))

        # Create findings for each KV Store entry
        for key, value in sorted(model.kv.items()):
            self._validate_and_format_kv(key, value, report)

        # CONSOLIDATED STRUCTURAL INTEGRITY CHECKS
        report.add(
            "structural_integrity:magic_version", True, f"GGUF v{model.version} ({model.endian})"
        )
        report.add("structural_integrity:Magic_Bytes", True, "Region: [0, 8)")
        report.add(
            "structural_integrity:GGUF_Header", True, f"Region: [8, {model.header_end_offset})"
        )
        report.add(
            "structural_integrity:KV_Store",
            True,
            f"Region: [{model.header_end_offset}, {model.kv_end_offset}) (Count: {model.n_kv})",
        )
        report.add(
            "structural_integrity:Tensor_Info",
            True,
            f"Region: [{model.kv_end_offset}, {model.tensor_info_end_offset}) (Count: {model.n_tensors})",
        )

        report.add(
            "structural_integrity:metadata_offset_bounds",
            model.tensor_info_end_offset <= file_size,
            f"Region: [0, {model.tensor_info_end_offset})",
        )
        ok_align = model.alignment != 0 and (model.alignment & (model.alignment - 1)) == 0
        report.add(
            "structural_integrity:alignment_power_of_two", ok_align, f"alignment={model.alignment}"
        )
        report.add(
            "structural_integrity:data_offset_bounds",
            model.data_offset <= file_size,
            f"Region: [{model.data_offset}, {file_size})",
        )

        order = sorted(model.tensors, key=lambda t: t.offset)
        report.add(
            "structural_integrity:tensor_offsets_sorted",
            all(order[i].offset <= order[i + 1].offset for i in range(len(order) - 1)),
            "non-decreasing offsets",
        )

        bounds: List[Tuple[str, int, int]] = []
        non_overlap = True

        # CONSOLIDATED TENSOR LAYOUT & SIZE ANALYSIS
        for i, ti in enumerate(order):
            start = model.data_offset + ti.offset
            next_start_abs = (
                (model.data_offset + order[i + 1].offset) if i + 1 < len(order) else file_size
            )
            end = next_start_abs
            bounds.append((ti.name, start, end))

            # Bounds Check
            in_file = 0 <= start <= end <= file_size

            # Size Consistency Check
            info = QUANTIZATION_MAP.get(ti.ggml_type)
            on_disk_size = end - start
            expected_size = info.get_expected_size(ti.n_elements) if info else -1
            size_ok = (expected_size == on_disk_size) if expected_size != -1 else False

            # Add a single, comprehensive finding for this tensor
            report.add(
                f"tensor_layout:{ti.name}",
                ok=(in_file and size_ok),
                details="",  # Details are now in columns
                **{
                    "start": start,
                    "end": end,
                    "on_disk": on_disk_size,
                    "expected": expected_size if expected_size != -1 else "N/A",
                    "type": ti.ggml_type.name,
                    "dims": str(ti.dims),
                },
            )

        for i in range(len(bounds) - 1):
            if bounds[i][2] > bounds[i + 1][1]:
                non_overlap = False
                break
        report.add(
            "structural_integrity:tensor_non_overlap",
            non_overlap,
            "no overlapping tensor data regions",
        )

        # Final Boundary and Profile Checks
        last_tensor_end = bounds[-1][2] if bounds else model.data_offset
        coverage_ok = last_tensor_end == file_size
        report.add(
            "structural_integrity:file_address_space_boundary",
            coverage_ok,
            f"Last data address {last_tensor_end} vs file size {file_size}",
        )

        quantization_mix = {
            f.context.get("type"): 0 for f in report.findings if f.name.startswith("tensor_layout:")
        }
        for f in report.findings:
            if f.name.startswith("tensor_layout:"):
                quantization_mix[f.context.get("type")] += 1
        profile_str = ", ".join(
            [f"{qt}: {count}" for qt, count in sorted(quantization_mix.items())]
        )
        if profile_str:
            report.add("structural_integrity:quantization_profile", True, profile_str)
