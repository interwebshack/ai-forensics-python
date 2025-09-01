# ai_forensics/analysis/gguf_analyzer.py
"""
GGUF analyzer: version-aware structural verification + reason matrix.
"""

from __future__ import annotations

import re
import struct
from typing import Any, Dict, List, Tuple

from ai_forensics.analysis.analyzer import Analyzer
from ai_forensics.analysis.base import AnalysisReport
from ai_forensics.model_formats.gguf.gguf import GGUFKV, GGUFModel, GGUFParseError
from ai_forensics.model_formats.gguf.gguf_quantization import QUANTIZATION_MAP
from ai_forensics.model_formats.gguf.gguf_rules import KNOWN_KEYS
from ai_forensics.model_formats.gguf.gguf_versions import (
    T_ARRAY,
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

KV_TYPE_MAP = {
    T_UINT8: "UInt8",
    T_INT8: "Int8",
    T_UINT16: "UInt16",
    T_INT16: "Int16",
    T_UINT32: "UInt32",
    T_INT32: "Int32",
    T_UINT64: "UInt64",
    T_INT64: "Int64",
    T_FLOAT32: "Float32",
    T_FLOAT64: "Float64",
    T_BOOL: "Bool",
    T_STRING: "String",
    T_ARRAY: "Array",
}


class GGUFAnalyzer(Analyzer):
    """Analyzer implementation for GGUF files."""

    def get_format_name(self) -> str:
        return "gguf"

    def _perform_kv_analysis(self, model: GGUFModel, report: AnalysisReport) -> None:
        """
        Performs a comprehensive, 'always-on deep' validation of the KV store,
        combining structural, content, and spec-conformance checks.
        """
        seen_keys: Dict[str, GGUFKV] = {}

        # Iterate in the natural on-disk order preserved by the parser
        for v in model.kv.values():
            status = "PASS"
            checks: List[str] = []

            # 1. Structural Checks (bounds are implicit from successful parsing)
            checks.append("bounds")

            # 2. Key Name Pattern Check
            if re.match(r"^[a-z0-9][a-z0-9._-]*$", v.key):
                checks.append("key_pattern")
            else:
                status = "WARN"
                checks.append("key_pattern_fail")

            # 3. Type and Value Decoding
            type_name = KV_TYPE_MAP.get(v.type, "Unknown")
            is_numeric = v.type not in (T_STRING, T_BOOL, T_ARRAY)
            decoded_value: Any = None
            value_preview = ""
            try:
                if is_numeric and isinstance(v.value, bytes):
                    if v.type in (T_FLOAT32, T_FLOAT64):
                        fmt = "<f" if v.type == T_FLOAT32 else "<d"
                        decoded_value = struct.unpack(fmt, v.value)[0]
                    else:
                        signed = v.type in (T_INT8, T_INT16, T_INT32, T_INT64)
                        decoded_value = int.from_bytes(v.value, "little", signed=signed)
                    value_preview = str(decoded_value)
                elif v.is_array:
                    value_preview = f"Array (Count={len(v.value)})"
                else:  # String or Bool
                    decoded_value = (
                        v.value.decode("utf-8") if isinstance(v.value, bytes) else v.value
                    )
                    value_preview = str(decoded_value)
                checks.append("decode")
            except Exception:
                status = "FAIL"
                checks.append("decode_fail")

            # 4. Spec Conformance (Registry) Checks
            spec = KNOWN_KEYS.get(v.key)
            if spec:
                checks.append("known_key")
                # Type check
                if spec["type"] == type_name:
                    checks.append("spec_type")
                else:
                    status = "FAIL"
                    checks.append("spec_type_fail")
                # Range/value checks
                if decoded_value is not None:
                    if "min" in spec and decoded_value < spec["min"]:
                        status = "FAIL"
                        checks.append("range_fail")
                    elif "multiple_of" in spec and decoded_value % spec["multiple_of"] != 0:
                        status = "FAIL"
                        checks.append("multiple_fail")
                    else:
                        checks.append("range")
            else:
                status = "WARN"
                checks.append("unknown_key")

            # 5. Duplicate Check
            if v.key in seen_keys:
                # This is a simplified check; a true deep check would compare raw bytes
                if seen_keys[v.key].value == v.value:
                    status = "WARN" if status != "FAIL" else "FAIL"
                    checks.append("duplicate_ok")
                else:
                    status = "FAIL"
                    checks.append("duplicate_conflict")
            seen_keys[v.key] = v

            # Create the final finding
            report.add(
                f"kv_analysis:{v.key}",
                ok=(status != "FAIL"),
                details="",
                **{
                    "status": status,
                    "key": v.key,
                    "start": v.offset_start,
                    "end": v.offset_end,
                    "size": v.offset_end - v.offset_start,
                    "type": f"{type_name} ({v.type})",
                    "value_preview": (
                        value_preview.replace("Ġ", " ")[:40] + "..."
                        if len(value_preview) > 40
                        else value_preview.replace("Ġ", " ")
                    ),
                    "checks": ", ".join(checks),
                },
            )

    def _perform_analysis(self, mv: memoryview, report: AnalysisReport) -> None:
        """Runs structural and deep validation checks."""
        try:
            parsed = parse_gguf_versioned(mv, file_size=report.file_size)
            if parsed.model is None:
                report.add("parse", False, "No GGUF version parser accepted this file")
                for mm in parsed.mismatches:
                    report.add_reason(f"gguf {mm.version}", mm.reason)
                return
            model = parsed.model
        except GGUFParseError as e:
            report.add("parse", False, f"GGUF parse error: {e}")
            return

        if "structure" in report.stages_run:
            # --- CONSOLIDATED STRUCTURAL INTEGRITY CHECKS ---
            report.add(
                "structural_integrity:magic_version",
                True,
                f"GGUF v{model.version} ({model.endian})",
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
                model.tensor_info_end_offset <= report.file_size,
                f"Region: [0, {model.tensor_info_end_offset})",
            )
            ok_align = model.alignment != 0 and (model.alignment & (model.alignment - 1)) == 0
            report.add(
                "structural_integrity:alignment_power_of_two",
                ok_align,
                f"alignment={model.alignment}",
            )
            report.add(
                "structural_integrity:data_offset_bounds",
                model.data_offset <= report.file_size,
                f"Region: [{model.data_offset}, {report.file_size})",
            )

            order = sorted(model.tensors, key=lambda t: t.offset)
            report.add(
                "structural_integrity:tensor_offsets_sorted",
                all(order[i].offset <= order[i + 1].offset for i in range(len(order) - 1)),
                "non-decreasing offsets",
            )

            bounds: List[Tuple[str, int, int]] = []
            non_overlap = True

            for i, ti in enumerate(order):
                start = model.data_offset + ti.offset
                next_start_abs = (
                    (model.data_offset + order[i + 1].offset)
                    if i + 1 < len(order)
                    else report.file_size
                )
                end = next_start_abs
                bounds.append((ti.name, start, end))

                in_file = 0 <= start <= end <= report.file_size
                info = QUANTIZATION_MAP.get(ti.ggml_type)
                on_disk_size = end - start
                expected_size = info.get_expected_size(ti.n_elements) if info else -1
                size_ok = (expected_size == on_disk_size) if expected_size != -1 else False

                report.add(
                    f"tensor_layout:{ti.name}",
                    ok=(in_file and size_ok),
                    details="",
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

            last_tensor_end = bounds[-1][2] if bounds else model.data_offset
            coverage_ok = last_tensor_end == report.file_size
            report.add(
                "structural_integrity:file_address_space_boundary",
                coverage_ok,
                f"Last data address {last_tensor_end} vs file size {report.file_size}",
            )

            # --- Run the comprehensive KV analysis ---
            self._perform_kv_analysis(model, report)
            # A cross-field check could be added here later if needed

            quantization_mix = {}
            for f in report.findings:
                if f.name.startswith("tensor_layout:"):
                    qt = f.context.get("type")
                    if qt:
                        quantization_mix[qt] = quantization_mix.get(qt, 0) + 1
            profile_str = ", ".join(
                [f"{qt}: {count}" for qt, count in sorted(quantization_mix.items())]
            )
            if profile_str:
                report.add("structural_integrity:quantization_profile", True, profile_str)
