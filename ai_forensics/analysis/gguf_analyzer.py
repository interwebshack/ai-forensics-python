# ai_forensics/analysis/gguf_analyzer.py
"""
GGUF analyzer: version-aware structural verification + reason matrix.
"""

from __future__ import annotations

import json
import struct
from typing import List, Tuple

from ai_forensics.analysis.analyzer import Analyzer
from ai_forensics.analysis.base import AnalysisReport
from ai_forensics.model_formats.gguf.gguf import GGUFKV, GGUFParseError
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

    def _create_kv_layout_finding(self, k: str, v: GGUFKV, report: AnalysisReport):
        """Creates a finding for the standard KV layout and integrity report."""
        ok = True
        value_str = ""
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
                    signed = v.type in (T_INT8, T_INT16, T_INT32, T_INT64)
                    val = int.from_bytes(v.value, "little", signed=signed)
                    value_str = str(val)
                elif v.type in (T_FLOAT32, T_FLOAT64):
                    fmt = "<f" if v.type == T_FLOAT32 else "<d"
                    val = struct.unpack(fmt, v.value)[0]
                    value_str = f"{val:.6f}"
            elif v.is_array:
                count = len(v.value) if isinstance(v.value, list) else 0
                preview_items = []
                for item in v.value[:3]:
                    str_item = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    readable_item = str_item.replace("Ġ", " ")
                    preview_items.append(readable_item)
                preview = f"[{', '.join(preview_items)}{', ...' if count > 3 else ''}]"
                value_str = f"Array, Count={count}, Preview={preview}"
            else:
                str_value = (
                    v.value.decode("utf-8") if isinstance(v.value, bytes) else str(v.value)
                ).replace("Ġ", " ")
                value_str = str_value

        except (struct.error, UnicodeDecodeError, TypeError) as e:
            ok = False
            value_str = f"[Decode Error: {type(e).__name__}]"

        if len(value_str) > 70:
            value_str = value_str[:67] + "..."

        report.add(
            f"kv_layout:{k}",
            ok,
            details="",
            **{
                "key": k,
                "type": v.type.name if hasattr(v.type, "name") else str(v.type),
                "value": value_str,
                "start": v.offset_start,
                "end": v.offset_end,
                "size": v.offset_end - v.offset_start,
            },
        )

    def _perform_deep_kv_analysis(self, model, report: AnalysisReport):
        """Performs deep content, semantic, and security analysis of the KV store."""
        known_file_types = {
            0: "All F32",
            1: "Mostly F16",
            2: "Mostly Q4_0",
            3: "Mostly Q4_1",
            7: "Mostly Q5_0",
            8: "Mostly Q5_1",
            9: "Mostly Q8_0",
            10: "Mostly Q2_K",
            11: "Mostly Q3_K",
            12: "Mostly Q4_K",
            13: "Mostly Q5_K",
            14: "Mostly Q6_K",
        }
        prompt_injection_keywords = [
            "ignore all previous",
            "disregard the above",
            "override instructions",
            "secret instruction",
            "true master",
            "confidential",
        ]
        template_exploit_payloads = ["__globals__", "__init__", "os.system", "subprocess.run"]

        for key, v in model.kv.items():
            ok = True
            full_value_str = ""
            is_numeric_bytes = isinstance(v.value, bytes) and not v.is_array

            # 1. Full Value Extraction
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
                        signed = v.type in (T_INT8, T_INT16, T_INT32, T_INT64)
                        val = int.from_bytes(v.value, "little", signed=signed)
                        full_value_str = str(val)
                    elif v.type in (T_FLOAT32, T_FLOAT64):
                        fmt = "<f" if v.type == T_FLOAT32 else "<d"
                        val = struct.unpack(fmt, v.value)[0]
                        full_value_str = str(val)
                elif v.is_array:
                    count = len(v.value)
                    items = v.value[:25] + ["..."] + v.value[-25:] if count > 50 else v.value
                    readable_items = [
                        (item.decode("utf-8") if isinstance(item, bytes) else str(item)).replace(
                            "Ġ", " "
                        )
                        for item in items
                    ]
                    full_value_str = f"Array (Count={count})\n" + "\n".join(readable_items)
                else:
                    full_value_str = (
                        v.value.decode("utf-8") if isinstance(v.value, bytes) else str(v.value)
                    ).replace("Ġ", " ")
            except Exception as e:
                ok = False
                full_value_str = f"[Extraction Error: {e}]"

            # 2. Semantic and Cybersecurity Validation
            if key == "general.file_type" and is_numeric_bytes:
                val = int(full_value_str)
                if val not in known_file_types:
                    ok = False
                full_value_str = f"{val} ({known_file_types.get(val, 'Unknown Type!')})"

            if key == "tokenizer.chat_template":
                if full_value_str.strip().startswith("{"):
                    try:
                        full_value_str = json.dumps(json.loads(full_value_str), indent=2)
                    except json.JSONDecodeError:
                        pass
                for keyword in prompt_injection_keywords + template_exploit_payloads:
                    if keyword in full_value_str.lower():
                        ok = False
                        full_value_str += (
                            f"\n\n[CYBERSECURITY ALERT: Suspicious keyword '{keyword}' found!]"
                        )

            report.add(
                f"deep_kv_store:{key}",
                ok,
                details=full_value_str,
                **{
                    "key": key,
                    "start": v.offset_start,
                    "end": v.offset_end,
                    "size": v.offset_end - v.offset_start,
                    "type": v.type.name if hasattr(v.type, "name") else str(v.type),
                },
            )

    def _perform_analysis(self, mv: memoryview, report: AnalysisReport) -> None:
        """Runs structural and (optionally) deep analysis."""
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

            # Standard KV Scan (for the layout table)
            for key, value in model.kv.items():
                self._create_kv_layout_finding(key, value, report)

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

        if "deep_scan" in report.stages_run:
            self._perform_deep_kv_analysis(model, report)
