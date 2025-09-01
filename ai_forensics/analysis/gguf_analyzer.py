# ai_forensics/analysis/gguf_analyzer.py
"""
GGUF analyzer: version-aware structural verification + reason matrix.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from loguru import logger

from ai_forensics.analysis.analyzer import Analyzer
from ai_forensics.analysis.base import AnalysisReport
from ai_forensics.model_formats.gguf.gguf import GGUFParseError
from ai_forensics.model_formats.gguf.gguf_quantization import QUANTIZATION_MAP
from ai_forensics.model_formats.gguf.gguf_versions import parse_gguf_versioned


class GGUFAnalyzer(Analyzer):
    """Analyzer implementation for GGUF files."""

    def get_format_name(self) -> str:
        return "gguf"

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

        # --- CONSOLIDATED STRUCTURAL INTEGRITY CHECKS ---
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

        # --- CONSOLIDATED TENSOR LAYOUT & SIZE ANALYSIS ---
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
