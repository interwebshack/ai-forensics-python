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
            report.add_reason("gguf v1/v2/v3", str(e))
            return

        if parsed.model is None:
            for mm in parsed.mismatches:
                report.add_reason(f"gguf {mm.version}", mm.reason)
            report.add("parse", False, "No version parser accepted this file")
            return

        model = parsed.model
        # Update the report metadata
        report.metadata.update(
            {
                "version": model.version,
                "endian": model.endian,
                "alignment": model.alignment,
                "n_kv": model.n_kv,
                "n_tensors": model.n_tensors,
                "data_offset": model.data_offset,
            }
        )

        # Metadata Layout Checks
        report.add(
            "structural_integrity:magic_version", True, f"GGUF v{model.version} ({model.endian})"
        )
        report.add(
            "structural_integrity:GGUF_Header", True, f"Region: [8, {model.header_end_offset})"
        )
        report.add(
            "structural_integrity:KV_Store",
            True,
            f"Region: [{model.header_end_offset}, {model.kv_end_offset})",
        )
        report.add(
            "structural_integrity:Tensor_Info",
            True,
            f"Region: [{model.kv_end_offset}, {model.tensor_info_end_offset})",
        )

        # Offset and Alignment Checks
        report.add(
            "structural_integrity:metadata_offset_bounds",
            model.tensor_info_end_offset <= file_size,
            f"End of metadata: {model.tensor_info_end_offset}",
        )
        ok_align = model.alignment != 0 and (model.alignment & (model.alignment - 1)) == 0
        report.add(
            "structural_integrity:alignment_power_of_two", ok_align, f"alignment={model.alignment}"
        )
        report.add(
            "structural_integrity:data_offset_bounds",
            model.data_offset <= file_size,
            f"Start of data: {model.data_offset}",
        )

        # Tensor Structure Checks
        order = sorted(model.tensors, key=lambda t: t.offset)
        report.add(
            "structural_integrity:tensor_offsets_sorted",
            all(order[i].offset <= order[i + 1].offset for i in range(len(order) - 1)),
            "non-decreasing offsets",
        )

        bounds: List[Tuple[str, int, int]] = []
        for i, ti in enumerate(order):
            start = model.data_offset + ti.offset
            next_start_abs = (
                (model.data_offset + order[i + 1].offset) if i + 1 < len(order) else file_size
            )
            # The on-disk size of the tensor data is the space between its start and
            # the next tensor's start
            end = next_start_abs
            in_file = 0 <= start <= end <= file_size

            # We now pass the details as structured context for better reporting.
            report.add(
                f"tensor_bounds:{ti.name}",
                in_file,
                f"[{start},{end})",  # The detail string is now simpler
                **{
                    "start": start,
                    "end": end,
                    "type": ti.ggml_type.name,
                    "dims": str(ti.dims),
                },
            )
            bounds.append((ti.name, start, end))

        non_overlap = True
        for i in range(len(bounds) - 1):
            _, s0, e0 = bounds[i]
            _, s1, _ = bounds[i + 1]
            if e0 > s1:
                non_overlap = False
                break
        report.add(
            "structural_integrity:tensor_non_overlap", non_overlap, "no overlapping tensor regions"
        )

        # Quantization Profile (Informational)
        quantization_mix = {k: v for k, v in report.metadata.get("profile", {}).items()}
        profile_str = ", ".join([f"{qt}: {count}" for qt, count in quantization_mix.items()])
        if profile_str:
            report.add("structural_integrity:quantization_profile", True, profile_str)

        # Final File Boundary Check
        last_tensor_end = bounds[-1][2] if bounds else model.data_offset
        coverage_ok = last_tensor_end == file_size
        details = (
            f"Last data address {last_tensor_end} matches file size {file_size}."
            if coverage_ok
            else f"Last data address {last_tensor_end} does not match file size {file_size}."
        )
        report.add("structural_integrity:file_address_space_boundary", coverage_ok, details)
