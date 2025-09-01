# ai_forensics/analysis/safetensors_analyzer.py
"""
SafeTensors analyzer: structural verification + reason matrix.
"""

from __future__ import annotations

from loguru import logger

from ai_forensics.analysis.analyzer import Analyzer
from ai_forensics.analysis.base import AnalysisReport
from ai_forensics.model_formats.safetensors.safetensors import (
    SafeTensorsParseError,
    parse_safetensors,
)


class SafeTensorsAnalyzer(Analyzer):
    """Analyzer implementation for SafeTensors files."""

    def get_format_name(self) -> str:
        return "safetensors"

    def _perform_analysis(self, mv: memoryview, report: AnalysisReport) -> None:
        """Core SafeTensors analysis logic."""
        file_size = report.file_size  # Get file size from the report

        try:
            model = parse_safetensors(mv, file_size=file_size)
        except SafeTensorsParseError as e:
            report.add("parse", False, f"SafeTensors parse error: {e}")
            report.add_reason("safetensors v1", str(e))
            return

        report.metadata.update(
            {
                "header_size": model.header_size,
                "n_tensors": len(model.tensors),
                "data_start": model.data_start,
            }
        )

        # File Layout and Coverage Analysis
        report.add(
            "file_layout:header_and_metadata",
            True,
            "SafeTensors Header (JSON)",
            **{"start": 0, "end": model.data_start},
        )
        report.add(
            "structural_integrity:metadata_offset_bounds",
            model.data_start <= file_size,
            f"Metadata region: [0, {model.data_start})",
        )
        report.add(
            "structural_integrity:data_offset_bounds",
            model.data_start <= file_size,
            f"Tensor data region: [{model.data_start}, {file_size})",
        )

        # Existing checks moved to the 'structural_integrity' group for consistency
        report.add("structural_integrity:header_basic", True, f"header_size={model.header_size}")

        prev_end = model.data_start
        ok_order = True
        ok_bounds = True
        for t in sorted(model.tensors, key=lambda x: x.data_offsets[0]):
            b, e = t.data_offsets
            abs_b = model.data_start + b
            abs_e = model.data_start + e
            in_file = 0 <= abs_b <= abs_e <= file_size
            if not in_file:
                ok_bounds = False
            if abs_b < prev_end:
                ok_order = False

            # Pass details as structured context for better reporting.
            report.add(
                f"tensor_bounds:{t.name}",
                in_file,
                f"[{abs_b},{abs_e})",  # Simpler detail string
                **{
                    "start": abs_b,
                    "end": abs_e,
                    "type": t.dtype,
                    "dims": str(t.shape),
                },
            )
            prev_end = max(prev_end, abs_e)

        report.add("tensor_order_non_overlapping", ok_order, "Non-overlapping, increasing offsets")
        report.add("tensor_bounds_all_valid", ok_bounds, "All tensor extents lie within file")

        # Overall file coverage check
        last_tensor_end = prev_end  # The loop already calculates this
        coverage_ok = last_tensor_end == file_size
        details = (
            "The end of the last tensor aligns perfectly with the end of the file."
            if coverage_ok
            else f"There are {file_size - last_tensor_end} bytes of unaccounted-for data at the end of the file."
        )
        report.add("overall_result:file_coverage", coverage_ok, details)
