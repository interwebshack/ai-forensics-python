# ai_forensics/analysis/safetensors_analyzer.py
"""
SafeTensors analyzer: structural verification + reason matrix.
"""

from __future__ import annotations

import hashlib

from loguru import logger

from ai_forensics.analysis.base import AnalysisReport

from ..formats.safetensors import SafeTensorsParseError, parse_safetensors
from ..io.file_reader import LocalFileSource
from ..observability import Timer


def _sha256_mv(mv: memoryview) -> str:
    h = hashlib.sha256()
    h.update(mv)
    return h.hexdigest()


def analyze_file(path: str, *, debug: bool = False, max_workers: int = 8) -> AnalysisReport:
    """Analyze a SafeTensors file (treated as v1) and populate reason matrix on failure."""
    src = LocalFileSource(path)
    with src.open() as mf:
        mv = mf.view
        file_size = mf.size

        with Timer("sha256") as t_hash:
            sha256_hex = _sha256_mv(mv)
        logger.debug("SHA256 computed in {ms:.2f}ms", ms=t_hash.duration_ms)

        try:
            with Timer("parse") as t_parse:
                model = parse_safetensors(mv, file_size=file_size)
            logger.debug("Parsed SafeTensors in {ms:.2f}ms", ms=t_parse.duration_ms)
        except SafeTensorsParseError as e:
            rep = AnalysisReport(
                file_path=path,
                file_size=file_size,
                sha256_hex=sha256_hex,
                format="safetensors",
                metadata={},
            )
            rep.add("parse", False, f"SafeTensors parse error: {e}")
            rep.add_reason("safetensors v1", str(e))
            return rep

        report = AnalysisReport(
            file_path=path,
            file_size=file_size,
            sha256_hex=sha256_hex,
            format="safetensors",
            metadata={
                "header_size": model.header_size,
                "n_tensors": len(model.tensors),
                "data_start": model.data_start,
            },
        )
        report.add(
            "header_basic", True, f"header_size={model.header_size} data_start={model.data_start}"
        )

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
            report.add(f"tensor_bounds:{t.name}", in_file, f"[{abs_b},{abs_e}) {t.dtype} {t.shape}")
            prev_end = max(prev_end, abs_e)

        report.add("tensor_order_non_overlapping", ok_order, "Non-overlapping, increasing offsets")
        report.add("tensor_bounds_all_valid", ok_bounds, "All tensor extents lie within file")

        return report
        return report
        return report
        return report
