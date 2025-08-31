# ai_forensics/analysis/gguf_analyzer.py
"""
GGUF analyzer: version-aware structural verification + reason matrix.
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

from loguru import logger

from ..formats.gguf.gguf import GGUFParseError
from ..formats.gguf.gguf_versions import parse_gguf_versioned
from ..io.file_reader import LocalFileSource
from ..observability import Timer
from .base import AnalysisReport


def _sha256_mv(mv: memoryview) -> str:
    h = hashlib.sha256()
    h.update(mv)
    return h.hexdigest()


def analyze_file(path: str, *, debug: bool = False, max_workers: int = 8) -> AnalysisReport:
    """Analyze a GGUF file, including version identification and mismatch reasons."""
    src = LocalFileSource(path)
    with src.open() as mf:
        mv = mf.view
        file_size = mf.size

        with Timer("sha256") as t_hash:
            sha256_hex = _sha256_mv(mv)
        logger.debug("SHA256 computed in {ms:.2f}ms", ms=t_hash.duration_ms)

        with Timer("parse") as t_parse:
            try:
                parsed = parse_gguf_versioned(mv, file_size=file_size)
            except GGUFParseError as e:
                rep = AnalysisReport(
                    file_path=path,
                    file_size=file_size,
                    sha256_hex=sha256_hex,
                    format="gguf",
                    metadata={},
                )
                rep.add("parse", False, f"GGUF parse error: {e}")
                rep.add_reason("gguf v1/v2/v3", str(e))
                return rep
        logger.debug("Parsed GGUF in {ms:.2f}ms", ms=t_parse.duration_ms)

        if parsed.model is None:
            rep = AnalysisReport(
                file_path=path,
                file_size=file_size,
                sha256_hex=sha256_hex,
                format="gguf",
                metadata={},
            )
            for mm in parsed.mismatches:
                rep.add_reason(f"gguf {mm.version}", mm.reason)
            rep.add("parse", False, "No version parser accepted this file")
            return rep

        model = parsed.model
        report = AnalysisReport(
            file_path=path,
            file_size=file_size,
            sha256_hex=sha256_hex,
            format="gguf",
            metadata={
                "version": model.version,
                "endian": model.endian,
                "alignment": model.alignment,
                "n_kv": model.n_kv,
                "n_tensors": model.n_tensors,
                "data_offset": model.data_offset,
            },
        )

        # Basic checks
        report.add("magic_version", True, f"GGUF v{model.version} ({model.endian})")
        ok_align = model.alignment != 0 and (model.alignment & (model.alignment - 1)) == 0
        report.add("alignment_power_of_two", ok_align, f"alignment={model.alignment}")
        report.add(
            "data_offset_bounds",
            model.data_offset <= file_size,
            f"data_start={model.data_offset}, file_size={file_size}",
        )

        # Tensor region checks (bounds & non-overlap via next-start rule)
        order = sorted(model.tensors, key=lambda t: t.offset)
        ok_sorted = all(order[i].offset <= order[i + 1].offset for i in range(len(order) - 1))
        report.add("tensor_offsets_sorted", ok_sorted, "non-decreasing offsets")

        bounds: List[Tuple[str, int, int]] = []
        for i, ti in enumerate(order):
            start = model.data_offset + ti.offset
            next_start_abs = (
                (model.data_offset + order[i + 1].offset) if i + 1 < len(order) else file_size
            )
            end = next_start_abs
            in_file = 0 <= start <= end <= file_size
            report.add(
                f"tensor_bounds:{ti.name}",
                in_file,
                f"[{start},{end}) type={ti.ggml_type} dims={ti.dims}",
            )
            bounds.append((ti.name, start, end))

        non_overlap = True
        for i in range(len(bounds) - 1):
            _, s0, e0 = bounds[i]
            _, s1, _ = bounds[i + 1]
            if e0 > s1:
                non_overlap = False
                break
        report.add("tensor_non_overlap", non_overlap, "no overlapping tensor regions")

        return report
        return report
