# ai_forensics/analysis/analyzer.py
"""
Base Analyzer class to handle common file operations.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import List

from loguru import logger

from ai_forensics.analysis.base import AnalysisReport
from ai_forensics.io.file_reader import LocalFileSource
from ai_forensics.observability import Timer


class Analyzer(ABC):
    """Abstract base class for file format analyzers."""

    def __init__(self, path: str):
        self.path = path
        self.src = LocalFileSource(path)

    def run(self, stages: List[str]) -> AnalysisReport:
        """
        Orchestrates the analysis process, running only the specified stages.

        Args:
            stages: A list of stages to run (e.g., ["sha256", "structure"]).
        """
        with self.src.open() as mf:
            mv = mf.view
            file_size = mf.size

            # Prepare a preliminary report.
            # sha256_hex defaults to a value indicating it wasn't run.
            report = AnalysisReport(
                file_path=self.path,
                file_size=file_size,
                sha256_hex="not_run",
                format=self.get_format_name(),
                metadata={},
            )

            # --- Conditionally execute SHA256 Stage ---
            if "sha256" in stages:
                with Timer("sha256") as t_hash:
                    try:
                        h = hashlib.sha256()
                        h.update(mv)
                        report.sha256_hex = h.hexdigest()
                    except Exception as e:
                        logger.error(
                            "Failed to compute SHA256 for {path}: {error}", path=self.path, error=e
                        )
                        report.sha256_hex = "computation_failed"
                logger.debug("SHA256 computed in {ms:.2f}ms", ms=t_hash.duration_ms)

            # --- Conditionally execute Structure Stage ---
            if "structure" in stages:
                with Timer("parse_and_analyze") as t_core:
                    # Delegate to the concrete implementation for the core analysis
                    self._perform_analysis(mv, report)

                logger.debug(
                    "{format} analysis completed in {ms:.2f}ms",
                    format=self.get_format_name().upper(),
                    ms=t_core.duration_ms,
                )

            return report

    @abstractmethod
    def _perform_analysis(self, mv: memoryview, report: AnalysisReport) -> None:
        """
        Format-specific analysis logic to be implemented by subclasses.
        This method should populate the given report object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the string name of the format (e.g., 'gguf')."""
        raise NotImplementedError
