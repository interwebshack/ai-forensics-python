# ai_forensics/reporting/safetensors_reporter.py
"""
SafeTensors-specific console reporting functions.
"""
from __future__ import annotations

from rich.console import Console

from ai_forensics.analysis.base import AnalysisReport

# (Import other rich components and base classes as needed)


console = Console()


def render_report(rep: AnalysisReport) -> None:
    """Renders a console report for a SafeTensors file."""
    # For now, we can use a simpler, generic reporting style.
    # This can be built out with SafeTensors-specific tables later.

    # (A simplified version of the old console reporter would go here)
    # For example, just printing the summary and findings tree:
    from .gguf_reporter import _render_generic_table, _render_reason_matrix, _render_summary

    _render_summary(rep)
    if rep.findings:
        _render_generic_table("Analysis Findings", rep.findings)
    _render_reason_matrix(rep)
