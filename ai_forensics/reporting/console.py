# ai_forensics/reporting/console.py
"""
Console reporting functions for analysis results.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from rich import box
from rich.console import Console
from rich.table import Table

from ai_forensics.analysis.base import AnalysisReport, Finding

console = Console()


def render_summary(rep: AnalysisReport) -> None:
    """Render a high-level summary table."""
    t = Table(title="AI Forensics Summary", box=box.SIMPLE_HEAVY)
    t.add_column("Field", style="bold")
    t.add_column("Value")
    t.add_row("Path", rep.file_path)
    t.add_row("Size (bytes)", str(rep.file_size))
    t.add_row("Format", rep.format)
    t.add_row("SHA-256", rep.sha256_hex)
    for k, v in rep.metadata.items():
        # Special formatting for the quantization profile for better readability
        if k == "profile" and isinstance(v, dict):
            profile_str = ", ".join([f"{qt}: {count}" for qt, count in v.items()])
            t.add_row("Quantization Profile", profile_str)
        else:
            t.add_row(k, str(v))
    console.print(t)


def render_findings(rep: AnalysisReport) -> None:
    """
    Render findings grouped into dynamically created tables for each check type.
    """
    if not rep.findings:
        return

    # 1. Group the findings by category
    groups = defaultdict(list)
    # Define which check names should be grouped into the main structural integrity table
    general_check_names = {
        "parse",
        "magic_version",
        "alignment_power_of_two",
        "data_offset_bounds",
        "tensor_offsets_sorted",
        "tensor_non_overlap",
        "header_basic",
        "tensor_order_non_overlapping",
        "tensor_bounds_all_valid",
    }

    # The quantization_profile is informational and displayed in the summary, so we exclude it here.
    findings_to_render = [f for f in rep.findings if f.name != "quantization_profile"]

    for f in findings_to_render:
        if ":" in f.name:
            # Group per-tensor checks by their prefix (e.g., "tensor_bounds", "quantization_known")
            group_name = f.name.split(":", 1)[0]
            title = group_name.replace("_", " ").title() + " Checks"
            groups[title].append(f)
        elif f.name in general_check_names:
            groups["Overall Structural Integrity"].append(f)
        else:
            groups["Miscellaneous Checks"].append(f)  # Fallback for any other checks

    # 2. Render a table for each group of findings
    for title, findings_in_group in sorted(groups.items()):
        table = Table(title=title, box=box.ROUNDED, show_lines=False, title_style="bold magenta")

        table.add_column("Status", justify="center", width=8)
        table.add_column("Check", style="cyan", no_wrap=True)
        table.add_column("Details", style="white")

        # Dynamically add a "Context" column only if any finding in the group has context data
        has_context = any(f.context for f in findings_in_group)
        if has_context:
            table.add_column("Context", style="yellow")

        # 3. Populate the table rows
        for f in sorted(findings_in_group, key=lambda x: x.name):
            status = "[green]PASS[/green]" if f.ok else "[bold red]FAIL[/bold red]"

            # Use the specific part of the name for per-tensor checks for brevity
            check_name = f.name.split(":", 1)[1] if ":" in f.name else f.name

            row_data = [status, check_name, f.details]

            if has_context:
                context_str = "\n".join([f"[dim]{k}:[/dim] {v}" for k, v in f.context.items()])
                row_data.append(context_str)

            table.add_row(*row_data)

        console.print(table)


def render_reason_matrix(rep: AnalysisReport) -> None:
    """Render the reason matrix table (why certain versions/specs failed)."""
    if not rep.reason_matrix:
        return
    rt = Table(
        title="Reason Matrix (Version/Spec Mismatch Explanations)",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    rt.add_column("Target Version/Spec", style="bold")
    rt.add_column("Reason")
    for entry in rep.reason_matrix:
        rt.add_row(entry.target, entry.reason)
    console.print(rt)


def render_report(rep: AnalysisReport) -> None:
    """Renders the full console report using the new table-based format."""
    render_summary(rep)
    render_findings(rep)
    render_reason_matrix(rep)
