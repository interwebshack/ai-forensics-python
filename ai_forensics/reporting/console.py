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


def _render_tensor_bounds_table(title: str, findings: List[Finding]) -> None:
    """Specialized renderer for the 'Tensor Bounds' table with custom columns and sorting."""
    table = Table(title=title, box=box.ROUNDED, show_lines=False, title_style="bold magenta")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Tensor Name", style="cyan", no_wrap=True)
    table.add_column("Start Address", justify="right", style="white")
    table.add_column("End Address", justify="right", style="white")
    table.add_column("Type", justify="left", style="yellow")
    table.add_column("Dimensions", justify="left", style="green")

    # Sort findings by the 'start' address stored in the context
    findings.sort(key=lambda f: f.context.get("start", 0))

    for f in findings:
        status = "[green]PASS[/green]" if f.ok else "[bold red]FAIL[/bold red]"
        tensor_name = f.name.split(":", 1)[1]

        # Extract data from context, with fallbacks for safety
        ctx = f.context
        start = str(ctx.get("start", "N/A"))
        end = str(ctx.get("end", "N/A"))
        dtype = ctx.get("type", "N/A")
        dims = ctx.get("dims", "N/A")

        table.add_row(status, tensor_name, start, end, dtype, dims)

    console.print(table)


def _render_generic_table(title: str, findings: List[Finding]) -> None:
    """Generic renderer for all other finding groups."""
    table = Table(title=title, box=box.ROUNDED, show_lines=False, title_style="bold magenta")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Details", style="white")

    has_context = any(f.context for f in findings)
    if has_context:
        table.add_column("Context", style="yellow")

    for f in sorted(findings, key=lambda x: x.name):
        status = "[green]PASS[/green]" if f.ok else "[bold red]FAIL[/bold red]"
        check_name = f.name.split(":", 1)[1] if ":" in f.name else f.name
        row_data = [status, check_name, f.details]
        if has_context:
            context_str = "\n".join([f"[dim]{k}:[/dim] {v}" for k, v in f.context.items()])
            row_data.append(context_str)
        table.add_row(*row_data)

    console.print(table)


def render_findings(rep: AnalysisReport) -> None:
    """Render findings grouped into dynamically created tables for each check type."""
    if not rep.findings:
        return

    groups = defaultdict(list)
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
    findings_to_render = [f for f in rep.findings if f.name != "quantization_profile"]

    for f in findings_to_render:
        if ":" in f.name:
            group_name = f.name.split(":", 1)[0]
            groups[group_name].append(f)
        elif f.name in general_check_names:
            groups["structural_integrity"].append(f)
        else:
            groups["miscellaneous"].append(f)

    # Render a table for each group, using the specialized renderer where appropriate
    for group_name, findings_in_group in sorted(groups.items()):
        title = group_name.replace("_", " ").title() + " Checks"

        if group_name == "tensor_bounds":
            _render_tensor_bounds_table(title, findings_in_group)
        else:
            _render_generic_table(title, findings_in_group)


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
