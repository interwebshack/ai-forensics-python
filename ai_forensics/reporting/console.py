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


def _render_file_layout_table(title: str, findings: List[Finding]) -> None:
    """Specialized renderer for the File Layout table."""
    table = Table(title=title, box=box.ROUNDED, show_lines=False, title_style="bold magenta")
    table.add_column("Section", style="cyan")
    table.add_column("Start Address", justify="right", style="white")
    table.add_column("End Address", justify="right", style="white")
    table.add_column("Description", style="white")

    # This table is informational, so we sort by start address
    findings.sort(key=lambda f: f.context.get("start", 0))

    for f in findings:
        section_name = f.name.split(":", 1)[1].replace("_", " ").title()
        ctx = f.context
        start = str(ctx.get("start", "N/A"))
        end = str(ctx.get("end", "N/A"))
        table.add_row(section_name, start, end, f.details)

    console.print(table)


def _render_overall_result_table(title: str, findings: List[Finding]) -> None:
    """Specialized renderer for the Overall Result table."""
    table = Table(title=title, box=box.HEAVY_HEAD, show_header=False, title_style="bold green")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Details")

    for f in findings:
        status = "[green]PASS[/green]" if f.ok else "[bold red]FAIL[/bold red]"
        table.add_row(status, f.details)

    console.print(table)


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


def _render_size_consistency_table(title: str, findings: List[Finding]) -> None:
    """Specialized renderer for the 'Tensor Size Consistency' table."""
    table = Table(title=title, box=box.ROUNDED, show_lines=False, title_style="bold magenta")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Tensor Name", style="cyan", no_wrap=True)
    table.add_column("On-Disk Size", justify="right", style="white")
    table.add_column("Expected Size", justify="right", style="white")
    table.add_column("GGML Type", justify="left", style="yellow")

    # Sort findings by the 'start' address to match the bounds table
    findings.sort(key=lambda f: f.context.get("start", 0))

    for f in findings:
        status = "[green]PASS[/green]" if f.ok else "[bold red]FAIL[/bold red]"
        tensor_name = f.name.split(":", 1)[1]

        ctx = f.context
        on_disk = str(ctx.get("on_disk", "N/A"))
        expected = str(ctx.get("expected", "N/A"))
        ggml_type = ctx.get("ggml_type", "N/A")

        # Add a subtle visual cue if the sizes do not match
        if not f.ok:
            on_disk = f"[red]{on_disk}[/red]"
            expected = f"[yellow]{expected}[/yellow]"

        table.add_row(status, tensor_name, on_disk, expected, ggml_type)

    console.print(table)


def _render_generic_table(
    title: str, findings: List[Finding], *, custom_sort_order: List[str] = None
) -> None:
    """Generic renderer for finding groups, with optional custom sorting."""
    table = Table(title=title, box=box.ROUNDED, show_lines=False, title_style="bold magenta")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Details", style="white")

    # Custom Sorting Logic
    if custom_sort_order:
        # Create a mapping from name to sort index, defaulting to a large number
        sort_map = {name: i for i, name in enumerate(custom_sort_order)}
        findings.sort(key=lambda f: sort_map.get(f.name.split(":", 1)[-1], 999))
    else:
        findings.sort(key=lambda x: x.name)

    for f in findings:
        status = "[green]PASS[/green]" if f.ok else "[bold red]FAIL[/bold red]"
        check_name = f.name.split(":", 1)[-1].replace("_", " ").title()

        # Special case for "KV Store" to keep it uppercase
        if check_name == "Kv Store":
            check_name = "KV Store"

        row_data = [status, check_name, f.details]
        table.add_row(*row_data)

    console.print(table)


def render_findings(rep: AnalysisReport) -> None:
    """Render findings grouped into dynamically created tables for each check type."""
    if not rep.findings:
        return

    groups = defaultdict(list)
    for f in rep.findings:
        if ":" in f.name:
            group_name = f.name.split(":", 1)[0]
            groups[group_name].append(f)

    # Simplified group order and defined custom sort for integrity table ---
    group_order = [
        "structural_integrity",
        "tensor_bounds",
        "tensor_size_consistency",
        "quantization_known",
        "quantization_alignment",
    ]

    integrity_sort_order = [
        "alignment_power_of_two",
        "magic_version",
        "metadata_offset_bounds",
        "GGUF_Header",
        "KV_Store",
        "Tensor_Info",
        "data_offset_bounds",
        "tensor_offsets_sorted",
        "tensor_non_overlap",
        "quantization_profile",
        "file_address_space_boundary",
    ]

    for group_name in group_order:
        if group_name in groups:
            findings_in_group = groups[group_name]
            title = group_name.replace("_", " ").title() + " Checks"

            if group_name == "structural_integrity":
                _render_generic_table(
                    title, findings_in_group, custom_sort_order=integrity_sort_order
                )
            elif group_name == "tensor_bounds":
                _render_tensor_bounds_table(title, findings_in_group)
            elif group_name == "tensor_size_consistency":
                _render_size_consistency_table(title, findings_in_group)
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
