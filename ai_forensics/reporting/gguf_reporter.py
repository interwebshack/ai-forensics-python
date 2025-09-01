# ai_forensics/reporting/gguf_reporter.py
"""
GGUF-specific console reporting functions.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_forensics.analysis.base import AnalysisReport, Finding

console = Console()

STATUS_STYLES = {
    "PASS": "[green]PASS[/green]",
    "WARN": "[yellow]WARN[/yellow]",
    "FAIL": "[bold red]FAIL[/bold red]",
}


def _render_summary(rep: AnalysisReport) -> None:
    """Render a high-level summary table with redundant info removed."""
    t = Table(title="AI Forensics Summary", box=box.SIMPLE_HEAVY)
    t.add_column("Field", style="bold")
    t.add_column("Value")
    t.add_row("Path", rep.file_path)
    t.add_row("Size (bytes)", str(rep.file_size))
    t.add_row("Format", rep.format)
    t.add_row("SHA-256", rep.sha256_hex)
    console.print(t)


def _render_kv_forensic_grid(findings: List[Finding]) -> None:
    """Renders a detailed 'Forensic Grid' for the KV store validation results."""
    console.print(
        Panel("[bold]Key-Value Store Deep Validation[/bold]", style="bold magenta", expand=False)
    )

    findings.sort(key=lambda f: f.context.get("start", 0))

    for index, f in enumerate(findings, start=1):
        ctx = f.context
        sub_checks = ctx.get("sub_checks", [])

        # Create the title for the outer panel
        title = (
            f"Key: [cyan]{ctx.get('key', 'N/A')}[/cyan]  "
            f"[dim]Index: {index}, Addr: {ctx.get('start', '?')}-{ctx.get('end', '?')} ({ctx.get('size', '?')} B)[/dim]"
        )

        # Create the inner table for the sub-checks
        sub_table = Table(box=None, show_header=False, padding=(0, 1))
        sub_table.add_column("Status", width=8)
        sub_table.add_column("Check", style="cyan", width=15)
        sub_table.add_column("Details", style="white")

        for sc in sub_checks:
            status_styled = STATUS_STYLES.get(sc.status, "[bold red]FAIL[/bold red]")
            sub_table.add_row(status_styled, sc.name, sc.details)

        # Determine the border style based on the overall status
        overall_status = "PASS"
        if any(sc.status == "FAIL" for sc in sub_checks):
            overall_status = "FAIL"
        elif any(sc.status == "WARN" for sc in sub_checks):
            overall_status = "WARN"

        border_style = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(overall_status, "red")

        console.print(Panel(sub_table, title=title, border_style=border_style, expand=False))


def _render_combined_tensor_table(title: str, findings: List[Finding]) -> None:
    """Renders a single, combined table for tensor layout, bounds, and size integrity."""
    table = Table(title=title, box=box.ROUNDED, show_lines=False, title_style="bold magenta")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Index", justify="right", style="dim")
    table.add_column("Tensor Name", style="cyan", no_wrap=True)
    table.add_column("Start Address", justify="right", style="white")
    table.add_column("End Address", justify="right", style="white")
    table.add_column("On-Disk Size", justify="right", style="white")
    table.add_column("Expected Size", justify="right", style="white")
    table.add_column("GGML Type", justify="left", style="yellow")
    table.add_column("Dimensions", justify="left", style="green")

    findings.sort(key=lambda f: f.context.get("start", 0))

    for index, f in enumerate(findings, start=1):
        status = "[green]PASS[/green]" if f.ok else "[bold red]FAIL[/bold red]"
        tensor_name = f.name.split(":", 1)[1]
        ctx = f.context

        on_disk = str(ctx.get("on_disk", "N/A"))
        expected = str(ctx.get("expected", "N/A"))
        if not f.ok and on_disk != expected:
            on_disk = f"[red]{on_disk}[/red]"
            expected = f"[yellow]{expected}[/yellow]"

        table.add_row(
            status,
            str(index),
            tensor_name,
            str(ctx.get("start", "N/A")),
            str(ctx.get("end", "N/A")),
            on_disk,
            expected,
            ctx.get("type", "N/A"),
            ctx.get("dims", "N/A"),
        )

    console.print(table)


def _render_generic_table(
    title: str, findings: List[Finding], *, custom_sort_order: List[str] = None
) -> None:
    """Generic renderer for finding groups, with optional custom sorting."""
    table = Table(title=title, box=box.ROUNDED, show_lines=False, title_style="bold magenta")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Details", style="white")

    if custom_sort_order:
        sort_map = {name: i for i, name in enumerate(custom_sort_order)}
        findings.sort(key=lambda f: sort_map.get(f.name.split(":", 1)[-1], 999))
    else:
        findings.sort(key=lambda x: x.name)

    for f in findings:
        status = "[green]PASS[/green]" if f.ok else "[bold red]FAIL[/bold red]"
        check_name = f.name.split(":", 1)[-1].replace("_", " ").title()
        if check_name == "Kv Store":
            check_name = "KV Store"

        table.add_row(status, check_name, f.details)

    console.print(table)


def _render_reason_matrix(rep: AnalysisReport) -> None:
    """Render the reason matrix table."""
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
    """Renders the full, detailed console report specifically for a GGUF file."""
    _render_summary(rep)

    groups = defaultdict(list)
    for f in rep.findings:
        if ":" in f.name:
            groups[f.name.split(":", 1)[0]].append(f)

    # Define the order for the main structural integrity table
    integrity_sort_order = [
        "alignment_power_of_two",
        "magic_version",
        "metadata_offset_bounds",
        "Magic_Bytes",
        "GGUF_Header",
        "KV_Store",
        "Tensor_Info",
        "data_offset_bounds",
        "tensor_offsets_sorted",
        "tensor_non_overlap",
        "file_address_space_boundary",
        "quantization_profile",
    ]

    # Structural Integrity
    if "structural_integrity" in groups:
        _render_generic_table(
            "Structural Integrity Checks",
            groups["structural_integrity"],
            custom_sort_order=integrity_sort_order,
        )

    # KV table
    if "kv_analysis" in groups:
        _render_kv_forensic_grid(groups["kv_analysis"])

    # Tensor Layout
    if "tensor_layout" in groups:
        _render_combined_tensor_table(
            "Tensor Layout & Size Integrity Checks", groups["tensor_layout"]
        )

    _render_reason_matrix(rep)
