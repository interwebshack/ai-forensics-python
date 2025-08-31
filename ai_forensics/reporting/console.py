# ai_forensics/reporting/console.py
"""
Console reporting functions for analysis results.
"""
from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

console = Console()


def render_summary(rep) -> None:
    """Render a high-level summary table."""
    t = Table(title="AI Forensics Summary", box=box.SIMPLE_HEAVY)
    t.add_column("Field", style="bold")
    t.add_column("Value")
    t.add_row("Path", rep.file_path)
    t.add_row("Size (bytes)", str(rep.file_size))
    t.add_row("Format", rep.format)
    t.add_row("SHA-256", rep.sha256_hex)
    for k, v in rep.metadata.items():
        t.add_row(k, str(v))
    console.print(t)


def render_findings(rep) -> None:
    """Render findings as a tree."""
    tree = Tree("Findings", guide_style="bright_black")
    for f in rep.findings:
        status = "[green]PASS[/green]" if f.ok else "[red]FAIL[/red]"
        node = tree.add(f"{status} — {f.name} — {f.details}")
        if f.context:
            for ck, cv in f.context.items():
                node.add(f"[dim]{ck}[/dim]: {cv}")
    console.print(tree)


def render_reason_matrix(rep) -> None:
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


def render_report(rep) -> None:
    """Renders the full console report."""
    render_summary(rep)
    render_findings(rep)
    render_reason_matrix(rep)
