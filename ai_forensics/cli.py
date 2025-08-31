# ai_forensics/cli.py
"""
cli.py

Rich console CLI:
- Default: show ASCII banner (your current behavior)
- scan:    analyze a .gguf or .safetensors file, print summary, findings, and
           reason matrix.
- version: show the package version.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from ai_forensics import __version__
from ai_forensics.analysis import gguf_analyzer, safetensors_analyzer
from ai_forensics.ascii import AsciiArtDisplayer
from ai_forensics.logging import configure_logging
from ai_forensics.reporting.console import render_report
from ai_forensics.reporting.json_reporter import write_json

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aifx",
        description="AI Forensics (Python) — GGUF & SafeTensors inspection with zero-copy IO.",
        formatter_class=argparse.RawTextHelpFormatter,  # Keeps formatting clean
    )
    # This is the key change to make the help text user-friendly
    sub = p.add_subparsers(dest="cmd", title="Available Commands", metavar="<command>")

    # default banner (no args) handled in main()

    # "scan" subcommand
    sp_scan = sub.add_parser("scan", help="Scan a local .gguf or .safetensors file")
    sp_scan.add_argument("path", help="Path to model file (.gguf | .safetensors)")
    sp_scan.add_argument("--debug", action="store_true", help="Enable debug logging")
    sp_scan.add_argument(
        "--json-out", type=str, default=None, help="Write JSON report to this path"
    )

    # "version" subcommand
    sub.add_parser("version", help="Show the version of ai-forensics")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # No subcommand → preserve your current behavior (ASCII banner)
    if not args.cmd:
        AsciiArtDisplayer().display()
        return 0

    if args.cmd == "version":
        console.print(f"Ai Forensics Version {__version__}")
        return 0

    if args.cmd == "scan":
        configure_logging(debug=args.debug)
        path = args.path
        if not os.path.exists(path):
            console.print(f"[red]File not found:[/red] {path}")
            return 2

        ext = os.path.splitext(path)[1].lower()
        if ext == ".gguf":
            rep = gguf_analyzer.analyze_file(path, debug=args.debug)
        elif ext in (".safetensors", ".safetensor"):
            rep = safetensors_analyzer.analyze_file(path, debug=args.debug)
        else:
            console.print(
                f"[red]Unknown/unsupported file type:[/red] {ext} (use .gguf or .safetensors)"
            )
            return 2

        console.print(
            Panel(
                f"[bold]Result:[/bold] {'[green]OK[/green]' if rep.ok else '[red]FAILED[/red]'}",
                style="bold cyan",
            )
        )
        # Generate console report
        render_report(rep)

        if args.json_out:
            write_json(rep, args.json_out)
            console.print(f"[dim]Wrote JSON report → {args.json_out}[/dim]")

        return 0

    # Fallback for unknown commands, argparse handles this automatically,
    # but we'll be explicit.
    parser.print_help()
    return 1  # Return a non-zero exit code for invalid commands
