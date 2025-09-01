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
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel

from ai_forensics import __version__
from ai_forensics.analysis import gguf_analyzer, safetensors_analyzer
from ai_forensics.ascii import AsciiArtDisplayer
from ai_forensics.logging import configure_logging
from ai_forensics.reporting import gguf_reporter, safetensors_reporter
from ai_forensics.reporting.json_reporter import write_json

console = Console()

# Define the available stages for analysis. This makes it easy to add more in the future.
AVAILABLE_STAGES: List[str] = ["sha256", "structure"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aifx",
        description="AI Forensics (Python) — GGUF & SafeTensors inspection with zero-copy IO.",
        formatter_class=argparse.RawTextHelpFormatter,  # Keeps formatting clean
    )
    # This is the key change to make the help text user-friendly
    sub = p.add_subparsers(dest="cmd", title="Available Commands", metavar="<command>")

    # default banner (no args) handled in main()

    # Subcommand "scan"
    sp_scan = sub.add_parser("scan", help="Scan a local .gguf or .safetensors file")
    sp_scan.add_argument("path", help="Path to model file (.gguf | .safetensors)")
    sp_scan.add_argument("--debug", action="store_true", help="Enable debug logging")
    sp_scan.add_argument(
        "--json-out", type=str, default=None, help="Write JSON report to this path"
    )
    sp_scan.add_argument(
        "--stage",
        nargs="+",  # This allows for one or more arguments
        choices=AVAILABLE_STAGES,
        metavar="STAGE",
        help=(
            f"Run only specific analysis stages. Defaults to all stages if not provided.\n"
            f"Available stages: {', '.join(AVAILABLE_STAGES)}.\n"
            f"Can be combined, e.g., --stage sha256 structure"
        ),
    )

    # Subcommand "version"
    sub.add_parser("version", help="Show the version of ai-forensics")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # No subcommand → preserve your current behavior (ASCII banner)
    if not args.cmd:
        AsciiArtDisplayer().display()
        return 0

    # Subcommand "version"
    if args.cmd == "version":
        console.print(f"Ai Forensics Version {__version__}")
        return 0

    # Subcommand "scan"
    if args.cmd == "scan":
        configure_logging(debug=args.debug)
        path = args.path
        if not os.path.exists(path):
            console.print(f"[red]File not found:[/red] {path}")
            return 2

        ext = os.path.splitext(path)[1].lower()

        # Instantiate the appropriate analyzer
        if ext == ".gguf":
            analyzer = gguf_analyzer.GGUFAnalyzer(path)
            reporter = gguf_reporter
        elif ext in (".safetensors", ".safetensor"):
            analyzer = safetensors_analyzer.SafeTensorsAnalyzer(path)
            reporter = safetensors_reporter
        else:
            console.print(f"[red]Unknown file type:[/red] {ext}")
            return 2

        # Run the analysis
        # Determine which stages to run. If the user provided none, default to all.
        stages_to_run = args.stage or AVAILABLE_STAGES

        console.print(f"[dim]Running stages: {', '.join(stages_to_run)}...[/dim]")

        # Run the analysis with the selected stages
        rep = analyzer.run(stages=stages_to_run)

        # The reporting logic doesn't need to change; it will render whatever
        # information is present in the final report object.
        console.print(
            Panel(
                f"[bold]Result:[/bold] {'[green]OK[/green]' if rep.ok else '[red]FAILED[/red]'}",
                style="bold cyan",
            )
        )

        # Dispatch to the correct reporter
        reporter.render_report(rep)

        if args.json_out:
            write_json(rep, args.json_out)
            console.print(f"[dim]Wrote JSON report → {args.json_out}[/dim]")

        return 0

    # Fallback for unknown commands, argparse handles this automatically,
    # but we'll be explicit.
    parser.print_help()
    return 1  # Return a non-zero exit code for invalid commands
