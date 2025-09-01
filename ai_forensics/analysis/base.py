# ai_forensics/analysis/base.py
"""
Base analysis models and “reason matrix” support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Finding:
    """Single check result."""

    name: str
    ok: bool
    details: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasonEntry:
    """Explains why a target version/spec failed to parse/validate."""

    target: str  # e.g., "gguf v1-LE", "gguf v3-BE", "safetensors v1"
    reason: str


@dataclass
class AnalysisReport:
    """Aggregate analysis report with a reason matrix."""

    file_path: str
    file_size: int
    sha256_hex: str
    format: str  # "gguf" | "safetensors" | "unknown"
    metadata: Dict[str, Any]
    findings: List[Finding] = field(default_factory=list)
    reason_matrix: List[ReasonEntry] = field(default_factory=list)
    stages_run: List[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, details: str = "", **context: Any) -> None:
        self.findings.append(Finding(name=name, ok=ok, details=details, context=context))

    def add_reason(self, target: str, reason: str) -> None:
        self.reason_matrix.append(ReasonEntry(target=target, reason=reason))

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.findings) if self.findings else True
