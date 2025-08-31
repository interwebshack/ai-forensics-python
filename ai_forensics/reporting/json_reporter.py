# ai_forensics/reporting/json_reporter.py
"""
JSON reporting utilities (optional).
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ai_forensics.observability import to_dict


def to_json_dict(report) -> Dict[str, Any]:
    """Convert AnalysisReport to a JSON-serializable dict."""
    return to_dict(report)


def write_json(report, path: str) -> None:
    """Write report to a file as pretty JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_dict(report), f, indent=2)
