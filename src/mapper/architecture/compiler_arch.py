"""Shared compiler_arch.json loading and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

REQUIRED_KEYS = [
    "schema_version",
    "fabric_contexts",
    "features",
    "layout_hash",
    "routing_programming",
]


def load_and_validate_compiler_arch(path: str) -> Dict[str, Any]:
    """Load and validate a compiler_arch.json file."""
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"compiler_arch.json not found: {path}")

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in compiler_arch.json: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("compiler_arch.json root must be an object")

    for key in REQUIRED_KEYS:
        if key not in payload:
            raise ValueError(f"compiler_arch.json missing required key '{key}'")

    if int(payload.get("fabric_contexts", 0)) < 1:
        raise ValueError("fabric_contexts must be >= 1")

    return payload
