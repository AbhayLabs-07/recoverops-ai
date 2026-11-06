"""Path-jailed filesystem isolation helper for RecoverBrain artifacts and checkpoints."""

from __future__ import annotations

import os
from pathlib import Path


def get_sandbox_root() -> Path:
    """Return the absolute path to the brain/sandbox directory."""
    # Anchored relative to this file
    root = (Path(__file__).parent / "sandbox").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_brain_path(relative_path: str) -> Path:
    """Resolve a path strictly inside brain/sandbox/.

    Rejects:
      - Non-string input
      - Empty or whitespace strings
      - Windows drive letters (C:, D:)
      - Absolute paths
      - Parent directory traversal tokens ('..')
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("Path traversal outside brain sandbox is forbidden")

    clean = relative_path.strip()
    drive, _ = os.path.splitdrive(clean)
    if drive:
        raise ValueError("Path traversal outside brain sandbox is forbidden")

    p = Path(clean)
    if p.is_absolute():
        raise ValueError("Path traversal outside brain sandbox is forbidden")

    for part in p.parts:
        if part == "..":
            raise ValueError("Path traversal outside brain sandbox is forbidden")

    root = get_sandbox_root()
    candidate = (root / p).resolve()

    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("Path traversal outside brain sandbox is forbidden")

    return candidate

