"""Locate the project-local Google Research Football package."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_local_football_on_path() -> None:
    """Prefer the checked-out GRF package bundled with this project."""

    football_dir = Path(__file__).resolve().parents[1] / "football"
    if football_dir.exists() and str(football_dir) not in sys.path:
        sys.path.insert(0, str(football_dir))
