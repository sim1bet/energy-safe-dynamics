"""Resolve the benchmark data bundled with this release."""
from __future__ import annotations

import os
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def dataset_root() -> Path:
    """Return the self-contained dataset root, with an explicit override hook."""
    configured = os.environ.get("EBM_DATA_ROOT")
    root = Path(configured).expanduser().resolve() if configured else RELEASE_ROOT / "datasets"
    if not root.is_dir():
        raise FileNotFoundError(f"EBM dataset root does not exist: {root}")
    return root
