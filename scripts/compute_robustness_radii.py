#!/usr/bin/env python3
"""Reviewer-facing launcher for the integrated runtime_hner99 package."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_hner99.runtime import main
if __name__ == "__main__":
    main()

