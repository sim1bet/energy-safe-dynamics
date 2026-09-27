#!/usr/bin/env python3
"""Reviewer-facing launcher for fixed-budget runtime HNER99 audits."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_hner99.audit import main
if __name__ == "__main__":
    main()

