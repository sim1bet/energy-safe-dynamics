# Author: Simone Betteti
"""Compatibility launcher for the canonical three-link PortHNN-u trainer."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    runpy.run_path(str(root / "baselines/porthnn_u/scripts/train_three_link.py"), run_name="__main__")
