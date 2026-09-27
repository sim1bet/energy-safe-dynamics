# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Compatibility launcher for the canonical Duffing PortHNN-u trainer."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    runpy.run_path(str(root / "baselines/porthnn_u/scripts/train_duffing.py"), run_name="__main__")
