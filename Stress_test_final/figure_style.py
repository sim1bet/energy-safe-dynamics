# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Single source of truth for theory-aligned publication figure styling."""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

DOUBLE_COLUMN_IN = 7.2
ENERGY_CAMERA_READY = LinearSegmentedColormap.from_list(
    "stress_test_energy",
    ["#047D7A", "#2FA7A2", "#8FD4CE", "#DCEFEA", "#F3E5B8", "#EBAE6A", "#D96C55"],
    N=256,
)
SAMPLED_RADIUS = "#0072B2"
REGULAR_RADIUS = "#D55E00"
COMPARISON_SHELL = "#009E73"
NOMINAL_SHELL = "#555555"
GRID = "#D9D9D9"


def publication_rc() -> dict:
    return {
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "axes.titlesize": 8.0, "axes.labelsize": 8.0, "axes.linewidth": 0.65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 400,
    }