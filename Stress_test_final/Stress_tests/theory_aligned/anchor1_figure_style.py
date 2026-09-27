# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Central visual identity for the Duffing Anchor 1 figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory


DOUBLE_COLUMN_IN = 7.2
SINGLE_COLUMN_IN = 3.5
GROUND_TRUTH = "#242424"
PORTHNN = "#D55E00"
PHEBM = "#0072B2"
CERTIFICATE = "#009E73"
VIOLATION = "#CC79A7"
NEUTRAL = "#8A8A8A"
METHOD_COLORS = {"truth": GROUND_TRUTH, "porthnn_u": PORTHNN, "ph_ebm": PHEBM}
METHOD_STYLES = {"truth": "-", "porthnn_u": "-.", "ph_ebm": "--"}
METHOD_MARKERS = {"truth": "o", "porthnn_u": "^", "ph_ebm": "s"}
METHOD_LABELS = {"truth": "Ground truth", "porthnn_u": "PortHNN-u", "ph_ebm": "pH-EBM"}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.2,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "legend.fontsize": 6.7,
        "legend.frameon": False,
        "lines.linewidth": 1.25,
        "savefig.transparent": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def panel_heading(
    label_axis,
    title_axis,
    label: str,
    title: str,
    y: float = 1.10,
    label_x: float = -0.14,
    vertical_axis=None,
) -> None:
    vertical_axis = label_axis if vertical_axis is None else vertical_axis
    label_transform = blended_transform_factory(label_axis.transAxes, vertical_axis.transAxes)
    title_transform = blended_transform_factory(title_axis.transAxes, vertical_axis.transAxes)
    label_axis.text(label_x, y, label.lower(), transform=label_transform, fontsize=9,
                    fontweight="bold", va="center", ha="left", clip_on=False)
    title_axis.text(0.5, y, title, transform=title_transform,
                    fontsize=8.2, fontweight="bold", va="center", ha="center", clip_on=False)
