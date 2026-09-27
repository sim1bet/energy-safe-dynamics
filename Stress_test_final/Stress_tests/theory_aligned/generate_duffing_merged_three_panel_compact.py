# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Render the compact-height twin of the three-panel Duffing figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Stress_tests.theory_aligned.anchor1_figure_style import apply_style
from Stress_tests.theory_aligned.generate_duffing_merged_three_panel import (
    OUTPUT_DIR,
    generate,
)


OUTPUT = OUTPUT_DIR / "figures" / "main" / "duffing_merged_three_panel_compact"


def main() -> None:
    apply_style()
    print(json.dumps(generate(
        OUTPUT,
        figure_height=4.38,
        bottom_height_ratio=1.5942,
        bottom_width_ratios=(0.4, 0.6),
        panel_c_shift=0.01,
        panel_c_label_to_ylabel=True,
        center_saddle_marker=True,
        compact_panel_b_legend=True,
        panel_a_heading_y=0.97,
    ), indent=2))


if __name__ == "__main__":
    main()