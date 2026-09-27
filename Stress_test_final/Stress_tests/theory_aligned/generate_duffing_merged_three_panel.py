# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Render the compact three-panel Duffing comparison figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Stress_tests.theory_aligned.anchor1_figure_style import (
    DOUBLE_COLUMN_IN,
    GROUND_TRUTH,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    METHOD_STYLES,
    PHEBM,
    PORTHNN,
    apply_style,
    panel_heading,
)
from Stress_tests.theory_aligned.generate_anchor1_main_figure import _energy_artifacts
from Stress_tests.theory_aligned.generate_duffing_merged_main_figure import (
    EQUAL_DIMENSIONS_ANCHOR,
    EQUAL_DIMENSIONS_COLOR,
    OUTPUT_DIR,
    _draw_panel_b,
    _legacy_data,
)

OUTPUT = OUTPUT_DIR / "figures" / "main" / "duffing_merged_three_panel"


def _draw_energy_geometry(fig, axes, colorbar_axis, legacy, grids, critical, qq, pp) -> None:
    levels = np.linspace(0.0, 1.25, 11)
    for index, (axis, method) in enumerate(zip(axes, ("truth", "porthnn_u", "ph_ebm"))):
        axis.contourf(
            qq, pp, np.clip(grids[method], 0.0, 1.25),
            levels=levels, cmap="cividis", extend="max",
        )
        axis.contour(qq, pp, grids[method], levels=[1.0], colors="white", linewidths=0.9)
        points = critical[method]
        axis.scatter(
            [points["left_minimum"]["state"][0], points["right_minimum"]["state"][0]],
            [points["left_minimum"]["state"][1], points["right_minimum"]["state"][1]],
            color="white", edgecolor=GROUND_TRUTH, linewidth=0.5, s=15,
            marker=METHOD_MARKERS[method], zorder=5,
        )
        axis.scatter(
            [points["saddle"]["state"][0]], [points["saddle"]["state"][1]],
            color="white", marker="x", s=20, zorder=5,
        )
        axis.set(xlim=(-1.6, 1.6), ylim=(-1.08, 1.08), aspect="equal", xlabel=r"$q$")
        axis.text(
            0.5, 1.01, METHOD_LABELS[method], transform=axis.transAxes,
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )
        if index == 0:
            axis.set_ylabel(r"$p$")
        else:
            axis.set_yticklabels([])

    pointwise = legacy["pointwise"]
    finite = pointwise[np.isfinite(pointwise) & (pointwise > 0)]
    floor = float(np.min(finite))
    ceiling = max(float(np.quantile(finite, 0.96)), floor * 1.01)
    phebm_axis = axes[2]
    boundary = phebm_axis.scatter(
        legacy["boundary"][:, 0], legacy["boundary"][:, 1],
        c=np.clip(pointwise, floor, ceiling), cmap="viridis",
        norm=LogNorm(vmin=floor, vmax=ceiling), s=8, linewidths=0,
        rasterized=True, zorder=6,
    )
    for summary in legacy["summaries"]:
        mask = legacy["components"] == int(summary["component_index"])
        critical_index = int(np.argmin(np.where(mask, pointwise, np.inf)))
        phebm_axis.scatter(
            *legacy["boundary"][critical_index], marker="x", s=26,
            color="#ff2a00", linewidth=1.2, zorder=7,
        )
    phebm_axis.text(
        0.5, 0.91, r"$\mathbf{\times}$  minimum-tolerance boundary points",
        transform=phebm_axis.transAxes, color="#ff2a00", fontsize=6.2,
        fontweight="bold", ha="center", va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        zorder=8,
    )
    colorbar = fig.colorbar(boundary, cax=colorbar_axis, orientation="vertical")
    ticks = np.geomspace(floor, ceiling, 4)
    colorbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    colorbar.ax.set_yticklabels([f"{value:.2g}" for value in ticks])
    colorbar.set_label(r"pointwise admissible radius $\rho_{\mathrm{CBF}}$", labelpad=2.0)


def _draw_forced_rollouts(axis, anchor, equal_dimensions_anchor) -> None:
    for method, key in (
        ("truth", "stress_truth"),
        ("porthnn_u", "stress_porthnn_u"),
        ("ph_ebm", "stress_ph_ebm"),
    ):
        axis.plot(
            anchor["stress_time"], anchor[key][:, 0],
            color=METHOD_COLORS[method], ls=METHOD_STYLES[method],
            label=METHOD_LABELS[method],
        )
    axis.plot(
        equal_dimensions_anchor["stress_time"],
        equal_dimensions_anchor["stress_porthnn_u"][:, 0],
        color=EQUAL_DIMENSIONS_COLOR, ls=":",
    )
    axis.set(xlabel="time [s]", ylabel=r"position $q(t)$")
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
        fontsize=6.0, handlelength=2.0, columnspacing=0.8,
    )
    inset = axis.inset_axes([0.105, 0.08, 0.855, 0.34])
    inset.set_facecolor("white")
    inset.plot(
        anchor["certificate_storage_gap"], anchor["porthnn_u_uniform_radius"],
        color=PORTHNN, ls="-", label=r"$\rho_{\mathrm{HNN}}$",
    )
    inset.plot(
        anchor["certificate_storage_gap"], anchor["ph_ebm_uniform_radius"],
        color=PHEBM, ls="-", label=r"$\rho_{\mathrm{EBM}}$",
    )
    inset.plot(
        equal_dimensions_anchor["certificate_storage_gap"],
        equal_dimensions_anchor["porthnn_u_uniform_radius"],
        color=EQUAL_DIMENSIONS_COLOR, ls=":",
    )
    inset.set(
        xlim=(0.0, float(anchor["certificate_storage_gap"][-1])),
        ylim=(-2e-6, 0.06),
    )
    inset.set_yscale("symlog", linthresh=1e-5, linscale=0.8)
    inset.set_yticks(
        [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        labels=["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"],
    )
    inset.tick_params(labelsize=5.2)
    inset.legend(
        loc="center", ncol=2, fontsize=5.5,
        bbox_to_anchor=(0.5 * float(anchor["certificate_storage_gap"][-1]), 1e-3),
        bbox_transform=inset.transData, handlelength=2.1, columnspacing=1.2,
    )


def _save(fig, output_base: Path, axes: dict[str, object]) -> dict[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        suffix: str(output_base.with_suffix(f".{suffix}"))
        for suffix in ("png", "pdf", "svg")
    }
    fig.savefig(outputs["png"], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    geometry = {
        label: [float(value) for value in axis.get_window_extent(renderer).bounds]
        for label, axis in axes.items()
    }
    manifest = output_base.with_suffix(".json")
    manifest.write_text(json.dumps({
        "panel_mapping": {
            "a": "old D with old A overlay in pH-EBM subpanel",
            "b": "old B",
            "c": "old G",
        },
        "axes_pixel_bounds": geometry,
        "energy_subpanel_height_spread_pixels": (
            max(geometry[f"a_{index}"][3] for index in range(3))
            - min(geometry[f"a_{index}"][3] for index in range(3))
        ),
        "colorbar_energy_height_difference_pixels": abs(
            geometry["a_colorbar"][3] - geometry["a_2"][3]
        ),
        "bottom_panel_height_difference_pixels": abs(
            geometry["b"][3] - geometry["c"][3]
        ),
    }, indent=2) + "\n")
    outputs["manifest"] = str(manifest)
    plt.close(fig)
    return outputs


def generate(
    output_base: Path = OUTPUT,
    *,
    figure_height: float = 5.32,
    bottom_height_ratio: float = 2.646,
    bottom_width_ratios: tuple[float, float] | None = None,
    panel_c_shift: float = 0.0,
    panel_c_label_x: float = -0.14,
    panel_c_label_to_ylabel: bool = False,
    center_saddle_marker: bool = False,
    compact_panel_b_legend: bool = False,
    panel_a_heading_y: float = 1.015,
) -> dict[str, str]:
    legacy = _legacy_data()
    q_axis, p_axis, grids, critical = _energy_artifacts(OUTPUT_DIR)
    qq, pp = np.meshgrid(q_axis, p_axis)
    anchor = np.load(OUTPUT_DIR / "figures" / "main" / "duffing_anchor1_main.npz")
    equal_dimensions_anchor = np.load(EQUAL_DIMENSIONS_ANCHOR)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, figure_height), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=(1.62, bottom_height_ratio), hspace=0.11)
    top = outer[0, 0].subgridspec(1, 3, width_ratios=(1.0, 1.0, 1.08), wspace=0.08)
    energy_axes = [fig.add_subplot(top[0, index]) for index in range(3)]
    divider = make_axes_locatable(energy_axes[2])
    colorbar_axis = divider.append_axes("right", size="4%", pad=0.08)
    _draw_energy_geometry(fig, energy_axes, colorbar_axis, legacy, grids, critical, qq, pp)
    fig.text(0.018, panel_a_heading_y, "a", fontsize=9, fontweight="bold", va="center", ha="left")
    fig.text(
        0.5, panel_a_heading_y, "Recovered energy geometry and local input capacity",
        fontsize=8.2, fontweight="bold", va="center", ha="center",
    )

    bottom = outer[1, 0].subgridspec(
        1, 2, width_ratios=bottom_width_ratios or (1.0, 1.0), wspace=0.18,
    )
    axis_b = fig.add_subplot(bottom[0, 0])
    axis_c = fig.add_subplot(bottom[0, 1])
    _draw_panel_b(axis_b, legacy)
    if center_saddle_marker:
        saddle_marker = next(
            line for line in axis_b.lines if line.get_label() == "Saddle energy shell"
        )
        saddle_energy = np.asarray(saddle_marker.get_xdata())
        sweep_energy = np.asarray(legacy["sweep"]["epsilon_minus_min_h"])
        sampled = np.interp(saddle_energy, sweep_energy, legacy["sweep"]["sampled_radius"])
        regular = np.interp(saddle_energy, sweep_energy, legacy["sweep"]["regular_radius"])
        saddle_marker.set_ydata(np.sqrt(sampled * regular))
    axis_b.set_box_aspect(None)
    _draw_forced_rollouts(axis_c, anchor, equal_dimensions_anchor)
    panel_b_heading_start = len(axis_b.texts)
    panel_heading(axis_b, axis_b, "B", "Radius across energy shells", y=1.19)
    panel_heading(
        axis_c, axis_c, "C", "Forced empirical rollouts",
        y=1.16, label_x=panel_c_label_x,
    )

    fig.canvas.draw()
    panel_b_position = axis_b.get_position()
    panel_c_position = axis_c.get_position()
    panel_b_left = energy_axes[0].get_position().x0
    axis_b.set_in_layout(False)
    axis_c.set_in_layout(False)
    if bottom_width_ratios is None:
        panel_b_width = (
            axis_b.get_legend().get_window_extent(fig.canvas.get_renderer())
            .transformed(fig.transFigure.inverted()).width
        )
        panel_c_left = 0.6985048092845753
        panel_c_width = 0.290682544013558
    else:
        bottom_right = 0.9891873532981333
        bottom_gap = 0.045
        available_width = bottom_right - panel_b_left - bottom_gap
        ratio_total = sum(bottom_width_ratios)
        panel_b_width = available_width * bottom_width_ratios[0] / ratio_total
        panel_c_width = available_width * bottom_width_ratios[1] / ratio_total
        panel_c_left = panel_b_left + panel_b_width + bottom_gap
    axis_b.set_position([
        panel_b_left, panel_b_position.y0, panel_b_width, panel_b_position.height,
    ])
    axis_c.set_position([
        panel_c_left + panel_c_shift, panel_c_position.y0,
        panel_c_width, panel_c_position.height,
    ])
    if compact_panel_b_legend:
        handles, _ = axis_b.get_legend_handles_labels()
        labels = [r"Sampled $\rho^\ast$", r"Regular $\rho$", "Saddle shell"]
        axis_b.get_legend().remove()
        axis_b.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01),
            ncol=3, handlelength=1.5, columnspacing=0.55, borderaxespad=0.0,
            prop={"size": 6.2, "weight": "bold"},
        )
    for text in axis_b.texts[panel_b_heading_start:]:
        text.set_visible(False)
        text.set_in_layout(False)
    panel_b_title_y = panel_b_position.y0 + 1.16 * panel_b_position.height
    fig.text(0.018, panel_b_title_y, "b", fontsize=9, fontweight="bold",
             va="center", ha="left")
    axis_b.text(0.5, 1.16, "Radius across energy shells", transform=axis_b.transAxes,
                fontsize=8.2, fontweight="bold", va="center", ha="center", clip_on=False)
    if panel_c_label_to_ylabel:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ylabel_bounds = axis_c.yaxis.label.get_window_extent(renderer)
        ylabel_center_x = ylabel_bounds.x0 + 0.5 * ylabel_bounds.width
        label_x = axis_c.transAxes.inverted().transform((ylabel_center_x, 0.0))[0]
        panel_c_label = next(text for text in axis_c.texts if text.get_text() == "c")
        panel_c_label.set_x(label_x)
        panel_c_label.set_ha("center")
    return _save(fig, output_base, {
        "a_0": energy_axes[0], "a_1": energy_axes[1], "a_2": energy_axes[2],
        "a_colorbar": colorbar_axis, "b": axis_b, "c": axis_c,
    })


def main() -> None:
    apply_style()
    print(json.dumps(generate(), indent=2))


if __name__ == "__main__":
    main()
