# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Render the merged Duffing certificate and architecture comparison natively."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, NullLocator
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
    NEUTRAL,
    PHEBM,
    PORTHNN,
    apply_style,
    panel_heading,
)
from Stress_tests.theory_aligned.duffing_certificate_figure import _energy_grid
from Stress_tests.theory_aligned.generate_anchor1_main_figure import _energy_artifacts
from Stress_tests.theory_aligned.visualization import saddle_shell_minimum_indices

LEGACY_RUN = (
    ROOT / "results" / "duffing_doublewell" / "runs"
    / "duffing_v5_champion_exact_seed4"
)
OUTPUT_DIR = ROOT / "results" / "duffing_doublewell" / "certificate_experiment_PHNN"
OUTPUT = OUTPUT_DIR / "figures" / "main" / "duffing_merged_main"
EQUAL_DIMENSIONS_ANCHOR = (
    ROOT / "results" / "duffing_doublewell" / "certificate_experiment_PHNN_equal_dimensions"
    / "figures" / "main" / "duffing_anchor1_main.npz"
)
EQUAL_DIMENSIONS_COLOR = "#E7A46A"


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _legacy_data() -> dict:
    theory_dir = LEGACY_RUN / "stress_tests" / "theory_aligned"
    radius = np.load(theory_dir / "radius_certificate.npz")
    stress = np.load(LEGACY_RUN / "stress_tests" / "arrays" / "stress_arrays.npz")
    config = _json(LEGACY_RUN / "config" / "config.json")
    audit = _json(LEGACY_RUN / "audits" / "audit.json")
    summaries = _json(theory_dir / "radius_certificate.json")["component_summaries"]
    boundary = np.asarray(radius["boundary_states"])
    wells = np.asarray(radius["component_wells"])
    components = np.asarray(radius["component_indices"], dtype=int)
    pointwise = np.asarray(radius["pointwise_radius"], dtype=float)
    all_states = np.concatenate([boundary, wells], axis=0)
    q_pad = max(0.18 * np.ptp(all_states[:, 0]), 0.2)
    p_pad = max(0.22 * np.ptp(all_states[:, 1]), 0.15)
    q_axis = np.linspace(all_states[:, 0].min() - q_pad, all_states[:, 0].max() + q_pad, 181)
    p_axis = np.linspace(all_states[:, 1].min() - p_pad, all_states[:, 1].max() + p_pad, 151)
    qq, pp = np.meshgrid(q_axis, p_axis)
    energy = _energy_grid(LEGACY_RUN / "checkpoints" / "params.pkl", config, qq, pp)
    epsilon = float(audit["metrics"]["stress"]["epsilon"])
    alpha = float(audit["metrics"]["stress"]["config"].get("relative_energy_alpha") or 0.8)
    h_min = float(np.min(stress["minima_energies"]))
    display_energy = alpha * (energy - h_min) / max(epsilon - h_min, 1e-8)
    angles = np.arctan2(
        boundary[:, 1] - wells[components, 1],
        boundary[:, 0] - wells[components, 0],
    )
    sweep = np.load(theory_dir / "eps_sweep_input.npz")
    sweep_meta = _json(theory_dir / "eps_sweep_input.json")
    return {
        "boundary": boundary, "wells": wells, "components": components,
        "pointwise": pointwise, "angles": angles, "summaries": summaries,
        "qq": qq, "pp": pp, "display_energy": display_energy,
        "shell_level": alpha, "sweep": sweep, "sweep_meta": sweep_meta,
    }


def _draw_panel_a(axis, colorbar_axis, data: dict, fig) -> None:
    pointwise = data["pointwise"]
    finite = pointwise[np.isfinite(pointwise) & (pointwise > 0)]
    floor = float(np.min(finite))
    ceiling = max(float(np.quantile(finite, 0.96)), floor * 1.01)
    display = data["display_energy"]
    levels = np.linspace(float(np.nanmin(display)), max(float(np.nanquantile(display, 0.94)), 0.92), 24)
    axis.contourf(data["qq"], data["pp"], display, levels=levels, cmap="Blues_r", extend="max")
    axis.contour(data["qq"], data["pp"], display, levels=[data["shell_level"]], colors=GROUND_TRUTH, linewidths=0.8)
    points = axis.scatter(
        data["boundary"][:, 0], data["boundary"][:, 1],
        c=np.clip(pointwise, floor, ceiling), cmap="viridis",
        norm=LogNorm(vmin=floor, vmax=ceiling), s=7, linewidths=0,
        rasterized=True, zorder=4,
    )
    axis.scatter(data["wells"][:, 0], data["wells"][:, 1], marker="o", s=15,
                 color="white", edgecolor=GROUND_TRUTH, linewidth=0.5, zorder=5)
    for summary in data["summaries"]:
        mask = data["components"] == int(summary["component_index"])
        local = np.where(mask, pointwise, np.inf)
        critical = int(np.argmin(local))
        axis.scatter(*data["boundary"][critical], marker="x", s=23, color="#ff2a00",
                     linewidth=1.2, zorder=6)
    axis.text(0.5, 0.91, r"$\mathbf{\times}$  minimum-tolerance boundary points",
              transform=axis.transAxes, color="#ff2a00", fontsize=6.2,
              fontweight="bold", ha="center", va="center",
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2})
    axis.set(xlabel=r"position $q$", ylabel=r"momentum $p$")
    axis.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(points, cax=colorbar_axis, orientation="horizontal")
    ticks = np.geomspace(floor, ceiling, 4)
    colorbar.ax.xaxis.set_major_locator(FixedLocator(ticks))
    colorbar.ax.xaxis.set_minor_locator(NullLocator())
    colorbar.ax.set_xticklabels([f"{value:.2g}" for value in ticks])
    colorbar.set_label(r"pointwise admissible radius $\rho_{\mathrm{CBF}}$", labelpad=1.5)


def _draw_panel_b(axis, data: dict) -> None:
    sweep = data["sweep"]
    energy = np.asarray(sweep["epsilon_minus_min_h"])
    sampled = np.asarray(sweep["sampled_radius"])
    regular = np.asarray(sweep["regular_radius"])
    axis.plot(energy, sampled, color=PHEBM, linestyle=METHOD_STYLES["ph_ebm"],
              linewidth=1.25, label=r"Sampled $\rho^\ast$")
    axis.plot(energy, regular, color="#8B72B1", linestyle=METHOD_STYLES["porthnn_u"],
              linewidth=1.25, label=r"Regular-shell estimate $\rho$")
    saddle_indices = saddle_shell_minimum_indices(energy, regular)
    if len(saddle_indices):
        axis.plot(energy[saddle_indices], regular[saddle_indices], linestyle="none",
                  marker="D", markersize=3.5, markerfacecolor="#8BD646",
                  markeredgecolor="#34452B", markeredgewidth=0.45,
                  label="Saddle energy shell", zorder=5)
    nominal = float(data["sweep_meta"]["nominal_epsilon_minus_min_h"])
    axis.axvspan(nominal, float(np.max(energy)), color="#f2f2f2", linewidth=0, zorder=0)
    axis.axvline(nominal, color=NEUTRAL, linewidth=0.7, linestyle=":")
    axis.text(nominal, 0.97, r"nominal $\epsilon$", transform=axis.get_xaxis_transform(),
              ha="right", va="top", fontsize=6.2, color=NEUTRAL)
    axis.text(0.80, 0.97, "beyond nominal", transform=axis.transAxes,
              ha="center", va="top", fontsize=6.2, color=NEUTRAL)
    axis.set_yscale("log")
    axis.set(xlim=(0.0, 1.025 * float(np.max(energy))),
             xlabel=r"relative shell energy, $\epsilon-H_{\min}$",
             ylabel="admissible input radius")
    axis.set_box_aspect(0.62)
    axis.set_anchor("N")
    axis.grid(axis="y", which="major", color="#d9d9d9", linewidth=0.45)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
                handlelength=2.0, columnspacing=0.9, borderaxespad=0.0,
                prop={"size": 6.7, "weight": "bold"})


def _draw_panel_c(axis, data: dict) -> None:
    colors = (PHEBM, "#5F9E72")
    for index, summary in enumerate(data["summaries"]):
        component = int(summary["component_index"])
        mask = data["components"] == component
        order = np.argsort(data["angles"][mask])
        angle = data["angles"][mask][order]
        radius = data["pointwise"][mask][order]
        axis.plot(angle, radius, color=colors[index], linewidth=1.25)
        axis.axhline(float(summary["sampled_exact_radius"]), color=colors[index],
                     linewidth=1.25, linestyle=":")
        axis.axhline(float(summary["regular_boundary_lower_estimate"]), color=colors[index],
                     linewidth=1.25, linestyle="--")
    axis.set_yscale("log")
    axis.set_ylim(0.0065, 3.2)
    axis.set(xlabel="boundary angle [rad]", ylabel=r"radius $\rho$")
    axis.set_box_aspect(0.6284)
    axis.grid(axis="y", which="both", color="0.9", linewidth=0.45)
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6}
    axis.text(-2.85, 0.42, "well 2", color="#5F9E72", fontsize=5.8,
              fontweight="bold", bbox=label_box)
    axis.text(-2.85, 0.12, "well 1", color=PHEBM, fontsize=5.8,
              fontweight="bold", bbox=label_box)
    axis.text(3.05, 0.034, r"sampled $\rho^\ast$", color="0.25", fontsize=5.2,
              ha="right", va="bottom", bbox=label_box)
    axis.text(3.05, 0.0092, "regular-shell", color="0.25", fontsize=5.2,
              ha="right", va="bottom", bbox=label_box)


def _save(fig, output_base: Path, axes: dict[str, object]) -> dict[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = {suffix: str(output_base.with_suffix(f".{suffix}")) for suffix in ("png", "pdf", "svg")}
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
        "panel_mapping": {"a": "legacy A", "b": "legacy C", "c": "legacy B", "d-g": "Anchor 1 B-E"},
        "axes_pixel_bounds": geometry,
        "c_d_plane_height_difference_pixels": abs(geometry["c"][3] - geometry["d_plane"][3]),
    }, indent=2) + "\n")
    outputs["manifest"] = str(manifest)
    plt.close(fig)
    return outputs


def generate(output_base: Path = OUTPUT) -> dict[str, str]:
    legacy = _legacy_data()
    q_axis, p_axis, grids, critical = _energy_artifacts(OUTPUT_DIR)
    qq, pp = np.meshgrid(q_axis, p_axis)
    anchor = np.load(OUTPUT_DIR / "figures" / "main" / "duffing_anchor1_main.npz")
    equal_dimensions_anchor = np.load(EQUAL_DIMENSIONS_ANCHOR)
    shell_metrics = _json(OUTPUT_DIR / "metrics" / "critical_and_boundary_metrics.json")["shells"]
    selected_alphas = (0.40, 0.80, 0.95, 0.98)
    boundaries = {
        alpha: np.load(OUTPUT_DIR / "boundaries" / f"alpha_{alpha:.2f}_right.npz")
        for alpha in selected_alphas
    }

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 7.35), constrained_layout=True)
    outer = fig.add_gridspec(
        3, 16, height_ratios=(2.80, 1.904, 2.646), hspace=0.01,
    )
    top_grid = outer[0, :].subgridspec(1, 2, width_ratios=(1.0, 1.0))
    a_grid = top_grid[0, 0].subgridspec(2, 1, height_ratios=(1.0, 0.055), hspace=0.08)
    axis_a = fig.add_subplot(a_grid[0, 0])
    colorbar_a = fig.add_subplot(a_grid[1, 0])
    axis_b = fig.add_subplot(top_grid[0, 1])
    _draw_panel_a(axis_a, colorbar_a, legacy, fig)
    _draw_panel_b(axis_b, legacy)
    panel_heading(axis_a, axis_a, "A", "Energy shell and local input capacity", y=1.25)
    panel_heading(axis_b, axis_b, "B", "Radius across energy shells", y=1.25,
                  vertical_axis=axis_a)

    axis_c = fig.add_subplot(outer[1, :4])
    _draw_panel_c(axis_c, legacy)
    panel_heading(axis_c, axis_c, "C", "Largest uniform radius", y=1.23, label_x=-0.28)

    d_grid = outer[1, 4:].subgridspec(1, 3, wspace=0.08)
    energy_axes = [fig.add_subplot(d_grid[0, index]) for index in range(3)]
    levels = np.linspace(0.0, 1.25, 11)
    for index, (axis, method) in enumerate(zip(energy_axes, ("truth", "porthnn_u", "ph_ebm"))):
        axis.contourf(qq, pp, np.clip(grids[method], 0.0, 1.25), levels=levels, cmap="cividis", extend="max")
        axis.contour(qq, pp, grids[method], levels=[1.0], colors="white", linewidths=0.9)
        points = critical[method]
        axis.scatter([points["left_minimum"]["state"][0], points["right_minimum"]["state"][0]],
                     [points["left_minimum"]["state"][1], points["right_minimum"]["state"][1]],
                     color="white", edgecolor=GROUND_TRUTH, linewidth=0.5, s=13, marker=METHOD_MARKERS[method])
        axis.scatter([points["saddle"]["state"][0]], [points["saddle"]["state"][1]], color="white", marker="x", s=18)
        axis.set(xlim=(-1.6, 1.6), ylim=(-1.08, 1.08), aspect="equal", xlabel=r"$q$")
        axis.text(0.5, 1.01, METHOD_LABELS[method], transform=axis.transAxes,
                  ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        if index == 0:
            axis.set_ylabel(r"$p$")
        else:
            axis.set_yticklabels([])
    panel_heading(energy_axes[0], energy_axes[1], "D", "Recovered energy geometry",
                  y=1.23, vertical_axis=axis_c)

    axis_e = fig.add_subplot(outer[2, :4])
    for alpha_index, alpha in enumerate(selected_alphas):
        boundary = boundaries[alpha]
        opacity = 0.35 + 0.2 * alpha_index
        for method, key in (("truth", "truth_states"), ("porthnn_u", "porthnn_u_states"), ("ph_ebm", "ph_ebm_states")):
            states = boundary[key]
            axis_e.plot(states[:, 0], states[:, 1], color=METHOD_COLORS[method],
                        ls=METHOD_STYLES[method], lw=0.75, alpha=opacity,
                        label=METHOD_LABELS[method] if alpha_index == 0 else None)
    axis_e.scatter([0], [0], marker="x", color=GROUND_TRUTH, s=24, zorder=5)
    axis_e.annotate(r"$\alpha: 0.40\rightarrow0.98$", xy=(0.35, -0.48), xytext=(0.70, -1.05),
                    arrowprops={"arrowstyle": "->", "lw": 0.65, "color": NEUTRAL},
                    color=NEUTRAL, fontsize=6.2, ha="center")
    axis_e.set(xlabel=r"position $q$", ylabel=r"momentum $p$", xlim=(-0.05, 1.55), ylim=(-0.78, 0.78))
    axis_e.set_aspect("equal", adjustable="datalim")
    axis_e.legend(loc="upper left", fontsize=6.0, handlelength=2.0, borderaxespad=0.4)
    panel_heading(axis_e, axis_e, "E", "Shells approach the saddle", label_x=-0.28)

    axis_f = fig.add_subplot(outer[2, 4:8])
    alphas = sorted({row["alpha"] for row in shell_metrics})
    for method, key in (("truth", "truth_kappa"), ("porthnn_u", "porthnn_u_kappa"), ("ph_ebm", "ph_ebm_kappa")):
        values = [np.mean([row[key] for row in shell_metrics if np.isclose(row["alpha"], alpha)]) for alpha in alphas]
        axis_f.plot(alphas, values, color=METHOD_COLORS[method], ls=METHOD_STYLES[method],
                    marker=METHOD_MARKERS[method], ms=3.1, label=METHOD_LABELS[method])
    axis_f.axvline(1.0, color=NEUTRAL, lw=0.7, ls=":")
    axis_f.text(0.99, 0.98, "saddle level", transform=axis_f.get_xaxis_transform(),
                ha="right", va="top", fontsize=6.2, color=NEUTRAL)
    axis_f.set(xlabel=r"relative shell $\alpha$", ylabel=r"shell slope $\kappa(\alpha)$",
               xlim=(0.18, 1.015), ylim=(0.0, None))
    axis_f.legend(loc="lower left", fontsize=6.0, handlelength=2.0, borderaxespad=0.4)
    panel_heading(axis_f, axis_f, "F", "Regularity collapse")

    axis_g = fig.add_subplot(outer[2, 8:])
    for method, key in (("truth", "stress_truth"), ("porthnn_u", "stress_porthnn_u"), ("ph_ebm", "stress_ph_ebm")):
        axis_g.plot(anchor["stress_time"], anchor[key][:, 0], color=METHOD_COLORS[method],
                    ls=METHOD_STYLES[method], label=METHOD_LABELS[method])
    axis_g.plot(equal_dimensions_anchor["stress_time"],
                equal_dimensions_anchor["stress_porthnn_u"][:, 0],
                color=EQUAL_DIMENSIONS_COLOR, ls=":")
    axis_g.set(xlabel="time [s]", ylabel=r"position $q(t)$")
    axis_g.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                  fontsize=6.0, handlelength=2.0, columnspacing=0.8)
    inset = axis_g.inset_axes([0.105, 0.08, 0.855, 0.34])
    inset.set_facecolor("white")
    inset.plot(anchor["certificate_storage_gap"], anchor["porthnn_u_uniform_radius"],
               color=PORTHNN, ls="-", label=r"$\rho_{\mathrm{HNN}}$")
    inset.plot(anchor["certificate_storage_gap"], anchor["ph_ebm_uniform_radius"],
               color=PHEBM, ls="-", label=r"$\rho_{\mathrm{EBM}}$")
    inset.plot(equal_dimensions_anchor["certificate_storage_gap"],
               equal_dimensions_anchor["porthnn_u_uniform_radius"],
               color=EQUAL_DIMENSIONS_COLOR, ls=":")
    inset.set(xlim=(0.0, float(anchor["certificate_storage_gap"][-1])),
              ylim=(-2e-6, 0.06))
    inset.set_yscale("symlog", linthresh=1e-5, linscale=0.8)
    inset.set_yticks([0.0, 1e-5, 1e-4, 1e-3, 1e-2], labels=["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
    inset.tick_params(labelsize=5.2)
    inset.legend(loc="center", ncol=2, fontsize=5.5,
                 bbox_to_anchor=(0.5 * float(anchor["certificate_storage_gap"][-1]), 1e-3),
                 bbox_transform=inset.transData, handlelength=2.1, columnspacing=1.2)
    panel_heading(axis_g, axis_g, "G", "Forced empirical rollouts")

    fig.canvas.draw()
    return _save(fig, output_base, {
        "c": axis_c, "d_plane": energy_axes[0],
        "e": axis_e, "f": axis_f, "g": axis_g,
    })


def main() -> None:
    apply_style()
    print(json.dumps(generate(), indent=2))


if __name__ == "__main__":
    main()
