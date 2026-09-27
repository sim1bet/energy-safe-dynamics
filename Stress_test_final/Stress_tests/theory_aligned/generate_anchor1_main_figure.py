# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Regenerate all Anchor 1 figures from saved numerical artifacts only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Stress_tests.theory_aligned.anchor1_figure_style import (
    CERTIFICATE,
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


OUTPUT = ROOT / "results" / "duffing_doublewell" / "certificate_experiment_PHNN"


def _json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _critical_map(path: Path):
    return {point["label"]: point for point in _json(path)["critical_points"]}


def _save(fig, base: Path, arrays: dict | None = None):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if arrays is not None:
        np.savez_compressed(base.with_suffix(".npz"), **arrays)


def _relative_energy(energy, critical):
    minimum = float(critical["right_minimum"]["energy"])
    saddle = float(critical["saddle"]["energy"])
    return (energy - minimum) / max(saddle - minimum, 1e-12)


def _energy_artifacts(output_dir):
    truth = np.load(output_dir / "ground_truth" / "energy_grid.npz")
    port = np.load(output_dir / "external_porthnn" / "energy_grid.npz")
    ebm = np.load(output_dir / "ph_ebm" / "energy_grid.npz")
    critical = {
        "truth": _critical_map(output_dir / "ground_truth" / "critical_points.json"),
        "porthnn_u": _critical_map(output_dir / "external_porthnn" / "critical_points.json"),
        "ph_ebm": _critical_map(output_dir / "ph_ebm" / "critical_points.json"),
    }
    grids = {
        "truth": _relative_energy(truth["energy"], critical["truth"]),
        "porthnn_u": _relative_energy(port["energy"], critical["porthnn_u"]),
        "ph_ebm": _relative_energy(ebm["energy"], critical["ph_ebm"]),
    }
    return truth["q"], truth["p"], grids, critical


def generate_main(output_dir: Path):
    q_axis, p_axis, grids, critical = _energy_artifacts(output_dir)
    qq, pp = np.meshgrid(q_axis, p_axis)
    stress = np.load(output_dir / "predictions" / "common_forcing_stress.npz")
    radius_sweep = np.load(output_dir / "metrics" / "cbf_radius_vs_storage_gap.npz")
    shell_metrics = _json(output_dir / "metrics" / "critical_and_boundary_metrics.json")["shells"]
    selected_alphas = (0.40, 0.80, 0.95, 0.98)
    boundary_data = {
        alpha: np.load(output_dir / "boundaries" / f"alpha_{alpha:.2f}_right.npz")
        for alpha in selected_alphas
    }

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4.55), constrained_layout=True)
    outer = fig.add_gridspec(2, 16, height_ratios=[0.72, 1.0])
    axis_a = fig.add_subplot(outer[0, :4])
    energy_grid = grids["truth"]
    axis_a.contour(qq, pp, energy_grid, levels=[0.2, 0.4, 0.6, 0.8, 1.0], colors=NEUTRAL, linewidths=0.55)
    for index in (0, 3):
        trajectory = stress["truth"][index]
        axis_a.plot(trajectory[:, 0], trajectory[:, 1], color=GROUND_TRUTH, lw=1.0, alpha=0.82)
    axis_a.scatter([-1, 1], [0, 0], color=GROUND_TRUTH, marker="o", s=18, zorder=4)
    axis_a.scatter([0], [0], color=GROUND_TRUTH, marker="x", s=28, zorder=4)
    axis_a.annotate("left well", (-1, 0), xytext=(-1.0, -0.9), ha="center",
                    arrowprops={"arrowstyle": "-", "lw": 0.6}, fontsize=6.5)
    axis_a.annotate("right well", (1, 0), xytext=(1.0, -0.9), ha="center",
                    arrowprops={"arrowstyle": "-", "lw": 0.6}, fontsize=6.5)
    axis_a.annotate("saddle", (0, 0), xytext=(-0.34, 0.42), arrowprops={"arrowstyle": "-", "lw": 0.6}, fontsize=6.5)
    axis_a.annotate("force $u$", (1.42, 0.42), xytext=(1.15, 0.82), ha="center",
                    arrowprops={"arrowstyle": "-|>", "color": CERTIFICATE, "lw": 1.1}, color=CERTIFICATE, fontsize=6.5)
    axis_a.set(xlabel=r"position $q$", ylabel=r"momentum $p$", xlim=(-1.62, 1.62), ylim=(-1.08, 1.08), aspect="equal")
    panel_heading(axis_a, axis_a, "A", "Double-well energy barrier", y=1.23, label_x=-0.28)

    subgrid = outer[0, 4:].subgridspec(1, 3, wspace=0.08)
    energy_axes = [fig.add_subplot(subgrid[0, index]) for index in range(3)]
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
    panel_heading(
        energy_axes[0], energy_axes[1], "B", "Recovered energy geometry",
        y=1.23, vertical_axis=axis_a,
    )

    axis_c = fig.add_subplot(outer[1, :4])
    for alpha_index, alpha in enumerate(selected_alphas):
        boundary = boundary_data[alpha]
        opacity = 0.35 + 0.2 * alpha_index
        for method, key in (("truth", "truth_states"), ("porthnn_u", "porthnn_u_states"), ("ph_ebm", "ph_ebm_states")):
            states = boundary[key]
            axis_c.plot(states[:, 0], states[:, 1], color=METHOD_COLORS[method], ls=METHOD_STYLES[method], lw=0.75, alpha=opacity,
                        label=METHOD_LABELS[method] if alpha_index == 0 else None)
    axis_c.scatter([0], [0], marker="x", color=GROUND_TRUTH, s=24, zorder=5)
    axis_c.annotate(r"$\alpha: 0.40\rightarrow0.98$", xy=(0.35, -0.48), xytext=(0.70, -1.05),
                    arrowprops={"arrowstyle": "->", "lw": 0.65, "color": NEUTRAL},
                    color=NEUTRAL, fontsize=6.2, ha="center")
    axis_c.set(xlabel=r"position $q$", ylabel=r"momentum $p$", xlim=(-0.05, 1.55), ylim=(-0.78, 0.78))
    axis_c.set_aspect("equal", adjustable="datalim")
    axis_c.legend(loc="upper left", ncol=1, fontsize=6.0,
                  handlelength=2.0, borderaxespad=0.4)
    panel_heading(axis_c, axis_c, "C", r"Shells approach the saddle", label_x=-0.28)

    axis_d = fig.add_subplot(outer[1, 4:8])
    for method, key in (("truth", "truth_kappa"), ("porthnn_u", "porthnn_u_kappa"), ("ph_ebm", "ph_ebm_kappa")):
        values = []
        for alpha in sorted({row["alpha"] for row in shell_metrics}):
            rows = [row for row in shell_metrics if np.isclose(row["alpha"], alpha)]
            values.append(np.mean([row[key] for row in rows]))
        axis_d.plot(sorted({row["alpha"] for row in shell_metrics}), values,
                    color=METHOD_COLORS[method], ls=METHOD_STYLES[method],
                    marker=METHOD_MARKERS[method], ms=3.1, label=METHOD_LABELS[method])
    axis_d.axvline(1.0, color=NEUTRAL, lw=0.7, ls=":")
    axis_d.text(0.99, 0.98, "saddle level", transform=axis_d.get_xaxis_transform(), ha="right", va="top", fontsize=6.2, color=NEUTRAL)
    axis_d.set(xlabel=r"relative shell $\alpha$", ylabel=r"shell slope $\kappa(\alpha)$", xlim=(0.18, 1.015), ylim=(0.0, None))
    axis_d.legend(loc="lower left", ncol=1, fontsize=6.0,
                  handlelength=2.0, borderaxespad=0.4)
    panel_heading(axis_d, axis_d, "D", "Regularity collapse")

    axis_e = fig.add_subplot(outer[1, 8:])
    case = 4
    time = stress["time"]
    for method, key in (("truth", "truth"), ("porthnn_u", "porthnn_u"), ("ph_ebm", "ph_ebm")):
        axis_e.plot(time, stress[key][case, :, 0], color=METHOD_COLORS[method], ls=METHOD_STYLES[method], label=METHOD_LABELS[method])
    axis_e.set(xlabel="time [s]", ylabel=r"position $q(t)$")
    axis_e.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                  fontsize=6.0, handlelength=2.0, columnspacing=0.8)
    inset = axis_e.inset_axes([0.105, 0.08, 0.855, 0.34])
    inset.set_facecolor("white")
    inset.plot(radius_sweep["storage_gap"], radius_sweep["porthnn_u_uniform_radius"],
               color=PORTHNN, ls="-",
               label=r"$\rho_{\mathrm{HNN}}$")
    inset.plot(radius_sweep["storage_gap"], radius_sweep["ph_ebm_uniform_radius"],
               color=PHEBM, ls="-",
               label=r"$\rho_{\mathrm{EBM}}$")
    inset.set(xlim=(0.0, float(radius_sweep["storage_gap"][-1])),
              ylim=(-2e-6, 0.06))
    inset.set_yscale("symlog", linthresh=1e-5, linscale=0.8)
    inset.set_yticks([0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                     labels=["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
    inset.tick_params(labelsize=5.2)
    inset.legend(loc="center", ncol=2, fontsize=5.5,
                 bbox_to_anchor=(0.5 * float(radius_sweep["storage_gap"][-1]), 1e-3),
                 bbox_transform=inset.transData, handlelength=2.1, columnspacing=1.2)
    panel_heading(axis_e, axis_e, "E", "Forced empirical rollouts")

    main_base = output_dir / "figures" / "main" / "duffing_anchor1_main"
    _save(fig, main_base, {
        "q_axis": q_axis, "p_axis": p_axis,
        "truth_relative_energy": grids["truth"],
        "porthnn_u_relative_energy": grids["porthnn_u"],
        "ph_ebm_relative_energy": grids["ph_ebm"],
        "stress_time": stress["time"], "stress_truth": stress["truth"][case],
        "stress_porthnn_u": stress["porthnn_u"][case], "stress_ph_ebm": stress["ph_ebm"][case],
        "certificate_storage_gap": radius_sweep["storage_gap"],
        "porthnn_u_uniform_radius": radius_sweep["porthnn_u_uniform_radius"],
        "ph_ebm_uniform_radius": radius_sweep["ph_ebm_uniform_radius"],
    })
    return main_base


def generate_supplements(output_dir: Path):
    supplement = output_dir / "figures" / "supplement"
    port_predictions = np.load(output_dir / "external_porthnn" / "predictions.npz")
    ebm_predictions = np.load(output_dir / "ph_ebm" / "predictions.npz")
    truth = port_predictions["truth"]
    initial_energy = 0.5 * truth[:, 0, 1] ** 2 + 0.25 * (truth[:, 0, 0] ** 2 - 1.0) ** 2
    selected = [int(np.argmin(initial_energy)), int(np.argmin(np.abs(initial_energy - 0.245)))]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 4.4), constrained_layout=True)
    for row, index in enumerate(selected):
        time = np.arange(truth.shape[1]) * 0.02
        for method, values in (("truth", truth), ("porthnn_u", port_predictions["predictions"]), ("ph_ebm", ebm_predictions["predictions"])):
            axes[row, 0].plot(time, values[index, :, 0], color=METHOD_COLORS[method], ls=METHOD_STYLES[method], label=METHOD_LABELS[method])
            axes[row, 1].plot(values[index, :, 0], values[index, :, 1], color=METHOD_COLORS[method], ls=METHOD_STYLES[method])
        axes[row, 0].set(ylabel=r"$q(t)$", title=("Intra-well" if row == 0 else "Near separatrix"))
        axes[row, 1].set(xlabel=r"$q$", ylabel=r"$p$", aspect="equal")
    axes[-1, 0].set_xlabel("time [s]")
    axes[0, 0].legend(ncol=3)
    _save(fig, supplement / "S1_training_and_reconstruction", {"selected_indices": np.asarray(selected), "truth": truth[selected], "porthnn_u": port_predictions["predictions"][selected], "ph_ebm": ebm_predictions["predictions"][selected]})

    q_axis, p_axis, grids, critical = _energy_artifacts(output_dir)
    force = np.load(output_dir / "external_porthnn" / "forcing_curve.npz")
    port_metrics = _json(output_dir / "external_porthnn" / "metrics.json")
    phebm_decomposition = np.load(output_dir / "ph_ebm" / "decomposition.npz")
    input_matrix = np.asarray(phebm_decomposition["input_matrix"])
    damping_matrix = np.asarray(phebm_decomposition["damping_matrix"])
    input_axis = force["input"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 4.5), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(force["input"], force["true_force"], color=GROUND_TRUTH, label="truth")
    axes[0].plot(force["input"], force["learned_force"], color=PORTHNN, ls="-.", label="PortHNN-u")
    axes[0].plot(input_axis, input_matrix[1, 0] * input_axis,
                 color=PHEBM, ls="--", label=r"pH-EBM $(Gu)_p$")
    axes[0].set(xlabel="measured input $u$", ylabel="momentum-channel force",
                title="Input-force reconstruction")
    axes[0].legend()
    input_positions = np.arange(2)
    input_width = 0.32
    truth_input = np.asarray([0.0, 1.0])
    phebm_input = input_matrix[:, 0]
    axes[1].bar(input_positions - input_width / 2, truth_input, input_width,
                color=GROUND_TRUTH, label="truth")
    axes[1].bar(input_positions + input_width / 2, phebm_input, input_width,
                color=PHEBM, label="pH-EBM")
    for position, value in zip(input_positions + input_width / 2, phebm_input):
        axes[1].annotate(f"{value:.4f}", (position, value), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=6.0)
    axes[1].set(xticks=input_positions, xticklabels=[r"$G_q$", r"$G_p$"],
                yscale="symlog", ylabel="input gain", title="Learned input direction")
    axes[1].legend()
    damping_labels = [r"$R_{qq}$", r"$R_{qp}$", r"$R_{pp}$"]
    damping_positions = np.arange(len(damping_labels))
    damping_width = 0.24
    truth_damping = np.asarray([0.0, 0.0, 0.4])
    porthnn_damping = np.asarray([0.0, 0.0, -port_metrics["learned_damping_parameter_N"]])
    phebm_damping = np.asarray([damping_matrix[0, 0], damping_matrix[0, 1], damping_matrix[1, 1]])
    for offset, (values, color, label) in enumerate((
        (truth_damping, GROUND_TRUTH, "truth"),
        (porthnn_damping, PORTHNN, "PortHNN-u"),
        (phebm_damping, PHEBM, "pH-EBM"),
    )):
        axes[2].bar(damping_positions + (offset - 1) * damping_width, values,
                    damping_width, color=color, label=label)
    axes[2].set(xticks=damping_positions, xticklabels=damping_labels,
                ylabel="dissipation coefficient", title="Damping decomposition")
    axes[2].legend()
    p_zero = int(np.argmin(np.abs(p_axis)))
    for method in ("truth", "porthnn_u", "ph_ebm"):
        axes[3].plot(q_axis, grids[method][p_zero], color=METHOD_COLORS[method], ls=METHOD_STYLES[method], label=METHOD_LABELS[method])
    axes[3].set(xlabel="position $q$ at $p=0$", ylabel="relative energy",
                ylim=(-0.08, 1.35), title="Recovered double-well profile")
    axes[3].legend()
    _save(fig, supplement / "S2_porthnn_decomposition", {
        "input": input_axis, "true_force": force["true_force"],
        "porthnn_u_force": force["learned_force"],
        "ph_ebm_input_matrix": input_matrix,
        "truth_damping_matrix": np.diag([0.0, 0.4]),
        "porthnn_u_damping_matrix": np.diag([0.0, -port_metrics["learned_damping_parameter_N"]]),
        "ph_ebm_damping_matrix": damping_matrix, "q": q_axis,
    })

    qq, pp = np.meshgrid(q_axis, p_axis)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.4), constrained_layout=True)
    for axis, method in zip(axes, ("truth", "porthnn_u", "ph_ebm")):
        axis.contour(qq, pp, grids[method], levels=[0.2, 0.4, 0.6, 0.8, 1.0], colors=NEUTRAL, linewidths=0.65)
        points = critical[method]
        for label, marker in (("left_minimum", "o"), ("right_minimum", "o"), ("saddle", "x")):
            axis.scatter(*np.asarray(points[label]["state"]), color=METHOD_COLORS[method], marker=marker, s=22)
        axis.set(title=METHOD_LABELS[method], xlabel=r"$q$", ylabel=r"$p$", xlim=(-1.6, 1.6), ylim=(-1.05, 1.05), aspect="equal")
    _save(fig, supplement / "S3_critical_point_accuracy", {"q": q_axis, "p": p_axis})

    convergence = _json(output_dir / "metrics" / "boundary_resolution_convergence.json")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.55), constrained_layout=True)
    for method, key in (("truth", "truth_kappa"), ("porthnn_u", "porthnn_u_kappa"), ("ph_ebm", "ph_ebm_kappa")):
        rows = [row for row in convergence if row["well"] == "right"]
        axes[0].semilogx([row["resolution"] for row in rows], [row[key] for row in rows], color=METHOD_COLORS[method], ls=METHOD_STYLES[method], marker=METHOD_MARKERS[method], label=METHOD_LABELS[method])
    rows = [row for row in convergence if row["well"] == "right"]
    axes[1].semilogx([row["resolution"] for row in rows], [row["truth_sampled_uniform_radius"] for row in rows], color=GROUND_TRUTH, marker="o", label="truth sampled")
    port_radius = np.asarray([row["porthnn_u_uniform_zero_force_centered_radius"] for row in rows])
    axes[1].semilogx([row["resolution"] for row in rows], port_radius, color=PORTHNN, ls="-.", marker="^", label="PortHNN-u centered")
    axes[1].semilogx([row["resolution"] for row in rows], [row["ph_ebm_sampled_uniform_radius"] for row in rows], color=PHEBM, ls="--", marker="s", label="pH-EBM sampled")
    axes[0].set(xlabel="boundary samples", ylabel=r"$\kappa_{0.8}$", title="Shell-slope convergence")
    axes[1].set(xlabel="boundary samples", ylabel="sampled centered radius", title="Certificate discretization", ylim=(-0.001, None))
    axes[0].legend(); axes[1].legend()
    for axis in axes:
        axis.set_xticks([128, 512, 2048], labels=["128", "512", "2048"])
        axis.xaxis.set_minor_locator(NullLocator())
    axes[1].text(0.98, 0.16, "centered at learned zero force", transform=axes[1].transAxes, ha="right", fontsize=6.0, color=PORTHNN)
    _save(fig, supplement / "S4_boundary_resolution", {"resolution": np.asarray([row["resolution"] for row in rows]), "truth_radius": np.asarray([row["truth_sampled_uniform_radius"] for row in rows]), "porthnn_u_centered_radius": np.asarray([row["porthnn_u_uniform_centered_radius"] for row in rows]), "ph_ebm_radius": np.asarray([row["ph_ebm_sampled_uniform_radius"] for row in rows])})

    fidelity = _json(output_dir / "audits" / "cbf_formula_implementation_audit.json")
    labels = list(fidelity)
    fig, axis = plt.subplots(figsize=(3.8, 2.55), constrained_layout=True)
    x = np.arange(len(labels)); width = 0.24
    for offset, (metric, title) in enumerate((("absolute_gap_median", "median"), ("absolute_gap_p95", "95th"), ("absolute_gap_maximum", "maximum"))):
        axis.bar(x + (offset - 1) * width, [fidelity[label][metric] for label in labels], width, label=title)
    axis.set(xticks=x, xticklabels=["validation", "boundary", "stress"], yscale="log", ylabel=r"$|\dot h_{analytic}-\dot h_{implemented}|$", title="pH-EBM formula-to-code audit")
    axis.legend(ncol=3)
    _save(fig, supplement / "S5_cbf_implementation_audit", {"values": np.asarray([[fidelity[label][key] for key in ("absolute_gap_median", "absolute_gap_p95", "absolute_gap_maximum")] for label in labels])})

    timestep = _json(output_dir / "audits" / "timestep_convergence.json")["rows"]
    fig, axis = plt.subplots(figsize=(3.8, 2.55), constrained_layout=True)
    for method in ("truth", "porthnn_u", "ph_ebm"):
        rows = sorted((row for row in timestep if row["method"] == method), key=lambda row: row["dt"])
        reference = np.asarray(rows[0]["terminal_state"])
        error = [np.linalg.norm(np.asarray(row["terminal_state"]) - reference) for row in rows]
        axis.plot([row["dt"] for row in rows], error, color=METHOD_COLORS[method], ls=METHOD_STYLES[method], marker=METHOD_MARKERS[method], label=METHOD_LABELS[method])
    axis.set(xlabel="RK4 timestep [s]", ylabel="terminal-state difference from dt/4", title="Timestep convergence")
    axis.legend()
    _save(fig, supplement / "S6_timestep_convergence", {"rows": np.asarray([[row["dt"], row["maximum_true_energy"]] for row in timestep])})

    mismatch = _json(output_dir / "metrics" / "damping_mismatch.json")["rows"]
    fig, axis = plt.subplots(figsize=(3.8, 2.55), constrained_layout=True)
    axis.plot([100 * row["relative_damping_reduction"] for row in mismatch], [row["maximum_support_loss"] for row in mismatch], color=CERTIFICATE, marker="s")
    axis.set(xlabel="true damping reduction [%]", ylabel=r"maximum support loss $\Delta_r p^2$", title="Controlled damping mismatch")
    _save(fig, supplement / "S7_damping_mismatch", {"relative_reduction": np.asarray([row["relative_damping_reduction"] for row in mismatch]), "maximum_support_loss": np.asarray([row["maximum_support_loss"] for row in mismatch])})


def _previews(main_base: Path, output_dir: Path):
    preview_dir = output_dir / "figures" / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(main_base.with_suffix(".png")).convert("RGB")
    image.save(preview_dir / "main_figure_double_column.png")
    single_width = int(round(image.width * 3.5 / 7.2))
    image.resize((single_width, int(round(image.height * single_width / image.width))), Image.Resampling.LANCZOS).save(preview_dir / "main_figure_single_column.png")
    image.convert("L").save(preview_dir / "main_figure_grayscale.png")
    image.convert("L").save(preview_dir / "main_figure_grayscale_preview.png")


def _caption(output_dir: Path):
    caption = (
        "Nonconvex energy geometry determines where robustness can be certified in the forced Duffing double well. "
        "The known plant obeys qdot=p and pdot=q-q^3-0.4p+u, with stable wells at q=+/-1 and a saddle at the origin. "
        "Ground truth is black, the input-conditioned PortHNN-u reimplementation of Desai et al. is vermilion dash-dot, and pH-EBM is blue dashed throughout. "
        "(A) Exact energy contours and representative forced trajectories establish the two-regime physical problem and momentum-channel actuation. "
        "(B) Matched physical-coordinate maps use per-model relative energy, with zero at the right minimum and one at the saddle; markers locate recovered critical points. "
        "(C) Right-well components at alpha=0.40, 0.80, 0.95, and 0.98 elongate toward the saddle. "
        "(D) All three Hamiltonians reproduce the collapse of the learned energy-shell slope kappa as alpha approaches one. "
        "(E) The same 0.4-amplitude, 0.35-Hz measured force is applied to all systems from a near-separatrix state. The inset compares the largest raw-input ball about each model's zero-force command that is valid over each sampled right-well shell, using 100 absolute storage gaps epsilon-H(z_min) on the common learned range. "
        "PortHNN-u's radius is centered at its learned zero-force command, while pH-EBM's is centered at zero; a symmetric-log scale resolves the much smaller PortHNN-u sampled tolerance. Both learned-model radii tend to zero at the collapsed shell and neither is transferred to the true plant. "
        "The exact complete-shell centered radius is zero because local capacity tends to zero near mechanical turning points; outside the guaranteed envelope means uncertified, not necessarily unsafe."
    )
    (output_dir / "figures" / "MAIN_FIGURE_CAPTION.txt").write_text(caption + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    apply_style()
    main_base = generate_main(args.output_dir)
    generate_supplements(args.output_dir)
    _previews(main_base, args.output_dir)
    _caption(args.output_dir)
    print(json.dumps({"main_figure": str(main_base), "supplementary_figures": 7}, indent=2))


if __name__ == "__main__":
    main()