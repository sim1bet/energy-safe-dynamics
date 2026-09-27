#!/usr/bin/env python3
# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Render theory-aligned energy and radius figures from stress-suite artifacts."""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT, ROOT / "Stress_test_final"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Stress_test_final.case import StressTestCase
from Stress_test_final.figure_style import COMPARISON_SHELL, DOUBLE_COLUMN_IN, ENERGY_CAMERA_READY, GRID, NOMINAL_SHELL, REGULAR_RADIUS, SAMPLED_RADIUS, publication_rc


def load_case(module_name: str) -> StressTestCase:
    case = importlib.import_module(module_name).build_case(smoke=False)
    if not isinstance(case, StressTestCase): raise TypeError("build_case must return StressTestCase")
    return case


def save(fig, output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": str(output.with_suffix(".png")), "pdf": str(output.with_suffix(".pdf"))}
    fig.savefig(paths["png"], bbox_inches="tight"); fig.savefig(paths["pdf"], bbox_inches="tight"); plt.close(fig)
    return paths


def render_energy(case: StressTestCase, arrays, output: Path) -> dict[str, str]:
    required = {"landscape_xx", "landscape_yy", "landscape_energy", "plane_center", "plane_basis"}
    if not required.issubset(arrays.files): raise ValueError(f"Missing energy-view arrays: {sorted(required - set(arrays.files))}")
    from Stress_tests.model_adapter import EBMStressAdapter
    xx, yy, profile = np.asarray(arrays["landscape_xx"]), np.asarray(arrays["landscape_yy"]), np.asarray(arrays["landscape_energy"])
    z = np.column_stack((xx.ravel(), yy.ravel()))
    states = np.asarray(arrays["plane_center"]) + z @ np.asarray(arrays["plane_basis"]).T
    affine = np.asarray(EBMStressAdapter(case.params, case.layers, case.d, case.m, case.dt).energy_batch(states)).reshape(xx.shape)
    low, high = float(np.nanmin([affine, profile])), float(np.nanmax([affine, profile]))
    with plt.rc_context(publication_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 3.3), constrained_layout=True)
        for axis, energy, title in zip(axes, (affine, profile), ("Affine slice", "Profiled envelope")):
            image = axis.contourf(xx, yy, energy, levels=40, cmap=ENERGY_CAMERA_READY, vmin=low, vmax=high)
            axis.contour(xx, yy, energy, levels=[case.epsilon], colors="#242424", linewidths=0.7)
            axis.set(title=title, xlabel=r"projected coordinate $z_1$", ylabel=r"projected coordinate $z_2$")
        fig.colorbar(image, ax=axes, shrink=0.82, label="energy")
        figures = {"2d": save(fig, output / "hamiltonian_2d")}
        fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 3.8), constrained_layout=True)
        for index, (energy, title) in enumerate(((affine, "Affine slice"), (profile, "Profiled envelope")), start=1):
            axis = fig.add_subplot(1, 2, index, projection="3d")
            surface = axis.plot_surface(xx, yy, energy, cmap=ENERGY_CAMERA_READY, vmin=low, vmax=high, linewidth=0, antialiased=True)
            axis.set(title=title, xlabel=r"$z_1$", ylabel=r"$z_2$", zlabel=r"$H$")
        fig.colorbar(surface, ax=fig.axes, shrink=0.7, label="energy")
        figures["3d"] = save(fig, output / "hamiltonian_3d")
        return figures


def render_boundary_composite(case: StressTestCase, arrays, output: Path) -> dict[str, str]:
    """Render projected sampled boundary evidence and stack it over the sweep."""
    from Stress_tests.boundary import project_to_energy_level
    from Stress_tests.certificate import compute_uniform_radius_certificate
    from Stress_tests.config import UncertaintySet
    from Stress_tests.model_adapter import EBMStressAdapter
    from Stress_tests.theory_aligned.combined_figure import compose_stacked_figures
    from Stress_tests.theory_aligned.visualization import ProjectedEnergyView, RadiusCurveView, plot_projected_radius_certificate
    required = {"boundary_states", "plane_center", "plane_basis", "landscape_xx", "landscape_yy", "landscape_energy"}
    if not required.issubset(arrays.files): raise ValueError(f"Missing boundary-view arrays: {sorted(required - set(arrays.files))}")
    adapter = EBMStressAdapter(case.params, case.layers, case.d, case.m, case.dt)
    boundary, _ = project_to_energy_level(adapter, np.asarray(arrays["boundary_states"]), case.epsilon)
    component = np.zeros(len(boundary), dtype=int)
    uncertainty = case.config.uncertainty
    radius = compute_uniform_radius_certificate(adapter, boundary, component, UncertaintySet(**asdict(uncertainty)))
    projected = (boundary - np.asarray(arrays["plane_center"])) @ np.asarray(arrays["plane_basis"])
    xx, yy, energy = np.asarray(arrays["landscape_xx"]), np.asarray(arrays["landscape_yy"]), np.asarray(arrays["landscape_energy"])
    normalized = (energy - np.nanmin(energy)) / (case.epsilon - np.nanmin(energy))
    parameter = np.arctan2(projected[:, 1], projected[:, 0])
    with plt.rc_context(publication_rc()):
        top = plot_projected_radius_certificate(
            ProjectedEnergyView(xx, yy, normalized, 1.0, projected, radius.pointwise_radius, "Projected input stress on boundary shell", r"projected coordinate $z_1$", r"projected coordinate $z_2$"),
            RadiusCurveView(parameter, radius.pointwise_radius, component, radius.component_summaries, "projected boundary ray angle [rad]", ["sampled shell"], connect_points=False, title="Theoretical and empirical radii versus ray"),
            output / "projected_input_stress", figure_size=(DOUBLE_COLUMN_IN, 4.45),
        )
    combined = compose_stacked_figures(output / "projected_input_stress", output / "epsilon_theoretical_vs_empirical_radius", output / "boundary_radius_composite")
    return {"projected": top, "combined": combined}


def render_sweep(case: StressTestCase, arrays, summary: dict, output: Path, count: int) -> dict:
    from Stress_tests.config import StressTestConfig, UncertaintySet
    from Stress_tests.geometry import MinimaResult
    from Stress_tests.model_adapter import EBMStressAdapter
    from Stress_tests.theory_aligned.eps_sweep_input import compute_epsilon_radius_sweep
    minima = MinimaResult(np.asarray(arrays["minima_states"]), np.asarray(arrays["minima_energies"]), np.asarray(arrays["minima_multiplicities"]), np.asarray(arrays["minima_states"]), np.asarray(arrays["minima_energies"]))
    minimum, gap = float(np.min(minima.energies)), case.epsilon - float(np.min(minima.energies))
    saved = summary["config"]
    config = StressTestConfig(seed=int(saved["seed"]), n_boundary_directions=int(saved["n_boundary_directions"]), boundary_bisection_steps=int(saved["boundary_bisection_steps"]), boundary_initial_radius=float(saved["boundary_initial_radius"]), boundary_max_radius=1.5 * float(saved["boundary_max_radius"]), boundary_energy_tolerance=float(saved["boundary_energy_tolerance"]), uncertainty=UncertaintySet(**saved["uncertainty"]))
    epsilon = minimum + np.linspace(0.10, 5.0 / 3.0, count) * gap
    points = compute_epsilon_radius_sweep(EBMStressAdapter(case.params, case.layers, case.d, case.m, case.dt), minima, epsilon, config.uncertainty, config)
    relative = np.asarray([point.epsilon - minimum for point in points]); sampled = np.asarray([point.sampled_radius for point in points]); regular = np.asarray([point.regular_radius for point in points])
    with plt.rc_context(publication_rc()):
        fig, axis = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 3.7), constrained_layout=True)
        axis.plot(relative, sampled, color=SAMPLED_RADIUS, linewidth=1.45, label=r"Sampled $\rho^\ast$")
        axis.plot(relative, regular, color=REGULAR_RADIUS, linewidth=1.3, linestyle=(0, (4, 2)), label=r"Regular-shell estimate $\rho$")
        axis.axvline(gap, color=NOMINAL_SHELL, linewidth=0.85, linestyle=(0, (1.5, 2)), label="Stress shell")
        if case.comparison_epsilon is not None: axis.axvline(case.comparison_epsilon - minimum, color=COMPARISON_SHELL, linewidth=1.0, linestyle=(0, (1, 1.5)), label="Comparison shell")
        finite = np.r_[sampled, regular]; finite = finite[np.isfinite(finite)]
        if len(finite) and np.all(finite > 0): axis.set_yscale("log")
        axis.set(xlim=(0, 1.025 * relative.max()), xlabel=r"Relative shell energy, $\epsilon-H_{\min}$", ylabel="Admissible input radius")
        axis.grid(axis="y", color=GRID, linewidth=0.45); axis.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3)
        figures = save(fig, output / "epsilon_theoretical_vs_empirical_radius")
    np.savez_compressed(output / "epsilon_theoretical_vs_empirical_radius.npz", epsilon=np.asarray([point.epsilon for point in points]), epsilon_minus_min_h=relative, min_h=minimum, empirical_sampled_radius=sampled, theoretical_regular_radius=regular, n_boundary=np.asarray([point.n_boundary for point in points]), failed_ray_fraction=np.asarray([point.failed_ray_fraction for point in points]))
    return {**figures, "count": count, "failed_shells": sum(point.error is not None for point in points), "points": [asdict(point) for point in points]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter"); parser.add_argument("--stress-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path); parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args(); case = load_case(args.adapter); output = args.output_dir or args.stress_dir / "theory_aligned"
    summary = json.loads((args.stress_dir / "summary.json").read_text())
    with np.load(args.stress_dir / "arrays" / "stress_arrays.npz") as arrays:
        energy = render_energy(case, arrays, output)
        sweep = render_sweep(case, arrays, summary, output, args.count)
        boundary = render_boundary_composite(case, arrays, output)
    (output / "visualization_summary.json").write_text(json.dumps({"schema": "dataset_agnostic_theory_aligned_figures_v1", "energy": energy, "sweep": sweep, "boundary": boundary, "epistemic_status": "Finite sampled stress-test evidence; not a continuous-shell proof."}, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"decision": "RENDERED_THEORY_ALIGNED_FIGURES", "output_dir": str(output)}, sort_keys=True))


if __name__ == "__main__": main()
