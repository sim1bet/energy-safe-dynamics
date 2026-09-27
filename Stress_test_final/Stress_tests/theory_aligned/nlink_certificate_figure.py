# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Theory-honest projected certificate figures for deep-dissipative n-link runs.

The n-link model has no known true Hamiltonian in its learned latent state.
Accordingly, this module compares a literal affine slice of the learned energy
with its stored profiled lower envelope and treats the sampled boundary as one
shell. It does not infer equilibrium components from nonstationary minima.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "EBM_model", ROOT / "Interface_code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DeepDissipative_NLink_main import _build_layers
from Stress_tests.boundary import project_to_energy_level
from Stress_tests.certificate import compute_uniform_radius_certificate
from Stress_tests.config import ProjectionConfig, UncertaintySet
from Stress_tests.geometry import (
    MinimaResult,
    Plane,
    profile_energy_grid,
    slice_energy_grid,
)
from Stress_tests.integration import apply_runtime_config
from Stress_tests.model_adapter import EBMStressAdapter
from Stress_tests.theory_aligned.visualization import (
    EnergySurfaceView,
    ProjectedEnergyView,
    RadiusCurveView,
    map_points_to_level_contour,
    plot_energy_surface_comparison,
    plot_projected_radius_certificate,
    select_contour_coverage_samples,
    sublevel_view_limits,
    MIN_CERTIFICATE_BOUNDARY_RAYS,
)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _build_adapter(run_dir: Path, config: dict, arrays) -> EBMStressAdapter:
    with (run_dir / "checkpoints" / "params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    reference_inputs = np.asarray(arrays["reference_inputs"])
    if reference_inputs.ndim < 2 or reference_inputs.shape[-1] < 1:
        raise ValueError("Cannot infer n-link input dimension from reference_inputs")
    apply_runtime_config(config)
    return EBMStressAdapter(
        params,
        _build_layers(config),
        d=int(config["d"]),
        m=int(reference_inputs.shape[-1]),
        dt=1.0,
    )


def _normalize_to_shell(energy: np.ndarray, epsilon: float) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    minimum = float(np.nanmin(energy))
    scale = epsilon - minimum
    if scale <= 0:
        raise ValueError("epsilon must exceed the displayed energy minimum")
    return (energy - minimum) / scale


def _comparison_radius_point(
    adapter: EBMStressAdapter, summary: dict, arrays, comparison_epsilon: float
) -> dict:
    """Sample a single boundary shell at an externally-supplied epsilon (e.g.
    the certificate threshold reported in the final comparison figure's panel
    d) using the SAME ray-sampling machinery as ``eps_sweep_input.py``, so it
    is directly comparable to the run's own nominal-epsilon radius curve."""
    from Stress_tests.boundary import sample_energy_boundary
    from Stress_tests.certificate import compute_uniform_radius_certificate
    from Stress_tests.config import StressTestConfig

    minima = MinimaResult(
        states=np.asarray(arrays["minima_states"]), energies=np.asarray(arrays["minima_energies"]),
        multiplicities=np.asarray(arrays["minima_multiplicities"]),
        all_terminal_states=np.asarray(arrays["minima_states"]), all_terminal_energies=np.asarray(arrays["minima_energies"]),
    )
    uncertainty = UncertaintySet(**summary["config"]["uncertainty"])
    stress_config = StressTestConfig(
        seed=int(summary["config"]["seed"]),
        n_boundary_directions=int(summary["config"]["n_boundary_directions"]),
        boundary_bisection_steps=int(summary["config"]["boundary_bisection_steps"]),
        boundary_initial_radius=float(summary["config"]["boundary_initial_radius"]),
        boundary_max_radius=float(summary["config"]["boundary_max_radius"]),
        boundary_energy_tolerance=float(summary["config"]["boundary_energy_tolerance"]),
        uncertainty=uncertainty,
    )
    boundary = sample_energy_boundary(adapter, minima, float(comparison_epsilon), stress_config)
    shell_indices = np.zeros(len(boundary.states), dtype=int)
    radius = compute_uniform_radius_certificate(adapter, boundary.states, shell_indices, uncertainty)
    component = radius.component_summaries[0]
    return {
        "sampled": float(component["sampled_exact_radius"]),
        "regular": float(component["regular_boundary_lower_estimate"]),
        "label": rf"comparison fig. $\epsilon={comparison_epsilon:.1f}$",
        "epsilon": float(comparison_epsilon),
        "n_boundary": int(len(boundary.states)),
        "failed_ray_fraction": float(boundary.failed_fraction),
        "source": (
            "results/deep_dissipative_nlink/comparison/ours/metrics.json "
            "(final pH-EBM vs. DDM comparison figure, panel d certificate threshold)"
        ),
    }


def render_nlink_theory_figure(run_dir: str | Path, *, comparison_epsilon: float | None = None) -> dict:
    """Compute sampled radius evidence and render n-link theory figures."""
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config" / "config.json")
    if config.get("experiment_name") != "deep_dissipative_nlink":
        raise ValueError("Expected a deep_dissipative_nlink run")
    summary = _load_json(run_dir / "stress_tests" / "summary.json")
    arrays = np.load(run_dir / "stress_tests" / "arrays" / "stress_arrays.npz")
    adapter = _build_adapter(run_dir, config, arrays)

    epsilon = float(summary["epsilon"])
    boundary_raw = np.asarray(arrays["boundary_states"])
    boundary, boundary_residual = project_to_energy_level(
        adapter, boundary_raw, epsilon
    )
    component_indices = np.zeros(len(boundary), dtype=int)
    uncertainty = UncertaintySet(**summary["config"]["uncertainty"])
    radius = compute_uniform_radius_certificate(
        adapter, boundary, component_indices, uncertainty
    )

    plane_center = np.asarray(arrays["plane_center"])
    plane_basis = np.asarray(arrays["plane_basis"])
    boundary_xy = (boundary - plane_center) @ plane_basis
    parameter = np.arctan2(boundary_xy[:, 1], boundary_xy[:, 0])

    if {"landscape_xx", "landscape_yy", "landscape_energy"}.issubset(arrays.files):
        xx = np.asarray(arrays["landscape_xx"])
        yy = np.asarray(arrays["landscape_yy"])
        profile_energy = np.asarray(arrays["landscape_energy"])
        coordinates = np.stack([xx.ravel(), yy.ravel()], axis=1)
        slice_states = plane_center + coordinates @ plane_basis.T
        slice_energy = np.asarray(adapter.energy_batch(slice_states)).reshape(xx.shape)
    else:
        projection_config = ProjectionConfig(**summary["config"]["projection"])
        plane_summary = summary["plane"]
        plane = Plane(
            center=plane_center,
            basis=plane_basis,
            complement=np.asarray(arrays["plane_complement"]),
            method=str(plane_summary["method"]),
            x_limits=tuple(plane_summary["x_limits"]),
            y_limits=tuple(plane_summary["y_limits"]),
        )
        minima = MinimaResult(
            states=np.asarray(arrays["minima_states"]),
            energies=np.asarray(arrays["minima_energies"]),
            multiplicities=np.asarray(arrays["minima_multiplicities"]),
            all_terminal_states=np.asarray(arrays["minima_states"]),
            all_terminal_energies=np.asarray(arrays["minima_energies"]),
        )
        slice_data = slice_energy_grid(adapter, plane, projection_config)
        profile_data = profile_energy_grid(adapter, plane, minima, projection_config)
        xx = np.asarray(profile_data["xx"])
        yy = np.asarray(profile_data["yy"])
        profile_energy = np.asarray(profile_data["energy"])
        slice_energy = np.asarray(slice_data["energy"])
    normalized_profile = _normalize_to_shell(profile_energy, epsilon)
    boundary_display_xy, display_source_indices, display_mapping_distance, display_angular_error = (
        select_contour_coverage_samples(
            xx, yy, normalized_profile, 1.0, boundary_xy, n_display=128
        )
    )
    x_limits, y_limits = sublevel_view_limits(
        xx, yy, normalized_profile, max_level=2.0, padding_fraction=0.02
    )

    finite_radius = np.asarray(radius.pointwise_radius)
    critical = int(np.argmin(np.where(np.isfinite(finite_radius), finite_radius, np.inf)))
    critical_display_xy = map_points_to_level_contour(
        xx, yy, normalized_profile, 1.0, boundary_xy[[critical]]
    )[0][0]
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    critical_label_xy = (0.5 * (x_min + x_max), y_min + 0.90 * (y_max - y_min))

    output_dir = run_dir / "stress_tests" / "theory_aligned"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_radius = (
        _comparison_radius_point(adapter, summary, arrays, comparison_epsilon)
        if comparison_epsilon is not None else None
    )
    n_boundary = int(len(boundary))
    if n_boundary >= MIN_CERTIFICATE_BOUNDARY_RAYS:
        certificate_boundary_status = (
            f"OK: {n_boundary} successful boundary rays (>= {MIN_CERTIFICATE_BOUNDARY_RAYS} threshold)"
        )
        certificate_figure = plot_projected_radius_certificate(
            ProjectedEnergyView(
                xx=xx,
                yy=yy,
                display_energy=normalized_profile,
                shell_level=1.0,
                boundary_xy=boundary_display_xy,
                pointwise_radius=radius.pointwise_radius[display_source_indices],
                title="Profiled safe-set boundary and mapped shell samples",
                x_label=r"projected latent coordinate $z_1$",
                y_label=r"projected latent coordinate $z_2$",
                critical_label=r"$\mathbf{\times}$  minimum sampled input tolerance",
                critical_label_xy=critical_label_xy,
                critical_xy=tuple(critical_display_xy),
                x_limits=x_limits,
                y_limits=y_limits,
            ),
            RadiusCurveView(
                parameter=parameter,
                pointwise_radius=radius.pointwise_radius,
                component_indices=component_indices,
                component_summaries=radius.component_summaries,
                parameter_label="projected polar angle [rad]",
                component_labels=["sampled shell"],
                connect_points=False,
                title="Sampled uniform radius on the traced shell",
            ),
            output_dir / "nlink_projected_radius_figure",
            comparison_radius=comparison_radius,
        )
    else:
        certificate_boundary_status = (
            f"ELIMINATED: only {n_boundary} successful boundary rays (< {MIN_CERTIFICATE_BOUNDARY_RAYS} "
            "threshold) -- the traced shell is too sparse to be representative geometry evidence, so "
            "the certificate-boundary panel is not rendered. Existing nlink_projected_radius_figure.* "
            "files (if any, from a prior run) are removed to avoid publishing stale evidence."
        )
        for suffix in (".png", ".pdf"):
            stale = output_dir / f"nlink_projected_radius_figure{suffix}"
            stale.unlink(missing_ok=True)
        certificate_figure = None
    energy_figure = plot_energy_surface_comparison(
        [
            EnergySurfaceView(
                xx, yy, slice_energy, "cividis",
                "Affine slice",
                r"$H(c+Uz)$", r"latent coordinate $z_1$", r"latent coordinate $z_2$",
            ),
            EnergySurfaceView(
                xx, yy, profile_energy, "viridis",
                "Profiled envelope",
                r"$\widetilde H(z)$", r"latent coordinate $z_1$", r"latent coordinate $z_2$",
            ),
        ],
        run_dir / "stress_tests" / "figures" / "nlink_slice_vs_profile_hamiltonian",
    )

    radius_npz = output_dir / "nlink_sampled_radius.npz"
    np.savez_compressed(
        radius_npz,
        boundary_states=boundary,
        boundary_states_before_projection=boundary_raw,
        boundary_projected=boundary_xy,
        boundary_display_on_profile_contour=boundary_display_xy,
        boundary_display_source_indices=display_source_indices,
        boundary_display_mapping_distance=display_mapping_distance,
        boundary_display_angular_error=display_angular_error,
        boundary_parameter=parameter,
        boundary_energy_residual=boundary_residual,
        component_indices=component_indices,
        pointwise_radius=radius.pointwise_radius,
        grad_norm=radius.grad_norm,
        residual_margin=radius.residual_margin,
        input_dual_norm=radius.input_dual_norm,
        normal_dissipation=radius.normal_dissipation,
        normal_input_gain=radius.normal_input_gain,
        normal_disturbance_support=radius.normal_disturbance_support,
        slice_energy=slice_energy,
        profile_energy=profile_energy,
        # panel-a display geometry, saved verbatim so other figures can
        # re-embed this exact panel without recomputing planes/minima/grids.
        xx=xx,
        yy=yy,
        normalized_profile=normalized_profile,
        critical_display_xy=np.asarray(critical_display_xy),
        critical_label_xy=np.asarray(critical_label_xy),
    )
    profile_diagnostics = summary.get("profile_optimization") or {}
    payload = {
        "run_id": run_dir.name,
        "experiment_name": "deep_dissipative_nlink",
        "theory_formula": "rho_star = inf_{Gamma, a_H != 0} delta/||a_H||_*",
        "component_policy": (
            "One sampled shell; no equilibrium components are claimed because "
            "the stored minimum is not a verified stationary point."
        ),
        "epistemic_status": (
            "Sampled projected-shell evidence, not a continuous-shell proof. "
            "The profile is an optimization-derived lower envelope and its "
            "stationarity residual must be reported with the figure."
        ),
        "component_summaries": radius.component_summaries,
        "certificate_boundary_status": certificate_boundary_status,
        "boundary_projection": {
            "n_points": int(len(boundary)),
            "max_abs_residual_after": float(np.max(np.abs(boundary_residual))),
            "median_abs_residual_after": float(np.median(np.abs(boundary_residual))),
        },
        "display_mapping": {
            "policy": (
                "Panel a uses evenly spaced arc-length targets on the longest "
                "profiled H_tilde=epsilon contour. Each target is paired with a "
                "unique full-state shell sample of nearest projected polar direction; "
                "its color is the radius evaluated at that source state. Panel b "
                "continues to show all source samples."
            ),
            "n_display_samples": int(len(display_source_indices)),
            "max_projected_distance": float(np.max(display_mapping_distance)),
            "median_projected_distance": float(np.median(display_mapping_distance)),
            "max_angular_mismatch_rad": float(np.max(display_angular_error)),
            "median_angular_mismatch_rad": float(np.median(display_angular_error)),
            "panel_x_limits": list(x_limits),
            "panel_y_limits": list(y_limits),
            "panel_support_max_normalized_energy": 2.0,
        },
        "source_diagnostics": {
            "failed_ray_fraction": summary.get("boundary", {}).get("failed_ray_fraction"),
            "fraction_beyond_runtime_state_clip": summary.get("boundary", {}).get(
                "fraction_beyond_runtime_state_clip"
            ),
            "stored_well_max_grad_norm": summary.get("wells", {}).get("max_grad_norm"),
            "profile_median_orthogonal_grad_norm": profile_diagnostics.get(
                "median_orthogonal_grad_norm"
            ),
            "profile_p99_orthogonal_grad_norm": profile_diagnostics.get(
                "p99_orthogonal_grad_norm"
            ),
        },
        "critical_boundary_index": critical,
        "comparison_radius": comparison_radius,
        "certificate_figure": certificate_figure,
        "energy_figure": energy_figure,
        "arrays": str(radius_npz),
    }
    json_path = output_dir / "nlink_sampled_radius.json"
    json_path.write_text(json.dumps(payload, default=_jsonable, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--comparison-epsilon", type=float,
                       help="Reference epsilon from an external figure (e.g. the final "
                            "comparison figure's certificate panel); overlaid on panel b "
                            "as a second, distinctly-styled reference certificate.")
    arguments = parser.parse_args()
    print(json.dumps(
        render_nlink_theory_figure(arguments.run_dir, comparison_epsilon=arguments.comparison_epsilon),
        indent=2,
    ))
