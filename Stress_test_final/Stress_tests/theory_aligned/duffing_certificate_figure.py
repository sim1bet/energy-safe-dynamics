# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Publication figure for Duffing Hamiltonian and input-radius certificates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Stress_tests.theory_aligned.visualization import (
    EnergySurfaceView,
    ProjectedEnergyView,
    RadiusCurveView,
    plot_energy_surface_comparison,
    plot_projected_radius_certificate,
)

CERT_DIR = Path(__file__).resolve().parent / "certificates"
if str(CERT_DIR) not in sys.path:
    sys.path.insert(0, str(CERT_DIR))

from checkpoint_io import load_checkpoint_numpy_only
from unforced_dissipativity_probe import energy_EBM_numpy


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _load_run_config(run_dir: Path) -> dict:
    """Load the archived run configuration, including legacy Duffing runs.

    Some completed stage-v5 runs predate the per-run ``config/`` archive but
    retain their immutable queue record.  Falling back to that record permits
    a figure-only rerender without changing the checkpoint or numerical data.
    """
    archived = run_dir / "config" / "config.json"
    if archived.exists():
        return _load_json(archived)
    queue_record = (
        run_dir.parents[1] / "configs_queue" / "stagev5" / "processed"
        / f"{run_dir.name}.json"
    )
    if queue_record.exists():
        return _load_json(queue_record)
    raise FileNotFoundError(
        f"No archived config for {run_dir}; checked {archived} and {queue_record}"
    )


def _energy_grid(checkpoint: Path, config: dict, q: np.ndarray, p: np.ndarray) -> np.ndarray:
    params = load_checkpoint_numpy_only(str(checkpoint))
    weights = [np.asarray(value, dtype=np.float64) for value in params.ebm_weights]
    biases = [np.asarray(value, dtype=np.float64) for value in params.ebm_biases]
    layer_args = [(float(config["p_1"]),), (float(config["p_2"]),)]
    points = np.stack([q.ravel(), p.ravel()], axis=1)
    values = [energy_EBM_numpy(point, weights, biases, layer_args) for point in points]
    return np.asarray(values).reshape(q.shape)


def render_duffing_hamiltonian_comparison(
    run_dir: str | Path,
    q_axis: np.ndarray | None = None,
    p_axis: np.ndarray | None = None,
) -> dict:
    """Render contour and 3D views of analytic and learned Duffing energies."""
    run_dir = Path(run_dir)
    config = _load_run_config(run_dir)
    arrays = np.load(run_dir / "stress_tests" / "arrays" / "stress_arrays.npz")
    boundary = np.asarray(arrays["boundary_states"])

    if q_axis is None or p_axis is None:
        q_pad = max(0.18 * np.ptp(boundary[:, 0]), 0.2)
        p_pad = max(0.22 * np.ptp(boundary[:, 1]), 0.15)
        q_axis = np.linspace(boundary[:, 0].min() - q_pad, boundary[:, 0].max() + q_pad, 201)
        p_axis = np.linspace(boundary[:, 1].min() - p_pad, boundary[:, 1].max() + p_pad, 181)

    qq, pp = np.meshgrid(q_axis, p_axis)
    true_energy = 0.5 * pp**2 + 0.25 * (qq**2 - 1.0) ** 2
    learned_energy = _energy_grid(run_dir / "checkpoints" / "params.pkl", config, qq, pp)

    return plot_energy_surface_comparison(
        [
            EnergySurfaceView(
                qq, pp, true_energy, "magma", "True Duffing Hamiltonian",
                r"$H_{\mathrm{true}}(q,p)$", r"position $q$", r"momentum $p$",
                surface_x_label=r"position $q$", surface_y_label=r"momentum $p$",
            ),
            EnergySurfaceView(
                qq, pp, learned_energy, "viridis", "Reconstructed Hamiltonian",
                r"$H_{\mathrm{EBM}}(q,p)$", r"position $q$", r"momentum $p$",
                surface_x_label=r"position $q$", surface_y_label=r"momentum $p$",
            ),
        ],
        run_dir / "stress_tests" / "figures" / "hamiltonian_true_vs_reconstructed",
    )


def render_duffing_certificate_figure(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    config = _load_run_config(run_dir)
    audit = _load_json(run_dir / "audits" / "audit.json")
    theory_dir = run_dir / "stress_tests" / "theory_aligned"
    arrays = np.load(run_dir / "stress_tests" / "arrays" / "stress_arrays.npz")
    radius_path = theory_dir / "radius_certificate.npz"
    if not radius_path.exists():
        from Stress_tests.theory_aligned.duffing_radius_certificate import compute_duffing_radius_artifact
        compute_duffing_radius_artifact(run_dir)
    radius_arrays = np.load(radius_path)
    radius_summary = _load_json(theory_dir / "radius_certificate.json")
    summaries = radius_summary["component_summaries"]

    boundary = np.asarray(radius_arrays["boundary_states"])
    wells = np.asarray(
        radius_arrays["component_wells"]
        if "component_wells" in radius_arrays.files
        else arrays["minima_states"][:len(summaries)]
    )
    epsilon = float(audit["metrics"]["stress"]["epsilon"])
    alpha_shell = float(
        audit["metrics"]["stress"]["config"].get("relative_energy_alpha") or 0.8
    )
    h_min = float(np.min(arrays["minima_energies"]))
    energy_scale = max(epsilon - h_min, 1e-8)
    pointwise_radius = np.asarray(radius_arrays["pointwise_radius"], dtype=float)
    components = np.asarray(radius_arrays["component_indices"], dtype=int)
    all_states = np.concatenate([boundary, wells], axis=0)
    q_pad = max(0.18 * np.ptp(all_states[:, 0]), 0.2)
    p_pad = max(0.22 * np.ptp(all_states[:, 1]), 0.15)
    q_axis = np.linspace(np.min(all_states[:, 0]) - q_pad, np.max(all_states[:, 0]) + q_pad, 181)
    p_axis = np.linspace(np.min(all_states[:, 1]) - p_pad, np.max(all_states[:, 1]) + p_pad, 151)
    qq, pp = np.meshgrid(q_axis, p_axis)
    energy = _energy_grid(run_dir / "checkpoints" / "params.pkl", config, qq, pp)
    relative_energy = alpha_shell * (energy - h_min) / energy_scale
    nearest = components
    angles = np.arctan2(
        boundary[:, 1] - wells[nearest, 1],
        boundary[:, 0] - wells[nearest, 0],
    )
    saddle_label_q = float(np.mean(wells[:, 0]))
    saddle_label_p = float(p_axis[0] + 0.90 * (p_axis[-1] - p_axis[0]))
    figure = plot_projected_radius_certificate(
        ProjectedEnergyView(
            xx=qq, yy=pp, display_energy=relative_energy,
            shell_level=alpha_shell, boundary_xy=boundary,
            pointwise_radius=pointwise_radius,
            title="Energy shell and local input capacity",
            x_label=r"position $q$", y_label=r"momentum $p$",
            minima_xy=wells,
            critical_label=r"$\mathbf{\times}$  minimum-tolerance boundary points",
            critical_label_xy=(saddle_label_q, saddle_label_p),
        ),
        RadiusCurveView(
            parameter=angles, pointwise_radius=pointwise_radius,
            component_indices=components, component_summaries=summaries,
            parameter_label="boundary angle [rad]",
            component_labels=[f"well {index + 1}" for index in range(len(summaries))],
        ),
        theory_dir / "duffing_certificate_figure",
    )
    comparison = render_duffing_hamiltonian_comparison(run_dir, q_axis=q_axis, p_axis=p_axis)
    return {
        **figure,
        "hamiltonian_comparison": comparison,
        "component_summaries": summaries,
    }


if __name__ == "__main__":
    print(json.dumps(render_duffing_certificate_figure(sys.argv[1]), indent=2))
