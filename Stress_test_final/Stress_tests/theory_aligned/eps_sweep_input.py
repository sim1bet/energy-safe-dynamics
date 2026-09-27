# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Standalone input-radius sweep over energy-barrier zero level sets."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "EBM_model", ROOT / "Interface_code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Stress_tests.boundary import sample_energy_boundary
from Stress_tests.certificate import compute_uniform_radius_certificate
from Stress_tests.config import StressTestConfig, UncertaintySet
from Stress_tests.geometry import MinimaResult
from Stress_tests.integration import apply_runtime_config
from Stress_tests.model_adapter import EBMStressAdapter
from Stress_tests.theory_aligned.visualization import plot_epsilon_radius_sweep


@dataclass(frozen=True)
class EpsilonRadiusPoint:
    epsilon: float
    sampled_radius: float
    regular_radius: float
    n_boundary: int
    failed_ray_fraction: float
    component_summaries: list[dict]
    error: str | None = None


def compute_epsilon_radius_sweep(
    adapter: EBMStressAdapter,
    minima: MinimaResult,
    epsilon_values: Sequence[float],
    uncertainty: UncertaintySet,
    stress_config: StressTestConfig,
) -> list[EpsilonRadiusPoint]:
    """Evaluate conservative shell radii independently at each epsilon."""
    points: list[EpsilonRadiusPoint] = []
    for epsilon in np.sort(np.unique(np.asarray(epsilon_values, dtype=float))):
        try:
            boundary = sample_energy_boundary(adapter, minima, float(epsilon), stress_config)
            shell_indices = np.zeros(len(boundary.states), dtype=int)
            radius = compute_uniform_radius_certificate(
                adapter, boundary.states, shell_indices, uncertainty
            )
            summaries = radius.component_summaries
            sampled = float(summaries[0]["sampled_exact_radius"])
            regular = float(summaries[0]["regular_boundary_lower_estimate"])
            points.append(EpsilonRadiusPoint(
                epsilon=float(epsilon), sampled_radius=sampled,
                regular_radius=regular, n_boundary=int(len(boundary.states)),
                failed_ray_fraction=float(boundary.failed_fraction),
                component_summaries=summaries,
            ))
        except Exception as exc:  # Preserve failed shells as explicit gaps.
            points.append(EpsilonRadiusPoint(
                epsilon=float(epsilon), sampled_radius=float("nan"),
                regular_radius=float("nan"), n_boundary=0,
                failed_ray_fraction=1.0, component_summaries=[], error=str(exc),
            ))
    return points


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _build_adapter(run_dir: Path, config: dict, arrays) -> EBMStressAdapter:
    with (run_dir / "checkpoints" / "params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    apply_runtime_config(config)
    experiment = config.get("experiment_name")
    if experiment == "duffing_doublewell":
        from Duffing_DoubleWell_main import _build_layers
        return EBMStressAdapter(
            params, _build_layers(config), d=int(config["d"]),
            m=int(config["m_ports"]), dt=0.02,
        )
    if experiment == "deep_dissipative_nlink":
        from DeepDissipative_NLink_main import _build_layers
        reference_inputs = np.asarray(arrays["reference_inputs"])
        return EBMStressAdapter(
            params, _build_layers(config), d=int(config["d"]),
            m=int(reference_inputs.shape[-1]), dt=1.0,
        )
    raise ValueError(
        "Automatic epsilon policy currently supports duffing_doublewell and "
        "deep_dissipative_nlink; pass prepared objects to "
        "compute_epsilon_radius_sweep for another experiment."
    )


def _minima_from_arrays(arrays) -> MinimaResult:
    states = np.asarray(arrays["minima_states"])
    energies = np.asarray(arrays["minima_energies"])
    multiplicities = np.asarray(arrays["minima_multiplicities"])
    return MinimaResult(
        states=states, energies=energies, multiplicities=multiplicities,
        all_terminal_states=states, all_terminal_energies=energies,
    )


def _default_epsilon_values(
    adapter: EBMStressAdapter,
    config: dict,
    summary: dict,
    arrays,
    count: int,
) -> tuple[np.ndarray, dict]:
    nominal = float(summary["epsilon"])
    h_ref = float(np.min(np.asarray(arrays["minima_energies"])))
    nominal_gap = nominal - h_ref
    if nominal_gap <= 0.0:
        raise ValueError("nominal epsilon must exceed min H")
    shell_fraction = np.linspace(0.10, 5.0 / 3.0, count)
    epsilon = h_ref + shell_fraction * nominal_gap
    if config.get("experiment_name") == "duffing_doublewell":
        nominal_alpha = float(summary["config"].get("relative_energy_alpha") or 0.8)
        h_saddle = h_ref + (nominal - h_ref) / nominal_alpha
        return epsilon, {
            "policy": "Relative-energy sweep extending two-thirds beyond nominal epsilon",
            "shell_fraction_of_nominal_gap": shell_fraction.tolist(),
            "nominal_shell_fraction": 1.0,
            "maximum_shell_fraction": 5.0 / 3.0,
            "H_ref": h_ref,
            "H_saddle": h_saddle,
            "saddle_shell_fraction": (h_saddle - h_ref) / nominal_gap,
            "topology_note": (
                "The range crosses the inferred saddle; each level is therefore "
                "evaluated as one global sampled shell, independent of anchor IDs."
            ),
        }
    return epsilon, {
        "policy": "Relative-energy sweep extending two-thirds beyond nominal epsilon",
        "shell_fraction_of_nominal_gap": shell_fraction.tolist(),
        "nominal_shell_fraction": 1.0,
        "maximum_shell_fraction": 5.0 / 3.0,
        "H_ref": h_ref,
        "epistemic_status": (
            "Operational sampled shells; H_ref is not asserted to be a "
            "stationary minimum. Values beyond nominal epsilon are an explicit "
            "sensitivity sweep, not an expanded safety claim."
        ),
    }


def render_epsilon_input_sweep(
    run_dir: str | Path,
    *,
    count: int = 300,
    epsilon_min: float | None = None,
    epsilon_max: float | None = None,
    comparison_epsilon: float | None = None,
) -> dict:
    """Compute and render ``eps_sweep_input`` for one completed run."""
    if count < 2:
        raise ValueError("count must be at least 2")
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config" / "config.json")
    summary = _load_json(run_dir / "stress_tests" / "summary.json")
    arrays = np.load(run_dir / "stress_tests" / "arrays" / "stress_arrays.npz")
    adapter = _build_adapter(run_dir, config, arrays)
    minima = _minima_from_arrays(arrays)
    uncertainty = UncertaintySet(**summary["config"]["uncertainty"])
    original_boundary_max_radius = float(summary["config"]["boundary_max_radius"])
    sweep_boundary_max_radius = 1.5 * original_boundary_max_radius
    sweep_config = StressTestConfig(
        seed=int(summary["config"]["seed"]),
        n_boundary_directions=int(summary["config"]["n_boundary_directions"]),
        boundary_bisection_steps=int(summary["config"]["boundary_bisection_steps"]),
        boundary_initial_radius=float(summary["config"]["boundary_initial_radius"]),
        boundary_max_radius=sweep_boundary_max_radius,
        boundary_energy_tolerance=float(summary["config"]["boundary_energy_tolerance"]),
        uncertainty=uncertainty,
    )
    if (epsilon_min is None) != (epsilon_max is None):
        raise ValueError("epsilon_min and epsilon_max must be supplied together")
    if epsilon_min is None:
        epsilon, policy = _default_epsilon_values(
            adapter, config, summary, arrays, count
        )
    else:
        if not float(epsilon_min) < float(epsilon_max):
            raise ValueError("epsilon_min must be less than epsilon_max")
        epsilon = np.linspace(float(epsilon_min), float(epsilon_max), count)
        policy = {"policy": "Explicit linear epsilon range"}

    points = compute_epsilon_radius_sweep(
        adapter, minima, epsilon, uncertainty, sweep_config
    )
    epsilon_array = np.asarray([point.epsilon for point in points])
    min_h = float(np.min(np.asarray(arrays["minima_energies"])))
    epsilon_minus_min_h = epsilon_array - min_h
    nominal_epsilon_minus_min_h = float(summary["epsilon"]) - min_h
    comparison_epsilon_minus_min_h = (
        float(comparison_epsilon) - min_h if comparison_epsilon is not None else None
    )
    sampled = np.asarray([point.sampled_radius for point in points])
    regular = np.asarray([point.regular_radius for point in points])
    output_dir = run_dir / "stress_tests" / "theory_aligned"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "eps_sweep_input"
    figure = plot_epsilon_radius_sweep(
        epsilon_minus_min_h, regular, sampled, output_base,
        nominal_epsilon_minus_min_h=nominal_epsilon_minus_min_h,
        nominal_label=rf"nominal $\epsilon={float(summary['epsilon']):.1f}$",
        comparison_epsilon_minus_min_h=comparison_epsilon_minus_min_h,
        comparison_label=(
            rf"comparison-fig. $\epsilon={float(comparison_epsilon):.1f}$"
            if comparison_epsilon is not None else r"comparison-figure $\epsilon$"
        ),
    )
    npz_path = output_base.with_suffix(".npz")
    np.savez_compressed(
        npz_path, epsilon=epsilon_array,
        epsilon_minus_min_h=epsilon_minus_min_h, min_h=np.asarray(min_h),
        sampled_radius=sampled,
        regular_radius=regular,
        n_boundary=np.asarray([point.n_boundary for point in points]),
        failed_ray_fraction=np.asarray([point.failed_ray_fraction for point in points]),
    )
    payload = {
        "run_id": run_dir.name,
        "experiment_name": config.get("experiment_name"),
        "nominal_epsilon": float(summary["epsilon"]),
        "min_h": min_h,
        "nominal_epsilon_minus_min_h": nominal_epsilon_minus_min_h,
        "comparison_epsilon": (float(comparison_epsilon) if comparison_epsilon is not None else None),
        "comparison_epsilon_minus_min_h": comparison_epsilon_minus_min_h,
        "comparison_epsilon_source": (
            "results/deep_dissipative_nlink/comparison/ours/metrics.json (final pH-EBM vs. "
            "DDM comparison figure, panel d certificate threshold)" if comparison_epsilon is not None else None
        ),
        "x_axis": "epsilon_minus_min_h",
        "epsilon_minus_min_h": epsilon_minus_min_h.tolist(),
        "boundary_search": {
            "original_max_radius": original_boundary_max_radius,
            "sweep_max_radius": sweep_boundary_max_radius,
            "reason": "Expanded by 1.5x to trace shells beyond nominal epsilon.",
        },
        "epsilon_selection": policy,
        "component_reduction": (
            "all traced states treated as one global level set; conservative "
            "sampled extrema remain valid across topology changes"
        ),
        "sampled_formula": "min_Gamma delta/||a_H||_* over sampled states",
        "regular_formula": "(r_min*kappa-w_perp_max)/g_perp_max from sampled extrema",
        "epistemic_status": (
            "Both curves use finite shell samples; neither is a certified "
            "continuous-shell bound without independent extrema verification."
        ),
        "points": _jsonable([asdict(point) for point in points]),
        "figure": figure,
        "arrays": str(npz_path),
    }
    json_path = output_base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--epsilon-min", type=float)
    parser.add_argument("--epsilon-max", type=float)
    parser.add_argument("--comparison-epsilon", type=float,
                       help="Reference epsilon from an external figure (e.g. the final "
                            "comparison figure's certificate panel); drawn as a second, "
                            "distinctly-styled level set alongside the run's own nominal epsilon.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = render_epsilon_input_sweep(
        args.run_dir, count=args.count,
        epsilon_min=args.epsilon_min, epsilon_max=args.epsilon_max,
        comparison_epsilon=args.comparison_epsilon,
    )
    print(json.dumps(result, indent=2, allow_nan=False))