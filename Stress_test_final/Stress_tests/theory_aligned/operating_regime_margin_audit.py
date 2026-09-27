# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Compare held-out test inputs with shell-wise input-radius estimates."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "EBM_model", ROOT / "Interface_code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Stress_tests.integration import apply_runtime_config
from Stress_tests.theory_aligned.eps_sweep_input import _build_adapter
from Stress_tests.theory_aligned.visualization import saddle_shell_minimum_indices


def _interpolate_without_gaps(
    query: np.ndarray, coordinates: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Linearly interpolate only between adjacent finite sweep samples."""
    query = np.asarray(query, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    values = np.asarray(values, dtype=float)
    result = np.full(query.shape, np.nan, dtype=float)
    if len(coordinates) < 2:
        return result
    exact_index = np.searchsorted(coordinates, query, side="left")
    exact_in_range = exact_index < len(coordinates)
    exact = np.zeros(query.shape, dtype=bool)
    exact[exact_in_range] = (
        coordinates[exact_index[exact_in_range]] == query[exact_in_range]
    )
    exact &= np.isfinite(values[np.clip(exact_index, 0, len(values) - 1)])
    result[exact] = values[exact_index[exact]]
    upper = np.searchsorted(coordinates, query, side="right")
    lower = upper - 1
    valid = (
        ~exact & (lower >= 0) & (upper < len(coordinates))
        & np.isfinite(query)
        & np.isfinite(values[np.clip(lower, 0, len(values) - 1)])
        & np.isfinite(values[np.clip(upper, 0, len(values) - 1)])
    )
    lower_valid = lower[valid]
    upper_valid = upper[valid]
    span = coordinates[upper_valid] - coordinates[lower_valid]
    weight = np.divide(
        query[valid] - coordinates[lower_valid], span,
        out=np.ones_like(span), where=span > 0.0,
    )
    result[valid] = (
        values[lower_valid] + weight * (values[upper_valid] - values[lower_valid])
    )
    return result


def classify_margin_samples(
    relative_energy: np.ndarray,
    input_norm: np.ndarray,
    sweep_energy: np.ndarray,
    regular_radius: np.ndarray,
    sampled_radius: np.ndarray,
    nominal_energy: float,
) -> dict:
    """Classify samples globally and in saddle-separated energy regimes."""
    relative_energy = np.asarray(relative_energy, dtype=float).reshape(-1)
    input_norm = np.asarray(input_norm, dtype=float).reshape(-1)
    if relative_energy.shape != input_norm.shape:
        raise ValueError("relative_energy and input_norm must have equal length")
    order = np.argsort(sweep_energy)
    sweep_energy = np.asarray(sweep_energy, dtype=float)[order]
    regular_radius = np.asarray(regular_radius, dtype=float)[order]
    sampled_radius = np.asarray(sampled_radius, dtype=float)[order]
    regular_at_state = _interpolate_without_gaps(
        relative_energy, sweep_energy, regular_radius
    )
    sampled_at_state = _interpolate_without_gaps(
        relative_energy, sweep_energy, sampled_radius
    )
    assessable = (
        np.isfinite(relative_energy) & np.isfinite(input_norm)
        & np.isfinite(regular_at_state) & np.isfinite(sampled_at_state)
    )
    below_regular = assessable & (input_norm <= regular_at_state)
    between = assessable & ~below_regular & (input_norm <= sampled_at_state)
    above_sampled = assessable & (input_norm > sampled_at_state)

    saddle_energy = sweep_energy[
        saddle_shell_minimum_indices(sweep_energy, regular_radius)
    ]
    boundaries = np.unique(np.concatenate([
        sweep_energy[[0, -1]],
        saddle_energy,
        np.asarray([nominal_energy], dtype=float),
    ]))
    boundaries = boundaries[
        (boundaries >= sweep_energy[0]) & (boundaries <= sweep_energy[-1])
    ]

    def summarize(mask: np.ndarray) -> dict:
        count = int(np.sum(mask))
        assessed_count = int(np.sum(mask & assessable))
        denominator = max(assessed_count, 1)
        return {
            "sample_count": count,
            "assessed_count": assessed_count,
            "below_regular_count": int(np.sum(mask & below_regular)),
            "between_regular_and_sampled_count": int(np.sum(mask & between)),
            "above_sampled_count": int(np.sum(mask & above_sampled)),
            "below_regular_fraction_of_assessed": float(np.sum(mask & below_regular) / denominator),
            "below_sampled_fraction_of_assessed": float(np.sum(mask & (below_regular | between)) / denominator),
            "above_sampled_fraction_of_assessed": float(np.sum(mask & above_sampled) / denominator),
        }

    regimes = []
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = assessable & (relative_energy >= lower)
        mask &= relative_energy <= upper if index == len(boundaries) - 2 else relative_energy < upper
        regime = summarize(mask)
        regime.update({
            "lower_relative_energy": float(lower),
            "upper_relative_energy": float(upper),
            "nominal_status": "below nominal" if upper <= nominal_energy else (
                "beyond nominal" if lower >= nominal_energy else "crosses nominal"
            ),
        })
        regimes.append(regime)

    overall = summarize(np.ones(len(relative_energy), dtype=bool))
    overall.update({
        "outside_sweep_count": int(np.sum(~assessable)),
        "below_sweep_count": int(np.sum(np.isfinite(relative_energy) & (relative_energy < sweep_energy[0]))),
        "above_sweep_count": int(np.sum(np.isfinite(relative_energy) & (relative_energy > sweep_energy[-1]))),
    })
    return {
        "overall": overall,
        "regimes": regimes,
        "saddle_relative_energies": saddle_energy.tolist(),
        "regular_radius_at_state": regular_at_state,
        "sampled_radius_at_state": sampled_at_state,
        "assessment_mask": assessable,
    }


def _load_duffing_samples(run_dir: Path, stress_arrays) -> tuple[np.ndarray, np.ndarray, dict]:
    signal_paths = sorted((run_dir / "signals").glob("test_*_traj*.npz"))
    if not signal_paths:
        raise FileNotFoundError("A persisted Duffing test signal is required to recover trajectory length")
    with np.load(signal_paths[0]) as signal:
        steps = int(len(signal["input"]))
    states_flat = np.asarray(stress_arrays["reference_states"])
    inputs_flat = np.asarray(stress_arrays["reference_inputs"])
    if len(states_flat) % steps:
        raise ValueError("Duffing reference states do not form complete test trajectories")
    trajectories = len(states_flat) // steps
    states = states_flat.reshape(trajectories, steps, -1)
    inputs = inputs_flat.reshape(trajectories, steps, -1)
    return (
        states[:, :-1].reshape(-1, states.shape[-1]),
        inputs[:, 1:].reshape(-1, inputs.shape[-1]),
        {
            "alignment": "u[t] paired with learned state preceding its integration step; first input omitted because x0 was not persisted",
            "trajectory_count": trajectories,
            "steps_per_trajectory_used": steps - 1,
        },
    )


def _load_nlink_samples(
    run_dir: Path, config: dict, dataset_stem: Path
) -> tuple[np.ndarray, np.ndarray, dict]:
    import jax.numpy as jnp
    from EBM_param_fields import encode_x0_batch, make_grad_energy
    from EBM_rollout import make_batch_rollout_fn
    from DeepDissipative_NLink_main import _build_layers

    observations = np.load(f"{dataset_stem}.test.obs.npy").astype(np.float32)
    inputs = np.load(f"{dataset_stem}.test.input.npy").astype(np.float32)
    init_window = int(config["init_win"])
    apply_runtime_config(config)
    with (run_dir / "checkpoints" / "params.pkl").open("rb") as stream:
        params = pickle.load(stream)
    encoder_window = np.concatenate([
        observations[:, :init_window], inputs[:, :init_window]
    ], axis=-1)
    initial_state = encode_x0_batch(params, jnp.asarray(encoder_window))
    layers = _build_layers(config)
    rollout = make_batch_rollout_fn(
        make_grad_energy(layers), int(config["d"]), inputs.shape[-1], 0.05,
        integrator=str(config["rollout_integrator"]),
    )
    latent, _, _ = rollout(
        params, initial_state, jnp.asarray(inputs[:, init_window:]), stop_grad=True
    )
    pre_step_state = np.concatenate([
        np.asarray(initial_state)[:, None, :], np.asarray(latent)[:, :-1, :]
    ], axis=1)
    aligned_inputs = inputs[:, init_window:]
    return (
        pre_step_state.reshape(-1, pre_step_state.shape[-1]),
        aligned_inputs.reshape(-1, aligned_inputs.shape[-1]),
        {
            "alignment": "u[t] paired with exact learned latent state preceding its RK4 integration step",
            "trajectory_count": int(len(inputs)),
            "steps_per_trajectory_used": int(aligned_inputs.shape[1]),
            "dataset_stem": str(dataset_stem),
        },
    )


def run_operating_regime_margin_audit(
    run_dir: str | Path, *, nlink_dataset_stem: str | Path | None = None
) -> dict:
    """Run and persist the test-input versus shell-margin audit."""
    run_dir = Path(run_dir)
    theory_dir = run_dir / "stress_tests" / "theory_aligned"
    with (run_dir / "config" / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    with np.load(run_dir / "stress_tests" / "arrays" / "stress_arrays.npz") as stress_arrays:
        adapter = _build_adapter(run_dir, config, stress_arrays)
        min_h = float(np.min(stress_arrays["minima_energies"]))
        if config["experiment_name"] == "duffing_doublewell":
            states, inputs, provenance = _load_duffing_samples(run_dir, stress_arrays)
        elif config["experiment_name"] == "deep_dissipative_nlink":
            if nlink_dataset_stem is None:
                raise ValueError("nlink_dataset_stem is required for output-only n-link test reconstruction")
            states, inputs, provenance = _load_nlink_samples(
                run_dir, config, Path(nlink_dataset_stem)
            )
        else:
            raise ValueError(f"Unsupported experiment {config['experiment_name']!r}")

    relative_energy = np.asarray(adapter.energy_batch(states), dtype=float) - min_h
    input_norm = np.max(np.abs(inputs), axis=1)
    sweep_base = theory_dir / "eps_sweep_input"
    with np.load(sweep_base.with_suffix(".npz")) as sweep:
        with sweep_base.with_suffix(".json").open(encoding="utf-8") as stream:
            sweep_metadata = json.load(stream)
        classification = classify_margin_samples(
            relative_energy,
            input_norm,
            sweep["epsilon_minus_min_h"],
            sweep["regular_radius"],
            sweep["sampled_radius"],
            float(sweep_metadata["nominal_epsilon_minus_min_h"]),
        )

    report = {
        "experiment_name": config["experiment_name"],
        "run_id": run_dir.name,
        "epistemic_status": (
            "Empirical comparison against interpolated sampled shell-wide margins; "
            "an input above a margin is not evidence that the observed transition was unsafe."
        ),
        "input_metric": "linf distance from zero input center (absolute value for scalar input)",
        "state_regime": "learned pre-step relative energy H(x)-min(H)",
        "interpolation": "linear only between adjacent finite sweep shells; no extrapolation across range or failures",
        "provenance": provenance,
        "overall": classification["overall"],
        "saddle_relative_energies": classification["saddle_relative_energies"],
        "regimes": classification["regimes"],
        "energy_quantiles": np.quantile(relative_energy[np.isfinite(relative_energy)], [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "input_norm_quantiles": np.quantile(input_norm[np.isfinite(input_norm)], [0, 0.1, 0.5, 0.9, 1]).tolist(),
    }
    json_path = theory_dir / "operating_regime_margin_audit.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    csv_path = theory_dir / "operating_regime_margin_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["regimes"][0].keys()))
        writer.writeheader()
        writer.writerows(report["regimes"])
    markdown_path = theory_dir / "operating_regime_margin_audit.md"
    overall = report["overall"]
    assessed_fraction = overall["assessed_count"] / max(overall["sample_count"], 1)
    lines = [
        "# Operating-regime input-margin audit",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Assessable samples: {overall['assessed_count']:,}/{overall['sample_count']:,} "
        f"({100.0 * assessed_fraction:.1f}%).",
        "",
        "| Relative-energy regime | Status | Samples | <= regular rho | "
        "regular rho < input <= sampled rho* | > sampled rho* |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for regime in report["regimes"]:
        count = regime["assessed_count"]
        if not count:
            continue
        lines.append(
            f"| [{regime['lower_relative_energy']:.6g}, "
            f"{regime['upper_relative_energy']:.6g}) | {regime['nominal_status']} | "
            f"{count:,} | {100.0 * regime['below_regular_fraction_of_assessed']:.2f}% | "
            f"{100.0 * (regime['below_sampled_fraction_of_assessed'] - regime['below_regular_fraction_of_assessed']):.2f}% | "
            f"{100.0 * regime['above_sampled_fraction_of_assessed']:.2f}% |"
        )
    lines.extend([
        "",
        "`rho` and `rho*` are interpolated only between adjacent finite sweep "
        "shells; samples outside that support are not extrapolated. Exceeding a "
        "shell-wide margin means the input lies outside that guaranteed ball; "
        "it does not show that the observed transition was unsafe.",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        **report,
        "artifacts": {
            "json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--nlink-dataset-stem", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run_operating_regime_margin_audit(
        arguments.run_dir, nlink_dataset_stem=arguments.nlink_dataset_stem
    ), indent=2))