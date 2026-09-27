# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Generate saved numerical artifacts for the Duffing Anchor 1 comparison."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root
from scipy.spatial.distance import directed_hausdorff

ROOT = Path(__file__).resolve().parents[2]
for module_path in (ROOT, ROOT / "EBM_model", ROOT / "Interface_code"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import EBM_param_fields as pf
from Stress_tests.experiments.duffing_ground_truth_certificate import (
    SADDLE_ENERGY,
    TRUE_MINIMA,
    TRUE_SADDLE,
    boundary_states as true_boundary,
    hamiltonian as true_hamiltonian,
    high_accuracy_kappa,
    sampled_uniform_interval,
)
from Stress_tests.theory_aligned.duffing_certificate_experiment import (
    DEFAULT_RUN,
    _learned_boundary as phebm_boundary,
    _learned_terms as phebm_terms,
    learned_critical_points as phebm_critical_points,
    load_adapter,
)
from Stress_tests.theory_aligned.porthnn_u_jax import (
    forcing as porthnn_forcing,
    hamiltonian as porthnn_hamiltonian,
    load_parameters,
    rollout as porthnn_rollout,
    vector_field as porthnn_vector_field,
)


OUTPUT = ROOT / "results" / "duffing_doublewell" / "certificate_experiment_PHNN"
DATASET = ROOT / "results" / "duffing_doublewell" / "datasets" / "duffing_v1.npz"
PH_PARAMS = OUTPUT / "external_porthnn" / "parameters.npz"
ALPHAS = np.asarray([0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.98])
BOUNDARY_RESOLUTION = 2048


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    return float(max(directed_hausdorff(first, second)[0], directed_hausdorff(second, first)[0]))


def _critical_points(energy, gradient, hessian) -> list[dict]:
    output = []
    seeds = (("left_minimum", [-1.0, 0.0]), ("saddle", [0.0, 0.0]), ("right_minimum", [1.0, 0.0]))
    for label, seed in seeds:
        solved = root(lambda state: np.asarray(gradient(jnp.asarray(state, dtype=jnp.float32))), seed)
        state = np.asarray(solved.x, dtype=float)
        gradient_norm = float(np.linalg.norm(np.asarray(gradient(jnp.asarray(state, dtype=jnp.float32)))))
        matrix = np.asarray(hessian(jnp.asarray(state, dtype=jnp.float32)), dtype=float)
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        classification = "minimum" if np.all(eigenvalues > 0.0) else "saddle" if eigenvalues[0] < 0.0 < eigenvalues[-1] else "other"
        output.append({
            "label": label, "state": state, "energy": float(energy(jnp.asarray(state, dtype=jnp.float32))),
            "gradient_norm": gradient_norm, "hessian_eigenvalues": eigenvalues,
            "classification": classification,
            "solver_success": bool(solved.success or gradient_norm < 1e-6),
            "solver_message": str(solved.message),
        })
    return output


def _first_crossing_boundary(energy_batch, minimum, epsilon, resolution=BOUNDARY_RESOLUTION, max_radius=4.0):
    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    directions = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    lower = np.zeros(resolution, dtype=np.float32)
    upper = np.zeros(resolution, dtype=np.float32)
    unresolved = np.ones(resolution, dtype=bool)
    radial_grid = np.linspace(0.0, max_radius, 1025, dtype=np.float32)
    for start in range(1, len(radial_grid), 16):
        stop = min(start + 16, len(radial_grid))
        radii = radial_grid[start:stop]
        states = minimum[None, None, :] + radii[:, None, None] * directions[None, :, :]
        energies = np.asarray(energy_batch(jnp.asarray(states.reshape(-1, 2)))).reshape(len(radii), resolution)
        hits = energies >= epsilon
        for offset, radius in enumerate(radii):
            newly = unresolved & hits[offset]
            upper[newly] = radius
            lower[newly] = radial_grid[start + offset - 1]
            unresolved[newly] = False
        if not np.any(unresolved):
            break
    if np.any(unresolved):
        raise RuntimeError(f"first crossing missing on {int(np.sum(unresolved))}/{resolution} rays")
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        energies = np.asarray(energy_batch(jnp.asarray(minimum + middle[:, None] * directions)))
        outside = energies >= epsilon
        upper = np.where(outside, middle, upper)
        lower = np.where(outside, lower, middle)
    radii = 0.5 * (lower + upper)
    states = minimum + radii[:, None] * directions
    energies = np.asarray(energy_batch(jnp.asarray(states)))
    return {"theta": theta, "states": states, "energies": energies, "energy_residual": energies - epsilon}


def _phebm_rollout_batch(adapter, initial_states, inputs, dt):
    def clip_state(state):
        state = jnp.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        norm = jnp.linalg.norm(state)
        return state * jnp.minimum(1.0, pf.STATE_NORM_CLIP / (norm + 1e-8))

    def one(initial_state, input_trajectory):
        def step(state, measured_input):
            state = clip_state(state)
            field = lambda value: adapter.vector_field(value, jnp.reshape(measured_input, (1,)))
            k1 = field(state)
            k2 = field(clip_state(state + 0.5 * dt * k1))
            k3 = field(clip_state(state + 0.5 * dt * k2))
            k4 = field(clip_state(state + dt * k3))
            next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            next_state = clip_state(next_state)
            return next_state, next_state
        _, states = jax.lax.scan(step, initial_state, input_trajectory)
        return jnp.concatenate([initial_state[None, :], states], axis=0)
    return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(initial_states), jnp.asarray(inputs)))


def _true_rollout(initial_state, inputs, dt):
    from Duffing_DoubleWell_main import simulate_duffing_trajectory
    return simulate_duffing_trajectory(np.asarray(initial_state), np.asarray(inputs), dt=dt)


def _prediction_metrics(truth, prediction):
    errors = prediction - truth
    per_trajectory = np.sqrt(np.mean(errors**2, axis=(1, 2)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    scale = float(np.std(truth))
    return {
        "rmse": rmse, "nrmse": rmse / scale,
        "median_trajectory_rmse": float(np.median(per_trajectory)),
        "maximum_trajectory_rmse": float(np.max(per_trajectory)),
    }


def _porthnn_certificate(params, gradients, input_limit=0.65, input_resolution=8193):
    """Compute the raw-input preimage of the PortHNN-u storage inequality."""
    gradients = np.asarray(gradients, dtype=np.float64)
    storage_velocity = gradients[:, 1]
    damping = float(params["damping"])
    input_grid = np.linspace(-input_limit, input_limit, input_resolution, dtype=np.float32)
    force_grid = np.asarray(jax.vmap(lambda value: porthnn_forcing(params, value))(jnp.asarray(input_grid)))
    force_derivative = np.diff(force_grid) / np.diff(input_grid)
    if np.any(force_derivative <= 0.0):
        raise RuntimeError("PortHNN-u force map is not monotone on the declared input range")

    lower = np.full(len(storage_velocity), -input_limit, dtype=np.float64)
    upper = np.full(len(storage_velocity), input_limit, dtype=np.float64)
    feasible = np.ones(len(storage_velocity), dtype=bool)
    threshold = -damping * storage_velocity
    positive = storage_velocity > 1e-10
    negative = storage_velocity < -1e-10

    upper[positive] = np.interp(threshold[positive], force_grid, input_grid)
    lower[negative] = np.interp(threshold[negative], force_grid, input_grid)
    feasible[positive & (threshold < force_grid[0])] = False
    feasible[negative & (threshold > force_grid[-1])] = False
    upper[positive & (threshold >= force_grid[-1])] = input_limit
    lower[negative & (threshold <= force_grid[0])] = -input_limit
    lower[~feasible] = np.nan
    upper[~feasible] = np.nan

    zero_feasible = feasible & (lower <= 0.0) & (upper >= 0.0)
    centered_radius = np.zeros(len(storage_velocity), dtype=np.float64)
    centered_radius[zero_feasible] = np.minimum(-lower[zero_feasible], upper[zero_feasible])
    uniform_lower = float(np.nanmax(lower)) if np.all(feasible) else None
    uniform_upper = float(np.nanmin(upper)) if np.all(feasible) else None
    uniform_nonempty = bool(
        uniform_lower is not None and uniform_upper is not None and uniform_lower <= uniform_upper
    )
    zero_force_input = float(np.interp(0.0, force_grid, input_grid))
    zero_force_centered_radius = (
        max(0.0, min(zero_force_input - uniform_lower, uniform_upper - zero_force_input))
        if uniform_nonempty and uniform_lower <= zero_force_input <= uniform_upper else 0.0
    )
    return {
        "lower": lower,
        "upper": upper,
        "centered_radius": centered_radius,
        "zero_input_feasible": zero_feasible,
        "storage_velocity": storage_velocity,
        "damping": damping,
        "force_at_zero": float(np.interp(0.0, input_grid, force_grid)),
        "zero_force_input": zero_force_input,
        "force_derivative_min": float(np.min(force_derivative)),
        "force_derivative_max": float(np.max(force_derivative)),
        "sampled_uniform_interval": [uniform_lower, uniform_upper] if uniform_nonempty else None,
        "sampled_uniform_centered_radius": (
            max(0.0, min(-uniform_lower, uniform_upper))
            if uniform_nonempty and uniform_lower <= 0.0 <= uniform_upper else 0.0
        ),
        "sampled_uniform_zero_force_centered_radius": zero_force_centered_radius,
    }


def _porthnn_certificate_audit(params, states, gradients, certificate, input_limit=0.65):
    states = np.asarray(states, dtype=np.float32)
    gradients = np.asarray(gradients, dtype=np.float32)
    velocity = certificate["storage_velocity"]
    worst_input = np.where(velocity >= 0.0, certificate["upper"], certificate["lower"])
    valid = np.isfinite(worst_input)
    force_values = np.asarray(jax.vmap(lambda value: porthnn_forcing(params, value))(
        jnp.asarray(worst_input[valid], dtype=jnp.float32)
    ))
    formula_hdot = certificate["damping"] * velocity[valid] ** 2 + velocity[valid] * force_values
    fields = np.asarray(jax.vmap(lambda state, value: porthnn_vector_field(params, state, value))(
        jnp.asarray(states[valid]), jnp.asarray(worst_input[valid], dtype=jnp.float32)
    ))
    implemented_hdot = np.sum(gradients[valid] * fields, axis=1)

    outward = np.where(velocity[valid] >= 0.0, 1.0, -1.0)
    outside_input = np.clip(worst_input[valid] + 0.01 * outward, -input_limit, input_limit)
    can_step_outside = np.abs(outside_input - worst_input[valid]) > 1e-7
    outside_force = np.asarray(jax.vmap(lambda value: porthnn_forcing(params, value))(
        jnp.asarray(outside_input, dtype=jnp.float32)
    ))
    outside_hdot = certificate["damping"] * velocity[valid] ** 2 + velocity[valid] * outside_force
    return {
        "sample_count": int(np.sum(valid)),
        "formula_implementation_gap_maximum": float(np.max(np.abs(formula_hdot - implemented_hdot))),
        "admissible_endpoint_hdot_maximum": float(np.max(formula_hdot)),
        "admissible_endpoint_pass_fraction": float(np.mean(formula_hdot <= 2e-5)),
        "just_outside_test_count": int(np.sum(can_step_outside)),
        "just_outside_violation_fraction": float(np.mean(outside_hdot[can_step_outside] > 0.0)),
    }


def _porthnn_applied_input_audit(params, states, inputs, input_limit=0.65):
    states = np.asarray(states, dtype=np.float32).reshape(-1, 2)
    inputs = np.asarray(inputs, dtype=np.float32).reshape(-1)
    gradients = np.asarray(jax.vmap(jax.grad(porthnn_hamiltonian, argnums=1), in_axes=(None, 0))(
        params, jnp.asarray(states)
    ))
    certificate = _porthnn_certificate(params, gradients, input_limit=input_limit)
    inside = (
        np.isfinite(certificate["lower"])
        & (inputs >= certificate["lower"] - 1e-8)
        & (inputs <= certificate["upper"] + 1e-8)
    )
    velocity = certificate["storage_velocity"]
    force_values = np.asarray(jax.vmap(lambda value: porthnn_forcing(params, value))(jnp.asarray(inputs)))
    storage_hdot = certificate["damping"] * velocity**2 + velocity * force_values
    return {
        "sample_count": len(states),
        "applied_input_inside_fraction": float(np.mean(inside)),
        "positive_storage_derivative_fraction": float(np.mean(storage_hdot > 1e-8)),
        "storage_derivative_maximum": float(np.max(storage_hdot)),
        "storage_derivative_median": float(np.median(storage_hdot)),
    }


def _fidelity(adapter, states, inputs):
    states = np.asarray(states, dtype=np.float32).reshape(-1, 2)
    inputs = np.asarray(inputs, dtype=np.float32).reshape(-1)
    count = min(4096, len(states))
    indices = np.linspace(0, len(states) - 1, count, dtype=int)
    states = states[indices]
    inputs = inputs[indices]

    def one(state, measured_input):
        a_h, d_h, _, _, _ = adapter.theory_energy_terms(state)
        analytic = d_h - a_h[0] * measured_input
        implemented = adapter.implemented_hdot(state, jnp.reshape(measured_input, (1,)))
        return analytic, implemented

    analytic, implemented = jax.jit(jax.vmap(one))(jnp.asarray(states), jnp.asarray(inputs))
    gap = np.abs(np.asarray(analytic) - np.asarray(implemented))
    return {
        "sample_count": count,
        "absolute_gap_median": float(np.median(gap)),
        "absolute_gap_p95": float(np.quantile(gap, 0.95)),
        "absolute_gap_maximum": float(np.max(gap)),
    }


def _write_tables(output_dir: Path, rows: list[tuple[str, str, str, str]]) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "architecture_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Quantity", "Truth", "PortHNN-u", "pH-EBM"])
        writer.writerows(rows)
    lines = [
        "\\begin{tabular}{llll}", "\\toprule",
        "Quantity & Truth & PortHNN-u & pH-EBM \\\\", "\\midrule",
    ]
    for quantity, truth, porthnn, phebm in rows:
        escaped = [value.replace("%", "\\%").replace("_", "\\_") for value in (quantity, truth, porthnn, phebm)]
        lines.append(" & ".join(escaped) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (table_dir / "architecture_comparison.tex").write_text("\n".join(lines))


def run(output_dir: Path = OUTPUT, porthnn_params_path: Path | None = None) -> dict:
    porthnn_params_path = porthnn_params_path or output_dir / "external_porthnn" / "parameters.npz"
    for directory in (
        "manifests", "external_porthnn", "ground_truth", "ph_ebm", "metrics",
        "predictions", "boundaries", "figures/main", "figures/supplement",
        "figures/previews", "tables", "audits", "logs",
    ):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    data = np.load(DATASET)
    porthnn_params = load_parameters(porthnn_params_path)
    porthnn_hidden_dims = [int(layer["weight"].shape[1]) for layer in porthnn_params["hamiltonian"][:-1]]
    adapter, phebm_config = load_adapter(DEFAULT_RUN)
    _, _, _, phebm_damping, phebm_input = adapter.theory_energy_terms(
        np.zeros(2, dtype=np.float32)
    )
    np.savez_compressed(
        output_dir / "ph_ebm" / "decomposition.npz",
        damping_matrix=np.asarray(phebm_damping),
        input_matrix=np.asarray(phebm_input),
    )
    ph_energy = jax.jit(lambda state: porthnn_hamiltonian(porthnn_params, state))
    ph_grad = jax.jit(jax.grad(ph_energy))
    ph_hessian = jax.jit(jax.hessian(ph_energy))
    ph_energy_batch = jax.jit(jax.vmap(ph_energy))
    ph_grad_batch = jax.jit(jax.vmap(ph_grad))

    truth_critical = [
        {"label": "left_minimum", "state": TRUE_MINIMA[0], "energy": 0.0, "classification": "minimum"},
        {"label": "saddle", "state": TRUE_SADDLE, "energy": SADDLE_ENERGY, "classification": "saddle"},
        {"label": "right_minimum", "state": TRUE_MINIMA[1], "energy": 0.0, "classification": "minimum"},
    ]
    ph_critical = _critical_points(ph_energy, ph_grad, ph_hessian)
    phebm_critical = phebm_critical_points(adapter)
    _write_json(output_dir / "ground_truth" / "critical_points.json", {"critical_points": truth_critical})
    _write_json(output_dir / "external_porthnn" / "critical_points.json", {"critical_points": ph_critical})
    _write_json(output_dir / "ph_ebm" / "critical_points.json", {"critical_points": phebm_critical})

    q_axis = np.linspace(-1.6, 1.6, 241, dtype=np.float32)
    p_axis = np.linspace(-1.1, 1.1, 181, dtype=np.float32)
    qq, pp = np.meshgrid(q_axis, p_axis)
    grid = np.column_stack([qq.ravel(), pp.ravel()]).astype(np.float32)
    truth_energy = true_hamiltonian(grid).reshape(qq.shape)
    ph_energy_grid = np.asarray(ph_energy_batch(jnp.asarray(grid))).reshape(qq.shape)
    phebm_energy_grid = np.asarray(adapter.energy_batch(jnp.asarray(grid))).reshape(qq.shape)
    np.savez_compressed(output_dir / "ground_truth" / "energy_grid.npz", q=q_axis, p=p_axis, energy=truth_energy)
    np.savez_compressed(output_dir / "external_porthnn" / "energy_grid.npz", q=q_axis, p=p_axis, energy=ph_energy_grid)
    np.savez_compressed(output_dir / "ph_ebm" / "energy_grid.npz", q=q_axis, p=p_axis, energy=phebm_energy_grid)
    grid_eigenvalues = np.linalg.eigvalsh(np.asarray(jax.vmap(ph_hessian)(jnp.asarray(grid))))
    force_diagnostics = _porthnn_certificate(porthnn_params, np.ones((1, 2)))
    _write_json(output_dir / "metrics" / "porthnn_u_architecture_diagnostics.json", {
        "hamiltonian_hessian_eigenvalue_minimum": float(np.min(grid_eigenvalues)),
        "hamiltonian_hessian_eigenvalue_maximum": float(np.max(grid_eigenvalues)),
        "hamiltonian_hessian_negative_fraction": float(np.mean(grid_eigenvalues < 0.0)),
        "learned_damping": float(porthnn_params["damping"]),
        "physical_damping": -0.4,
        "force_map_monotone_on_declared_domain": True,
        "force_derivative_minimum": force_diagnostics["force_derivative_min"],
        "force_derivative_maximum": force_diagnostics["force_derivative_max"],
    })

    by_method = {
        "porthnn_u": {point["label"]: point for point in ph_critical},
        "ph_ebm": {point["label"]: point for point in phebm_critical},
    }

    sweep_resolution = 1024
    ph_right = by_method["porthnn_u"]["right_minimum"]
    ebm_right = by_method["ph_ebm"]["right_minimum"]
    ph_gap_max = float(by_method["porthnn_u"]["saddle"]["energy"] - ph_right["energy"])
    ebm_gap_max = float(by_method["ph_ebm"]["saddle"]["energy"] - ebm_right["energy"])
    storage_gaps = np.linspace(0.0, min(ph_gap_max, ebm_gap_max), 100)
    ph_uniform_radius = np.zeros_like(storage_gaps)
    ebm_uniform_radius = np.zeros_like(storage_gaps)
    for index, storage_gap in enumerate(storage_gaps[1:], start=1):
        ph_shell = _first_crossing_boundary(
            ph_energy_batch,
            np.asarray(ph_right["state"]),
            float(ph_right["energy"] + storage_gap),
            resolution=sweep_resolution,
        )
        ph_shell_gradients = np.asarray(ph_grad_batch(jnp.asarray(ph_shell["states"])))
        ph_uniform_radius[index] = _porthnn_certificate(
            porthnn_params, ph_shell_gradients
        )["sampled_uniform_zero_force_centered_radius"]

        ebm_shell = phebm_boundary(
            adapter,
            np.asarray(ebm_right["state"]),
            float(ebm_right["energy"] + storage_gap),
            sweep_resolution,
        )
        ebm_uniform_radius[index] = phebm_terms(
            adapter, ebm_shell["states"]
        )["sampled_symmetric_radius"]

    np.savez_compressed(
        output_dir / "metrics" / "cbf_radius_vs_storage_gap.npz",
        storage_gap=storage_gaps,
        porthnn_u_uniform_radius=ph_uniform_radius,
        ph_ebm_uniform_radius=ebm_uniform_radius,
    )
    _write_json(output_dir / "metrics" / "cbf_radius_vs_storage_gap.json", {
        "well": "right",
        "sample_count": len(storage_gaps),
        "boundary_resolution": sweep_resolution,
        "storage_gap_range": [float(storage_gaps[0]), float(storage_gaps[-1])],
        "radius_definition": "largest raw-input ball about each model's zero-force command valid over the sampled complete shell",
        "porthnn_u_zero_force_center": float(_porthnn_certificate(porthnn_params, np.ones((1, 2)))["zero_force_input"]),
        "ph_ebm_zero_force_center": 0.0,
        "porthnn_u_storage_gap_to_saddle": ph_gap_max,
        "ph_ebm_storage_gap_to_saddle": ebm_gap_max,
    })

    shell_metrics = []
    porthnn_certificate_audits = []
    for well, label in (("left", "left_minimum"), ("right", "right_minimum")):
        for alpha in ALPHAS:
            truth = true_boundary(float(alpha), well, BOUNDARY_RESOLUTION)
            ph_minimum = np.asarray(by_method["porthnn_u"][label]["state"])
            ph_minimum_energy = float(by_method["porthnn_u"][label]["energy"])
            ph_saddle_energy = float(by_method["porthnn_u"]["saddle"]["energy"])
            ph_epsilon = ph_minimum_energy + float(alpha) * (ph_saddle_energy - ph_minimum_energy)
            ph_boundary = _first_crossing_boundary(ph_energy_batch, ph_minimum, ph_epsilon)
            ph_gradients = np.asarray(ph_grad_batch(jnp.asarray(ph_boundary["states"])))
            ph_certificate = _porthnn_certificate(porthnn_params, ph_gradients)
            ph_certificate_audit = _porthnn_certificate_audit(
                porthnn_params, ph_boundary["states"], ph_gradients, ph_certificate
            )
            porthnn_certificate_audits.append({
                "well": well, "alpha": float(alpha), **ph_certificate_audit,
            })

            ebm_minimum = np.asarray(by_method["ph_ebm"][label]["state"])
            ebm_minimum_energy = float(by_method["ph_ebm"][label]["energy"])
            ebm_saddle_energy = float(by_method["ph_ebm"]["saddle"]["energy"])
            ebm_epsilon = ebm_minimum_energy + float(alpha) * (ebm_saddle_energy - ebm_minimum_energy)
            ebm_boundary = phebm_boundary(adapter, ebm_minimum, ebm_epsilon, BOUNDARY_RESOLUTION)
            ebm_gradients = np.asarray(adapter.grad_energy_batch(jnp.asarray(ebm_boundary["states"])))
            ebm_certificate = phebm_terms(adapter, ebm_boundary["states"])
            truth_kappa = high_accuracy_kappa(float(alpha), well)["kappa"]
            row = {
                "well": well, "alpha": float(alpha), "truth_kappa": truth_kappa,
                "porthnn_u_kappa": float(np.min(np.linalg.norm(ph_gradients, axis=1))),
                "ph_ebm_kappa": float(np.min(np.linalg.norm(ebm_gradients, axis=1))),
                "porthnn_u_hausdorff": _hausdorff(truth.states, ph_boundary["states"]),
                "ph_ebm_hausdorff": _hausdorff(truth.states, ebm_boundary["states"]),
                "porthnn_u_max_energy_residual": float(np.max(np.abs(ph_boundary["energy_residual"]))),
                "ph_ebm_max_energy_residual": float(np.max(np.abs(ebm_boundary["energy_residual"]))),
                "ph_ebm_sampled_input_interval": ebm_certificate["sampled_interval"],
                "ph_ebm_sampled_uniform_radius": ebm_certificate["sampled_symmetric_radius"],
                "porthnn_u_sampled_input_interval": ph_certificate["sampled_uniform_interval"],
                "porthnn_u_sampled_uniform_centered_radius": ph_certificate["sampled_uniform_centered_radius"],
                "porthnn_u_sampled_uniform_zero_force_centered_radius": ph_certificate["sampled_uniform_zero_force_centered_radius"],
                "porthnn_u_zero_force_input": ph_certificate["zero_force_input"],
                "porthnn_u_zero_input_feasible_fraction": float(np.mean(ph_certificate["zero_input_feasible"])),
                "porthnn_u_certificate": "nonlinear raw-input preimage; sampled model estimate",
            }
            shell_metrics.append(row)
            np.savez_compressed(
                output_dir / "boundaries" / f"alpha_{alpha:.2f}_{well}.npz",
                theta=truth.theta, truth_states=truth.states,
                porthnn_u_states=ph_boundary["states"], ph_ebm_states=ebm_boundary["states"],
                truth_grad_norm=truth.grad_norm,
                porthnn_u_grad_norm=np.linalg.norm(ph_gradients, axis=1),
                ph_ebm_grad_norm=np.linalg.norm(ebm_gradients, axis=1),
                porthnn_u_interval_lower=ph_certificate["lower"],
                porthnn_u_interval_upper=ph_certificate["upper"],
                porthnn_u_pointwise_centered_radius=ph_certificate["centered_radius"],
                porthnn_u_zero_input_feasible=ph_certificate["zero_input_feasible"],
                porthnn_u_storage_velocity=ph_certificate["storage_velocity"],
                ph_ebm_pointwise_radius=ebm_certificate["pointwise_radius"],
                ph_ebm_a_h=ebm_certificate["a_h"],
                ph_ebm_d_h=ebm_certificate["d_h"],
            )
    _write_json(output_dir / "metrics" / "critical_and_boundary_metrics.json", {"shells": shell_metrics})

    convergence_rows = []
    for well, label in (("left", "left_minimum"), ("right", "right_minimum")):
        for resolution in (128, 256, 512, 1024, 2048):
            alpha = 0.8
            ph_minimum = np.asarray(by_method["porthnn_u"][label]["state"])
            ph_minimum_energy = float(by_method["porthnn_u"][label]["energy"])
            ph_epsilon = ph_minimum_energy + alpha * (
                float(by_method["porthnn_u"]["saddle"]["energy"]) - ph_minimum_energy
            )
            ph_shell = _first_crossing_boundary(
                ph_energy_batch, ph_minimum, ph_epsilon, resolution=resolution
            )
            ph_kappa = float(np.min(np.linalg.norm(
                np.asarray(ph_grad_batch(jnp.asarray(ph_shell["states"]))), axis=1
            )))
            ph_shell_gradients = np.asarray(ph_grad_batch(jnp.asarray(ph_shell["states"])))
            ph_certificate = _porthnn_certificate(porthnn_params, ph_shell_gradients)
            ebm_minimum = np.asarray(by_method["ph_ebm"][label]["state"])
            ebm_minimum_energy = float(by_method["ph_ebm"][label]["energy"])
            ebm_epsilon = ebm_minimum_energy + alpha * (
                float(by_method["ph_ebm"]["saddle"]["energy"]) - ebm_minimum_energy
            )
            ebm_shell = phebm_boundary(adapter, ebm_minimum, ebm_epsilon, resolution)
            ebm_gradient = np.asarray(adapter.grad_energy_batch(jnp.asarray(ebm_shell["states"])))
            ebm_certificate = phebm_terms(adapter, ebm_shell["states"])
            true_interval = sampled_uniform_interval(alpha, well, resolution)
            convergence_rows.append({
                "well": well, "resolution": resolution,
                "truth_kappa": high_accuracy_kappa(alpha, well)["kappa"],
                "porthnn_u_kappa": ph_kappa,
                "ph_ebm_kappa": float(np.min(np.linalg.norm(ebm_gradient, axis=1))),
                "truth_sampled_uniform_radius": true_interval["sampled_symmetric_radius"],
                "truth_exact_uniform_radius": 0.0,
                "porthnn_u_uniform_centered_radius": ph_certificate["sampled_uniform_centered_radius"],
                "porthnn_u_uniform_zero_force_centered_radius": ph_certificate["sampled_uniform_zero_force_centered_radius"],
                "porthnn_u_uniform_interval": ph_certificate["sampled_uniform_interval"],
                "ph_ebm_sampled_uniform_radius": ebm_certificate["sampled_symmetric_radius"],
            })
    _write_json(output_dir / "metrics" / "boundary_resolution_convergence.json", {
        "alpha": 0.8, "rows": convergence_rows,
        "epistemic_status": "PortHNN-u values are sampled nonlinear raw-input preimages under its learned storage function.",
    })
    mismatch_states = true_boundary(0.8, "right", BOUNDARY_RESOLUTION).states
    mismatch_rows = []
    for ratio in (0.0, 0.05, 0.10, 0.20):
        delta_r = 0.4 * ratio
        support_loss = delta_r * mismatch_states[:, 1] ** 2
        mismatch_rows.append({
            "relative_damping_reduction": ratio, "delta_r": delta_r,
            "maximum_support_loss": float(np.max(support_loss)),
            "analytic_support": "Delta_r*p^2",
        })
    _write_json(output_dir / "metrics" / "damping_mismatch.json", {"rows": mismatch_rows})

    test_truth = np.asarray(data["z_test"])
    test_inputs = np.asarray(data["u_test"])[..., 0]
    porthnn_predictions = np.asarray(np.load(output_dir / "external_porthnn" / "predictions.npz")["predictions"])
    phebm_predictions = _phebm_rollout_batch(adapter, test_truth[:, 0], test_inputs, 0.02)
    np.savez_compressed(
        output_dir / "ph_ebm" / "predictions.npz", truth=test_truth,
        predictions=phebm_predictions, inputs=test_inputs,
    )
    dynamics_metrics = {
        "porthnn_u": _prediction_metrics(test_truth, porthnn_predictions),
        "ph_ebm": _prediction_metrics(test_truth, phebm_predictions),
    }
    _write_json(output_dir / "metrics" / "trajectory_metrics.json", dynamics_metrics)

    duration = 20.0
    base_dt = 0.02
    time_axis = np.arange(int(duration / base_dt)) * base_dt
    initial_conditions = np.asarray([[-1.15, 0.15], [0.15, 0.0]], dtype=np.float32)
    amplitudes = np.asarray([0.2, 0.4, 0.6])
    stress_inputs = np.stack([amplitude * np.sin(2.0 * np.pi * 0.35 * time_axis) for amplitude in amplitudes])
    stress_initial = np.repeat(initial_conditions, len(amplitudes), axis=0)
    stress_forcing = np.tile(stress_inputs, (len(initial_conditions), 1)).astype(np.float32)
    truth_stress = np.stack([_true_rollout(state, values, base_dt) for state, values in zip(stress_initial, stress_forcing)])
    ph_stress = np.asarray(jax.jit(jax.vmap(lambda state, values: porthnn_rollout(porthnn_params, state, values, base_dt)))(jnp.asarray(stress_initial), jnp.asarray(stress_forcing)))
    ebm_stress = _phebm_rollout_batch(adapter, stress_initial, stress_forcing, base_dt)
    np.savez_compressed(
        output_dir / "predictions" / "common_forcing_stress.npz",
        time=np.arange(truth_stress.shape[1]) * base_dt, initial_conditions=stress_initial,
        amplitudes=np.tile(amplitudes, len(initial_conditions)), inputs=stress_forcing,
        truth=truth_stress, porthnn_u=ph_stress, ph_ebm=ebm_stress,
    )
    stress_rows = []
    for index, (initial, amplitude) in enumerate(zip(stress_initial, np.tile(amplitudes, len(initial_conditions)))):
        for method, prediction in (("porthnn_u", ph_stress[index]), ("ph_ebm", ebm_stress[index])):
            energy = true_hamiltonian(prediction)
            stress_rows.append({
                "case": index, "method": method, "initial_state": initial,
                "forcing_amplitude": amplitude,
                "trajectory_rmse": float(np.sqrt(np.mean((prediction - truth_stress[index]) ** 2))),
                "maximum_true_energy_along_prediction": float(np.max(energy)),
                "crossed_well": bool(np.any(np.sign(prediction[:, 0]) != np.sign(initial[0]))),
            })
    _write_json(output_dir / "metrics" / "common_forcing_stress.json", {"rows": stress_rows})
    porthnn_stress_certificate = _porthnn_applied_input_audit(
        porthnn_params, ph_stress[:, :-1], stress_forcing
    )
    _write_json(output_dir / "audits" / "porthnn_u_cbf_stress_audit.json", {
        "storage_derivative": "N*(partial_p H)^2 + (partial_p H)*F_theta(u)",
        "input_domain": [-0.65, 0.65],
        "shell_audits": porthnn_certificate_audits,
        "common_forcing_trajectories": porthnn_stress_certificate,
    })

    alpha_eight_boundary = np.load(output_dir / "boundaries" / "alpha_0.80_right.npz")
    boundary_states = alpha_eight_boundary["ph_ebm_states"]
    boundary_inputs = np.zeros(len(boundary_states))
    fidelity = {
        "validation_states": _fidelity(adapter, data["z_val"][:, :-1], data["u_val"]),
        "selected_boundary_states": _fidelity(adapter, boundary_states, boundary_inputs),
        "common_stress_trajectories": _fidelity(adapter, ebm_stress[:, :-1], stress_forcing),
    }
    _write_json(output_dir / "audits" / "cbf_formula_implementation_audit.json", fidelity)

    timestep_rows = []
    for dt in (0.02, 0.01, 0.005):
        repeat = int(round(base_dt / dt))
        refined_inputs = np.repeat(stress_inputs[1], repeat)
        initial = initial_conditions[1]
        truth_refined = _true_rollout(initial, refined_inputs, dt)
        ph_refined = np.asarray(porthnn_rollout(porthnn_params, jnp.asarray(initial), jnp.asarray(refined_inputs), dt))
        ebm_refined = _phebm_rollout_batch(adapter, initial[None, :], refined_inputs[None, :], dt)[0]
        for method, prediction in (("truth", truth_refined), ("porthnn_u", ph_refined), ("ph_ebm", ebm_refined)):
            timestep_rows.append({
                "dt": dt, "method": method, "terminal_state": prediction[-1],
                "maximum_true_energy": float(np.max(true_hamiltonian(prediction))),
            })
    _write_json(output_dir / "audits" / "timestep_convergence.json", {"rows": timestep_rows})

    def shell_row(alpha, well="right"):
        return next(row for row in shell_metrics if row["well"] == well and np.isclose(row["alpha"], alpha))

    ph_points = {point["label"]: point for point in ph_critical}
    ebm_points = {point["label"]: point for point in phebm_critical}
    ph_saddle_error = np.linalg.norm(np.asarray(ph_points["saddle"]["state"]) - TRUE_SADDLE)
    ebm_saddle_error = np.linalg.norm(np.asarray(ebm_points["saddle"]["state"]) - TRUE_SADDLE)
    ph_gap = 0.5 * sum(ph_points["saddle"]["energy"] - ph_points[label]["energy"] for label in ("left_minimum", "right_minimum"))
    ebm_gap = 0.5 * sum(ebm_points["saddle"]["energy"] - ebm_points[label]["energy"] for label in ("left_minimum", "right_minimum"))
    rows = [
        ("Test state RMSE", "0", f"{dynamics_metrics['porthnn_u']['rmse']:.6f}", f"{dynamics_metrics['ph_ebm']['rmse']:.6f}"),
        ("Long-horizon RMSE", "0", f"{dynamics_metrics['porthnn_u']['rmse']:.6f}", f"{dynamics_metrics['ph_ebm']['rmse']:.6f}"),
        ("Trainable parameters", "0", "162201", "2392"),
        ("Recovered minima", "2", "2", "2"),
        ("Saddle location error", "0", f"{ph_saddle_error:.6f}", f"{ebm_saddle_error:.6f}"),
        ("Relative saddle-gap error", "0", f"{abs(ph_gap-SADDLE_ENERGY)/SADDLE_ENERGY:.3%}", f"{abs(ebm_gap-SADDLE_ENERGY)/SADDLE_ENERGY:.3%}"),
        ("Boundary Hausdorff alpha=0.80", "0", f"{shell_row(.8)['porthnn_u_hausdorff']:.6f}", f"{shell_row(.8)['ph_ebm_hausdorff']:.6f}"),
        ("Boundary Hausdorff alpha=0.95", "0", f"{shell_row(.95)['porthnn_u_hausdorff']:.6f}", f"{shell_row(.95)['ph_ebm_hausdorff']:.6f}"),
        ("Kappa error alpha=0.80", "0", f"{abs(shell_row(.8)['porthnn_u_kappa']-shell_row(.8)['truth_kappa']):.6f}", f"{abs(shell_row(.8)['ph_ebm_kappa']-shell_row(.8)['truth_kappa']):.6f}"),
        ("Kappa error alpha=0.95", "0", f"{abs(shell_row(.95)['porthnn_u_kappa']-shell_row(.95)['truth_kappa']):.6f}", f"{abs(shell_row(.95)['ph_ebm_kappa']-shell_row(.95)['truth_kappa']):.6f}"),
        ("Explicit admissible-input certificate", "Analytic; uniform set degenerate", "Nonlinear preimage; sampled model estimate", "Affine set-valued; sampled model estimate"),
        ("Runtime controller required", "No", "No", "No"),
    ]
    _write_tables(output_dir, rows)
    _write_json(output_dir / "metrics" / "architecture_components.json", {
        "truth": {"H": "analytic", "J": "canonical analytic", "R": "diag(0,0.4)", "input": "G*u, G=(0,1)^T", "coercive": True, "explicit_cbf_set": "analytic"},
        "porthnn_u": {"H": f"unconstrained nonconvex {'/'.join(map(str, porthnn_hidden_dims))} tanh MLP", "J": "fixed canonical", "R": "one unconstrained scalar N", "input": "nonlinear F_theta(u)", "coercive": False, "explicit_cbf_set": "nonlinear raw-input preimage"},
        "ph_ebm": {"H": "coercive nonconvex EBM, layers 64/32", "J": "learned constant skew field", "R": "PSD parameterization", "input": "learned constant G*u", "coercive": True, "explicit_cbf_set": "yes"},
    })
    _write_json(output_dir / "manifests" / "analysis_manifest.json", {
        "alphas": ALPHAS, "boundary_resolution": BOUNDARY_RESOLUTION,
        "porthnn_params_path": str(porthnn_params_path.resolve()),
        "porthnn_hidden_dims": porthnn_hidden_dims,
        "common_forcing": "sinusoid, frequency 0.35 Hz, amplitudes 0.2/0.4/0.6",
        "energy_normalization": "per-model minimum and saddle gap",
        "porthnn_certificate": "sampled nonlinear raw-input preimage under learned storage; not a true-plant guarantee",
    })
    return {"trajectory_metrics": dynamics_metrics, "shell_rows": len(shell_metrics), "stress_rows": len(stress_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--porthnn-params", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.output_dir, porthnn_params_path=args.porthnn_params)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()