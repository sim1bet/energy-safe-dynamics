# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Primary known-system validation experiment for the Duffing certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar, root
from scipy.spatial.distance import directed_hausdorff

ROOT = Path(__file__).resolve().parents[2]
for module_path in (ROOT, ROOT / "EBM_model", ROOT / "Interface_code"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from Duffing_DoubleWell_main import _build_layers
from Stress_tests.experiments.duffing_ground_truth_certificate import (
    DAMPING,
    SADDLE_ENERGY,
    TRUE_MINIMA,
    TRUE_SADDLE,
    boundary_states as true_boundary_states,
    damping_mismatch_support,
    grad_hamiltonian,
    hamiltonian,
    high_accuracy_kappa,
    sampled_uniform_interval,
    vector_field as true_vector_field,
)
from Stress_tests.integration import apply_runtime_config
from Stress_tests.model_adapter import EBMStressAdapter


DEFAULT_RUN = ROOT / "results" / "duffing_doublewell" / "runs" / "duffing_v5_champion_exact_seed4"
DEFAULT_OUTPUT = ROOT / "results" / "duffing_doublewell" / "certificate_experiment"
ALPHAS = np.asarray([0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.98])
RESOLUTIONS = (128, 256, 512, 1024, 2048, 4096)
COLORS = {"true": "#202020", "learned": "#0072B2", "left": "#009E73", "right": "#D55E00"}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(fig: plt.Figure, base: Path, arrays: dict[str, np.ndarray]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(base.with_suffix(".npz"), **arrays)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.7, "lines.linewidth": 1.3,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })


def load_adapter(run_dir: Path) -> tuple[EBMStressAdapter, dict]:
    with (run_dir / "config" / "config.json").open() as handle:
        config = json.load(handle)
    apply_runtime_config(config)
    with (run_dir / "checkpoints" / "params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    return EBMStressAdapter(
        params, _build_layers(config), d=int(config["d"]),
        m=int(config["m_ports"]), dt=0.02,
    ), config


def learned_critical_points(adapter: EBMStressAdapter) -> list[dict]:
    grad = lambda state: np.asarray(adapter.grad_energy(np.asarray(state, dtype=np.float32)), dtype=float)
    hessian_jit = jax.jit(jax.jacfwd(adapter.grad_energy))
    points: list[dict] = []
    for label, seed in (("left_minimum", [-1.0, 0.0]), ("saddle", [0.0, 0.0]), ("right_minimum", [1.0, 0.0])):
        solved = root(grad, np.asarray(seed), method="hybr", options={"xtol": 1e-10})
        state = np.asarray(solved.x, dtype=float)
        hessian = np.asarray(hessian_jit(jnp.asarray(state, dtype=jnp.float32)), dtype=float)
        eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
        classification = "minimum" if np.all(eigenvalues > 0.0) else "saddle" if eigenvalues[0] < 0.0 < eigenvalues[-1] else "other"
        points.append({
            "label": label, "state": state, "energy": float(adapter.energy(state)),
            "gradient_norm": float(np.linalg.norm(grad(state))),
            "hessian_eigenvalues": eigenvalues, "classification": classification,
            "solver_success": bool(solved.success), "solver_message": str(solved.message),
        })
    return points


def _learned_boundary(
    adapter: EBMStressAdapter,
    minimum: np.ndarray,
    epsilon: float,
    resolution: int,
    max_radius: float = 4.0,
) -> dict:
    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    directions = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    lower = np.zeros(resolution, dtype=np.float32)
    upper = np.zeros(resolution, dtype=np.float32)
    unresolved = np.ones(resolution, dtype=bool)
    radial_grid = np.linspace(0.0, max_radius, 1025, dtype=np.float32)
    radial_chunk = 64
    for start in range(1, len(radial_grid), radial_chunk):
        stop = min(start + radial_chunk, len(radial_grid))
        radii = radial_grid[start:stop]
        states = minimum[None, None, :] + radii[:, None, None] * directions[None, :, :]
        energies = np.asarray(adapter.energy_batch(states.reshape(-1, 2))).reshape(len(radii), resolution)
        hits = energies >= epsilon
        for offset, radius in enumerate(radii):
            newly = unresolved & hits[offset]
            upper[newly] = radius
            lower[newly] = radial_grid[start + offset - 1]
            unresolved[newly] = False
        if not np.any(unresolved):
            break
    if np.any(unresolved):
        raise RuntimeError(f"learned boundary failed on {np.sum(unresolved)} of {resolution} rays")
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        energies = np.asarray(adapter.energy_batch(minimum + middle[:, None] * directions))
        outside = energies >= epsilon
        upper = np.where(outside, middle, upper)
        lower = np.where(outside, lower, middle)
    radii = 0.5 * (lower + upper)
    states = minimum + radii[:, None] * directions
    energies = np.asarray(adapter.energy_batch(states), dtype=float)
    return {
        "theta": theta, "states": states.astype(float), "radii": radii.astype(float),
        "energies": energies, "energy_residual": energies - epsilon,
    }


def _learned_terms(adapter: EBMStressAdapter, states: np.ndarray) -> dict:
    def one(state):
        a_h, d_h, gradient, _, _ = adapter.theory_energy_terms(state)
        return a_h[0], d_h, gradient
    a_h, d_h, gradients = jax.jit(jax.vmap(one))(jnp.asarray(states, dtype=jnp.float32))
    a_h = np.asarray(a_h, dtype=float)
    d_h = np.asarray(d_h, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    tolerance = 1e-12
    active = np.abs(a_h) > tolerance
    lower = np.full(len(states), -np.inf)
    upper = np.full(len(states), np.inf)
    lower[a_h < -tolerance] = d_h[a_h < -tolerance] / a_h[a_h < -tolerance]
    upper[a_h > tolerance] = d_h[a_h > tolerance] / a_h[a_h > tolerance]
    interval = [float(np.max(lower)), float(np.min(upper))]
    radius = max(0.0, min(interval[1], -interval[0]))
    pointwise = np.divide(d_h, np.abs(a_h), out=np.full_like(d_h, np.inf), where=active)
    return {
        "a_h": a_h, "d_h": d_h, "gradients": gradients,
        "grad_norm": np.linalg.norm(gradients, axis=1),
        "pointwise_radius": pointwise, "input_lower": lower, "input_upper": upper,
        "sampled_interval": interval, "sampled_symmetric_radius": float(radius),
        "coupling_tolerance": tolerance,
    }


def _learned_state_at_angle(
    adapter: EBMStressAdapter, minimum: np.ndarray, epsilon: float, theta: float
) -> np.ndarray:
    direction = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    radial_grid = np.linspace(0.0, 4.0, 2049, dtype=np.float32)
    states = minimum + radial_grid[:, None] * direction
    energies = np.asarray(adapter.energy_batch(states))
    crossings = np.flatnonzero(energies >= epsilon)
    if len(crossings) == 0:
        raise RuntimeError("learned scalar ray did not cross the requested shell")
    crossing = int(crossings[0])
    lower = float(radial_grid[max(0, crossing - 1)])
    upper = float(radial_grid[crossing])
    for _ in range(52):
        middle = 0.5 * (lower + upper)
        if float(adapter.energy(minimum + middle * direction)) >= epsilon:
            upper = middle
        else:
            lower = middle
    return np.asarray(minimum + 0.5 * (lower + upper) * direction, dtype=float)


def _refine_learned_kappa(
    adapter: EBMStressAdapter, minimum: np.ndarray, epsilon: float,
    theta: np.ndarray, grad_norm: np.ndarray,
) -> dict:
    candidate = int(np.argmin(grad_norm))
    step = 2.0 * np.pi / len(theta)
    center = float(theta[candidate])

    def objective(angle: float) -> float:
        state = _learned_state_at_angle(adapter, minimum, epsilon, angle)
        return float(np.linalg.norm(np.asarray(adapter.grad_energy(state))))

    solved = minimize_scalar(
        objective, bounds=(center - 2.0 * step, center + 2.0 * step),
        method="bounded", options={"xatol": 1e-10, "maxiter": 100},
    )
    state = _learned_state_at_angle(adapter, minimum, epsilon, float(solved.x))
    return {
        "kappa": objective(float(solved.x)), "state": state,
        "theta": float(np.mod(solved.x, 2.0 * np.pi)),
        "optimizer_success": bool(solved.success),
    }


def _dense_contour_crosscheck(
    adapter: EBMStressAdapter, minimum: np.ndarray, epsilon: float,
    ray_states: np.ndarray,
) -> dict:
    q = np.linspace(float(minimum[0]) - 1.35, float(minimum[0]) + 1.35, 401)
    p = np.linspace(float(minimum[1]) - 1.15, float(minimum[1]) + 1.15, 361)
    qq, pp = np.meshgrid(q, p)
    grid = np.column_stack([qq.ravel(), pp.ravel()]).astype(np.float32)
    energy = np.asarray(adapter.energy_batch(grid)).reshape(qq.shape)
    figure, axis = plt.subplots()
    contour = axis.contour(qq, pp, energy, levels=[epsilon])
    segments = contour.allsegs[0]
    plt.close(figure)
    candidates = [vertices for vertices in segments if len(vertices) >= 20]
    if not candidates:
        raise RuntimeError("dense contour extraction found no learned shell")
    vertices = min(candidates, key=lambda values: np.min(np.linalg.norm(values - minimum, axis=1)))
    return {
        "hausdorff": _hausdorff(ray_states, vertices),
        "n_contour_vertices": int(len(vertices)),
        "grid_shape": [int(len(p)), int(len(q))],
    }


def _hausdorff(left: np.ndarray, right: np.ndarray) -> float:
    return float(max(directed_hausdorff(left, right)[0], directed_hausdorff(right, left)[0]))


def _rk4_true(state: np.ndarray, forcing: float, dt: float) -> np.ndarray:
    k1 = true_vector_field(state, forcing)
    k2 = true_vector_field(state + 0.5 * dt * k1, forcing)
    k3 = true_vector_field(state + 0.5 * dt * k2, forcing)
    k4 = true_vector_field(state + dt * k3, forcing)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rollout_true(state: np.ndarray, factor: float, dt: float, duration: float = 2.0) -> dict:
    steps = int(round(duration / dt))
    states = np.empty((steps + 1, 2), dtype=float)
    inputs = np.empty(steps, dtype=float)
    states[0] = state
    for index in range(steps):
        p = states[index, 1]
        radius = DAMPING * abs(p)
        inputs[index] = factor * radius * (1.0 if p >= 0.0 else -1.0)
        states[index + 1] = _rk4_true(states[index], inputs[index], dt)
    return {"states": states, "inputs": inputs, "energy": hamiltonian(states), "dt": dt, "factor": factor}


def _critical_payload(learned: list[dict]) -> dict:
    exact = [
        {"label": "left_minimum", "state": TRUE_MINIMA[0], "energy": 0.0, "classification": "minimum"},
        {"label": "saddle", "state": TRUE_SADDLE, "energy": SADDLE_ENERGY, "classification": "saddle"},
        {"label": "right_minimum", "state": TRUE_MINIMA[1], "energy": 0.0, "classification": "minimum"},
    ]
    by_label = {point["label"]: point for point in learned}
    learned_left_gap = float(by_label["saddle"]["energy"] - by_label["left_minimum"]["energy"])
    learned_right_gap = float(by_label["saddle"]["energy"] - by_label["right_minimum"]["energy"])
    return {
        "ground_truth": exact, "learned": learned,
        "errors": {
            label: {
                "location_l2": float(np.linalg.norm(np.asarray(by_label[label]["state"]) - np.asarray(reference["state"]))),
                "energy_absolute_after_minimum_gauge": (
                    0.0 if "minimum" in label else
                    abs(0.5 * (learned_left_gap + learned_right_gap) - SADDLE_ENERGY)
                ),
            }
            for label, reference in ((point["label"], point) for point in exact)
        },
        "learned_saddle_gaps": {"left": learned_left_gap, "right": learned_right_gap},
        "true_saddle_gap": SADDLE_ENERGY,
    }


def run_experiment(run_dir: Path, output_dir: Path, max_resolution: int = 4096) -> dict:
    _style()
    directories = ["manifests", "ground_truth", "learned", "boundaries", "metrics", "predictions", "figures", "mismatch", "audits", "logs"]
    for name in directories:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    adapter, config = load_adapter(run_dir)
    checkpoint = run_dir / "checkpoints" / "params.pkl"
    dataset = ROOT / "results" / "duffing_doublewell" / "datasets" / "duffing_v1.npz"
    config_path = run_dir / "config" / "config.json"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "duffing_primary_certificate_validation",
        "run_id": run_dir.name, "run_directory": str(run_dir.relative_to(ROOT)),
        "git_commit": None, "git_status": "repository metadata unavailable",
        "checkpoint_sha256": _sha256(checkpoint), "config_sha256": _sha256(config_path),
        "dataset_sha256": _sha256(dataset), "sampling_interval": 0.02,
        "alphas": ALPHAS, "requested_max_resolution": int(max_resolution),
        "runtime_config": adapter.compatibility_report().runtime_flags,
        "ground_truth_module": "Stress_tests.experiments.duffing_ground_truth_certificate",
    }
    _write_json(output_dir / "manifests" / "experiment_manifest.json", manifest)
    _write_json(output_dir / "ground_truth" / "certificate_formula.json", {
        "energy_derivative": "dH/dt = -0.4*p^2 + p*(u+d)",
        "pointwise_half_space": "p*(u+d) <= 0.4*p^2",
        "pointwise_combined_forcing_radius": "rho(q,p) = 0.4*abs(p)",
        "bounded_disturbance_input_condition": "abs(u) + rho_d <= 0.4*abs(p)",
        "uniform_input_interval_on_every_regular_closed_component": [0.0, 0.0],
        "uniform_input_and_disturbance_conclusion": "No positive centered input or disturbance radius is uniform on a complete regular closed component.",
    })

    learned_critical = learned_critical_points(adapter)
    critical = _critical_payload(learned_critical)
    _write_json(output_dir / "ground_truth" / "critical_points.json", {"critical_points": critical["ground_truth"]})
    _write_json(output_dir / "learned" / "critical_points.json", {
        "critical_points": learned_critical, "errors": critical["errors"],
        "learned_saddle_gaps": critical["learned_saddle_gaps"],
        "true_saddle_gap": critical["true_saddle_gap"],
    })
    learned_by_label = {point["label"]: point for point in learned_critical}
    learned_saddle_energy = float(learned_by_label["saddle"]["energy"])

    available_resolutions = tuple(value for value in RESOLUTIONS if value <= max_resolution)
    if not available_resolutions:
        raise ValueError("max_resolution must be at least 128")
    shell_metrics: list[dict] = []
    boundary_cache: dict[tuple[str, str, float], dict] = {}
    convergence: list[dict] = []
    for well, label, true_minimum in (("left", "left_minimum", TRUE_MINIMA[0]), ("right", "right_minimum", TRUE_MINIMA[1])):
        learned_minimum = np.asarray(learned_by_label[label]["state"])
        learned_minimum_energy = float(learned_by_label[label]["energy"])
        learned_gap = learned_saddle_energy - learned_minimum_energy
        for alpha in ALPHAS:
            true_boundary = true_boundary_states(float(alpha), well, resolution=max_resolution)
            epsilon_learned = learned_minimum_energy + float(alpha) * learned_gap
            learned_boundary = _learned_boundary(adapter, learned_minimum, epsilon_learned, max_resolution)
            learned_terms = _learned_terms(adapter, learned_boundary["states"])
            learned_kappa = _refine_learned_kappa(
                adapter, learned_minimum, epsilon_learned,
                learned_boundary["theta"], learned_terms["grad_norm"],
            )
            boundary_cache[("true", well, float(alpha))] = {"boundary": true_boundary}
            boundary_cache[("learned", well, float(alpha))] = {"boundary": learned_boundary, "terms": learned_terms}
            true_kappa = high_accuracy_kappa(float(alpha), well)
            metric = {
                "well": well, "alpha": float(alpha),
                "true_energy": float(alpha * SADDLE_ENERGY), "learned_energy": epsilon_learned,
                "true_kappa": true_kappa["kappa"],
                "learned_sampled_kappa": float(np.min(learned_terms["grad_norm"])),
                "learned_refined_kappa": learned_kappa["kappa"],
                "true_kappa_state": true_kappa["state"],
                "learned_kappa_state": learned_kappa["state"],
                "true_distance_to_saddle": true_kappa["distance_to_saddle"],
                "learned_distance_to_saddle": float(np.min(np.linalg.norm(learned_boundary["states"] - np.asarray(learned_by_label["saddle"]["state"]), axis=1))),
                "boundary_hausdorff": _hausdorff(true_boundary.states, learned_boundary["states"]),
                "true_max_energy_residual": float(np.max(np.abs(true_boundary.energy_residual))),
                "learned_max_energy_residual": float(np.max(np.abs(learned_boundary["energy_residual"]))),
                "true_exact_uniform_interval": [0.0, 0.0],
                "true_exact_uniform_radius": 0.0,
                "learned_sampled_interval": learned_terms["sampled_interval"],
                "learned_sampled_radius": learned_terms["sampled_symmetric_radius"],
            }
            if np.isclose(alpha, 0.8):
                metric["ray_vs_dense_contour"] = _dense_contour_crosscheck(
                    adapter, learned_minimum, epsilon_learned, learned_boundary["states"]
                )
            shell_metrics.append(metric)
            np.savez_compressed(
                output_dir / "boundaries" / f"alpha_{alpha:.2f}_{well}.npz",
                true_states=true_boundary.states, learned_states=learned_boundary["states"],
                theta=true_boundary.theta, true_grad_norm=true_boundary.grad_norm,
                learned_grad_norm=learned_terms["grad_norm"],
                true_pointwise_radius=true_boundary.pointwise_radius,
                learned_pointwise_radius=learned_terms["pointwise_radius"],
                learned_a_h=learned_terms["a_h"], learned_d_h=learned_terms["d_h"],
            )
        for resolution in available_resolutions:
            alpha = 0.8
            epsilon_learned = learned_minimum_energy + alpha * learned_gap
            learned_boundary = _learned_boundary(adapter, learned_minimum, epsilon_learned, resolution)
            learned_terms = _learned_terms(adapter, learned_boundary["states"])
            true_interval = sampled_uniform_interval(alpha, well, resolution)
            convergence.append({
                "well": well, "resolution": resolution,
                "true_sampled_radius": true_interval["sampled_symmetric_radius"],
                "true_exact_radius": 0.0,
                "learned_sampled_radius": learned_terms["sampled_symmetric_radius"],
                "learned_sampled_interval": learned_terms["sampled_interval"],
                "learned_min_abs_coupling": float(np.min(np.abs(learned_terms["a_h"]))),
            })

    _write_json(output_dir / "metrics" / "shell_metrics.json", {"shells": shell_metrics})
    _write_json(output_dir / "metrics" / "uniform_radius_convergence.json", {"alpha": 0.8, "rows": convergence})

    alpha_boundary = boundary_cache[("learned", "right", 0.8)]
    states = alpha_boundary["boundary"]["states"]
    terms = alpha_boundary["terms"]
    finite_radius = np.where(np.isfinite(terms["pointwise_radius"]), terms["pointwise_radius"], 0.0)
    factors = np.asarray([0.9, 1.0, 1.1])
    formula_checks = []
    for factor in factors:
        inputs = factor * finite_radius * np.sign(terms["a_h"])
        theory_hdot = -terms["d_h"] + terms["a_h"] * inputs
        implemented_energy_hdot = -np.asarray(jax.vmap(adapter.implemented_hdot)(jnp.asarray(states, dtype=jnp.float32), jnp.asarray(inputs[:, None], dtype=jnp.float32)))
        gap = implemented_energy_hdot - theory_hdot
        worst = int(np.argmax(np.abs(gap)))
        formula_checks.append({
            "factor": factor, "theory_energy_derivative_quantiles": np.quantile(theory_hdot, [0.0, 0.5, 0.95, 1.0]),
            "implemented_energy_derivative_quantiles": np.quantile(implemented_energy_hdot, [0.0, 0.5, 0.95, 1.0]),
            "absolute_gap_quantiles": np.quantile(np.abs(gap), [0.5, 0.95, 0.99, 1.0]),
            "input_magnitude_quantiles": np.quantile(np.abs(inputs), [0.5, 0.95, 0.99, 1.0]),
            "positive_theory_derivative_fraction": float(np.mean(theory_hdot > 1e-8)),
            "positive_implemented_derivative_fraction": float(np.mean(implemented_energy_hdot > 1e-8)),
            "worst_gap_state": states[worst], "worst_gap_a_h": terms["a_h"][worst],
            "worst_gap_input": inputs[worst], "worst_absolute_gap": abs(gap[worst]),
        })
    _write_json(output_dir / "audits" / "formula_vs_implementation.json", {"alpha": 0.8, "well": "right", "rows": formula_checks})

    rollout_start = true_boundary_states(0.8, "right", 4096).states[1024]
    rollout_rows = []
    rollout_arrays = {}
    for factor in factors:
        for dt in (0.02, 0.01, 0.005):
            rollout = _rollout_true(rollout_start, float(factor), dt)
            key = f"factor_{factor:.1f}_dt_{dt:g}".replace(".", "p")
            rollout_arrays[f"{key}_states"] = rollout["states"]
            rollout_arrays[f"{key}_energy"] = rollout["energy"]
            rollout_arrays[f"{key}_inputs"] = rollout["inputs"]
            rollout_rows.append({
                "factor": factor, "dt": dt, "maximum_energy": float(np.max(rollout["energy"])),
                "final_energy": float(rollout["energy"][-1]),
                "escaped_initial_shell": bool(np.max(rollout["energy"]) > 0.8 * SADDLE_ENERGY + 1e-8),
            })
    np.savez_compressed(output_dir / "predictions" / "state_dependent_rollouts.npz", **rollout_arrays)
    _write_json(output_dir / "predictions" / "state_dependent_rollouts.json", {"rows": rollout_rows})

    mismatch_rows = []
    mismatch_states = true_boundary_states(0.8, "right", 4096).states
    for ratio in (0.0, 0.05, 0.10, 0.20):
        delta = ratio * DAMPING
        support = damping_mismatch_support(mismatch_states, delta)
        nominal_dissipation = DAMPING * mismatch_states[:, 1] ** 2
        mismatch_rows.append({
            "relative_damping_reduction": ratio, "delta_r": delta,
            "maximum_support_loss": float(np.max(support)),
            "minimum_remaining_dissipation": float(np.min(nominal_dissipation - support)),
            "analytic_support_identity_max_error": 0.0,
        })
    _write_json(output_dir / "mismatch" / "damping_mismatch.json", {"model": "r_plant = 0.4 - Delta_r", "rows": mismatch_rows})

    _render_figures(output_dir, adapter, learned_critical, shell_metrics, convergence, boundary_cache, rollout_arrays, rollout_rows, mismatch_rows)
    full_metrics = {
        "manifest": manifest, "critical_points": critical, "shell_metrics": shell_metrics,
        "uniform_radius_convergence": convergence, "formula_vs_implementation": formula_checks,
        "rollouts": rollout_rows, "damping_mismatch": mismatch_rows,
        "primary_conclusion": "The exact plant has pointwise nonzero capacity away from p=0 but zero continuous uniform symmetric radius on every regular closed shell.",
        "learned_radius_status": "Finite-resolution sampled estimate; not a continuous-boundary proof.",
    }
    _write_json(output_dir / "metrics" / "duffing_certificate_metrics.json", full_metrics)
    geometry_passed = all(point["classification"] == expected for point, expected in zip(learned_critical, ("minimum", "saddle", "minimum")))
    fidelity_passed = max(row["worst_absolute_gap"] for row in formula_checks) <= 1e-3
    audit = {
        "passed": bool(geometry_passed and fidelity_passed),
        "pipeline_completed": True,
        "checks": {
            "geometry_passed": geometry_passed,
            "certificate_fidelity_passed": fidelity_passed,
            "certificate_fidelity_max_gap_tolerance": 1e-3,
            "true_uniform_radius_exactly_zero": True,
            "learned_critical_point_types": [point["classification"] for point in learned_critical],
            "maximum_true_boundary_energy_residual": max(row["true_max_energy_residual"] for row in shell_metrics),
            "maximum_learned_boundary_energy_residual": max(row["learned_max_energy_residual"] for row in shell_metrics),
            "resolution_series": available_resolutions,
        },
        "limitations": [
            "Learned boundary extrema are sampled, not interval-certified.",
            "Ray tracing returns the first crossing from each learned minimum.",
            "Finite-time escape tests are numerical validation, not proof.",
        ],
    }
    _write_json(output_dir / "audits" / "audit.json", audit)
    return {"output_directory": str(output_dir), "audit": audit, "figures": 8}


def _render_figures(output_dir, adapter, learned_critical, shell_metrics, convergence, boundary_cache, rollout_arrays, rollout_rows, mismatch_rows):
    q = np.linspace(-1.65, 1.65, 241)
    p = np.linspace(-1.15, 1.15, 201)
    qq, pp = np.meshgrid(q, p)
    grid = np.column_stack([qq.ravel(), pp.ravel()]).astype(np.float32)
    true_h = hamiltonian(grid).reshape(qq.shape)
    learned_h = np.asarray(adapter.energy_batch(grid)).reshape(qq.shape).copy()
    learned_h -= min(float(item["energy"]) for item in learned_critical if item["classification"] == "minimum")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    levels_true = np.linspace(0.0, 0.32, 13)
    levels_learned = np.linspace(0.0, np.quantile(learned_h, 0.85), 13)
    axes[0].contourf(qq, pp, true_h, levels=levels_true, cmap="magma")
    axes[1].contourf(qq, pp, learned_h, levels=levels_learned, cmap="viridis")
    for axis, title in zip(axes, ("True Hamiltonian", "Learned Hamiltonian")):
        axis.set(title=title, xlabel=r"position $q$", ylabel=r"momentum $p$", aspect="equal")
    _save_figure(fig, output_dir / "figures" / "duffing_true_vs_learned_H", {"q": q, "p": p, "true_H": true_h, "learned_H_shifted": learned_h})

    fig, axis = plt.subplots(figsize=(3.7, 3.0), constrained_layout=True)
    for model, style in (("true", "-"), ("learned", "--")):
        for well in ("left", "right"):
            rows = [row for row in shell_metrics if row["well"] == well]
            values = [row["true_kappa"] if model == "true" else row["learned_sampled_kappa"] for row in rows]
            axis.plot(ALPHAS, values, style, color=COLORS[well], label=f"{model}, {well}")
    axis.set(xlabel=r"relative shell $\alpha$", ylabel=r"$\kappa=\inf_{\Gamma}\|\nabla H\|$")
    axis.legend(frameon=False, ncol=2)
    _save_figure(fig, output_dir / "figures" / "kappa_vs_shell", {"alpha": ALPHAS, "rows": np.asarray([[row["true_kappa"], row["learned_sampled_kappa"]] for row in shell_metrics])})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for axis, well in zip(axes, ("left", "right")):
        for alpha in ALPHAS:
            true_boundary = boundary_cache[("true", well, float(alpha))]["boundary"].states
            learned_boundary = boundary_cache[("learned", well, float(alpha))]["boundary"]["states"]
            axis.plot(true_boundary[:, 0], true_boundary[:, 1], color=COLORS["true"], alpha=0.15 + 0.7 * alpha, lw=0.8)
            axis.plot(learned_boundary[:, 0], learned_boundary[:, 1], color=COLORS["learned"], alpha=0.15 + 0.7 * alpha, lw=0.8, ls="--")
        axis.scatter([0.0], [0.0], marker="x", color="#CC79A7", zorder=4)
        axis.set(title=f"{well.capitalize()} well", xlabel=r"$q$", ylabel=r"$p$", aspect="equal")
    _save_figure(fig, output_dir / "figures" / "boundary_evolution_to_saddle", {"alpha": ALPHAS})

    true_boundary = boundary_cache[("true", "right", 0.8)]["boundary"]
    learned = boundary_cache[("learned", "right", 0.8)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(true_boundary.states[:, 0], true_boundary.states[:, 1], color=COLORS["true"], label="true")
    axes[0].plot(learned["boundary"]["states"][:, 0], learned["boundary"]["states"][:, 1], color=COLORS["learned"], ls="--", label="learned")
    axes[0].set(xlabel=r"$q$", ylabel=r"$p$", aspect="equal", title=r"Shell $\alpha=0.80$")
    axes[0].legend(frameon=False)
    axes[1].plot(true_boundary.theta, true_boundary.pointwise_radius, color=COLORS["true"], label="true")
    axes[1].plot(learned["boundary"]["theta"], learned["terms"]["pointwise_radius"], color=COLORS["learned"], ls="--", label="learned")
    axes[1].set(xlabel=r"boundary angle $\theta$", ylabel=r"pointwise radius $d_H/|a_H|$", title="State-dependent capacity")
    axes[1].set_ylim(bottom=0.0, top=np.nanquantile(learned["terms"]["pointwise_radius"][np.isfinite(learned["terms"]["pointwise_radius"])], 0.98) * 1.1)
    _save_figure(fig, output_dir / "figures" / "pointwise_input_certificate_alpha080", {"theta": true_boundary.theta, "true_radius": true_boundary.pointwise_radius, "learned_radius": learned["terms"]["pointwise_radius"]})

    fig, axis = plt.subplots(figsize=(3.7, 3.0), constrained_layout=True)
    for well in ("left", "right"):
        rows = [row for row in convergence if row["well"] == well]
        resolution = np.asarray([row["resolution"] for row in rows])
        axis.loglog(resolution, [row["true_sampled_radius"] for row in rows], "o-", color=COLORS[well], label=f"true sampled, {well}")
        axis.loglog(resolution, [row["learned_sampled_radius"] for row in rows], "s--", color=COLORS[well], label=f"learned sampled, {well}")
    axis.axhline(0.0, color="black", lw=0.7)
    axis.set(xlabel="boundary samples", ylabel="sampled uniform radius")
    axis.legend(frameon=False)
    _save_figure(fig, output_dir / "figures" / "uniform_radius_convergence", {"rows": np.asarray([[row["resolution"], row["true_sampled_radius"], row["learned_sampled_radius"]] for row in convergence])})

    fig, axis = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    for factor, color in zip((0.9, 1.0, 1.1), ("#009E73", "#0072B2", "#D55E00")):
        key = f"factor_{factor:.1f}_dt_0p005".replace(".", "p")
        energy = rollout_arrays[f"{key}_energy"]
        axis.plot(np.arange(len(energy)) * 0.005, energy, color=color, label=fr"${factor:.1f}\rho(q,p)$")
    axis.axhline(0.8 * SADDLE_ENERGY, color="black", ls=":", label="initial shell")
    axis.set(xlabel="time [s]", ylabel=r"$H^\star(q,p)$")
    axis.legend(frameon=False)
    _save_figure(fig, output_dir / "figures" / "state_dependent_certificate_rollouts", {"rows": np.asarray([[row["factor"], row["dt"], row["maximum_energy"], row["final_energy"]] for row in rollout_rows])})

    fig, axis = plt.subplots(figsize=(3.7, 3.0), constrained_layout=True)
    for factor, color in zip((0.9, 1.0, 1.1), ("#009E73", "#0072B2", "#D55E00")):
        rows = sorted((row for row in rollout_rows if np.isclose(row["factor"], factor)), key=lambda row: row["dt"])
        axis.plot([row["dt"] for row in rows], [row["maximum_energy"] - 0.8 * SADDLE_ENERGY for row in rows], "o-", color=color, label=fr"${factor:.1f}\rho(q,p)$")
    axis.axhline(0.0, color="black", lw=0.7, ls=":")
    axis.set(xlabel="RK4 timestep [s]", ylabel="maximum shell excess")
    axis.legend(frameon=False)
    _save_figure(fig, output_dir / "figures" / "timestep_convergence", {"rows": np.asarray([[row["factor"], row["dt"], row["maximum_energy"] - 0.8 * SADDLE_ENERGY] for row in rollout_rows])})

    fig, axis = plt.subplots(figsize=(3.7, 3.0), constrained_layout=True)
    ratios = np.asarray([row["relative_damping_reduction"] for row in mismatch_rows])
    losses = np.asarray([row["maximum_support_loss"] for row in mismatch_rows])
    axis.plot(100.0 * ratios, losses, "o-", color="#D55E00")
    axis.set(xlabel="damping reduction [%]", ylabel=r"maximum support loss $\Delta_r p^2$")
    _save_figure(fig, output_dir / "figures" / "duffing_damping_mismatch", {"relative_reduction": ratios, "maximum_support_loss": losses})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-resolution", type=int, default=4096, choices=RESOLUTIONS)
    args = parser.parse_args()
    print(json.dumps(_jsonable(run_experiment(args.run_dir, args.output_dir, args.max_resolution)), indent=2))


if __name__ == "__main__":
    main()