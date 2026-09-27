# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Top-level, paper-oriented stress-test orchestration.

Nothing in this module trains or modifies the EBM.  The suite is deliberately
post-hoc: given a finished parameter pytree, it interrogates the learned
Hamiltonian and the claimed energy-CBF certificate as aggressively as possible.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import jax.numpy as jnp

from .boundary import inset_boundary_points, sample_energy_boundary
from .certificate import audit_certificate
from .config import StressTestConfig
from .geometry import (
    choose_plane, discover_minima, filter_converged_wells, profile_energy_grid,
    slice_energy_grid,
)
from .model_adapter import EBMStressAdapter
from .plotting import (
    plot_adversarial_rollouts, plot_landscape_comparison, plot_profile_surface,
    plot_robustness_frontier, plot_profile_stationarity,
)
from .frontier import compute_robustness_frontier
from .rollouts import (
    adversarial_rollout_batch, forced_rollout_batch, random_piecewise_input,
)


def _jsonable(x):
    if isinstance(x, dict): return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, (np.bool_,)): return bool(x)
    return x


def _random_inputs(rng: np.random.Generator, n: int, m: int, spec) -> np.ndarray:
    center = np.zeros(m, dtype=np.float32) if spec.input_center is None else np.asarray(spec.input_center, dtype=np.float32)
    if spec.input_norm == "linf":
        r = np.asarray(spec.input_radius, dtype=np.float32)
        if r.ndim == 0: r = np.full(m, float(r), dtype=np.float32)
        return center + rng.uniform(-1.0, 1.0, size=(n, m)).astype(np.float32) * r
    r = float(np.asarray(spec.input_radius))
    v = rng.normal(size=(n, m)).astype(np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    rad = rng.random((n, 1), dtype=np.float32) ** (1.0 / max(m, 1))
    return center + r * rad * v


def _random_disturbance_sequences(rng, B, T, d, spec, hold):
    r = float(spec.state_disturbance_radius)
    if r <= 0:
        return np.zeros((B, T, d), dtype=np.float32)
    nblocks = int(np.ceil(T / max(hold, 1)))
    v = rng.normal(size=(B, nblocks, d)).astype(np.float32)
    if spec.state_disturbance_norm == "l2":
        v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        mag = rng.uniform(0.75, 1.0, size=(B, nblocks, 1)).astype(np.float32)
        v = r * mag * v
    else:
        v = r * np.sign(v) * rng.uniform(0.75, 1.0, size=(B, nblocks, d)).astype(np.float32)
    return np.repeat(v, max(hold, 1), axis=1)[:, :T]


def _sample_fidelity_states(rng, reference_states, boundary_states, n, d):
    pools = []
    if reference_states is not None and len(reference_states): pools.append(np.asarray(reference_states))
    if boundary_states is not None and len(boundary_states): pools.append(np.asarray(boundary_states))
    if not pools:
        raise ValueError("Need reference or boundary states for certificate fidelity audit")
    base = np.concatenate(pools, axis=0)
    idx = rng.choice(len(base), size=int(n), replace=len(base) < int(n))
    X = base[idx].astype(np.float32)
    # Tiny perturbation avoids evaluating only on a low-dimensional trajectory
    # manifold and catches numerical safeguards that activate just off-data.
    scale = np.std(base, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    X += 0.02 * rng.normal(size=X.shape).astype(np.float32) * scale[None, :]
    return X


def run_stress_suite(
    *,
    params,
    layers: tuple,
    d: int,
    m: int,
    dt: float,
    epsilon: float,
    gamma: float,
    reference_states: Optional[np.ndarray] = None,
    reference_inputs: Optional[np.ndarray] = None,
    output_dir: str | Path = "stress_results",
    config: Optional[StressTestConfig] = None,
) -> dict:
    """Run the complete Hamiltonian/certificate stress suite.

    Parameters
    ----------
    reference_states:
        Strongly recommended.  Pass latent states from ordinary train/test
        rollouts.  They are used only to choose an informative visualization
        plane and to seed well discovery; no labels are required.
    reference_inputs:
        Optional at present; retained in the saved NPZ so plots/analyses can be
        extended without rerunning the model.

    Returns
    -------
    A nested summary dict.  Full numerical arrays and figures are written to
    ``output_dir``.
    """
    cfg = StressTestConfig() if config is None else config
    out = Path(output_dir)
    figs = out / "figures"
    arrs = out / "arrays"
    figs.mkdir(parents=True, exist_ok=True)
    arrs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    adapter = EBMStressAdapter(params, layers, d=d, m=m, dt=dt)
    compat = adapter.compatibility_report()

    # 1) Discover Hamiltonian wells. Non-stationary multi-start artifacts are
    # dropped before anything ranks candidates by energy alone (see
    # filter_converged_wells / duffing_doublewell_run_analysis.md, 2026-08-19).
    minima = discover_minima(adapter, reference_states, cfg.projection, seed=cfg.seed)
    minima = filter_converged_wells(
        adapter, minima, grad_norm_tol=cfg.projection.minima_grad_norm_tol
    )

    # 2) Trace H=epsilon in the original d-dimensional state space.  Boundary
    #    geometry is available *before* selecting the paper plane, so the auto
    #    plane can use the certified set itself instead of arbitrary coordinates.
    boundary = sample_energy_boundary(adapter, minima, float(epsilon), cfg)
    plane = choose_plane(
        adapter, reference_states, minima, cfg.projection, boundary_states=boundary.states
    )

    # Projected boundary samples define/expand the visualization canvas.
    zb = plane.project(boundary.states)
    if len(zb):
        cur_lo = np.array([plane.x_limits[0], plane.y_limits[0]], dtype=float)
        cur_hi = np.array([plane.x_limits[1], plane.y_limits[1]], dtype=float)
        blo = np.min(zb, axis=0); bhi = np.max(zb, axis=0)
        span = np.maximum(np.maximum(cur_hi, bhi) - np.minimum(cur_lo, blo), 1e-3)
        pad = 0.08 * span
        lo = np.minimum(cur_lo, blo) - pad
        hi = np.maximum(cur_hi, bhi) + pad
        plane.x_limits = (float(lo[0]), float(hi[0]))
        plane.y_limits = (float(lo[1]), float(hi[1]))

    # 3) Literal slice and lower-envelope/profiled Hamiltonian.  Computing both
    #    is the most transparent publication choice: one is geometrically exact
    #    as a cross-section; the other faithfully projects sublevel-set topology.
    def compute_landscapes():
        sd = None; pd = None
        if cfg.projection.landscape_mode in ("slice", "both"):
            sd = slice_energy_grid(adapter, plane, cfg.projection)
        # For d=2 the plane has no orthogonal complement: profile == slice and
        # its stationarity residual is identically zero, so a second panel is
        # mathematically and visually redundant.
        if plane.complement.shape[1] > 0 and cfg.projection.landscape_mode in ("profile", "both"):
            pd = profile_energy_grid(adapter, plane, minima, cfg.projection)
        return sd, pd

    def safe_touches_edge(data):
        if data is None:
            return False
        E = np.asarray(data["energy"])
        edge = np.concatenate([E[0, :], E[-1, :], E[:, 0], E[:, -1]])
        return bool(np.any(edge <= float(epsilon)))

    slice_data, profile_data = compute_landscapes()
    extent_expansions = 0
    target = profile_data if profile_data is not None else slice_data
    while (cfg.projection.ensure_closed_safety_contour
           and safe_touches_edge(target)
           and extent_expansions < int(cfg.projection.max_extent_expansions)):
        fac = float(cfg.projection.extent_expansion_factor)
        def expand(lim):
            mid = 0.5 * (lim[0] + lim[1]); half = 0.5 * (lim[1] - lim[0]) * fac
            return (float(mid - half), float(mid + half))
        plane.x_limits = expand(plane.x_limits)
        plane.y_limits = expand(plane.y_limits)
        extent_expansions += 1
        slice_data, profile_data = compute_landscapes()
        target = profile_data if profile_data is not None else slice_data
    contour_touches_edge = safe_touches_edge(target)

    plot_landscape_comparison(
        plane, minima, float(epsilon), slice_data, profile_data,
        reference_states, figs / "hamiltonian_landscape.pdf",
    )
    plot_landscape_comparison(
        plane, minima, float(epsilon), slice_data, profile_data,
        reference_states, figs / "hamiltonian_landscape.png",
    )
    if profile_data is not None:
        plot_profile_surface(profile_data, float(epsilon), figs / "hamiltonian_profile_3d.pdf")
        plot_profile_surface(profile_data, float(epsilon), figs / "hamiltonian_profile_3d.png")
        plot_profile_stationarity(profile_data, figs / "profile_stationarity.pdf")
        plot_profile_stationarity(profile_data, figs / "profile_stationarity.png")

    # 4) Worst-case support-function audit on the traced high-dimensional boundary.
    cert = audit_certificate(
        adapter, boundary.states, float(epsilon), float(gamma), cfg.uncertainty,
        slack_tolerance=cfg.cbf_slack_tolerance,
        implemented_attack_steps=cfg.implemented_attack_steps,
        implemented_attack_restarts=cfg.implemented_attack_restarts,
        implemented_attack_lr=cfg.implemented_attack_lr,
        seed=cfg.seed + 211,
        include_implemented_diagnostics=cfg.run_non_theoretical_diagnostics,
    )

    frontier = None
    if cfg.run_non_theoretical_diagnostics:
        frontier = compute_robustness_frontier(
            adapter, boundary.states, float(epsilon), float(gamma), cfg.uncertainty,
            grid_size=cfg.frontier_grid_size,
            max_input_scale=cfg.frontier_max_input_scale,
            max_disturbance_scale=cfg.frontier_max_disturbance_scale,
        )
        plot_robustness_frontier(frontier, figs / "robustness_frontier.pdf")
        plot_robustness_frontier(frontier, figs / "robustness_frontier.png")

    # 5) Formula-vs-implemented derivative audit on off-data states and inputs.
    #    This is essential because the repository deliberately contains smooth
    #    numerical saturation maps around the ideal pH field.
    fidelity_states = np.empty((0, d), dtype=np.float32)
    fidelity_inputs = np.empty((0, m), dtype=np.float32)
    fidelity = None
    fidelity_summary = None
    if cfg.run_non_theoretical_diagnostics:
        fidelity_states = _sample_fidelity_states(
            rng, reference_states, boundary.states, cfg.n_fidelity_samples, d
        )
        fidelity_inputs = _random_inputs(rng, len(fidelity_states), m, cfg.uncertainty)
        fidelity = adapter.certificate_fidelity(
            fidelity_states, fidelity_inputs, float(epsilon), float(gamma)
        )
        fidelity_summary = {
            "max_abs_hdot_gap": float(np.max(fidelity["abs_hdot_gap"])),
            "median_abs_hdot_gap": float(np.median(fidelity["abs_hdot_gap"])),
            "p95_abs_hdot_gap": float(np.quantile(fidelity["abs_hdot_gap"], 0.95)),
            "p99_abs_hdot_gap": float(np.quantile(fidelity["abs_hdot_gap"], 0.99)),
            "max_relative_hdot_gap": float(np.max(fidelity["relative_hdot_gap"])),
            "formula_violation_fraction": float(np.mean(fidelity["slack_formula"] < -cfg.cbf_slack_tolerance)),
            "implemented_violation_fraction": float(np.mean(fidelity["slack_implemented"] < -cfg.cbf_slack_tolerance)),
        }

    # 6) Start just inside the high-dimensional boundary and attack with the
    #    state-dependent pointwise worst input/disturbance for a long horizon.
    timestep_results = {}
    base_attack = None
    random_attack = None
    random_summary = None
    if cfg.run_non_theoretical_diagnostics:
        x0_pool = inset_boundary_points(boundary, minima, cfg.boundary_inset)
        B = min(int(cfg.n_rollouts), len(x0_pool))
        ids = rng.choice(len(x0_pool), size=B, replace=False)
        X0 = x0_pool[ids]
        base_T = float(cfg.horizon_steps) * float(dt)
        for mult in cfg.timestep_multipliers:
            dti = float(dt) * float(mult)
            steps = max(1, int(round(base_T / dti)))
            attack = adversarial_rollout_batch(
                adapter, X0, float(epsilon), float(gamma), cfg.uncertainty,
                horizon_steps=steps, dt=dti, integrator="rk4",
                barrier_tolerance=cfg.barrier_tolerance,
            )
            timestep_results[str(mult)] = {
                "dt": dti, "steps": steps,
                "violation_fraction": float(np.mean(attack.violation_per_rollout)),
                "min_h": float(np.min(attack.min_h_per_rollout)),
                "median_min_h": float(np.median(attack.min_h_per_rollout)),
                "min_implemented_slack": float(np.min(attack.implemented_slack)),
                "max_state_clip_displacement": float(np.max(attack.state_clip_displacement)),
            }
            if abs(float(mult) - 1.0) < 1e-12:
                base_attack = attack
        if base_attack is None:
            base_attack = adversarial_rollout_batch(
                adapter, X0, float(epsilon), float(gamma), cfg.uncertainty,
                horizon_steps=cfg.horizon_steps, dt=float(dt), integrator="rk4",
                barrier_tolerance=cfg.barrier_tolerance,
            )

        U_rand = np.stack([
            random_piecewise_input(rng, cfg.horizon_steps, m, cfg.uncertainty, cfg.random_piecewise_hold)
            for _ in range(B)
        ])
        W_rand = _random_disturbance_sequences(
            rng, B, cfg.horizon_steps, d, cfg.uncertainty, cfg.random_piecewise_hold
        )
        random_attack = forced_rollout_batch(
            adapter, X0, U_rand, W_rand, float(epsilon), dt=float(dt), integrator="rk4",
            barrier_tolerance=cfg.barrier_tolerance,
        )
        random_summary = {
            "violation_fraction": float(np.mean(random_attack["violation_per_rollout"])),
            "min_h": float(np.min(random_attack["min_h_per_rollout"])),
            "median_min_h": float(np.median(random_attack["min_h_per_rollout"])),
            "max_state_clip_displacement": float(np.max(random_attack["state_clip_displacement"])),
        }

    if profile_data is not None and base_attack is not None:
        plot_adversarial_rollouts(
            plane, profile_data, float(epsilon), base_attack.states, base_attack.h,
            figs / "adversarial_rollouts.pdf",
        )
        plot_adversarial_rollouts(
            plane, profile_data, float(epsilon), base_attack.states, base_attack.h,
            figs / "adversarial_rollouts.png",
        )

    # Persist enough raw material to remake figures without rerunning JAX.
    np.savez_compressed(
        arrs / "stress_arrays.npz",
        minima_states=minima.states,
        minima_energies=minima.energies,
        minima_multiplicities=minima.multiplicities,
        plane_center=plane.center,
        plane_basis=plane.basis,
        plane_complement=plane.complement,
        boundary_states=boundary.states,
        boundary_anchor_indices=boundary.anchor_indices,
        boundary_h=boundary.barrier_residuals,
        boundary_formula_worst_inputs=cert.formula_worst_inputs,
        boundary_implemented_attack_inputs=cert.implemented_attack_inputs,
        boundary_robust_formula_slack=cert.robust_formula_slack,
        boundary_robust_implemented_slack=cert.robust_implemented_slack,
        boundary_projected=plane.project(boundary.states),
        landscape_xx=np.asarray(target["xx"]),
        landscape_yy=np.asarray(target["yy"]),
        landscape_energy=np.asarray(target["energy"]),
        relative_energy_alpha=(np.asarray([cfg.relative_energy_alpha], dtype=np.float32)
                       if cfg.relative_energy_alpha is not None else np.empty(0)),
        frontier_input_scales=(frontier["input_scales"] if frontier is not None else np.empty(0)),
        frontier_disturbance_scales=(frontier["disturbance_scales"] if frontier is not None else np.empty(0)),
        frontier_worst_slack=(frontier["worst_slack"] if frontier is not None else np.empty((0, 0))),
        frontier_critical_boundary_index=(frontier["critical_boundary_index"] if frontier is not None else np.empty((0, 0), dtype=int)),
        fidelity_states=fidelity_states,
        fidelity_inputs=fidelity_inputs,
        fidelity_hdot_formula=(fidelity["hdot_formula"] if fidelity is not None else np.empty(0)),
        fidelity_hdot_implemented=(fidelity["hdot_implemented"] if fidelity is not None else np.empty(0)),
        attack_states=(base_attack.states if base_attack is not None else np.empty((0, 0, d))),
        attack_inputs=(base_attack.inputs if base_attack is not None else np.empty((0, 0, m))),
        attack_disturbances=(base_attack.disturbances if base_attack is not None else np.empty((0, 0, d))),
        attack_h=(base_attack.h if base_attack is not None else np.empty((0, 0))),
        random_attack_h=(random_attack["h"] if random_attack is not None else np.empty((0, 0))),
        profile_stationarity=(np.asarray(profile_data["profile_stationarity"]) if profile_data is not None and "profile_stationarity" in profile_data else np.empty((0, 0))),
        reference_states=np.asarray(reference_states) if reference_states is not None else np.empty((0, d)),
        reference_inputs=np.asarray(reference_inputs) if reference_inputs is not None else np.empty((0, m)),
    )

    minima_grad_norms = np.linalg.norm(
        np.asarray(adapter.grad_energy_batch(jnp.asarray(minima.states))), axis=1
    ) if len(minima.states) else np.empty(0)
    profile_diag = None
    if profile_data is not None and "profile_stationarity" in profile_data:
        pr = np.asarray(profile_data["profile_stationarity"])
        profile_diag = {
            "median_orthogonal_grad_norm": float(np.median(pr)),
            "p99_orthogonal_grad_norm": float(np.quantile(pr, 0.99)),
            "max_orthogonal_grad_norm": float(np.max(pr)),
        }

    summary = {
        "epsilon": float(epsilon),
        "gamma": float(gamma),
        "config": asdict(cfg),
        "compatibility": {
            "certificate_affine_in_input": compat.certificate_affine_in_input,
            "runtime_flags": compat.runtime_flags,
            "warnings": compat.warnings,
        },
        "wells": {
            "count": int(len(minima.states)),
            "energies": minima.energies,
            "multiplicities": minima.multiplicities,
            "grad_norms": minima_grad_norms,
            "max_grad_norm": float(np.max(minima_grad_norms)) if len(minima_grad_norms) else None,
        },
        "profile_optimization": profile_diag,
        "plane": {
            "method": plane.method,
            "extent_expansions": int(extent_expansions),
            "safety_contour_touches_grid_edge": bool(contour_touches_edge),
            "center": plane.center,
            "basis": plane.basis,
            "x_limits": plane.x_limits,
            "y_limits": plane.y_limits,
        },
        "boundary": {
            "n_points": int(len(boundary.states)),
            "failed_ray_fraction": float(boundary.failed_fraction),
            "max_abs_energy_residual": float(np.max(np.abs(boundary.barrier_residuals))),
            "max_state_norm": float(np.max(np.linalg.norm(boundary.states, axis=1))),
            "fraction_beyond_runtime_state_clip": float(np.mean(
                np.linalg.norm(boundary.states, axis=1) > float(compat.runtime_flags["STATE_NORM_CLIP"])
            )),
        },
        "certificate_boundary_audit": cert.summary,
        "robustness_frontier": ({
            "nominal_grid_slack": float(frontier["nominal_grid_slack"]),
            "nominal_input_scale_grid": float(frontier["nominal_input_scale_grid"]),
            "nominal_disturbance_scale_grid": float(frontier["nominal_disturbance_scale_grid"]),
            "max_input_scale": float(cfg.frontier_max_input_scale),
            "max_disturbance_scale": float(cfg.frontier_max_disturbance_scale),
        } if frontier is not None else None),
        "certificate_fidelity": fidelity_summary,
        "greedy_adversarial_timestep_audit": timestep_results,
        "random_piecewise_stress": random_summary,
        "outputs": {
            "landscape_figure": str(figs / "hamiltonian_landscape.pdf"),
            "landscape_preview": str(figs / "hamiltonian_landscape.png"),
            "profile_3d_figure": str(figs / "hamiltonian_profile_3d.pdf") if profile_data is not None else None,
            "profile_stationarity_figure": str(figs / "profile_stationarity.pdf") if profile_data is not None else None,
            "adversarial_figure": str(figs / "adversarial_rollouts.pdf") if profile_data is not None and base_attack is not None else None,
            "robustness_frontier_figure": str(figs / "robustness_frontier.pdf") if frontier is not None else None,
            "arrays": str(arrs / "stress_arrays.npz"),
        },
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return _jsonable(summary)
