# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Worst-case CBF checks for bounded external inputs and perturbations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .config import UncertaintySet
from .model_adapter import EBMStressAdapter


@dataclass
class CertificateAudit:
    states: np.ndarray
    formula_worst_inputs: np.ndarray
    implemented_attack_inputs: np.ndarray
    worst_state_disturbances: np.ndarray
    h: np.ndarray
    formula_hdot: np.ndarray
    implemented_hdot_at_formula_input: np.ndarray
    implemented_hdot_attacked: np.ndarray
    robust_formula_slack: np.ndarray
    robust_implemented_slack_at_formula_input: np.ndarray
    robust_implemented_slack: np.ndarray
    summary: dict


@dataclass
class UniformRadiusCertificate:
    component_indices: np.ndarray
    pointwise_radius: np.ndarray
    grad_norm: np.ndarray
    residual_margin: np.ndarray
    input_dual_norm: np.ndarray
    normal_dissipation: np.ndarray
    normal_input_gain: np.ndarray
    normal_disturbance_support: np.ndarray
    component_summaries: list[dict]


def compute_uniform_radius_certificate(
    adapter: EBMStressAdapter,
    states: np.ndarray,
    component_indices: np.ndarray,
    uncertainty: UncertaintySet,
    coupling_tolerance: float = 1e-8,
) -> UniformRadiusCertificate:
    """Estimate the largest invariant input radius on sampled components.

    Implements the manuscript's component-wise exact formula
    ``rho* = inf_Gamma delta/||a_H||_*`` and its regular-boundary lower
    estimate ``(r*kappa-wbar)/gbar`` using raw-gradient theory terms.
    Both quantities remain sampled estimates until the full continuous shell
    extrema are bounded independently.
    """
    states = np.asarray(states, dtype=np.float32)
    components = np.asarray(component_indices, dtype=int).reshape(-1)
    if len(states) != len(components):
        raise ValueError("states and component_indices must have equal length")

    a_h, d_h, grad, R, G = jax.jit(jax.vmap(adapter.theory_energy_terms))(jnp.asarray(states))
    a_h = np.asarray(a_h)
    d_h = np.asarray(d_h)
    grad = np.asarray(grad)
    R = np.asarray(R)
    G = np.asarray(G)

    disturbance = worst_case_state_disturbance(grad, uncertainty)
    disturbance_support = np.sum(grad * disturbance, axis=-1)
    residual = d_h - disturbance_support
    if uncertainty.input_norm == "linf":
        dual = np.sum(np.abs(a_h), axis=-1)
    elif uncertainty.input_norm == "l2":
        dual = np.linalg.norm(a_h, axis=-1)
    else:
        raise ValueError(f"Unknown input_norm={uncertainty.input_norm}")

    pointwise = np.full(len(states), np.inf, dtype=np.float64)
    coupled = dual > float(coupling_tolerance)
    pointwise[coupled] = residual[coupled] / dual[coupled]
    pointwise[~coupled & (residual < 0.0)] = 0.0

    grad_norm = np.linalg.norm(grad, axis=-1)
    normals = grad / np.maximum(grad_norm[:, None], 1e-12)
    normal_dissipation = np.einsum("ni,nij,nj->n", normals, R, normals)
    normal_ports = np.einsum("nij,ni->nj", G, normals)
    if uncertainty.input_norm == "linf":
        normal_gain = np.sum(np.abs(normal_ports), axis=-1)
    else:
        normal_gain = np.linalg.norm(normal_ports, axis=-1)
    normal_disturbance = disturbance_support / np.maximum(grad_norm, 1e-12)

    summaries = []
    for component in np.unique(components):
        mask = components == component
        finite_ratio = pointwise[mask & np.isfinite(pointwise)]
        zero_input_feasible = bool(np.all(residual[mask] >= -1e-10))
        sampled_radius = (
            float(np.min(finite_ratio)) if len(finite_ratio)
            else (float("inf") if zero_input_feasible else 0.0)
        )
        kappa = float(np.min(grad_norm[mask]))
        r_min = float(np.min(normal_dissipation[mask]))
        g_max = float(np.max(normal_gain[mask]))
        w_max = float(np.max(normal_disturbance[mask]))
        numerator = r_min * kappa - w_max
        if not zero_input_feasible or numerator < 0.0:
            regular_bound = 0.0
        elif g_max <= float(coupling_tolerance):
            regular_bound = float("inf")
        else:
            regular_bound = float(numerator / g_max)
        summaries.append({
            "component_index": int(component),
            "n_boundary": int(np.sum(mask)),
            "zero_input_feasible": zero_input_feasible,
            "sampled_exact_radius": sampled_radius,
            "regular_boundary_lower_estimate": regular_bound,
            "kappa": kappa,
            "normal_dissipation_min": r_min,
            "normal_input_gain_max": g_max,
            "normal_disturbance_support_max": w_max,
        })

    return UniformRadiusCertificate(
        component_indices=components,
        pointwise_radius=pointwise,
        grad_norm=grad_norm,
        residual_margin=residual,
        input_dual_norm=dual,
        normal_dissipation=normal_dissipation,
        normal_input_gain=normal_gain,
        normal_disturbance_support=normal_disturbance,
        component_summaries=summaries,
    )


def _center(spec: UncertaintySet, m: int) -> np.ndarray:
    if spec.input_center is None:
        return np.zeros(m, dtype=np.float32)
    c = np.asarray(spec.input_center, dtype=np.float32).reshape(-1)
    if len(c) != m:
        raise ValueError(f"input_center has length {len(c)} but model has m={m} ports")
    return c


def _linf_radius(spec: UncertaintySet, m: int) -> np.ndarray:
    r = np.asarray(spec.input_radius, dtype=np.float32)
    if r.ndim == 0:
        return np.full(m, float(r), dtype=np.float32)
    r = r.reshape(-1)
    if len(r) != m:
        raise ValueError(f"input_radius has length {len(r)} but model has m={m} ports")
    return r


def worst_case_input_from_a(a: np.ndarray, spec: UncertaintySet) -> np.ndarray:
    """Maximize a^T u over the configured input set.

    Since hdot_formula = drift - a^T u, this is the pointwise worst input for
    the barrier derivative.
    """
    A = np.asarray(a, dtype=np.float32)
    m = A.shape[-1]
    c = _center(spec, m)
    if spec.input_norm == "linf":
        r = _linf_radius(spec, m)
        return c + np.sign(A) * r
    if spec.input_norm == "l2":
        r = np.asarray(spec.input_radius, dtype=np.float32)
        if r.ndim != 0:
            raise ValueError("l2 input_radius must be a scalar")
        n = np.linalg.norm(A, axis=-1, keepdims=True)
        return c + float(r) * A / np.maximum(n, 1e-12)
    raise ValueError(f"Unknown input_norm={spec.input_norm}")


def worst_case_state_disturbance(grad_energy: np.ndarray, spec: UncertaintySet) -> np.ndarray:
    """Worst additive w in xdot=f+w for h=epsilon-H.

    The disturbance contribution is -grad(H)^T w, hence the adversary aligns w
    with +grad(H).

    When ``spec.disturbance_map`` (E) is set, the disturbance is generalized to
    ``w = E d``, ``|d| <= state_disturbance_radius``.  The support function of
    the mapped set in the direction of ``grad(H)`` is
    ``sigma_W(grad H) = rho_d * |E^T grad H|_*``, attained at
    ``d* = rho_d * dual_argmax(E^T grad H)`` and mapped back via ``w* = E d*``.
    ``disturbance_map=None`` (default) is exactly the identity map (E=I) and
    reproduces the original behaviour bit-for-bit.
    """
    G = np.asarray(grad_energy, dtype=np.float32)
    r = float(spec.state_disturbance_radius)
    if r <= 0:
        return np.zeros_like(G)
    if spec.disturbance_map is None:
        Eg = G
    else:
        E = np.asarray(spec.disturbance_map, dtype=np.float32)  # [d, k]
        Eg = G @ E  # E^T grad(H), shape [..., k]
    if spec.state_disturbance_norm == "l2":
        n = np.linalg.norm(Eg, axis=-1, keepdims=True)
        d_star = r * Eg / np.maximum(n, 1e-12)
    elif spec.state_disturbance_norm == "linf":
        d_star = r * np.sign(Eg)
    else:
        raise ValueError(f"Unknown state_disturbance_norm={spec.state_disturbance_norm}")
    if spec.disturbance_map is None:
        return d_star
    return d_star @ E.T



def _pgd_implemented_worst_input(
    adapter: EBMStressAdapter,
    states: np.ndarray,
    formula_start: np.ndarray,
    spec: UncertaintySet,
    steps: int,
    restarts: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirically minimize implemented hdot over the admissible input set.

    For the ideal affine pH field the support-function solution is exact and PGD
    should return the same boundary point.  When numerical saturation or an
    optional nonlinear input map is active, this attack probes the *actual*
    implemented vector field rather than assuming affine dependence on raw u.
    """
    X = np.asarray(states, dtype=np.float32)
    U0 = np.asarray(formula_start, dtype=np.float32)
    n, m = U0.shape
    R = max(int(restarts), 1)
    rng = np.random.default_rng(seed)
    center = _center(spec, m)

    starts = np.empty((R, n, m), dtype=np.float32)
    starts[0] = U0
    for rix in range(1, R):
        if spec.input_norm == "linf":
            rad = _linf_radius(spec, m)
            starts[rix] = center + rng.uniform(-1.0, 1.0, size=(n, m)).astype(np.float32) * rad
        else:
            rad = float(np.asarray(spec.input_radius))
            v = rng.normal(size=(n, m)).astype(np.float32)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            rho = rng.random((n, 1), dtype=np.float32) ** (1.0 / max(m, 1))
            starts[rix] = center + rad * rho * v

    Xrep = np.repeat(X[None, :, :], R, axis=0).reshape(-1, adapter.d)
    Ustart = starts.reshape(-1, m)
    c = jnp.asarray(center)

    if spec.input_norm == "linf":
        rad = jnp.asarray(_linf_radius(spec, m))
        def project(u):
            return c + jnp.clip(u - c, -rad, rad)
    else:
        rad = float(np.asarray(spec.input_radius))
        def project(u):
            delta = u - c
            nn = jnp.linalg.norm(delta, axis=-1, keepdims=True)
            return c + delta * jnp.minimum(1.0, rad / (nn + 1e-8))

    def objective(x, u):
        return adapter.implemented_hdot(x, u)
    grad_u = jax.vmap(jax.grad(objective, argnums=1))
    obj_batch = jax.vmap(objective)

    @jax.jit
    def attack(xrep, u0):
        def body(u, _):
            g = jnp.nan_to_num(grad_u(xrep, u), nan=0.0, posinf=0.0, neginf=0.0)
            gn = jnp.linalg.norm(g, axis=-1, keepdims=True)
            g = g * jnp.minimum(1.0, 20.0 / (gn + 1e-8))
            return project(u - float(lr) * g), None
        uf, _ = jax.lax.scan(body, project(u0), xs=None, length=max(int(steps), 0))
        return uf, obj_batch(xrep, uf)

    uf, vals = attack(jnp.asarray(Xrep), jnp.asarray(Ustart))
    uf = np.asarray(uf).reshape(R, n, m)
    vals = np.asarray(vals).reshape(R, n)
    pick = np.argmin(vals, axis=0)
    cols = np.arange(n)
    return uf[pick, cols], vals[pick, cols]

def audit_certificate(
    adapter: EBMStressAdapter,
    states: np.ndarray,
    epsilon: float,
    gamma: float,
    uncertainty: UncertaintySet,
    slack_tolerance: float = 1e-5,
    implemented_attack_steps: int = 40,
    implemented_attack_restarts: int = 4,
    implemented_attack_lr: float = 0.08,
    seed: int = 0,
    include_implemented_diagnostics: bool = True,
) -> CertificateAudit:
    """Evaluate analytic and implemented worst-case CBF slack on given states."""
    X = jnp.asarray(states)

    def terms(x):
        a, drift = adapter.affine_terms(x)
        g = adapter.grad_energy(x)
        h = adapter.barrier(x, epsilon)
        return a, drift, g, h

    a, drift, grad, h = jax.jit(jax.vmap(terms))(X)
    a_np = np.asarray(a)
    drift_np = np.asarray(drift)
    grad_np = np.asarray(grad)
    h_np = np.asarray(h)

    u_adv = worst_case_input_from_a(a_np, uncertainty)
    w_adv = worst_case_state_disturbance(grad_np, uncertainty)
    state_penalty = np.sum(grad_np * w_adv, axis=-1)

    # Formula-side hdot is evaluated exactly at the analytic support point.
    formula_hdot = drift_np - np.sum(a_np * u_adv, axis=-1)

    robust_formula = formula_hdot - state_penalty + float(gamma) * h_np
    tol = float(slack_tolerance)

    summary = {
        "n_states": int(len(h_np)),
        "min_h": float(np.min(h_np)),
        "max_abs_boundary_h": float(np.max(np.abs(h_np))),
        "min_robust_formula_slack": float(np.min(robust_formula)),
        "formula_violation_fraction": float(np.mean(robust_formula < -tol)),
        "implemented_diagnostics_included": bool(include_implemented_diagnostics),
    }

    if include_implemented_diagnostics:
        def actual_one(x, u):
            return adapter.implemented_hdot(x, u)
        actual_formula_u = np.asarray(jax.jit(jax.vmap(actual_one))(X, jnp.asarray(u_adv)))
        u_attack, actual_attacked = _pgd_implemented_worst_input(
            adapter, np.asarray(states), u_adv, uncertainty,
            steps=implemented_attack_steps, restarts=implemented_attack_restarts,
            lr=implemented_attack_lr, seed=seed,
        )
        robust_actual_formula_u = actual_formula_u - state_penalty + float(gamma) * h_np
        robust_actual = actual_attacked - state_penalty + float(gamma) * h_np
        summary.update({
            "min_robust_implemented_slack_at_formula_input": float(np.min(robust_actual_formula_u)),
            "min_robust_implemented_slack": float(np.min(robust_actual)),
            "implemented_violation_fraction_at_formula_input": float(np.mean(robust_actual_formula_u < -tol)),
            "implemented_violation_fraction": float(np.mean(robust_actual < -tol)),
            "max_formula_vs_implemented_hdot_gap_at_formula_input": float(np.max(np.abs(formula_hdot - actual_formula_u))),
            "median_formula_vs_implemented_hdot_gap_at_formula_input": float(np.median(np.abs(formula_hdot - actual_formula_u))),
            "pgd_improvement_in_worst_hdot_max": float(np.max(actual_formula_u - actual_attacked)),
        })
    else:
        empty_inputs = np.empty((0, u_adv.shape[-1]), dtype=np.float32)
        empty_values = np.empty(0, dtype=np.float32)
        u_attack = empty_inputs
        actual_formula_u = empty_values
        actual_attacked = empty_values
        robust_actual_formula_u = empty_values
        robust_actual = empty_values

    return CertificateAudit(
        states=np.asarray(states),
        formula_worst_inputs=u_adv,
        implemented_attack_inputs=u_attack,
        worst_state_disturbances=w_adv,
        h=h_np,
        formula_hdot=formula_hdot,
        implemented_hdot_at_formula_input=actual_formula_u,
        implemented_hdot_attacked=actual_attacked,
        robust_formula_slack=robust_formula,
        robust_implemented_slack_at_formula_input=robust_actual_formula_u,
        robust_implemented_slack=robust_actual,
        summary=summary,
    )
