# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Long-horizon adversarial and randomized stress trajectories.

These are *empirical attacks*, not replacements for the certificate.  Their
purpose is to search aggressively for counterexamples, diagnose numerical
integration effects, and produce compelling trajectory overlays for figures.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import pathing  # noqa: F401
import EBM_param_fields as pf

from .certificate import worst_case_input_from_a, worst_case_state_disturbance
from .config import UncertaintySet
from .model_adapter import EBMStressAdapter


@dataclass
class RolloutResult:
    states: np.ndarray
    inputs: np.ndarray
    disturbances: np.ndarray
    h: np.ndarray
    formula_slack: np.ndarray
    implemented_slack: np.ndarray
    state_clip_displacement: np.ndarray
    min_h: float
    violation: bool


def _single_adversarial_input(a: jax.Array, spec: UncertaintySet, m: int) -> jax.Array:
    center = jnp.zeros((m,)) if spec.input_center is None else jnp.asarray(spec.input_center)
    if spec.input_norm == "linf":
        r = jnp.asarray(spec.input_radius)
        if r.ndim == 0:
            r = jnp.full((m,), r)
        return center + jnp.sign(a) * r
    r = jnp.asarray(spec.input_radius)
    return center + r * a / (jnp.linalg.norm(a) + 1e-8)


def _single_adversarial_disturbance(g: jax.Array, spec: UncertaintySet) -> jax.Array:
    r = float(spec.state_disturbance_radius)
    if r <= 0:
        return jnp.zeros_like(g)
    if spec.state_disturbance_norm == "linf":
        return r * jnp.sign(g)
    return r * g / (jnp.linalg.norm(g) + 1e-8)


def adversarial_rollout(
    adapter: EBMStressAdapter,
    x0: np.ndarray,
    epsilon: float,
    gamma: float,
    uncertainty: UncertaintySet,
    horizon_steps: int,
    dt: float | None = None,
    integrator: str = "rk4",
    barrier_tolerance: float = 1e-5,
) -> RolloutResult:
    """Greedy state-dependent worst-case forcing at every integration step.

    At x_k the input maximizes the affine CBF term a(x_k)^T u over the
    admissible input set and the additive disturbance aligns with grad H.  This
    is the exact pointwise adversary for the ideal affine certificate.  It is a
    strong empirical attack, though not a globally optimal finite-horizon
    adversary.
    """
    step_dt = float(adapter.dt if dt is None else dt)
    m = adapter.m

    def one_step(x, _):
        a, _ = adapter.affine_terms(x)
        u = _single_adversarial_input(a, uncertainty, m)
        g = adapter.grad_energy(x)
        w = _single_adversarial_disturbance(g, uncertainty)
        h0 = adapter.barrier(x, epsilon)
        sf = adapter.controller_hdot(x, u) - jnp.dot(g, w) + gamma * h0
        sa = adapter.implemented_hdot(x, u) - jnp.dot(g, w) + gamma * h0

        def f(xx):
            return adapter.vector_field(xx, u) + w

        if integrator == "euler":
            raw_next = x + step_dt * f(x)
        elif integrator == "rk4":
            k1 = f(x)
            k2 = f(pf._soft_clip_norm(pf._safe(x + 0.5 * step_dt * k1), pf.STATE_NORM_CLIP))
            k3 = f(pf._soft_clip_norm(pf._safe(x + 0.5 * step_dt * k2), pf.STATE_NORM_CLIP))
            k4 = f(pf._soft_clip_norm(pf._safe(x + step_dt * k3), pf.STATE_NORM_CLIP))
            raw_next = x + (step_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError("integrator must be 'euler' or 'rk4'")

        x_next = pf._soft_clip_norm(pf._safe(raw_next), pf.STATE_NORM_CLIP)
        clip_disp = jnp.linalg.norm(raw_next - x_next)
        h_next = adapter.barrier(x_next, epsilon)
        return x_next, (x_next, u, w, h_next, sf, sa, clip_disp)

    scan = jax.jit(lambda x: jax.lax.scan(one_step, x, xs=None, length=int(horizon_steps)))
    _, (X, U, W, h, sf, sa, cd) = scan(jnp.asarray(x0, dtype=jnp.float32))
    h_np = np.asarray(h)
    return RolloutResult(
        states=np.asarray(X),
        inputs=np.asarray(U),
        disturbances=np.asarray(W),
        h=h_np,
        formula_slack=np.asarray(sf),
        implemented_slack=np.asarray(sa),
        state_clip_displacement=np.asarray(cd),
        min_h=float(np.min(h_np)),
        violation=bool(np.min(h_np) < -float(barrier_tolerance)),
    )


def random_piecewise_input(
    rng: np.random.Generator,
    horizon_steps: int,
    m: int,
    spec: UncertaintySet,
    hold: int,
) -> np.ndarray:
    """Draw a high-amplitude piecewise-constant signal from the admissible set."""
    nblocks = int(np.ceil(horizon_steps / max(int(hold), 1)))
    center = np.zeros(m, dtype=np.float32) if spec.input_center is None else np.asarray(spec.input_center, dtype=np.float32)
    if spec.input_norm == "linf":
        r = np.asarray(spec.input_radius, dtype=np.float32)
        if r.ndim == 0: r = np.full(m, float(r), dtype=np.float32)
        # Bias draws toward the boundary of the input box; uniform interior
        # tests are usually too gentle to be useful stress tests.
        signs = rng.choice([-1.0, 1.0], size=(nblocks, m))
        mag = rng.uniform(0.75, 1.0, size=(nblocks, m))
        blocks = center + signs * mag * r
    else:
        r = float(np.asarray(spec.input_radius))
        v = rng.normal(size=(nblocks, m))
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        mag = rng.uniform(0.75, 1.0, size=(nblocks, 1))
        blocks = center + r * mag * v
    return np.repeat(blocks, max(int(hold), 1), axis=0)[:horizon_steps].astype(np.float32)

@dataclass
class BatchRolloutResult:
    states: np.ndarray                 # [B,T,d]
    inputs: np.ndarray                 # [B,T,m]
    disturbances: np.ndarray           # [B,T,d]
    h: np.ndarray                      # [B,T]
    formula_slack: np.ndarray          # [B,T]
    implemented_slack: np.ndarray      # [B,T]
    state_clip_displacement: np.ndarray# [B,T]
    min_h_per_rollout: np.ndarray      # [B]
    violation_per_rollout: np.ndarray  # [B]


def adversarial_rollout_batch(
    adapter: EBMStressAdapter,
    x0_batch: np.ndarray,
    epsilon: float,
    gamma: float,
    uncertainty: UncertaintySet,
    horizon_steps: int,
    dt: float | None = None,
    integrator: str = "rk4",
    barrier_tolerance: float = 1e-5,
) -> BatchRolloutResult:
    """Vectorized version of :func:`adversarial_rollout` for many x0."""
    step_dt = float(adapter.dt if dt is None else dt)
    m = adapter.m

    def scan_one(x0):
        def one_step(x, _):
            a, _ = adapter.affine_terms(x)
            u = _single_adversarial_input(a, uncertainty, m)
            g = adapter.grad_energy(x)
            w = _single_adversarial_disturbance(g, uncertainty)
            h0 = adapter.barrier(x, epsilon)
            sf = adapter.controller_hdot(x, u) - jnp.dot(g, w) + gamma * h0
            sa = adapter.implemented_hdot(x, u) - jnp.dot(g, w) + gamma * h0

            def f(xx):
                return adapter.vector_field(xx, u) + w

            if integrator == "euler":
                raw_next = x + step_dt * f(x)
            elif integrator == "rk4":
                k1 = f(x)
                k2 = f(pf._soft_clip_norm(pf._safe(x + 0.5 * step_dt * k1), pf.STATE_NORM_CLIP))
                k3 = f(pf._soft_clip_norm(pf._safe(x + 0.5 * step_dt * k2), pf.STATE_NORM_CLIP))
                k4 = f(pf._soft_clip_norm(pf._safe(x + step_dt * k3), pf.STATE_NORM_CLIP))
                raw_next = x + (step_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError("integrator must be 'euler' or 'rk4'")
            x_next = pf._soft_clip_norm(pf._safe(raw_next), pf.STATE_NORM_CLIP)
            cd = jnp.linalg.norm(raw_next - x_next)
            h_next = adapter.barrier(x_next, epsilon)
            return x_next, (x_next, u, w, h_next, sf, sa, cd)
        _, out = jax.lax.scan(one_step, x0, xs=None, length=int(horizon_steps))
        return out

    X0 = jnp.asarray(x0_batch, dtype=jnp.float32)
    X, U, W, h, sf, sa, cd = jax.jit(jax.vmap(scan_one))(X0)
    h_np = np.asarray(h)
    mins = np.min(h_np, axis=1)
    return BatchRolloutResult(
        states=np.asarray(X), inputs=np.asarray(U), disturbances=np.asarray(W),
        h=h_np, formula_slack=np.asarray(sf), implemented_slack=np.asarray(sa),
        state_clip_displacement=np.asarray(cd), min_h_per_rollout=mins,
        violation_per_rollout=mins < -float(barrier_tolerance),
    )

def forced_rollout_batch(
    adapter: EBMStressAdapter,
    x0_batch: np.ndarray,
    input_batch: np.ndarray,
    disturbance_batch: np.ndarray | None,
    epsilon: float,
    dt: float | None = None,
    integrator: str = "rk4",
    barrier_tolerance: float = 1e-5,
) -> dict:
    """Integrate prescribed input/disturbance sequences for Monte-Carlo stress tests."""
    step_dt = float(adapter.dt if dt is None else dt)
    X0 = jnp.asarray(x0_batch, dtype=jnp.float32)
    U = jnp.asarray(input_batch, dtype=jnp.float32)
    if disturbance_batch is None:
        W = jnp.zeros((U.shape[0], U.shape[1], adapter.d), dtype=jnp.float32)
    else:
        W = jnp.asarray(disturbance_batch, dtype=jnp.float32)

    def one_rollout(x0, u_seq, w_seq):
        def step(x, uw):
            u, w = uw
            def f(xx): return adapter.vector_field(xx, u) + w
            if integrator == "euler":
                raw_next = x + step_dt * f(x)
            else:
                k1 = f(x)
                k2 = f(pf._soft_clip_norm(pf._safe(x + 0.5*step_dt*k1), pf.STATE_NORM_CLIP))
                k3 = f(pf._soft_clip_norm(pf._safe(x + 0.5*step_dt*k2), pf.STATE_NORM_CLIP))
                k4 = f(pf._soft_clip_norm(pf._safe(x + step_dt*k3), pf.STATE_NORM_CLIP))
                raw_next = x + (step_dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            x_next = pf._soft_clip_norm(pf._safe(raw_next), pf.STATE_NORM_CLIP)
            return x_next, (x_next, adapter.barrier(x_next, epsilon), jnp.linalg.norm(raw_next-x_next))
        _, out = jax.lax.scan(step, x0, (u_seq, w_seq))
        return out

    X, h, cd = jax.jit(jax.vmap(one_rollout))(X0, U, W)
    h_np = np.asarray(h)
    mins = np.min(h_np, axis=1)
    return {
        "states": np.asarray(X),
        "h": h_np,
        "state_clip_displacement": np.asarray(cd),
        "min_h_per_rollout": mins,
        "violation_per_rollout": mins < -float(barrier_tolerance),
    }
