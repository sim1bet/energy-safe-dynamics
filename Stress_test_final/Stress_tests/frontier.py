# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Analytic robustness frontier for the affine pH energy-CBF certificate.

For h=epsilon-H and the controller-side pH formula

    hdot = drift(x) - a(x)^T u,

support functions make the worst-case boundary slack explicit for norm-bounded
inputs and additive state disturbances.  This module turns that structure into
a two-parameter phase diagram: how much can the claimed input set and
perturbation set be scaled before the worst boundary point loses CBF margin?
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .certificate import _center, _linf_radius
from .config import UncertaintySet
from .model_adapter import EBMStressAdapter


def compute_robustness_frontier(
    adapter: EBMStressAdapter,
    boundary_states: np.ndarray,
    epsilon: float,
    gamma: float,
    uncertainty: UncertaintySet,
    grid_size: int = 81,
    max_input_scale: float = 2.0,
    max_disturbance_scale: float = 2.0,
) -> dict:
    """Return worst robust CBF slack over the sampled boundary on a scale grid.

    Scale 1.0 on either axis means exactly the corresponding set in
    ``uncertainty``.  This parameterization also supports anisotropic Linf input
    boxes because the entire vector of per-channel radii is scaled together.
    """
    X = jnp.asarray(boundary_states)

    def one(x):
        a, drift = adapter.affine_terms(x)
        g = adapter.grad_energy(x)
        h = adapter.barrier(x, epsilon)
        return a, drift, g, h

    a, drift, g, h = jax.jit(jax.vmap(one))(X)
    a = np.asarray(a); drift = np.asarray(drift); g = np.asarray(g); h = np.asarray(h)
    c = _center(uncertainty, adapter.m)
    base = drift - np.sum(a * c[None, :], axis=1) + float(gamma) * h

    if uncertainty.input_norm == "linf":
        r = _linf_radius(uncertainty, adapter.m)
        input_penalty = np.sum(np.abs(a) * r[None, :], axis=1)
    else:
        r = float(np.asarray(uncertainty.input_radius))
        input_penalty = r * np.linalg.norm(a, axis=1)

    dr = float(uncertainty.state_disturbance_radius)
    if dr <= 0:
        state_penalty = np.zeros(len(X), dtype=np.float32)
    elif uncertainty.state_disturbance_norm == "linf":
        # Support function of an Linf ball is radius * ||g||_1.
        state_penalty = dr * np.sum(np.abs(g), axis=1)
    else:
        state_penalty = dr * np.linalg.norm(g, axis=1)

    si = np.linspace(0.0, float(max_input_scale), int(grid_size))
    sd = np.linspace(0.0, float(max_disturbance_scale), int(grid_size))
    # [D,I,N] -> min_N.  Vectorized NumPy is inexpensive even for ~1000
    # boundary states and a 101x101 frontier grid.
    slack_all = (
        base[None, None, :]
        - si[None, :, None] * input_penalty[None, None, :]
        - sd[:, None, None] * state_penalty[None, None, :]
    )
    worst = np.min(slack_all, axis=-1)
    critical = np.argmin(slack_all, axis=-1)

    # Nearest grid cell to the nominal (1,1) claimed set.
    ii = int(np.argmin(np.abs(si - 1.0)))
    jj = int(np.argmin(np.abs(sd - 1.0)))
    return {
        "input_scales": si,
        "disturbance_scales": sd,
        "worst_slack": worst,
        "critical_boundary_index": critical,
        "nominal_grid_slack": float(worst[jj, ii]),
        "nominal_input_scale_grid": float(si[ii]),
        "nominal_disturbance_scale_grid": float(sd[jj]),
        "base_slack": base,
        "input_penalty_at_scale1": input_penalty,
        "disturbance_penalty_at_scale1": state_penalty,
    }
