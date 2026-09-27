# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Sampling of the energy-CBF boundary H(x)=epsilon.

The routine traces the *first* outward crossing from discovered wells.  This is
preferable to drawing random Gaussian states and filtering by |H-epsilon|,
which badly undersamples narrow/nonconvex parts of a high-dimensional level
set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from .config import StressTestConfig
from .geometry import MinimaResult
from .model_adapter import EBMStressAdapter


@dataclass
class BoundaryResult:
    states: np.ndarray
    anchor_indices: np.ndarray
    directions: np.ndarray
    radii: np.ndarray
    energies: np.ndarray
    barrier_residuals: np.ndarray
    failed_fraction: float


def project_to_energy_level(
    adapter: EBMStressAdapter,
    states: np.ndarray,
    epsilon: float,
    steps: int = 6,
    max_step_norm: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply damped normal corrections to enforce ``H(x)=epsilon``."""
    corrected = np.asarray(states, dtype=np.float32).copy()
    for _ in range(max(int(steps), 0)):
        energy = np.asarray(adapter.energy_batch(jnp.asarray(corrected)))
        residual = float(epsilon) - energy
        grad = np.asarray(adapter.grad_energy_batch(jnp.asarray(corrected)))
        delta = residual[:, None] * grad / np.maximum(
            np.sum(grad * grad, axis=1, keepdims=True), 1e-12
        )
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        delta *= np.minimum(1.0, float(max_step_norm) / np.maximum(norm, 1e-12))
        corrected += delta.astype(np.float32)
    energy = np.asarray(adapter.energy_batch(jnp.asarray(corrected)))
    return corrected, float(epsilon) - energy


def sample_energy_boundary(
    adapter: EBMStressAdapter,
    minima: MinimaResult,
    epsilon: float,
    cfg: StressTestConfig,
) -> BoundaryResult:
    rng = np.random.default_rng(cfg.seed + 101)
    valid = np.flatnonzero(minima.energies < float(epsilon))
    if len(valid) == 0:
        raise ValueError(
            "No discovered Hamiltonian well lies below epsilon; cannot trace the "
            "safe-set boundary. Re-check epsilon or the minima search."
        )

    total = int(cfg.n_boundary_directions)
    per = int(np.ceil(total / len(valid)))
    anchors = np.repeat(minima.states[valid], per, axis=0)[:total]
    anchor_ids = np.repeat(valid, per)[:total]

    dirs = rng.normal(size=(total, adapter.d)).astype(np.float32)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)

    lo = np.zeros(total, dtype=np.float32)
    hi = np.full(total, float(cfg.boundary_initial_radius), dtype=np.float32)
    crossed = np.zeros(total, dtype=bool)
    max_r = float(cfg.boundary_max_radius)

    # Vectorized exponential bracketing.  A ray may never cross epsilon (for
    # example if it leaves the operational region before that level); those
    # rays are reported instead of silently replaced with arbitrary points.
    for _ in range(32):
        xh = anchors + hi[:, None] * dirs
        eh = np.asarray(adapter.energy_batch(jnp.asarray(xh)))
        newly = eh >= float(epsilon)
        crossed |= newly
        if np.all(crossed | (hi >= max_r)):
            break
        hi = np.where(crossed, hi, np.minimum(2.0 * hi, max_r))

    keep = crossed
    if not np.any(keep):
        raise RuntimeError(
            "Boundary tracing failed on every ray. epsilon may exceed all energy "
            "values in the search radius, or the Hamiltonian may not be radially "
            "increasing over the operational region."
        )

    a = anchors[keep]
    d = dirs[keep]
    l = lo[keep]
    r = hi[keep]
    ids = anchor_ids[keep]

    for _ in range(int(cfg.boundary_bisection_steps)):
        mid = 0.5 * (l + r)
        xm = a + mid[:, None] * d
        em = np.asarray(adapter.energy_batch(jnp.asarray(xm)))
        outside = em >= float(epsilon)
        r = np.where(outside, mid, r)
        l = np.where(outside, l, mid)

    radii = 0.5 * (l + r)
    xb = a + radii[:, None] * d
    xb, residual = project_to_energy_level(adapter, xb, float(epsilon))
    eb = float(epsilon) - residual

    return BoundaryResult(
        states=xb,
        anchor_indices=ids,
        directions=d,
        radii=radii,
        energies=eb,
        barrier_residuals=residual,
        failed_fraction=float(1.0 - np.mean(keep)),
    )


def inset_boundary_points(boundary: BoundaryResult, minima: MinimaResult, fraction: float) -> np.ndarray:
    """Move boundary points slightly toward their source wells.

    ``fraction`` is geometric rather than an energy offset.  The actual h(x)
    values should always be recomputed and reported because nonconvex H makes a
    fixed geometric inset correspond to different energy margins around the
    boundary.
    """
    frac = float(np.clip(fraction, 0.0, 0.5))
    anchors = minima.states[boundary.anchor_indices]
    return boundary.states + frac * (anchors - boundary.states)
