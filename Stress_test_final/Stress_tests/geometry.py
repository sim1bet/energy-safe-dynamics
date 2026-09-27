# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Geometry utilities for discovering Hamiltonian wells and choosing 2-D planes.

The key visualization principle is to separate:

* an affine *slice* H(c + U z), which is literal but can miss wells displaced
  in the hidden coordinates; and
* a *profile* min_w H(c + U z + V w), which is the lower envelope over the
  orthogonal complement.  The epsilon-sublevel set of the profile is exactly
  the planar projection of the epsilon-sublevel set, up to optimization error
  and the optional operational state-radius restriction.

That distinction should be stated explicitly in any paper caption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .config import ProjectionConfig
from .model_adapter import EBMStressAdapter


@dataclass
class MinimaResult:
    states: np.ndarray          # [n_wells, d]
    energies: np.ndarray        # [n_wells]
    multiplicities: np.ndarray  # number of starts converging to each well
    all_terminal_states: np.ndarray
    all_terminal_energies: np.ndarray


@dataclass
class Plane:
    center: np.ndarray          # [d]
    basis: np.ndarray           # [d, 2], orthonormal
    complement: np.ndarray      # [d, d-2], orthonormal
    method: str
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]

    def project(self, states: np.ndarray) -> np.ndarray:
        X = np.asarray(states)
        return (X - self.center) @ self.basis

    def lift(self, z: np.ndarray) -> np.ndarray:
        Z = np.asarray(z)
        return self.center + Z @ self.basis.T


def _project_rows_to_ball(x: jax.Array, radius: float) -> jax.Array:
    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, radius / (n + 1e-8))
    return x * scale


def discover_minima(
    adapter: EBMStressAdapter,
    reference_states: Optional[np.ndarray],
    cfg: ProjectionConfig,
    seed: int = 0,
) -> MinimaResult:
    """Multi-start Adam descent on the learned Hamiltonian.

    The search is bounded to the model's operational state ball by default.
    This is intentional: the rollout code itself softly restricts states to
    STATE_NORM_CLIP, so a beautiful well far outside that domain is not useful
    evidence about the dynamics actually being integrated.
    """
    rng = np.random.default_rng(seed)
    d = adapter.d
    n = int(cfg.n_minima_starts)
    radius = float(adapter.compatibility_report().runtime_flags["STATE_NORM_CLIP"])

    ref = None if reference_states is None else np.asarray(reference_states, dtype=np.float32)
    if ref is not None and len(ref) > 0:
        if ref.ndim != 2 or ref.shape[1] != d:
            raise ValueError(f"reference_states must have shape [N,{d}], got {ref.shape}")
        take = min(len(ref), max(n // 2, 1))
        idx = rng.choice(len(ref), size=take, replace=len(ref) < take)
        starts_ref = ref[idx]
        scale = np.std(ref, axis=0)
        scale = np.where(scale > 1e-5, scale, np.median(scale[scale > 1e-5]) if np.any(scale > 1e-5) else 1.0)
        center = np.median(ref, axis=0)
    else:
        take = 0
        starts_ref = np.empty((0, d), dtype=np.float32)
        scale = np.ones(d, dtype=np.float32)
        center = np.zeros(d, dtype=np.float32)

    n_rand = n - take
    starts_rand = center + rng.normal(size=(n_rand, d)).astype(np.float32) * scale[None, :] * 1.5
    starts = np.concatenate([starts_ref, starts_rand], axis=0)
    starts = np.asarray(_project_rows_to_ball(jnp.asarray(starts), radius))

    params = adapter.params
    value_fn = adapter.energy_value_fn

    def energy_one(x):
        return value_fn(x, params.ebm_weights, params.ebm_biases)

    grad_one = jax.grad(energy_one)
    grad_batch = jax.vmap(grad_one)
    energy_batch = jax.vmap(energy_one)

    beta1, beta2 = 0.9, 0.999
    lr = float(cfg.minima_lr)
    steps = int(cfg.minima_steps)

    @jax.jit
    def descend(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)

        def body(carry, t):
            x, m, v = carry
            g = jnp.nan_to_num(grad_batch(x), nan=0.0, posinf=0.0, neginf=0.0)
            # Gradient-norm limiting is only an optimizer safeguard.  It does
            # not change the energy being evaluated or any reported result.
            gn = jnp.linalg.norm(g, axis=-1, keepdims=True)
            g = g * jnp.minimum(1.0, 50.0 / (gn + 1e-8))
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            tt = t.astype(jnp.float32) + 1.0
            mh = m / (1.0 - beta1 ** tt)
            vh = v / (1.0 - beta2 ** tt)
            x = x - lr * mh / (jnp.sqrt(vh) + 1e-8)
            x = _project_rows_to_ball(x, radius)
            return (x, m, v), None

        (xf, _, _), _ = jax.lax.scan(body, (x0, m0, v0), jnp.arange(steps))
        return xf, energy_batch(xf)

    terminal, energies = descend(jnp.asarray(starts))
    terminal = np.asarray(terminal)
    energies = np.asarray(energies)

    # Greedy clustering in ascending energy order.  The tolerance is scaled by
    # the typical spread of observed states so it has similar meaning across
    # datasets whose latent coordinates have different numerical scales.
    order = np.argsort(energies)
    if ref is not None and len(ref) > 1:
        empirical_scale = float(np.sqrt(np.mean(np.var(ref, axis=0))))
        empirical_scale = max(empirical_scale, 1e-3)
    else:
        empirical_scale = max(float(np.sqrt(np.mean(np.var(terminal, axis=0)))), 1e-3)
    tol = float(cfg.minima_cluster_tol) * empirical_scale

    centers: list[np.ndarray] = []
    center_energies: list[float] = []
    counts: list[int] = []
    for idx in order:
        x = terminal[idx]
        e = float(energies[idx])
        if not centers:
            centers.append(x.copy()); center_energies.append(e); counts.append(1)
            continue
        dist = np.linalg.norm(np.stack(centers) - x[None, :], axis=1)
        k = int(np.argmin(dist))
        if dist[k] <= tol:
            counts[k] += 1
            # Keep the lowest-energy representative of the cluster.
            if e < center_energies[k]:
                centers[k] = x.copy(); center_energies[k] = e
        else:
            centers.append(x.copy()); center_energies.append(e); counts.append(1)

    centers_np = np.stack(centers)
    ce_np = np.asarray(center_energies)
    counts_np = np.asarray(counts, dtype=int)
    order2 = np.argsort(ce_np)
    limit = min(len(order2), int(cfg.max_wells_to_show))
    order2 = order2[:limit]

    return MinimaResult(
        states=centers_np[order2],
        energies=ce_np[order2],
        multiplicities=counts_np[order2],
        all_terminal_states=terminal,
        all_terminal_energies=energies,
    )


def filter_converged_wells(
    adapter: EBMStressAdapter, minima: MinimaResult, grad_norm_tol: float = 1.0,
) -> MinimaResult:
    """Discard clustered "wells" that are not converged stationary points.

    ``discover_minima`` clusters wherever its fixed-step multi-start descent
    terminates; a low energy value alone is not evidence of a critical point.
    On optimization-hostile landscapes some starts fail to converge and land
    far from any real equilibrium, yet can still have the lowest energy in
    the batch -- callers that rank candidates by energy alone (e.g. picking
    "the two lowest wells" for a saddle search) then silently treat a
    non-stationary artifact as a well. See
    ``duffing_doublewell_run_analysis.md`` (2026-08-19) for a concrete case:
    three such artifacts (|grad H| in [319, 341]) poisoned the safety epsilon
    by two orders of magnitude for one run. Always keeps at least the
    lowest-gradient-norm candidate so callers never receive an empty result.
    """
    if len(minima.states) == 0:
        return minima
    grad_norms = np.linalg.norm(np.asarray(adapter.grad_energy_batch(minima.states)), axis=-1)
    keep = grad_norms <= float(grad_norm_tol)
    if not np.any(keep):
        keep = np.zeros(len(grad_norms), dtype=bool)
        keep[int(np.argmin(grad_norms))] = True
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(minima.energies[idx])]
    return MinimaResult(
        states=minima.states[idx],
        energies=minima.energies[idx],
        multiplicities=minima.multiplicities[idx],
        all_terminal_states=minima.all_terminal_states,
        all_terminal_energies=minima.all_terminal_energies,
    )


def _orthonormalize_two(v1: np.ndarray, v2: np.ndarray, d: int) -> np.ndarray:
    e1 = np.asarray(v1, dtype=float)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.asarray(v2, dtype=float) - e1 * float(np.dot(e1, v2))
    if np.linalg.norm(e2) < 1e-10:
        # Deterministic fallback: choose the coordinate direction least aligned
        # with e1, then Gram-Schmidt it.
        k = int(np.argmin(np.abs(e1)))
        e2 = np.eye(d)[k] - e1 * e1[k]
    e2 /= max(np.linalg.norm(e2), 1e-12)
    return np.stack([e1, e2], axis=1)


def _pca_basis(points: np.ndarray, d: int) -> np.ndarray:
    X = np.asarray(points, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    if len(X) >= 2:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        if vh.shape[0] >= 2:
            return _orthonormalize_two(vh[0], vh[1], d)
    return np.eye(d)[:, :2]


def choose_plane(
    adapter: EBMStressAdapter,
    reference_states: Optional[np.ndarray],
    minima: MinimaResult,
    cfg: ProjectionConfig,
    boundary_states: Optional[np.ndarray] = None,
) -> Plane:
    """Choose an interpretable 2-D affine plane for the landscape figure."""
    d = adapter.d
    if d < 2:
        raise ValueError("Planar Hamiltonian visualization requires state dimension d >= 2.")

    wells = minima.states
    center = wells[0].copy() if len(wells) else np.zeros(d)
    ref = None if reference_states is None else np.asarray(reference_states)
    bnd = None if boundary_states is None else np.asarray(boundary_states)
    method = cfg.plane_method

    if method == "auto":
        # Multiple wells define the most scientifically interesting axis.  With
        # only one well, use the CBF boundary itself so the plane captures the
        # widest geometry of the certified set rather than the training cloud.
        method = "wells" if len(wells) >= 2 else ("safety_pca" if bnd is not None and len(bnd) >= 2 else "pca")

    if method == "wells" and len(wells) >= 2:
        if len(wells) >= 3 and np.linalg.matrix_rank(wells - wells.mean(axis=0)) >= 2:
            basis = _pca_basis(wells, d)
        else:
            e1 = wells[1] - wells[0]
            # Second axis is the dominant observed direction orthogonal to the
            # inter-well axis.  This produces a stable, interpretable view even
            # when exactly two wells are discovered.
            support_for_second = bnd if bnd is not None and len(bnd) >= 2 else ref
            if support_for_second is not None and len(support_for_second) >= 2:
                pca = _pca_basis(support_for_second, d)
                candidate = pca[:, 0]
                if abs(np.dot(candidate, e1) / (np.linalg.norm(e1) + 1e-12)) > 0.95:
                    candidate = pca[:, 1]
            else:
                candidate = np.eye(d)[int(np.argmin(np.abs(e1)))]
            basis = _orthonormalize_two(e1, candidate, d)
    elif method == "safety_pca":
        source = bnd if bnd is not None and len(bnd) >= 2 else ref
        if source is None or len(source) < 2:
            source = minima.all_terminal_states
        basis = _pca_basis(source, d)
    elif method == "readout":
        # First axis: strongest local output-sensitive direction at the lowest
        # discovered well.  For MIMO output this uses the leading right singular
        # vector of the output Jacobian.  Second axis: dominant safety/trajectory
        # direction after orthogonalization.
        J = np.asarray(jax.jacrev(lambda xx: adapter.output(xx))(jnp.asarray(center)))
        J = np.atleast_2d(J)
        _, _, vh = np.linalg.svd(J, full_matrices=False)
        e1 = vh[0] if vh.size else np.eye(d)[0]
        source = bnd if bnd is not None and len(bnd) >= 2 else ref
        if source is not None and len(source) >= 2:
            pca = _pca_basis(source, d)
            candidate = pca[:, 0]
            if abs(np.dot(candidate, e1) / (np.linalg.norm(e1) + 1e-12)) > 0.95:
                candidate = pca[:, 1]
        else:
            candidate = np.eye(d)[int(np.argmin(np.abs(e1)))]
        basis = _orthonormalize_two(e1, candidate, d)
    elif method == "hessian":
        Hxx = np.asarray(jax.hessian(lambda x: adapter.energy(x))(jnp.asarray(center)))
        vals, vecs = np.linalg.eigh(0.5 * (Hxx + Hxx.T))
        # Small |curvature| directions reveal flat/ring-like geometry; sorting
        # by magnitude is more informative than taking the stiffest axes.
        ii = np.argsort(np.abs(vals))[:2]
        basis = _orthonormalize_two(vecs[:, ii[0]], vecs[:, ii[1]], d)
    else:  # pca, or wells requested but only one well found
        if ref is not None and len(ref) >= 2:
            basis = _pca_basis(ref, d)
        elif len(minima.all_terminal_states) >= 2:
            basis = _pca_basis(minima.all_terminal_states, d)
        else:
            basis = np.eye(d)[:, :2]
        method = "pca" if cfg.plane_method != "hessian" else method

    # Complete orthonormal basis.  QR is robust even when d==2 (empty V).
    q, _ = np.linalg.qr(basis, mode="complete")
    # QR may flip signs; re-align the first two columns with the selected basis
    # for deterministic orientation in repeated plots.
    for j in range(2):
        if np.dot(q[:, j], basis[:, j]) < 0:
            q[:, j] *= -1
    basis = q[:, :2]
    complement = q[:, 2:]

    support = []
    if ref is not None and len(ref): support.append(ref)
    if bnd is not None and len(bnd): support.append(bnd)
    if len(wells): support.append(wells)
    points = np.concatenate(support, axis=0) if support else center[None, :]
    z = (points - center) @ basis

    qlo = float(cfg.extent_quantile_low)
    qhi = float(cfg.extent_quantile_high)
    lo = np.quantile(z, qlo, axis=0)
    hi = np.quantile(z, qhi, axis=0)
    span = np.maximum(hi - lo, 0.5)
    pad = float(cfg.padding_fraction) * span
    lo -= pad; hi += pad
    # Every discovered well is semantically important and must remain visible
    # even when it is an outlier relative to the visited-state cloud.
    if len(wells):
        zw = (wells - center) @ basis
        lo = np.minimum(lo, np.min(zw, axis=0) - 0.05 * span)
        hi = np.maximum(hi, np.max(zw, axis=0) + 0.05 * span)
    # Always keep the anchor/global minimum visible.
    lo = np.minimum(lo, -0.05 * span)
    hi = np.maximum(hi, 0.05 * span)

    return Plane(
        center=center,
        basis=basis,
        complement=complement,
        method=method,
        x_limits=(float(lo[0]), float(hi[0])),
        y_limits=(float(lo[1]), float(hi[1])),
    )


def make_plane_grid(plane: Plane, grid_size: int):
    gx = np.linspace(*plane.x_limits, int(grid_size))
    gy = np.linspace(*plane.y_limits, int(grid_size))
    xx, yy = np.meshgrid(gx, gy, indexing="xy")
    z = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return gx, gy, xx, yy, z


def slice_energy_grid(adapter: EBMStressAdapter, plane: Plane, cfg: ProjectionConfig) -> dict:
    gx, gy, xx, yy, z = make_plane_grid(plane, cfg.grid_size)
    x = plane.lift(z)
    e = np.asarray(adapter.energy_batch(jnp.asarray(x))).reshape(xx.shape)
    return {"gx": gx, "gy": gy, "xx": xx, "yy": yy, "energy": e}


def profile_energy_grid(
    adapter: EBMStressAdapter,
    plane: Plane,
    minima: MinimaResult,
    cfg: ProjectionConfig,
) -> dict:
    """Compute H_tilde(z)=min_w H(c+Uz+Vw) on a planar grid.

    Optimization is batched and warm-started from the orthogonal coordinates
    of discovered wells.  This is much more reliable than always starting at
    w=0 when the Hamiltonian has multiple disconnected wells.
    """
    gx, gy, xx, yy, z_all = make_plane_grid(plane, cfg.grid_size)
    k = plane.complement.shape[1]
    if k == 0:
        out = slice_energy_grid(adapter, plane, cfg)
        out["argmin_states"] = plane.lift(z_all).reshape(xx.shape + (adapter.d,))
        out["profile_stationarity"] = np.zeros(xx.shape, dtype=np.float32)
        return out

    params = adapter.params
    value_fn = adapter.energy_value_fn
    c = jnp.asarray(plane.center)
    U = jnp.asarray(plane.basis)
    V = jnp.asarray(plane.complement)
    radius = cfg.profile_state_radius
    radius = None if radius is None else float(radius)

    def energy_one(x):
        return value_fn(x, params.ebm_weights, params.ebm_biases)
    grad_one = jax.grad(energy_one)
    grad_batch = jax.vmap(grad_one)
    energy_batch = jax.vmap(energy_one)

    beta1, beta2 = 0.9, 0.999
    lr = float(cfg.profile_lr)
    steps = int(cfg.profile_steps)

    @jax.jit
    def optimize(z, w0):
        m0 = jnp.zeros_like(w0)
        v0 = jnp.zeros_like(w0)

        def body(carry, t):
            w, m, v = carry
            x = c + z @ U.T + w @ V.T
            gx_full = jnp.nan_to_num(grad_batch(x), nan=0.0, posinf=0.0, neginf=0.0)
            gw = gx_full @ V
            gn = jnp.linalg.norm(gw, axis=-1, keepdims=True)
            gw = gw * jnp.minimum(1.0, 50.0 / (gn + 1e-8))
            m = beta1 * m + (1.0 - beta1) * gw
            v = beta2 * v + (1.0 - beta2) * (gw * gw)
            tt = t.astype(jnp.float32) + 1.0
            mh = m / (1.0 - beta1 ** tt)
            vh = v / (1.0 - beta2 ** tt)
            w = w - lr * mh / (jnp.sqrt(vh) + 1e-8)

            # Optional theorem-domain restriction.  With radius=None the
            # minimization is unconstrained, so {z: H_tilde(z)<=epsilon} is the
            # genuine planar projection of the full energy sublevel set.
            if radius is not None:
                def shrink(_, ww):
                    xx = c + z @ U.T + ww @ V.T
                    n = jnp.linalg.norm(xx, axis=-1, keepdims=True)
                    fac = jnp.minimum(1.0, radius / (n + 1e-8))
                    return ww * fac
                w = jax.lax.fori_loop(0, 3, shrink, w)
            return (w, m, v), None

        (wf, _, _), _ = jax.lax.scan(body, (w0, m0, v0), jnp.arange(steps))
        xf = c + z @ U.T + wf @ V.T
        ef = energy_batch(xf)
        return wf, xf, ef

    # Candidate warm starts: zero plus discovered wells' complement coordinates.
    seeds = [np.zeros(k, dtype=np.float32)]
    if len(minima.states):
        wk = (minima.states - plane.center) @ plane.complement
        for row in wk:
            if len(seeds) >= int(cfg.profile_restarts):
                break
            seeds.append(np.asarray(row, dtype=np.float32))
    while len(seeds) < int(cfg.profile_restarts):
        seeds.append(seeds[-1].copy())
    seeds = np.stack(seeds[: int(cfg.profile_restarts)])

    best_e = np.empty(len(z_all), dtype=np.float32)
    best_x = np.empty((len(z_all), adapter.d), dtype=np.float32)
    chunk = int(cfg.profile_chunk_size)

    for start in range(0, len(z_all), chunk):
        z_np = z_all[start : start + chunk].astype(np.float32)
        b = len(z_np)
        # Flatten restart and point axes so all restarts share one compiled call.
        z_rep = np.repeat(z_np[None, :, :], len(seeds), axis=0).reshape(-1, 2)
        w_rep = np.repeat(seeds[:, None, :], b, axis=1).reshape(-1, k)
        _, xf, ef = optimize(jnp.asarray(z_rep), jnp.asarray(w_rep))
        ef = np.asarray(ef).reshape(len(seeds), b)
        xf = np.asarray(xf).reshape(len(seeds), b, adapter.d)
        pick = np.argmin(ef, axis=0)
        cols = np.arange(b)
        best_e[start : start + b] = ef[pick, cols]
        best_x[start : start + b] = xf[pick, cols]

    # First-order optimality diagnostic in the marginalized directions.  A
    # profiled figure is only trustworthy where V^T grad H is small; retain the
    # residual so final-paper settings can be tightened until convergence.
    gbest = np.asarray(adapter.grad_energy_batch(jnp.asarray(best_x)))
    stationarity = np.linalg.norm(gbest @ plane.complement, axis=1)

    return {
        "gx": gx,
        "gy": gy,
        "xx": xx,
        "yy": yy,
        "energy": best_e.reshape(xx.shape),
        "argmin_states": best_x.reshape(xx.shape + (adapter.d,)),
        "profile_stationarity": stationarity.reshape(xx.shape),
    }
