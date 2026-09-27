# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Thin, diagnostic-first adapter around the trained free-EBM pH model.

A central purpose of this module is to keep three notions separate:

1. ``H(x)``: the *actual* learned Hamiltonian/energy used in the paper.
2. the affine pH CBF derivative assembled by ``EBM_controller``;
3. the derivative of H along the *implemented* vector field, including the
   repository's numerical soft-clips and any optional nonlinear input gates.

Those expressions coincide in the ideal pH equations, but numerical safety
maps can make them differ.  A spotlight-grade certificate experiment should
measure that gap rather than silently assume it is zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import pathing  # noqa: F401  (sets import path before flat model imports)
import EBM_param_fields as pf
from EBM_class import energy_EBM
from EBM_param_fields import vector_field_and_output


@dataclass
class CompatibilityReport:
    certificate_affine_in_input: bool
    warnings: list[str]
    runtime_flags: dict[str, Any]


class EBMStressAdapter:
    """Expose energy, CBF and vector-field diagnostics with one stable API."""

    def __init__(self, params, layers: tuple, d: int, m: int, dt: float):
        self.params = params
        self.layers = layers
        self.d = int(d)
        self.m = int(m)
        self.dt = float(dt)

        # Mirror EBM_controller.make_energy_fns locally so importing this
        # post-hoc diagnostic package does not require Optax (a training-only
        # dependency of EBM_controller).  The formulas are intentionally kept
        # line-for-line equivalent to the controller implementation.
        def value_fn(x, weights, biases):
            return energy_EBM(x, weights, biases, layers)
        self.energy_value_fn = value_fn
        self.grad_energy_fn = jax.grad(value_fn, argnums=0)

        self._energy_jit = jax.jit(
            lambda x: self.energy_value_fn(
                x, self.params.ebm_weights, self.params.ebm_biases
            )
        )
        self._grad_jit = jax.jit(
            lambda x: self.grad_energy_fn(
                x, self.params.ebm_weights, self.params.ebm_biases
            )
        )
        self._vf_jit = jax.jit(
            lambda x, u: vector_field_and_output(
                self.params, x, u, self.grad_energy_fn, self.d, self.m
            )[0]
        )
        self._output_jit = jax.jit(
            lambda x, u: vector_field_and_output(
                self.params, x, u, self.grad_energy_fn, self.d, self.m
            )[2]
        )
        self._energy_batch_jit = jax.jit(jax.vmap(self._energy_jit))
        self._grad_batch_jit = jax.jit(jax.vmap(self._grad_jit))
        self._vf_batch_jit = jax.jit(jax.vmap(self._vf_jit))

    # ------------------------------------------------------------------
    # Hamiltonian / barrier
    # ------------------------------------------------------------------
    def energy(self, x) -> jax.Array:
        return self._energy_jit(jnp.asarray(x))

    def energy_batch(self, x) -> jax.Array:
        return self._energy_batch_jit(jnp.asarray(x))

    def grad_energy(self, x) -> jax.Array:
        """True autodiff gradient of H(x), before GRAD_E_CLIP."""
        return self._grad_jit(jnp.asarray(x))

    def grad_energy_batch(self, x) -> jax.Array:
        return self._grad_batch_jit(jnp.asarray(x))

    def barrier(self, x, epsilon: float) -> jax.Array:
        """Energy CBF h(x) = epsilon - H(x)."""
        return jnp.asarray(epsilon) - self.energy(x)

    def barrier_batch(self, x, epsilon: float) -> jax.Array:
        return jnp.asarray(epsilon) - self.energy_batch(x)

    # ------------------------------------------------------------------
    # CBF derivative used by the existing controller implementation
    # ------------------------------------------------------------------
    def affine_terms(self, x):
        """Return (a, drift) with hdot_formula = drift - a^T u."""
        x = jnp.asarray(x)
        tp = self.params.trunk
        raw_g = self.grad_energy_fn(x, self.params.ebm_weights, self.params.ebm_biases)
        g = pf._soft_clip_norm(pf._safe(raw_g), pf.GRAD_E_CLIP)
        rich_features = pf._field_features(tp, x)
        M = pf._assemble_M(
            tp.L_M, self.d, x=x, G_damp=tp.G_damp, h_damp=tp.h_damp,
            rich_features=rich_features, W_rich_M=tp.W_rich_M,
        )
        B = pf._assemble_input_matrix(tp, x, rich_features=rich_features)
        drift = pf.VF_SCALE * jnp.dot(g, M @ g)
        a = pf.VF_SCALE * (B.T @ g)
        return a, drift

    def theory_energy_terms(self, x):
        """Return raw-gradient terms used by the continuous-time theory.

        Unlike ``affine_terms``, this path does not soft-clip ``grad H``.
        It returns ``(a_H, d_H, grad_H, R, G)`` for
        ``hdot = d_H - a_H^T u`` under the ideal port-Hamiltonian model.
        """
        x = jnp.asarray(x)
        tp = self.params.trunk
        grad = self.grad_energy(x)
        rich_features = pf._field_features(tp, x)
        R = pf.VF_SCALE * pf._assemble_M(
            tp.L_M, self.d, x=x, G_damp=tp.G_damp, h_damp=tp.h_damp,
            rich_features=rich_features, W_rich_M=tp.W_rich_M,
        )
        G = pf.VF_SCALE * pf._assemble_input_matrix(
            tp, x, rich_features=rich_features
        )
        d_h = jnp.dot(grad, R @ grad)
        a_h = G.T @ grad
        return a_h, d_h, grad, R, G

    def controller_hdot(self, x, u) -> jax.Array:
        a, drift = self.affine_terms(x)
        return drift - jnp.dot(a, jnp.asarray(u))

    def controller_slack(self, x, u, epsilon: float, gamma: float) -> jax.Array:
        return self.controller_hdot(x, u) + jnp.asarray(gamma) * self.barrier(x, epsilon)

    # ------------------------------------------------------------------
    # Derivative along the code path that is actually integrated
    # ------------------------------------------------------------------
    def vector_field(self, x, u) -> jax.Array:
        return self._vf_jit(jnp.asarray(x), jnp.asarray(u))

    def vector_field_batch(self, x, u) -> jax.Array:
        return self._vf_batch_jit(jnp.asarray(x), jnp.asarray(u))

    def output(self, x, u=None) -> jax.Array:
        if u is None:
            u = jnp.zeros((self.m,))
        return self._output_jit(jnp.asarray(x), jnp.asarray(u))

    def implemented_hdot(self, x, u) -> jax.Array:
        """Compute dh/dt = -grad H(x)^T f_implemented(x,u).

        This is the derivative of the *declared barrier* along the vector field
        that EBM_rollout actually integrates.  It is the quantity to inspect
        whenever GRAD_E_CLIP/XDOT_CLIP/STATE_NORM_CLIP are not demonstrably
        inactive in the certified region.
        """
        g = self.grad_energy(x)
        f = self.vector_field(x, u)
        return -jnp.dot(g, f)

    def implemented_slack(self, x, u, epsilon: float, gamma: float) -> jax.Array:
        return self.implemented_hdot(x, u) + jnp.asarray(gamma) * self.barrier(x, epsilon)

    # ------------------------------------------------------------------
    # Numerics / compatibility audit
    # ------------------------------------------------------------------
    def compatibility_report(self) -> CompatibilityReport:
        flags = {
            "USE_INPUT_GAIN": bool(pf.USE_INPUT_GAIN),
            "USE_SATURATING_INPUT": bool(pf.USE_SATURATING_INPUT),
            "USE_STATE_DAMPING": bool(pf.USE_STATE_DAMPING),
            "USE_STATE_INTERCONNECTION": bool(pf.USE_STATE_INTERCONNECTION),
            "GRAD_E_CLIP": float(pf.GRAD_E_CLIP),
            "XDOT_CLIP": float(pf.XDOT_CLIP),
            "STATE_NORM_CLIP": float(pf.STATE_NORM_CLIP),
            "VF_SCALE": float(pf.VF_SCALE),
        }
        warnings: list[str] = []
        affine_ok = True

        if pf.USE_INPUT_GAIN:
            warnings.append(
                "USE_INPUT_GAIN=True: the input matrix is state dependent but remains "
                "affine in raw input; affine_terms evaluates the exact G(x)."
            )
        if pf.USE_SATURATING_INPUT:
            affine_ok = False
            warnings.append(
                "USE_SATURATING_INPUT=True: the implemented dynamics are nonlinear in "
                "the raw input, whereas the existing CBF affine terms use raw u."
            )

        warnings.append(
            "The controller formula uses the model's direction-preserving clipped "
            "energy gradient and omits the outer elementwise XDOT soft saturation. "
            "Use certificate_fidelity_audit to verify that these numerical safeguards "
            "are inactive/negligible throughout the claimed certified set."
        )
        return CompatibilityReport(affine_ok, warnings, flags)

    def certificate_fidelity(self, states, inputs, epsilon: float, gamma: float) -> dict[str, np.ndarray]:
        """Compare symbolic/affine CBF quantities to the implemented dynamics."""
        X = jnp.asarray(states)
        U = jnp.asarray(inputs)

        def one(x, u):
            h = self.barrier(x, epsilon)
            formula = self.controller_hdot(x, u)
            actual = self.implemented_hdot(x, u)
            return h, formula, actual, formula + gamma * h, actual + gamma * h

        h, hf, ha, sf, sa = jax.jit(jax.vmap(one))(X, U)
        hf_np, ha_np = np.asarray(hf), np.asarray(ha)
        abs_gap = np.abs(hf_np - ha_np)
        denom = np.maximum(np.abs(ha_np), 1e-8)
        return {
            "h": np.asarray(h),
            "hdot_formula": hf_np,
            "hdot_implemented": ha_np,
            "slack_formula": np.asarray(sf),
            "slack_implemented": np.asarray(sa),
            "abs_hdot_gap": abs_gap,
            "relative_hdot_gap": abs_gap / denom,
        }
