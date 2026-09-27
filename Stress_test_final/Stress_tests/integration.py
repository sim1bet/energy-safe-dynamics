# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Small helpers for wiring a trained model into :mod:`Stress_tests`.

These functions deliberately avoid importing EBM_rollout because that module
imports Optax for iterative x0 refinement.  Post-hoc stress testing only needs a
forward simulator, so the helpers below remain evaluation-only and lightweight.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import pathing  # noqa: F401
import EBM_param_fields as pf
from EBM_param_fields import encode_x0

from .model_adapter import EBMStressAdapter


def encode_x0_from_burnin(params, y_window, u_window=None):
    """Use the repository's trained x0 encoder on a burn-in window.

    If ``USE_INPUT_AWARE_ENCODER`` was enabled during training, ``u_window`` is
    required and concatenated exactly as in EBM_rollout._encoder_window.
    """
    y = jnp.asarray(y_window)
    feature_width = int(y.shape[-1]) + (int(params.trunk.B.shape[1]) if pf.USE_INPUT_AWARE_ENCODER else 0)
    required = int(params.trunk.W_enc.shape[1]) // max(feature_width, 1)
    if y.shape[0] < required:
        raise ValueError(f"burn-in has {y.shape[0]} steps but encoder requires {required}")
    y = y[-required:]
    if pf.USE_INPUT_AWARE_ENCODER:
        if u_window is None:
            raise ValueError("u_window is required because USE_INPUT_AWARE_ENCODER=True")
        u = jnp.asarray(u_window)[-required:]
        y = jnp.concatenate([y, u], axis=-1)
    return np.asarray(encode_x0(params, y))



def apply_runtime_config(config: dict) -> None:
    """Restore mutable EBM_param_fields globals when evaluating in a new process.

    ``run_training`` writes several model choices into module-level globals.  If
    stress testing is launched in a separate Python process after loading a
    checkpoint, call this helper *before* building EBMStressAdapter so the
    evaluated vector field matches the one used during training.
    """
    mapping = {
        "chol_clip_exp": "CHOL_CLIP_EXP",
        "damping_scale": "DAMPING_SCALE",
        "b_init_scale": "B_INIT_SCALE",
        "vf_scale": "VF_SCALE",
        "use_feedthrough": "USE_FEEDTHROUGH",
        "use_identity_readout": "USE_IDENTITY_READOUT",
        "use_nonlinear_readout": "USE_NONLINEAR_READOUT",
        "readout_hidden": "READOUT_HIDDEN",
        "use_state_damping": "USE_STATE_DAMPING",
        "use_state_interconnection": "USE_STATE_INTERCONNECTION",
        "use_quadratic_interconnection": "USE_QUADRATIC_INTERCONNECTION",
        "use_cubic_interconnection": "USE_CUBIC_INTERCONNECTION",
        "use_saturating_readout": "USE_SATURATING_READOUT",
        "sat_scale_init": "SAT_SCALE_INIT",
        "sat_scale_min": "SAT_SCALE_MIN",
        "use_input_gain": "USE_INPUT_GAIN",
        "input_gain_mode": "INPUT_GAIN_MODE",
        "use_rich_interconnection": "USE_RICH_INTERCONNECTION",
        "use_rich_damping": "USE_RICH_DAMPING",
        "use_rich_input_matrix": "USE_RICH_INPUT_MATRIX",
        "ph_field_width": "PH_FIELD_WIDTH",
        "input_matrix_structure": "INPUT_MATRIX_STRUCTURE",
        "use_saturating_input": "USE_SATURATING_INPUT",
        "sat_u_init": "SAT_U_INIT",
        "sat_u_min": "SAT_U_MIN",
        "use_input_aware_encoder": "USE_INPUT_AWARE_ENCODER",
        "grad_e_clip": "GRAD_E_CLIP",
        "xdot_clip": "XDOT_CLIP",
        "x0_norm_max": "X0_NORM_MAX",
        "state_norm_clip": "STATE_NORM_CLIP",
    }
    for key, attr in mapping.items():
        if key in config:
            setattr(pf, attr, config[key])

def collect_reference_rollout(
    adapter: EBMStressAdapter,
    x0,
    u_traj,
    integrator: str = "rk4",
) -> np.ndarray:
    """Roll out the implemented model and return latent states [T,d]."""
    dt = float(adapter.dt)
    # Preserve the checkpoint/state precision.  HNER99 enables x64 for its
    # numerical shell reconstruction, whereas standard model validation uses
    # float32; lax.scan requires both carry and step output to agree.
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.asarray(x0).dtype
    U = jnp.asarray(u_traj, dtype=dtype)

    def step(x, u):
        x = pf._soft_clip_norm(pf._safe(x), pf.STATE_NORM_CLIP)
        def f(xx): return adapter.vector_field(xx, u)
        if integrator == "euler":
            raw = x + dt * f(x)
        elif integrator == "rk4":
            k1 = f(x)
            k2 = f(pf._soft_clip_norm(pf._safe(x + 0.5*dt*k1), pf.STATE_NORM_CLIP))
            k3 = f(pf._soft_clip_norm(pf._safe(x + 0.5*dt*k2), pf.STATE_NORM_CLIP))
            k4 = f(pf._soft_clip_norm(pf._safe(x + dt*k3), pf.STATE_NORM_CLIP))
            raw = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError("integrator must be 'euler' or 'rk4'")
        xn = pf._soft_clip_norm(pf._safe(raw), pf.STATE_NORM_CLIP)
        return xn, xn

    _, X = jax.jit(lambda x: jax.lax.scan(step, x, U))(jnp.asarray(x0, dtype=dtype))
    return np.asarray(X)
