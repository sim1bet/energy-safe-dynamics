# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Application-agnostic, JAX-free (finite-difference) evaluation of H_theta
and its gradient, used only for the SAMPLED_ONLY obligation OB-T3-1 (unforced
dissipativity of the idealized formula, `-grad(H)^T M grad(H) <= 0`) and for
quantifying the GRAD_E_CLIP soft-clip's effect relative to the theory's
exact (unclipped) gradient.

This module re-implements ``EBM_class.py::energy_EBM`` and the two
activation/Lagrangian pairs used by ``copilotA1``
(``lagrangian_tanh``/``activation_tanh``, ``lagrangian_polynomial_stable``/
``activation_polynomial_stable``) in pure numpy, and approximates
grad(H) by a central finite difference (NOT autodiff -- no JAX dependency).

Evidence produced here is explicitly a SAMPLED, FINITE-DIFFERENCE
APPROXIMATION, not an exact or certified computation, and is labeled as such
everywhere it is reported. It exists to make an otherwise "NOT_VERIFIED
(no JAX in this environment)" obligation into an honestly-labeled
SAMPLED_ONLY / EMPIRICALLY_SUPPORTED one, without overstating what it shows.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from checkpoint_io import load_checkpoint_numpy_only
from structural_checks import assemble_M_numpy, assemble_A_numpy, _safe


def _softplus(x):
    # numerically stable softplus
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lagrangian_tanh_np(x, beta):
    bx = beta * x
    logcosh = _softplus(2.0 * bx) - bx - np.log(2.0)
    return float(np.sum(logcosh) / beta)


def activation_tanh_np(x, beta):
    return np.tanh(beta * x)


def lagrangian_polynomial_stable_np(x, p):
    return float(np.sum(_softplus(x) ** p) / p)


def activation_polynomial_stable_np(x, p):
    return (_softplus(x) ** (p - 1.0)) * _sigmoid(x)


def energy_EBM_numpy(x: np.ndarray, weights: list, biases: list, layer_args: list) -> float:
    """Re-implementation of EBM_class.py::energy_EBM for copilotA1's 2-layer
    (tanh, p_1) then (polynomial_stable, p_2) architecture. `layer_args` is
    [(beta,), (p,)] matching the config's first_layer_type/second_layer_type.
    """
    activations = [activation_tanh_np, activation_polynomial_stable_np]
    lagrangians = [lagrangian_tanh_np, lagrangian_polynomial_stable_np]
    K = len(weights)
    E = float(np.dot(x, x)) / 2.0
    pre_acts, acts = [], [x]
    for k in range(K):
        x_k = weights[k] @ acts[k]
        y_k = activations[k](x_k, *layer_args[k])
        pre_acts.append(x_k)
        acts.append(y_k)
    for k in range(K):
        x_k = pre_acts[k]
        y_k = acts[k + 1]
        L_k = lagrangians[k](x_k, *layer_args[k])
        # energy_layer_k: (x_k - b_k).y_k - L_k - y_k.x_k
        E += float(np.dot(x_k - biases[k], y_k) - L_k - np.dot(y_k, x_k))
    return E


def grad_energy_finite_diff(x: np.ndarray, weights, biases, layer_args, h: float = 1e-4) -> np.ndarray:
    d = x.shape[0]
    grad = np.zeros(d, dtype=np.float64)
    for i in range(d):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        grad[i] = (energy_EBM_numpy(xp, weights, biases, layer_args)
                  - energy_EBM_numpy(xm, weights, biases, layer_args)) / (2.0 * h)
    return grad


def soft_clip_norm(g: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(g)
    target = max_norm * np.tanh(norm / (max_norm + 1e-8))
    scale = target / (norm + 1e-8)
    return g * scale


@dataclass
class UnforcedDissipativityProbe:
    obligation_id: str
    method: str
    domain_sampled: str
    n_samples: int
    finite_diff_step: float
    results: list


def run_probe(checkpoint_path: str, config_path: str, n_samples: int = 8,
              sample_radius: float = 5.0, seed: int = 0) -> UnforcedDissipativityProbe:
    with open(config_path) as f:
        config = json.load(f)
    d = int(config["d"])
    chol_clip_exp = float(config["chol_clip_exp"])
    damping_scale = float(config["damping_scale"])
    grad_e_clip = float(config["grad_e_clip"])
    p1 = float(config["p_1"])
    p2 = float(config["p_2"])
    layer_args = [(p1,), (p2,)]

    params = load_checkpoint_numpy_only(checkpoint_path)
    weights = [np.asarray(w, dtype=np.float64) for w in params.ebm_weights]
    biases = [np.asarray(b, dtype=np.float64) for b in params.ebm_biases]
    M, _ = assemble_M_numpy(np.asarray(params.trunk.L_M), d, chol_clip_exp, damping_scale, False)

    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_samples):
        x = rng.normal(size=d).astype(np.float64) * sample_radius
        g_raw = grad_energy_finite_diff(x, weights, biases, layer_args)
        g_clipped = soft_clip_norm(g_raw, grad_e_clip)
        raw_norm = float(np.linalg.norm(g_raw))
        clipped_norm = float(np.linalg.norm(g_clipped))
        drift_raw = float(g_raw @ M @ g_raw)       # -grad(H)^T M grad(H) <= 0 <=> drift_raw >= 0
        drift_clipped = float(g_clipped @ M @ g_clipped)
        results.append({
            "sample_index": i,
            "state_norm": float(np.linalg.norm(x)),
            "raw_grad_norm_finite_diff": raw_norm,
            "clipped_grad_norm": clipped_norm,
            "clip_scale_factor": clipped_norm / raw_norm if raw_norm > 0 else None,
            "drift_raw_nonneg": drift_raw >= 0.0,
            "drift_raw_value": drift_raw,
            "drift_clipped_nonneg": drift_clipped >= 0.0,
            "drift_clipped_value": drift_clipped,
        })
    return UnforcedDissipativityProbe(
        obligation_id="OB-T3-1",
        method="Finite-difference (central, h=1e-4) approximation of grad(H_theta) "
               "at randomly sampled states, combined with the exact (checkpoint-"
               "derived) M=L@L.T. NOT an exact autodiff computation (JAX unavailable "
               "in this environment); reported as SAMPLED_ONLY / finite-difference-"
               "approximate evidence only.",
        domain_sampled=f"N(0, {sample_radius}^2 I_{d}) (unstructured Gaussian probe, "
                       "NOT restricted to the operational ball or anchored to any well)",
        n_samples=n_samples,
        finite_diff_step=1e-4,
        results=results,
    )


if __name__ == "__main__":
    ckpt, cfg = sys.argv[1], sys.argv[2]
    probe = run_probe(ckpt, cfg)
    print(json.dumps(asdict(probe), indent=2))
