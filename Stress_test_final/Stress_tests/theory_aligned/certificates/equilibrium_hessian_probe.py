# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Application-agnostic, JAX-free finite-difference Hessian check for the
local-PL/equilibrium proof obligations (OB-T5-1, OB-T12-1).

Evaluates a finite-difference approximation of grad(H_theta) and
Hessian(H_theta) at a SPECIFIC, ALREADY-STORED candidate state (the
"well" returned by ``Stress_tests.geometry.discover_minima`` for a given
run, read from that run's own ``stress_tests/arrays/stress_arrays.npz`` --
this script does not run any new optimization/search of its own).

This directly and independently cross-checks (via a completely different,
non-JAX computational path) the gradient-norm number already reported in
that run's ``audits/audit.json`` (``metrics.stress.wells.max_grad_norm``),
and additionally computes the local Hessian eigenvalues, which the
existing pipeline never computes at all (see cross-audit obligation
T12/Proposition `hessian_pl`, `iclr2027_conference_robust_extensions_red.tex`).

Both the gradient and the Hessian here are FINITE-DIFFERENCE
APPROXIMATIONS (not exact autodiff), and the underlying value is
double-differenced (Hessian from a first difference of a first
difference), so numerical error is larger than for a single gradient
evaluation. This is reported honestly as SAMPLED_ONLY /
finite-difference-approximate evidence, sufficient to indicate order of
magnitude and sign, not as a certified bound.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict

import numpy as np

from checkpoint_io import load_checkpoint_numpy_only
from unforced_dissipativity_probe import energy_EBM_numpy, grad_energy_finite_diff


def finite_diff_hessian(x: np.ndarray, weights, biases, layer_args, h: float = 1e-3) -> np.ndarray:
    d = x.shape[0]
    Hmat = np.zeros((d, d), dtype=np.float64)
    E0 = energy_EBM_numpy(x, weights, biases, layer_args)
    E_plus = np.zeros(d)
    E_minus = np.zeros(d)
    for i in range(d):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        E_plus[i] = energy_EBM_numpy(xp, weights, biases, layer_args)
        E_minus[i] = energy_EBM_numpy(xm, weights, biases, layer_args)
        Hmat[i, i] = (E_plus[i] - 2.0 * E0 + E_minus[i]) / (h * h)
    for i in range(d):
        for j in range(i + 1, d):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            val = (energy_EBM_numpy(xpp, weights, biases, layer_args)
                  - energy_EBM_numpy(xpm, weights, biases, layer_args)
                  - energy_EBM_numpy(xmp, weights, biases, layer_args)
                  + energy_EBM_numpy(xmm, weights, biases, layer_args)) / (4.0 * h * h)
            Hmat[i, j] = Hmat[j, i] = val
    return Hmat


@dataclass
class EquilibriumHessianProbe:
    obligation_ids: list
    candidate_state: list
    candidate_source: str
    audit_json_reported_grad_norm: float | None
    finite_diff_grad_norm: float
    finite_diff_grad_vs_audit_relative_diff: float | None
    hessian_eigenvalues: list
    hessian_symmetry_defect: float
    hessian_min_eig_positive: bool
    method: str
    interpretation: str


def run_probe(checkpoint_path: str, config_path: str, npz_path: str,
              audit_json_path: str | None = None) -> EquilibriumHessianProbe:
    with open(config_path) as f:
        config = json.load(f)
    p1, p2 = float(config["p_1"]), float(config["p_2"])
    layer_args = [(p1,), (p2,)]

    params = load_checkpoint_numpy_only(checkpoint_path)
    weights = [np.asarray(w, dtype=np.float64) for w in params.ebm_weights]
    biases = [np.asarray(b, dtype=np.float64) for b in params.ebm_biases]

    npz = np.load(npz_path, allow_pickle=True)
    x_star = np.asarray(npz["minima_states"][0], dtype=np.float64)

    reported_norm = None
    if audit_json_path:
        with open(audit_json_path) as f:
            audit = json.load(f)
        reported_norm = audit.get("metrics", {}).get("stress", {}).get("wells", {}).get("max_grad_norm")

    g = grad_energy_finite_diff(x_star, weights, biases, layer_args, h=1e-4)
    g_norm = float(np.linalg.norm(g))
    rel_diff = None
    if reported_norm:
        rel_diff = abs(g_norm - reported_norm) / reported_norm

    H = finite_diff_hessian(x_star, weights, biases, layer_args, h=1e-3)
    sym_defect = float(np.max(np.abs(H - H.T)))
    eigvals = np.linalg.eigvalsh((H + H.T) / 2.0)

    return EquilibriumHessianProbe(
        obligation_ids=["OB-T5-1", "OB-T12-1"],
        candidate_state=x_star.tolist(),
        candidate_source=f"minima_states[0] from {npz_path} (the SAME candidate "
                         "'well' already reported in this run's audit.json; not a "
                         "new search)",
        audit_json_reported_grad_norm=reported_norm,
        finite_diff_grad_norm=g_norm,
        finite_diff_grad_vs_audit_relative_diff=rel_diff,
        hessian_eigenvalues=eigvals.tolist(),
        hessian_symmetry_defect=sym_defect,
        hessian_min_eig_positive=bool(np.min(eigvals) > 0.0),
        method="Central finite-difference gradient (h=1e-4) and central "
               "finite-difference Hessian (double-differenced, h=1e-3) of the "
               "checkpoint's exact H_theta, evaluated at the run's own "
               "already-discovered candidate well. No JAX/autodiff used; no "
               "new optimization run.",
        interpretation=(
            "This DOES NOT establish OB-T5-1 (existence of a true equilibrium): "
            "finite-difference stationarity residual and Hessian are numerical "
            "diagnostics at one optimizer-returned candidate. A positive Hessian "
            "supports local minimum curvature only when the stationarity residual "
            "is sufficiently small for the intended tolerance; neither finite "
            "sampling nor this probe proves existence, isolation, or a PL bound."
        ),
    )


if __name__ == "__main__":
    ckpt, cfg, npz = sys.argv[1], sys.argv[2], sys.argv[3]
    audit = sys.argv[4] if len(sys.argv) > 4 else None
    probe = run_probe(ckpt, cfg, npz, audit)
    print(json.dumps(asdict(probe), indent=2))
