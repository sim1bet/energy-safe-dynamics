# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Convenience hook for the repository's normalized nonlinear-benchmark IO.

This is the least invasive way to attach stress testing to any current
``Interface_code/*_main.py`` file: call ``stress_from_normalized_io`` after
training, using the already-normalized test arrays returned by that dataset's
loader.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .config import StressTestConfig
from .integration import apply_runtime_config, collect_reference_rollout, encode_x0_from_burnin
from .model_adapter import EBMStressAdapter
from .suite import run_stress_suite


def stress_from_normalized_io(
    *,
    params,
    layers: tuple,
    config: dict,
    dt: float,
    u_test: np.ndarray,
    y_test: np.ndarray,
    init_len: int,
    epsilon: float,
    gamma: float,
    output_dir: str | Path,
    stress_config: Optional[StressTestConfig] = None,
    x0: Optional[np.ndarray] = None,
) -> dict:
    """Collect a representative latent test rollout and launch the suite.

    ``u_test``/``y_test`` must be in the same normalized coordinates seen by
    the model, exactly as returned by the current dataset interface loaders.

    This helper intentionally uses the trained linear burn-in encoder when
    ``x0`` is not provided.  If a paper experiment relies on iterative/multistart
    x0 refinement, pass that refined ``x0`` explicitly so the reference latent
    trajectory exactly matches the reported prediction path.
    """
    apply_runtime_config(config)
    d = int(config["d"])
    m = int(config["m_ports"])
    adapter = EBMStressAdapter(params, layers, d=d, m=m, dt=float(dt))

    u = np.asarray(u_test, dtype=np.float32)
    y = np.asarray(y_test, dtype=np.float32)
    if x0 is None:
        x0 = encode_x0_from_burnin(params, y[:init_len], u[:init_len])

    reference_states = collect_reference_rollout(
        adapter, x0, u[init_len:], integrator=str(config.get("rollout_integrator", "rk4"))
    )

    return run_stress_suite(
        params=params,
        layers=layers,
        d=d,
        m=m,
        dt=float(dt),
        epsilon=float(epsilon),
        gamma=float(gamma),
        reference_states=reference_states,
        reference_inputs=u[init_len:],
        output_dir=output_dir,
        config=stress_config,
    )
