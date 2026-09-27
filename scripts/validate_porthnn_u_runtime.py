#!/usr/bin/env python3
"""Aggregate finite-execution and port-Hamiltonian structure smoke tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from porthnn_u import duffing, three_link
from porthnn_u.champions import load_modular_champion
from porthnn_u.modular import (dissipation_apply, input_map_apply,
                               interconnection_apply, modular_decode,
                               modular_rollout, modular_vector_field, storage_apply)

CHAMPIONS = ROOT / "baselines" / "porthnn_u" / "champions"


def tree_finite(tree) -> bool:
    return all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(tree))


def load_three_link(path: Path):
    template = three_link.initialize(seed=0, input_dim=1)
    with np.load(path) as archive:
        for branch in ("hamiltonian", "j_head", "r_head", "g_head"):
            layers = []
            for index, layer in enumerate(template[branch]):
                layers.append({name: jnp.asarray(archive[f"{branch}_{index}_{name}"], dtype=jnp.float32)
                               if value is not None else None
                               for name, value in layer.items()})
            template[branch] = tuple(layers)
    return template


def physical_models() -> dict:
    report = {}
    d_run = CHAMPIONS / "duffing" / "seed0"
    d_params = duffing.load_parameters(d_run / "parameters.npz")
    d_data = np.load(ROOT / "datasets" / "duffing_doublewell" / "parameters" / "duffing_v1.npz")
    d_inputs = jnp.asarray(d_data["u_val"][0, :8, 0], dtype=jnp.float32)
    d_states = duffing.rollout(d_params, jnp.asarray(d_data["z_val"][0, 0], dtype=jnp.float32), d_inputs, 0.02)
    if not tree_finite((d_params, d_states)):
        raise FloatingPointError("Duffing produced nonfinite values")
    report["duffing"] = {"parameter_count": duffing.parameter_count(d_params),
                         "rollout_shape": list(d_states.shape),
                         "energy_start": float(duffing.hamiltonian(d_params, d_states[0]))}

    t_run = CHAMPIONS / "three_link" / "full_jrg" / "seed0"
    t_params = load_three_link(t_run / "checkpoint_best.npz")
    data_root = ROOT / "datasets" / "deep_dissipative_nlink_n3" / "parameters"
    raw_states = np.load(data_root / "ex2_n3nlink.train.state.npy")
    raw_inputs = np.load(data_root / "ex2_n3nlink.train.input.npy")
    # Mirror the training script's q,qdot -> q,p conversion without importing
    # its launcher, which globally enables JAX x64 and contaminates the
    # subsequent float32 modular-champion smoke tests.
    q = raw_states[0, 0, :3]
    velocity = raw_states[0, 0, 3:]
    index = np.arange(3)
    mass = ((3 - np.maximum(index[:, None], index[None, :])) / 54.0
            * np.cos(q[:, None] - q[None, :]))
    initial = jnp.asarray(np.concatenate((q, mass @ velocity)), dtype=jnp.float32)
    controls = jnp.asarray(raw_inputs[0, :8], dtype=jnp.float32)
    t_states = three_link.rollout(t_params, initial, controls, 0.05)
    J, R, G = three_link.matrices(t_params, t_states[-1])
    if not tree_finite((t_params, t_states, J, R, G)):
        raise FloatingPointError("three-link produced nonfinite values")
    report["three_link"] = {"parameter_counts": three_link.parameter_counts(t_params),
                            "rollout_shape": list(t_states.shape),
                            "j_skew_residual": float(jnp.max(jnp.abs(J + J.T))),
                            "r_min_eigenvalue": float(jnp.min(jnp.linalg.eigvalsh(R)))}
    return report


def modular_models() -> dict:
    report = {}
    for dataset in ("silverbox", "ced"):
        run, config, spec, params, normalization = load_modular_champion(dataset)
        state = jnp.zeros(spec.state_dim, dtype=jnp.float32)
        control = jnp.zeros(spec.input_dim, dtype=jnp.float32)
        J = interconnection_apply(params, state, spec)
        R = dissipation_apply(params, state, spec)
        G = input_map_apply(params, state, spec)
        values = (storage_apply(params, state, spec),
                  modular_vector_field(params, state, control, spec), J, R, G,
                  modular_decode(params, state[None], spec,
                                 normalization["output_mean"], normalization["output_scale"]),
                  modular_rollout(params, state, jnp.zeros((8, spec.input_dim)),
                                  float(0.0016384041943147375 if dataset == "silverbox" else 0.02), spec))
        if not tree_finite((params, values)):
            raise FloatingPointError(f"{dataset}: nonfinite runtime value")
        report[dataset] = {
            "config_id": config["config_id"], "state_dim": spec.state_dim,
            "input_dim": spec.input_dim, "j_skew_residual": float(jnp.max(jnp.abs(J + J.T))),
            "r_min_eigenvalue": float(jnp.min(jnp.linalg.eigvalsh(R))),
            "g_shape": list(G.shape), "rollout_shape": list(values[-1].shape),
            "checkpoint": str((run / "best_validation_parameters.npz").relative_to(ROOT)),
        }
    return report


def main() -> None:
    report = {"status": "PASS", "models": {}}
    try:
        report["models"].update(physical_models())
        report["models"].update(modular_models())
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(report["status"] != "PASS")


if __name__ == "__main__":
    main()
