#!/usr/bin/env python3
# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Recompute runtime HNER99 empirical robustness radii for contained champions.

The shell is the 99th percentile of Hamiltonian values on ordinary development
or validation rollouts. This is an empirical numerical diagnostic, not a formal
global invariance certificate. Frozen historical reports are kept separately
under ``certificates/hner99``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "EBM_model", ROOT / "dataset_interfaces",
             ROOT / "Stress_test_final"):
    sys.path.insert(0, str(path))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, qmc

from Stress_tests.integration import (apply_runtime_config, collect_reference_rollout,
                                      encode_x0_from_burnin)
from Stress_tests.model_adapter import EBMStressAdapter

Q = 0.99


@dataclass(frozen=True)
class Model:
    dataset: str
    name: str
    checkpoint: Path
    states: np.ndarray
    inputs: np.ndarray
    energy: object
    grad: object
    rhs: object
    affine: bool
    input_matrix: object | None
    runtime_modified: bool
    post_step: bool
    trajectories: int
    ph_parts: object | None = None


def vectorized(function, values, chunk=2048):
    return np.concatenate([
        np.asarray(jax.vmap(function)(jnp.asarray(values[start:start + chunk])))
        for start in range(0, len(values), chunk)
    ])


def ebm_model(dataset, name, checkpoint, params, layers, d, m, dt, states, inputs,
              trajectories, *, post_step=True):
    adapter = EBMStressAdapter(params, layers, d, m, dt)

    def ph_parts(state):
        _, _, grad_h, damping, _ = adapter.theory_energy_terms(state)
        return grad_h, damping

    return Model(dataset, name, checkpoint, np.asarray(states), np.asarray(inputs),
                 adapter.energy, adapter.grad_energy, adapter.vector_field,
                 False, None, True, post_step, trajectories, ph_parts)


def _json_config(path, defaults):
    payload = json.loads(Path(path).read_text())
    defaults.update(payload.get("config", payload))
    return defaults


def load_duffing_phebm():
    import Duffing_DoubleWell_main as module
    base = ROOT / "datasets" / "duffing_doublewell"
    config = _json_config(base / "champion_configuration/config.json", module.get_config())
    apply_runtime_config(config)
    with (base / "checkpoints/params.pkl").open("rb") as stream:
        params = pickle.load(stream)
    data = module.load_duffing_doublewell(base / "parameters")
    return ebm_model("duffing_doublewell", "phebm", base / "checkpoints/params.pkl",
                     params, module._build_layers(config), 2, 1,
                     float(data.get("_gen_config", {}).get("dt", 0.02)), data["z_val"].reshape(-1, 2),
                     data["u_val"].reshape(-1, 1), len(data["z_val"]))


def load_duffing_porthnn():
    from porthnn_u.duffing import hamiltonian, load_parameters, vector_field
    base = ROOT / "baselines/porthnn_u/champions/duffing/seed0"
    checkpoint = base / "parameters.npz"
    params = load_parameters(checkpoint)
    data = np.load(ROOT / "datasets/duffing_doublewell/parameters/duffing_v1.npz")
    energy = lambda state: hamiltonian(params, state)
    return Model("duffing_doublewell", "porthnn_u", checkpoint,
                 data["z_val"].reshape(-1, 2), data["u_val"].reshape(-1, 1),
                 energy, jax.grad(energy), lambda state, control: vector_field(params, state, control),
                 False, None, False, False, len(data["z_val"]))


def load_three_link_phebm():
    import DeepDissipative_NLink_main as module
    from EBM_param_fields import make_grad_energy
    from EBM_rollout import make_batch_rollout_fn
    base = ROOT / "datasets/deep_dissipative_nlink_n3"
    config = _json_config(base / "champion_configuration/config.json", module.get_config())
    apply_runtime_config(config)
    with (base / "checkpoints/params.pkl").open("rb") as stream:
        params = pickle.load(stream)
    data = module.load_nlink_dataset(base / "parameters", "ex2_n3")
    init = int(config["init_win"])
    encoder = np.concatenate([data["y_train"][:, :init], data["u_train"][:, :init]], -1)
    x0 = module.encode_x0_batch(params, jnp.asarray(encoder))
    rollout = make_batch_rollout_fn(make_grad_energy(module._build_layers(config)),
                                    int(config["d"]), int(data["in_dim"]), data["dt"],
                                    integrator=config["rollout_integrator"])
    states, _, _ = rollout(params, x0, jnp.asarray(data["u_train"][:, init:]), stop_grad=True)
    all_states = np.concatenate([np.asarray(x0)[:, None], np.asarray(states)], 1)
    return ebm_model("three_link_pendulum", "phebm", base / "checkpoints/params.pkl",
                     params, module._build_layers(config), int(config["d"]),
                     int(data["in_dim"]), data["dt"], all_states.reshape(-1, config["d"]),
                     data["u_train"][:, init:].reshape(-1, data["in_dim"]), len(all_states))


def _canonical_three_link(states):
    states = jnp.asarray(states)
    q = states[..., :3]
    idx = jnp.arange(3); i = idx[:, None]; k = idx[None, :]
    coefficient = (3 - jnp.maximum(i, k)) / 54.0
    mass = coefficient * jnp.cos(q[..., :, None] - q[..., None, :])
    momentum = jnp.einsum("...ij,...j->...i", mass, states[..., 3:])
    return np.asarray(jnp.concatenate((q, momentum), axis=-1))


def _load_three_link_params(checkpoint):
    from porthnn_u.three_link import initialize
    params = initialize(0, input_dim=1)
    with np.load(checkpoint) as archive:
        for branch in ("hamiltonian", "j_head", "r_head", "g_head"):
            params[branch] = tuple({name: jnp.asarray(archive[f"{branch}_{index}_{name}"])
                                    if value is not None else None
                                    for name, value in layer.items()}
                                   for index, layer in enumerate(params[branch]))
    return params


def load_three_link_porthnn():
    from porthnn_u import three_link as module
    checkpoint = ROOT / "baselines/porthnn_u/champions/three_link/full_jrg/seed0/checkpoint_best.npz"
    params = _load_three_link_params(checkpoint)
    data = ROOT / "datasets/deep_dissipative_nlink_n3/parameters"
    states = _canonical_three_link(np.load(data / "ex2_n3nlink.train.state.npy"))
    inputs = np.load(data / "ex2_n3nlink.train.input.npy").astype(float)
    energy = lambda state: module.hamiltonian(params, state)
    grad = jax.grad(energy)

    def ph_parts(state):
        _, damping, _ = module.matrices(params, state)
        return grad(state), damping

    return Model("three_link_pendulum", "porthnn_u", checkpoint,
                 states.reshape(-1, 6), inputs.reshape(-1, inputs.shape[-1]),
                 energy, grad, lambda state, control: module.vector_field(params, state, control),
                 True, lambda state: module.matrices(params, state)[2], False, False,
                 len(states), ph_parts)


def load_modular_porthnn(dataset):
    import yaml
    from porthnn_u.champions import load_modular_champion
    from porthnn_u.data import load_campaign_development
    from porthnn_u.model import encode_initial_state
    from porthnn_u.modular import (dissipation_apply, input_map_apply, modular_rollout,
                                   modular_vector_field, storage_apply)
    run, config, spec, params, _ = load_modular_champion(dataset)
    fold = 1 if dataset == "ced" else None
    bundle = load_campaign_development(dataset, init_window=config["init_window"],
                                       max_horizon=config["max_horizon"], fold=fold)
    dt = float(bundle.metadata["dt"])
    rollout = jax.jit(lambda state, controls: modular_rollout(params, state, controls, dt, spec))
    states, inputs = [], []
    for record in bundle.validation:
        initial = encode_initial_state(params, jnp.asarray(record.inputs[:record.init_window]),
                                       jnp.asarray(record.outputs[:record.init_window]), spec)
        pieces = [np.asarray(initial)[None]]
        state = initial
        for start in range(record.init_window, len(record.inputs), 64):
            values = rollout(state, jnp.asarray(record.inputs[start:start + 64]))
            pieces.append(np.asarray(values)); state = values[-1]
        states.append(np.concatenate(pieces)); inputs.append(record.inputs[record.init_window:])
    energy = jax.jit(lambda state: storage_apply(params, state, spec))
    grad = jax.jit(jax.grad(energy))
    return Model(dataset, "porthnn_u", run / "best_validation_parameters.npz",
                 np.concatenate(states), np.concatenate(inputs), energy, grad,
                 jax.jit(lambda state, control: modular_vector_field(params, state, control, spec)),
                 True, jax.jit(lambda state: input_map_apply(params, state, spec)),
                 False, False, len(states),
                 lambda state: (grad(state), dissipation_apply(params, state, spec)))


def load_ced_phebm():
    import nonlinear_benchmarks
    import Silverbox_main as module
    from EBM_param_fields import init_params
    from checkpoint_compat import load_legacy_checkpoint
    base = ROOT / "datasets/CED"
    config = _json_config(base / "champion_configuration/config.json", module.get_config())
    apply_runtime_config(config)
    bundled = base / "DATAUNIF.MAT"
    if bundled.is_file():
        from scipy.io import loadmat
        raw = loadmat(bundled)
        u = [np.asarray(raw[key], np.float32).reshape(-1, 1) for key in ("u11", "u12")]
        y = [np.asarray(raw[key], np.float32).reshape(-1, 1) for key in ("z11", "z12")]
        dt = 0.02
    else:
        records, _ = nonlinear_benchmarks.CED(dir_placement=ROOT / "datasets")
        u = [np.asarray(pair[0], np.float32).reshape(-1, 1) for pair in records]
        y = [np.asarray(pair[1], np.float32).reshape(-1, 1) for pair in records]
        dt = float(records[0].sampling_time)
    u_all, y_all = np.concatenate(u), np.concatenate(y)
    u_mean, u_scale = u_all.mean(0), u_all.std(0)
    y_mean, y_scale = y_all.mean(0), y_all.std(0)
    u = [(value - u_mean) / u_scale for value in u]
    y = [(value - y_mean) / y_scale for value in y]
    init = int(config["init_win"])
    template = init_params(jax.random.key(0), d=config["d"], m=config["m_ports"],
                           n_obs=config["n_obs"], init_win=init,
                           layer_dims=config["layer_dims"], hidden=config["hidden"], dt=dt)
    checkpoint = base / "checkpoints/params.pkl"
    params = load_legacy_checkpoint(checkpoint, template)
    adapter = EBMStressAdapter(params, module._build_layers(config), config["d"], config["m_ports"], dt)
    states, inputs = [], []
    for controls, outputs in zip(u, y):
        initial = encode_x0_from_burnin(params, outputs[:init], controls[:init])
        rolled = collect_reference_rollout(adapter, initial, controls[init:], config["rollout_integrator"])
        states.append(np.concatenate([initial[None], rolled])); inputs.append(controls[init:])
    return ebm_model("ced", "phebm", checkpoint, params, module._build_layers(config),
                     config["d"], config["m_ports"], dt, np.concatenate(states),
                     np.concatenate(inputs), len(states))


def load_nanodrone_phebm():
    import NanoDrone_main as module
    from EBM_param_fields import make_grad_energy
    from EBM_rollout import make_rollout_fn
    from nanodrone.data import fit_scalers, load_nanodrone_dataset, transform_inputs
    base = ROOT / "datasets/nanodrone_S3"
    config = _json_config(base / "champion_configuration/s3_ph_T2_seed47.json", module.get_config())
    module._apply_runtime_flags(config)
    with (base / "checkpoints/best_physical_mae.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    params = payload["params"] if isinstance(payload, dict) and "params" in payload else payload
    dataset = load_nanodrone_dataset(base / "parameters/raw_data", include_official_test=False)
    state_scaler, input_scaler = fit_scalers(
        dataset["development_train"], config["nanodrone_input_representation"])
    rollout = make_rollout_fn(make_grad_energy(module._build_layers(config)), 12,
                              int(config["m_ports"]), module.DT,
                              integrator=config["rollout_integrator"])
    states, inputs = [], []
    for record in dataset["development_validation"]:
        normalized_state = state_scaler.transform(record.y)
        normalized_input = input_scaler.transform(
            transform_inputs(record.u, config["nanodrone_input_representation"]))
        rolled, _, _ = rollout(params, jnp.asarray(normalized_state[0]),
                               jnp.asarray(normalized_input[:-1]))
        states.append(np.concatenate([normalized_state[:1], np.asarray(rolled)]))
        inputs.append(normalized_input[:-1])
    return ebm_model("nanodrone", "phebm_seed47", base / "checkpoints/best_physical_mae.pkl",
                     params, module._build_layers(config), 12, int(config["m_ports"]),
                     module.DT, np.concatenate(states), np.concatenate(inputs), len(states))


LOADERS = {
    "duffing_phebm": load_duffing_phebm,
    "duffing_porthnn": load_duffing_porthnn,
    "n3_phebm": load_three_link_phebm,
    "n3_porthnn": load_three_link_porthnn,
    "ced_phebm": load_ced_phebm,
    "ced_porthnn": lambda: load_modular_porthnn("ced"),
    "silverbox_porthnn": lambda: load_modular_porthnn("silverbox"),
    "nanodrone_phebm_seed47": load_nanodrone_phebm,
}
UNAVAILABLE = {
    "silverbox_phebm": "the supplied release has no trained Silverbox pH-EBM checkpoint",
    "nanodrone_phebm_seed23": "the historical HNER99 report used seed 23; the release contains seed 47",
}


def input_moment(inputs):
    moment = inputs.T @ inputs / len(inputs)
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    keep = eigenvalues / max(eigenvalues.max(), 1e-300) >= 1e-8
    factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    return factor, eigenvalues, int(keep.sum())


def farthest_shell_seeds(states, energies, epsilon, count):
    pool = np.argsort(abs(energies - epsilon))[:min(128, len(states))]
    candidates = states[pool]
    scale = states.std(0)
    normalized = candidates / np.where(scale > 1e-12, scale, 1)
    chosen = [int(np.argmin(abs(energies[pool] - epsilon)))]
    distance = np.sum((normalized - normalized[chosen[0]]) ** 2, 1)
    while len(chosen) < min(count, len(pool)):
        index = int(np.argmax(distance)); chosen.append(index)
        distance = np.minimum(distance, np.sum((normalized - normalized[index]) ** 2, 1))
    return candidates[chosen], scale


def reconstruct_shell(model, count):
    energies = vectorized(model.energy, model.states)
    epsilon = float(np.quantile(energies, Q))
    seeds, scale = farthest_shell_seeds(model.states, energies, epsilon, count)
    accepted, errors = [], []
    for seed in seeds:
        state = seed.astype(float).copy()
        for _ in range(3):
            gradient = np.asarray(model.grad(state), float)
            delta = (float(model.energy(state)) - epsilon) * gradient / (gradient @ gradient + 1e-12)
            normalized = np.linalg.norm(delta / np.where(scale > 1e-12, scale, 1))
            state -= delta * min(1.0, 0.25 / max(normalized, 1e-30))
            if abs(float(model.energy(state)) - epsilon) / max(1, abs(epsilon)) < 1e-6:
                break
        error = abs(float(model.energy(state)) - epsilon) / max(1, abs(epsilon))
        displacement = np.linalg.norm((state - seed) / np.where(scale > 1e-12, scale, 1))
        if np.linalg.norm(model.grad(state)) >= 1e-8 and error < 1e-4 and displacement <= 1:
            accepted.append(state); errors.append(error)
    return epsilon, np.asarray(accepted), max(errors, default=np.inf)


def flux(model, state, control):
    return float(np.asarray(model.grad(state)) @ np.asarray(model.rhs(state, control)))


def sobol_directions(dimension):
    values = qmc.Sobol(dimension, scramble=False).random_base2(5)
    values = norm.ppf(np.clip(values, 1e-9, 1 - 1e-9))
    values = values[np.linalg.norm(values, axis=1) > 1e-12][:16]
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return np.concatenate([values, -values])


def nonlinear_radius(model, state, factor, cap=4.0):
    zero = np.zeros(factor.shape[0])
    if flux(model, state, zero) >= 0:
        return 0.0, False
    directions = np.array([[-1.0], [1.0]]) if factor.shape[1] == 1 else sobol_directions(factor.shape[1])
    best = np.inf
    for direction in directions:
        for lower, upper in zip((0, .25, .5, 1, 2), (.25, .5, 1, 2, 4)):
            if flux(model, state, factor @ (upper * direction)) >= 0:
                best = min(best, brentq(
                    lambda radius: flux(model, state, factor @ (radius * direction)),
                    lower, upper, xtol=1e-4))
                break
    return best, bool(best > cap)


def affine_radius(model, state, factor):
    zero_flux = flux(model, state, np.zeros(factor.shape[0]))
    if zero_flux >= 0:
        return 0.0, False
    direction = np.asarray(model.input_matrix(state)).T @ np.asarray(model.grad(state))
    denominator = np.linalg.norm(factor.T @ direction)
    return (np.inf if denominator < 1e-14 else -zero_flux / denominator), False


def evaluate(model, shell_count):
    factor, eigenvalues, rank = input_moment(model.inputs)
    epsilon, points, max_error = reconstruct_shell(model, shell_count)
    if len(points) < 8:
        return {"reliable": False, "epsilon": epsilon, "accepted": len(points)}
    zero_flux = np.array([flux(model, state, np.zeros(model.inputs.shape[1])) for state in points])
    radii, censored = [], []
    for state in points:
        radius, hit_cap = (affine_radius(model, state, factor) if model.affine
                           else nonlinear_radius(model, state, factor))
        radii.append(radius); censored.append(hit_cap)
    gradients = vectorized(model.grad, points)
    residuals = []
    if model.ph_parts:
        for state in points:
            gradient, damping = model.ph_parts(jnp.asarray(state))
            expected = float(np.asarray(gradient) @ np.asarray(damping) @ np.asarray(gradient))
            residuals.append(abs(-flux(model, state, np.zeros(model.inputs.shape[1])) - expected))
    return {"reliable": True, "epsilon": epsilon, "points": points,
            "accepted": len(points), "max_error": max_error,
            "kappa": float(np.min(np.linalg.norm(gradients, axis=1))),
            "zero_outward": int(np.sum(zero_flux >= 0)),
            "max_zero_flux": float(np.max(zero_flux)), "radii": np.asarray(radii),
            "censored": all(censored), "rank": rank, "eigenvalues": eigenvalues.tolist(),
            "ph_residual": max(residuals, default=0.0)}


def compute(model):
    shell32, shell64 = evaluate(model, 32), evaluate(model, 64)
    relative_checkpoint = model.checkpoint.relative_to(ROOT).as_posix()
    if not shell32["reliable"] or not shell64["reliable"]:
        return {"dataset": model.dataset, "model": model.name,
                "checkpoint": relative_checkpoint, "q": Q,
                "status": "UNRELIABLE_SHELL_RECONSTRUCTION",
                "shell_32_result": shell32, "shell_64_result": shell64}
    radius32, radius64 = float(np.min(shell32["radii"])), float(np.min(shell64["radii"]))
    sensitive = abs(radius64 - radius32) / max(radius64, 1e-8) >= 0.1
    chosen = shell32 if sensitive and radius32 < radius64 else shell64
    radii = chosen["radii"]
    status = ("ZERO_RADIUS_ZERO_INPUT_OUTWARD" if chosen["zero_outward"] else
              "CENSORED_ABOVE_SEARCH_CAP" if chosen["censored"] else
              "POST_STEP_PROJECTION_DIAGNOSTIC_ONLY" if model.post_step else
              "EMPIRICAL_RUNTIME_HNER")
    index = int(np.argmin(radii))
    return {"dataset": model.dataset, "model": model.name,
            "checkpoint": relative_checkpoint, "q": Q, "epsilon_q": chosen["epsilon"],
            "n_validation_states": len(model.states), "input_dim": model.inputs.shape[1],
            "input_active_rank": chosen["rank"],
            "input_rms_or_second_moment": chosen["eigenvalues"], "input_center": "zero",
            "runtime_rhs_used": True, "runtime_modified": model.runtime_modified,
            "post_step_projection": model.post_step,
            "ph_identity_exact": chosen["ph_residual"] < 1e-5,
            "max_ph_identity_residual": chosen["ph_residual"],
            "k_shell_requested": 32 if chosen is shell32 else 64,
            "k_shell_accepted": chosen["accepted"], "max_shell_error": chosen["max_error"],
            "empirical_kappa": chosen["kappa"], "near_critical_shell": chosen["kappa"] < 1e-6,
            "n_zero_input_outward": chosen["zero_outward"],
            "max_zero_input_flux": chosen["max_zero_flux"],
            "input_structure": "affine" if model.affine else
                               ("nonlinear_scalar" if model.inputs.shape[1] == 1 else "nonlinear_multidimensional"),
            "radius_method": "closed_form" if model.affine else
                             ("signed_grid_brent" if model.inputs.shape[1] == 1 else "16_sobol_plus_negatives_directional_brent"),
            "input_search_cap": None if model.affine else 4,
            "hner99_min": float(np.min(radii)), "hner99_p10": float(np.quantile(radii, .1)),
            "hner99_median": float(np.median(radii)),
            "argmin_state": chosen["points"][index].tolist(),
            "shell_32_result": radius32, "shell_64_result": radius64,
            "shell_sampling_sensitive": sensitive,
            "uncertainty": "NOT_ESTIMATED_SINGLE_TRAJECTORY" if model.trajectories < 5 else "NOT_ESTIMATED_OPTIONAL",
            "status": status,
            "warnings": "Finite validation-derived shell; empirical numerical radius, not a formal global certificate."}


def write_report(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"{payload['dataset']}_{payload['model']}_hner99_runtime"
    encoder = lambda value: value.tolist() if hasattr(value, "tolist") else str(value)
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=encoder) + "\n")
    stem.with_suffix(".txt").write_text("\n".join(
        f"{key.upper()}: {json.dumps(value, default=encoder) if isinstance(value, (list, dict)) else value}"
        for key, value in payload.items()) + "\n")
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selectors", nargs="*", choices=sorted(LOADERS))
    parser.add_argument("--output", type=Path, default=ROOT / "results/hner99_runtime")
    parser.add_argument("--list", action="store_true", help="list available and unavailable selectors")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({"available": sorted(LOADERS), "unavailable": UNAVAILABLE}, indent=2))
        return
    selectors = args.selectors or list(LOADERS)
    for selector in selectors:
        payload = compute(LOADERS[selector]())
        stem = write_report(payload, args.output)
        print(f"{stem}: {payload['status']} hner99_min={payload.get('hner99_min')}")


if __name__ == "__main__":
    main()
