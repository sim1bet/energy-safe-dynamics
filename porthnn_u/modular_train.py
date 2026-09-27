"""Development-only training entry point for modular PortHNN-u configurations."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from .artifacts import save_parameters
from .campaign import config_hash, source_tree_hash
from .campaign_train import _fold_audit, _learning_rate, _schedule
from .data import load_campaign_development, make_campaign_windows
from .model import encode_initial_state
from .modular import ModularSpec, dissipation_apply, init_modular_params, input_map_apply, interconnection_apply, modular_decode, modular_rollout, power_diagnostics


def _spec(values: dict) -> ModularSpec:
    return ModularSpec(
        state_dim=int(values["latent_dim"]), input_dim=int(values["input_dim"]), output_dim=int(values["output_dim"]),
        storage_hidden_dims=tuple(values["storage_hidden_dims"]), encoder_hidden_dims=tuple(values["encoder_hidden_dims"]),
        matrix_hidden_dims=tuple(values["matrix_hidden_dims"]), activation=values["storage_activation"],
        interconnection=values["interconnection"]["type"], dissipation=values["dissipation"]["type"], input_map=values["input_map"]["type"],
        r_diagonal=float(values["r_diagonal"]), readout=values.get("readout", {}).get("type", "affine_normalized"),
    )


def _predict(params, batch, dt, spec, legacy, normalization=None):
    def one(u_init, y_init, u_rollout):
        state = encode_initial_state(params, u_init, y_init, spec)
        states = modular_rollout(params, state, u_rollout, dt, spec, legacy_porthnn_u=legacy)
        if normalization is None:
            return modular_decode(params, states, spec)
        return modular_decode(params, states, spec, normalization.output_mean, normalization.output_scale)
    return jax.vmap(one)(*batch[:3])


def _window_arrays(windows):
    """Keep the full window bank as NumPy; transfer only one batch at a time."""
    return (windows.u_init, windows.y_init, windows.u_rollout, windows.y_target)


def _validate(params, arrays_by_horizon, dt, spec, legacy, normalization, batch_size):
    sse = count = nonfinite = 0
    maximum = 0.0
    by_horizon = {}
    for horizon, arrays in arrays_by_horizon.items():
        hsse = hcount = hnonfinite = 0
        for start in range(0, arrays[0].shape[0], batch_size):
            batch = tuple(jnp.asarray(item[start:start + batch_size]) for item in arrays)
            prediction = np.asarray(jax.block_until_ready(_predict(params, batch, dt, spec, legacy, normalization)))
            target = np.asarray(batch[3])
            hnonfinite += int(prediction.size - np.isfinite(prediction).sum())
            if np.isfinite(prediction).all():
                hsse += float(np.sum((prediction - target) ** 2)); hcount += target.size
                maximum = max(maximum, float(np.max(np.abs(prediction))))
        sse += hsse; count += hcount; nonfinite += hnonfinite
        by_horizon[str(horizon)] = {"normalized_mse": hsse / hcount if hcount else float("inf"), "sample_count": hcount, "nonfinite_count": hnonfinite}
    mse = sse / count if count else float("inf")
    rmse = float(np.sqrt(mse))
    return {"validation_mse": mse, "normalized_rmse": rmse, "physical_rmse": rmse * float(normalization.output_scale[0]), "nrmse_std": rmse, "sample_count": count, "nonfinite_count": nonfinite, "max_abs_prediction": maximum, "per_horizon": by_horizon}


def _sequential_validation(params, realizations, dt, spec, legacy, normalization, chunk_size=64):
    """Evaluate contiguous validation in bounded chunks while carrying the latent state."""
    results, predictions = {}, {}
    for realization in realizations:
        u_init = jnp.asarray(realization.inputs[:realization.init_window])
        y_init = jnp.asarray(realization.outputs[:realization.init_window])
        inputs = jnp.asarray(realization.inputs[realization.init_window:])
        target = np.asarray(realization.outputs[realization.init_window:])
        state = encode_initial_state(params, u_init, y_init, spec)
        chunks = []
        for start in range(0, len(inputs), chunk_size):
            states = modular_rollout(params, state, inputs[start:start + chunk_size], dt, spec, legacy_porthnn_u=legacy)
            chunks.append(np.asarray(jax.block_until_ready(modular_decode(params, states, spec, normalization.output_mean, normalization.output_scale))))
            state = states[-1]
        prediction = np.concatenate(chunks, axis=0)
        predictions[realization.name] = prediction
        finite = np.isfinite(prediction).all()
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2))) if finite else float("inf")
        quarters = np.array_split(prediction - target, 4)
        quarter_nrmse = [float(np.sqrt(np.mean(item ** 2))) if item.size else float("inf") for item in quarters]
        results[realization.name] = {"normalized_rmse": rmse, "nrmse_std": rmse, "physical_rmse": rmse * float(normalization.output_scale[0]), "nonfinite_count": int(prediction.size - np.isfinite(prediction).sum()), "max_abs_prediction": float(np.max(np.abs(prediction))) if prediction.size else float("inf"), "quarter_nrmse": quarter_nrmse}
    return results, predictions


def _selection_score(metrics: dict, dataset: str) -> float:
    per_horizon = metrics["per_horizon"]
    horizon_nrmse = {int(key): float(np.sqrt(value["normalized_mse"])) for key, value in per_horizon.items()}
    full = [item["normalized_rmse"] for item in metrics["full_validation"].values()]
    if dataset == "silverbox":
        return 0.10 * horizon_nrmse[192] + 0.15 * horizon_nrmse[384] + 0.20 * horizon_nrmse[768] + 0.55 * float(np.mean(full))
    fold_score = 0.15 * horizon_nrmse[32] + 0.20 * horizon_nrmse[64] + 0.20 * horizon_nrmse[90] + 0.45 * float(np.mean(full))
    metrics["fold_continuous_score"] = fold_score
    return fold_score


def _reject(metrics: dict, params, realizations, spec, normalization) -> tuple[bool, list[str]]:
    reasons = []
    if metrics["nonfinite_count"] or not np.isfinite(metrics["validation_mse"]):
        reasons.append("nonfinite_window_validation")
    for result in metrics["full_validation"].values():
        quarters = result["quarter_nrmse"]
        if result["nonfinite_count"] or not np.isfinite(result["normalized_rmse"]): reasons.append("nonfinite_full_rollout")
        if result["max_abs_prediction"] > 10.0: reasons.append("output_bound")
        if quarters[-1] > quarters[0] * 5.0 and quarters[-1] - quarters[0] > 0.5: reasons.append("quarter_runaway")
        if result["normalized_rmse"] > 2.0 and metrics["normalized_rmse"] < result["normalized_rmse"] / 2.0: reasons.append("reset_full_discrepancy")
    state = encode_initial_state(
        params, jnp.asarray(realizations[0].inputs[:realizations[0].init_window]),
        jnp.asarray(realizations[0].outputs[:realizations[0].init_window]), spec,
    )
    measured_input = jnp.asarray(realizations[0].inputs[realizations[0].init_window])
    interconnection, dissipation = interconnection_apply(params, state, spec), dissipation_apply(params, state, spec)
    powers = power_diagnostics(params, state, measured_input, spec)
    tolerance = 1e-4 * (1.0 + abs(float(powers["dot_H"])))
    if not np.allclose(np.asarray(interconnection + interconnection.T), 0.0, atol=1e-5): reasons.append("J_not_skew")
    if not np.allclose(np.asarray(dissipation), np.asarray(dissipation).T, atol=1e-5) or np.min(np.linalg.eigvalsh(np.asarray(dissipation))) < -1e-6: reasons.append("R_not_psd")
    if float(powers["P_R"]) > tolerance: reasons.append("positive_dissipative_power")
    if abs(float(powers["power_balance_residual"])) > tolerance: reasons.append("power_balance")
    metrics["nonzero_input_power"] = {key: float(value) for key, value in powers.items()}
    return bool(reasons), sorted(set(reasons))


def _array_hash(*arrays) -> str:
    digest = sha256()
    for array in arrays:
        value = np.asarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _save_parameter_tree(path: Path, tree) -> None:
    arrays = {}
    def collect(value, name):
        if isinstance(value, dict):
            for key, item in value.items():
                collect(item, f"{name}_{key}" if name else key)
        elif isinstance(value, tuple):
            for index, item in enumerate(value):
                collect(item, f"{name}_{index}")
        else:
            arrays[name] = np.asarray(value)
    collect(tree, "")
    np.savez(path, **arrays)


def _parameter_counts(tree) -> dict:
    return {name: int(sum(np.asarray(value).size for value in jax.tree.leaves(branch))) for name, branch in tree.items()}


def train_modular_campaign(config_path: str | Path, output: str | Path, fold: int | None = None) -> dict:
    config_path, output = Path(config_path), Path(output)
    values = yaml.safe_load(config_path.read_text())
    if values.get("development_only") is not True or values.get("official_test_access") != "forbidden":
        raise ValueError("modular campaign is development-only and forbids official-test access")
    if values["dataset"] == "ced" and fold not in range(4):
        raise ValueError("CED modular campaign requires a fold")
    if values["dataset"] == "silverbox" and fold is not None:
        raise ValueError("Silverbox modular campaign has one fixed development split")
    spec, legacy = _spec(values), bool(values["legacy_porthnn_u"])
    bundle = load_campaign_development(values["dataset"], init_window=values["init_window"], max_horizon=values["max_horizon"], fold=fold)
    if bundle.train[0].inputs.shape[1] != spec.input_dim:
        raise ValueError("configured input dimension disagrees with data adapter")
    horizons = sorted({horizon for phase in values["training_schedule"] for horizon in phase["horizons"]} | set(values["validation_horizons"]))
    stride = int(values.get("training_window_stride", 1))
    train_windows = {horizon: make_campaign_windows(bundle, split="train", init_window=values["init_window"], horizon=horizon, stride=stride) for horizon in horizons}
    validation_windows = {horizon: make_campaign_windows(bundle, split="validation", init_window=values["init_window"], horizon=horizon, stride=stride) for horizon in values["validation_horizons"]}
    train_arrays = {key: _window_arrays(item) for key, item in train_windows.items()}
    validation_arrays = {key: _window_arrays(item) for key, item in validation_windows.items()}
    if values.get("symmetry_augmentation", False):
        train_arrays = {key: tuple(np.concatenate((item, -item), axis=0) for item in arrays) for key, arrays in train_arrays.items()}
    chained_arrays = None
    if values.get("chained_loss_probability", 0.0) > 0.0:
        chained_length = int(values["chained_horizon"]) * int(values["chained_chunks"])
        chained_windows = make_campaign_windows(bundle, split="train", init_window=values["init_window"], horizon=chained_length, stride=stride)
        chained_arrays = _window_arrays(chained_windows)
        if values.get("symmetry_augmentation", False):
            chained_arrays = tuple(np.concatenate((item, -item), axis=0) for item in chained_arrays)
    output.mkdir(parents=True, exist_ok=True); (output / "diagnostics").mkdir(exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(values, sort_keys=False))
    (output / "normalization.json").write_text(json.dumps(bundle.normalization.to_dict(), indent=2, sort_keys=True))
    (output / "fold_audit.json").write_text(json.dumps(_fold_audit(bundle, train_windows), indent=2, sort_keys=True))

    params = init_modular_params(jax.random.key(values["seed"]), spec, values["init_window"], legacy_porthnn_u=legacy)
    manifest = {"model_schema_version": 4, "module_types": {key: values[key]["type"] for key in ("storage", "interconnection", "dissipation", "input_map")}, "parameter_counts": _parameter_counts(params), "config_id": values["config_id"], "config_sha256": config_hash(values), "source_tree_hash": source_tree_hash(Path.cwd()), "job_id": os.environ.get("JOB_ID"), "fold": fold, "official_test_access": "forbidden", "started_utc": datetime.now(UTC).isoformat(), "data_hashes": {item.name: _array_hash(item.inputs, item.outputs) for item in (*bundle.train, *bundle.validation)}, "window_hashes": {str(horizon): _array_hash(windows.start) for horizon, windows in train_windows.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    optimizer = optax.chain(optax.clip_by_global_norm(values["gradient_clip_global_norm"]), optax.adam(lambda step: _learning_rate(step, values)))
    state = optimizer.init(params)

    def loss_fn(parameters, batch):
        prediction = _predict(parameters, batch, bundle.metadata["dt"], spec, legacy, bundle.normalization)
        weights = jnp.linspace(values.get("late_loss_weight_start", 1.0), values.get("late_loss_weight_end", 1.0), prediction.shape[1])[None, :, None]
        weights = weights / jnp.mean(weights)
        mse = jnp.mean(weights * (prediction - batch[3]) ** 2)
        terminal = float(values.get("terminal_loss_weight", 0.0)) * jnp.mean((prediction[:, -1] - batch[3][:, -1]) ** 2)
        if legacy:
            return mse + terminal
        matrix_parameters = [parameters[key] for key in ("J", "R", "G", "R_constant", "G_constant") if key in parameters]
        regularization = float(values["matrix_weight_decay"]) * sum(jnp.sum(item ** 2) for branch in matrix_parameters for item in jax.tree.leaves(branch))
        encoded = jax.vmap(lambda u, y: encode_initial_state(parameters, u, y, spec))(batch[0], batch[1])
        if spec.input_map == "full_matrix_state_dependent":
            g_jacobian = jax.vmap(jax.jacrev(lambda z: input_map_apply(parameters, z, spec)))(encoded)
            regularization = regularization + float(values["g_jacobian_weight"]) * jnp.mean(g_jacobian ** 2)
        return mse + terminal + regularization

    @jax.jit
    def step(parameters, optimizer_state, batch):
        loss, gradient = jax.value_and_grad(loss_fn)(parameters, batch)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, parameters)
        return optax.apply_updates(parameters, updates), optimizer_state, loss

    def chained_loss_fn(parameters, batch):
        state = jax.vmap(lambda u, y: encode_initial_state(parameters, u, y, spec))(batch[0], batch[1])
        chunk_length = int(values["chained_horizon"])
        predictions = []
        for start in range(0, batch[2].shape[1], chunk_length):
            input_chunk = batch[2][:, start:start + chunk_length]
            states = jax.vmap(lambda initial, inputs: modular_rollout(parameters, initial, inputs, bundle.metadata["dt"], spec, legacy_porthnn_u=legacy))(state, input_chunk)
            decoded = jax.vmap(lambda item: modular_decode(parameters, item, spec, bundle.normalization.output_mean, bundle.normalization.output_scale))(states)
            predictions.append(decoded)
            state = jax.lax.stop_gradient(states[:, -1])
        prediction = jnp.concatenate(predictions, axis=1)
        weights = jnp.linspace(1.0, 4.0, prediction.shape[1])[None, :, None]
        weights = weights / jnp.mean(weights)
        mse = jnp.mean(weights * (prediction - batch[3]) ** 2)
        terminal = float(values.get("terminal_loss_weight", 0.0)) * jnp.mean((prediction[:, -1] - batch[3][:, -1]) ** 2)
        return mse + terminal

    @jax.jit
    def chained_step(parameters, optimizer_state, batch):
        loss, gradient = jax.value_and_grad(chained_loss_fn)(parameters, batch)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, parameters)
        return optax.apply_updates(parameters, updates), optimizer_state, loss

    key = jax.random.key(values["seed"] + 1); best = float("inf"); best_params = params; last_improvement = 0
    history = {"step": [], "train_loss": [], "validation_mse": [], "selection_score": []}
    for update in range(1, values["updates"] + 1):
        active, probabilities = _schedule(values, update)
        key, choose_key, batch_key = jax.random.split(key, 3)
        horizon = active[int(jax.random.choice(choose_key, len(active), p=jnp.asarray(probabilities)))]
        arrays = train_arrays[horizon]
        use_chained = chained_arrays is not None and bool(jax.random.bernoulli(batch_key, float(values["chained_loss_probability"])))
        if use_chained:
            arrays, size = chained_arrays, int(values.get("batch_size_chained", 16))
        else:
            size = int(values.get(f"batch_size_h{horizon}", values["batch_size"]))
        indices = jax.random.randint(batch_key, (size,), 0, arrays[0].shape[0])
        host_indices = np.asarray(indices)
        batch = tuple(jnp.asarray(item[host_indices]) for item in arrays)
        params, state, loss = chained_step(params, state, batch) if use_chained else step(params, state, batch)
        if update % values["validation_interval"] == 0 or update == values["updates"]:
            metrics = _validate(params, validation_arrays, bundle.metadata["dt"], spec, legacy, bundle.normalization, values["validation_batch_size"])
            metrics["full_validation"], _ = _sequential_validation(params, bundle.validation, bundle.metadata["dt"], spec, legacy, bundle.normalization)
            score = _selection_score(metrics, values["dataset"])
            rejected, _ = _reject(metrics, params, bundle.validation, spec, bundle.normalization)
            history["step"].append(update); history["train_loss"].append(float(loss)); history["validation_mse"].append(metrics["validation_mse"]); history["selection_score"].append(score)
            if np.isfinite(score) and not rejected and score < best:
                best, best_params = score, jax.tree.map(lambda value: value.copy(), params)
                last_improvement = update
            if update >= int(values.get("minimum_updates_before_stopping", values["updates"])) and update - last_improvement >= int(values.get("early_stopping_patience", values["updates"])):
                break

    metrics = _validate(best_params, validation_arrays, bundle.metadata["dt"], spec, legacy, bundle.normalization, values["validation_batch_size"])
    metrics["full_validation"], continuous_predictions = _sequential_validation(best_params, bundle.validation, bundle.metadata["dt"], spec, legacy, bundle.normalization)
    metrics["selection_score"] = _selection_score(metrics, values["dataset"])
    if not legacy:
        probe = jnp.zeros((spec.state_dim,)); power = power_diagnostics(best_params, probe, jnp.zeros((spec.input_dim,)), spec)
        metrics["zero_input_power"] = {key: float(value) for key, value in power.items()}
    metrics["rejected"], metrics["rejection_reasons"] = _reject(metrics, best_params, bundle.validation, spec, bundle.normalization)
    if legacy:
        save_parameters(output / "best_validation_parameters.npz", best_params)
    else:
        _save_parameter_tree(output / "best_validation_parameters.npz", best_params)
        _save_parameter_tree(output / "final_parameters.npz", params)
    np.savez(output / "training_history.npz", **{key: np.asarray(value) for key, value in history.items()})
    np.savez(output / "continuous_validation_predictions.npz", **continuous_predictions)
    (output / "diagnostics" / "nonzero_input_power.json").write_text(json.dumps(metrics["nonzero_input_power"], indent=2, sort_keys=True))
    (output / "metrics_validation.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    required = ["resolved_config.yaml", "normalization.json", "fold_audit.json", "manifest.json", "best_validation_parameters.npz", "final_parameters.npz", "training_history.npz", "continuous_validation_predictions.npz", "metrics_validation.json", "diagnostics/nonzero_input_power.json"]
    verification = {"required_artifacts": required, "missing": [name for name in required if not (output / name).exists()]}
    verification["passed"] = not verification["missing"] and not metrics["rejected"]
    (output / "artifact_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True))
    manifest["finished_utc"] = datetime.now(UTC).isoformat(); manifest["artifact_verification_passed"] = verification["passed"]; (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output / "COMPLETED").write_text("completed\n")
    return metrics
