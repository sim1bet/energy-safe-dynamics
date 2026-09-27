"""Development-only trainer for the constrained PortHNN-u factorial campaign."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from .artifacts import save_parameters
from .campaign import config_hash, source_tree_hash
from .data import make_campaign_windows, load_campaign_development
from .model import ModelSpec, damping_vector, decode, encode_initial_state, forcing, init_params, rollout


def _spec(values: dict) -> ModelSpec:
    return ModelSpec(
        state_dim=int(values["latent_dim"]), input_dim=1, output_dim=1,
        hidden_dims=tuple(values["hidden_dims"]), activation=values["hamiltonian_activation"],
        damping=values["damping_parameterization"], encoder_hidden_dims=tuple(values["encoder_hidden_dims"]),
        forcing_hidden_dims=tuple(values["forcing_hidden_dims"]),
        forcing_parameterization=values["forcing_parameterization"], force_cap=values["force_cap"],
        damping_floor=float(values["damping_floor"]),
    )


def _predict(params, u_init, y_init, u_rollout, dt, spec):
    def one(inputs, outputs, rollout_inputs):
        state = encode_initial_state(params, inputs, outputs, spec)
        return decode(params, rollout(params, state, rollout_inputs, dt, spec))
    return jax.vmap(one)(u_init, y_init, u_rollout)


def _tree_l2(tree):
    return sum(jnp.sum(value ** 2) for value in jax.tree.leaves(tree))


def _learning_rate(step, values: dict):
    warmup = int(values["warmup_updates"])
    total = int(values["updates"])
    start, end = float(values["learning_rate"]), float(values["learning_rate_end"])
    progress = jnp.clip((step - warmup) / max(1, total - warmup), 0.0, 1.0)
    cosine = end + 0.5 * (start - end) * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.where(step < warmup, start * (step + 1) / warmup, cosine)


def _schedule(values: dict, step: int) -> tuple[tuple[int, ...], np.ndarray]:
    for phase in values["training_schedule"]:
        if step <= int(phase["until"]):
            return tuple(phase["horizons"]), np.asarray(phase["probabilities"], dtype=np.float64)
    raise ValueError("training schedule does not cover requested updates")


def _arrays(windows):
    """Retain complete overlapping window banks in host memory."""
    return windows.u_init, windows.y_init, windows.u_rollout, windows.y_target


def _fold_audit(bundle, windows_by_horizon):
    audit = {"fold": bundle.metadata.get("fold"), "guard": bundle.metadata.get("guard"), "realizations": {}}
    for realization in bundle.train:
        entry = {"sample_count": len(realization.inputs), "input_mean": float(realization.inputs.mean()), "input_std": float(realization.inputs.std()), "output_mean": float(realization.outputs.mean()), "output_std": float(realization.outputs.std()), "window_counts": {str(horizon): int(sum(name == realization.name for name in windows.realization)) for horizon, windows in windows_by_horizon.items()}}
        if bundle.train_window_starts is not None:
            for label, selectors in (("train", bundle.train_window_starts), ("validation", bundle.validation_window_starts)):
                indices = selectors[realization.name]
                entry[f"{label}_start_sha256"] = __import__("hashlib").sha256(indices.tobytes()).hexdigest()
                entry[f"{label}_start_count"] = len(indices)
        audit["realizations"][realization.name] = entry
    return audit


def _validate_prediction_only(params, arrays_by_horizon, dt, spec, normalization, batch_size=8):
    """Accumulate selection metrics without regularizers or device-wide outputs."""
    total_sse = total_count = nonfinite = 0
    maximum_prediction = maximum_latent_norm = 0.0
    per_horizon = {}
    for horizon, arrays in arrays_by_horizon.items():
        horizon_sse = horizon_count = horizon_nonfinite = 0
        for start in range(0, arrays[0].shape[0], batch_size):
            batch = tuple(jnp.asarray(value[start:start + batch_size]) for value in arrays)
            prediction = _predict(params, batch[0], batch[1], batch[2], dt, spec)
            prediction = np.asarray(jax.block_until_ready(prediction))
            target = np.asarray(batch[3])
            finite = np.isfinite(prediction)
            horizon_nonfinite += int(prediction.size - finite.sum())
            if not finite.all():
                continue
            residual = prediction - target
            horizon_sse += float(np.sum(residual ** 2))
            horizon_count += residual.size
            maximum_prediction = max(maximum_prediction, float(np.max(np.abs(prediction))))
        total_sse += horizon_sse
        total_count += horizon_count
        nonfinite += horizon_nonfinite
        per_horizon[str(horizon)] = {
            "normalized_mse": horizon_sse / horizon_count if horizon_count else float("inf"),
            "sample_count": horizon_count,
            "nonfinite_count": horizon_nonfinite,
        }
    normalized_mse = total_sse / total_count if total_count else float("inf")
    normalized_rmse = float(np.sqrt(normalized_mse))
    physical_rmse = normalized_rmse * float(normalization.output_scale[0])
    return {
        "validation_mse": normalized_mse,
        "normalized_rmse": normalized_rmse,
        "physical_rmse": physical_rmse,
        "nrmse_std": normalized_rmse,
        "sample_count": total_count,
        "nonfinite_count": nonfinite,
        "max_abs_prediction": maximum_prediction,
        "max_latent_norm": maximum_latent_norm,
        "per_horizon": per_horizon,
    }


def _sequential_validation(params, realizations, dt, spec, normalization, chunk_size=64):
    """Evaluate contiguous validation in bounded chunks while carrying the latent state."""
    result = {}
    for realization in realizations:
        u_init = jnp.asarray(realization.inputs[:realization.init_window])
        y_init = jnp.asarray(realization.outputs[:realization.init_window])
        state = encode_initial_state(params, u_init, y_init, spec)
        chunks = []
        inputs = realization.inputs[realization.init_window:]
        for start in range(0, len(inputs), chunk_size):
            states = rollout(params, state, jnp.asarray(inputs[start:start + chunk_size]), dt, spec)
            chunks.append(np.asarray(jax.block_until_ready(decode(params, states))))
            state = states[-1]
        prediction = np.concatenate(chunks, axis=0)
        target = realization.outputs[realization.init_window:]
        finite = np.isfinite(prediction).all()
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2))) if finite else float("inf")
        result[realization.name] = {"normalized_rmse": rmse, "nrmse_std": rmse, "physical_rmse": rmse * float(normalization.output_scale[0]), "nonfinite_count": int(prediction.size - np.isfinite(prediction).sum())}
    return result


def train_campaign(config_path: str | Path, output: str | Path, fold: int | None = None) -> dict:
    """Train one resolved development config, never materializing official tests."""
    config_path, output = Path(config_path), Path(output)
    values = yaml.safe_load(config_path.read_text())
    if values.get("development_only") is not True or values.get("official_test_access") != "forbidden":
        raise ValueError("campaign training requires a development-only, test-embargoed config")
    if values["dataset"] == "ced" and fold not in range(4):
        raise ValueError("CED campaign training requires --fold in {0,1,2,3}")
    if values["dataset"] == "silverbox" and fold is not None:
        raise ValueError("Silverbox campaign uses its fixed contiguous development split")
    if (output / "COMPLETED").exists():
        return {"status": "already_completed", "output": str(output)}

    spec = _spec(values)
    bundle = load_campaign_development(values["dataset"], init_window=int(values["init_window"]), max_horizon=int(values["max_horizon"]), fold=fold)
    horizons = sorted({horizon for phase in values["training_schedule"] for horizon in phase["horizons"]} | set(values["validation_horizons"]))
    train_windows = {horizon: make_campaign_windows(bundle, split="train", init_window=values["init_window"], horizon=horizon, stride=1) for horizon in horizons}
    validation_windows = {horizon: make_campaign_windows(bundle, split="validation", init_window=values["init_window"], horizon=horizon, stride=1) for horizon in values["validation_horizons"]}
    train_arrays = {horizon: _arrays(windows) for horizon, windows in train_windows.items()}
    validation_arrays = {horizon: _arrays(windows) for horizon, windows in validation_windows.items()}

    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics").mkdir(exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(values, sort_keys=False))
    (output / "normalization.json").write_text(json.dumps(bundle.normalization.to_dict(), indent=2, sort_keys=True))
    (output / "fold_audit.json").write_text(json.dumps(_fold_audit(bundle, train_windows), indent=2, sort_keys=True))
    manifest = {"config_id": values["config_id"], "config_sha256": config_hash(values), "source_tree_hash": source_tree_hash(Path.cwd()), "dataset": values["dataset"], "fold": fold, "job_id": os.environ.get("JOB_ID"), "started_utc": datetime.now(UTC).isoformat(), "official_test_access": "forbidden"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    params = init_params(jax.random.key(int(values["seed"])), spec, int(values["init_window"]))
    optimizer = optax.chain(optax.clip_by_global_norm(float(values["gradient_clip_global_norm"])), optax.adam(lambda step: _learning_rate(step, values)))
    opt_state = optimizer.init(params)

    def loss_fn(params, batch):
        u_init, y_init, u_rollout, y_target = batch
        prediction = _predict(params, u_init, y_init, u_rollout, bundle.metadata["dt"], spec)
        time_weight = jnp.linspace(values["late_loss_weight_start"], values["late_loss_weight_end"], y_target.shape[1])[None, :, None]
        time_weight = time_weight / jnp.mean(time_weight)
        mse = jnp.mean(time_weight * (prediction - y_target) ** 2)
        terminal = float(values.get("terminal_loss_weight", 0.0)) * jnp.mean((prediction[:, -1] - y_target[:, -1]) ** 2)
        force_values = jax.vmap(lambda inputs: forcing(params, inputs, spec))(u_rollout.reshape((-1, 1)))
        jacobian = jax.vmap(jax.jacrev(lambda inputs: forcing(params, inputs, spec)))(u_rollout.reshape((-1, 1)))
        regularization = (float(values["lambda_force_l1"]) * jnp.mean(jnp.abs(force_values)) + float(values["lambda_force_jacobian"]) * jnp.mean(jacobian ** 2) + float(values["lambda_encoder_l2"]) * _tree_l2(params["encoder"]) + float(values["lambda_readout_l2"]) * _tree_l2(params["readout"]))
        return mse + terminal + regularization, mse

    @jax.jit
    def step(params, opt_state, batch):
        (loss, mse), gradient = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(gradient, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, mse

    key = jax.random.key(int(values["seed"]) + 1)
    best_loss, best_params, last_improvement = float("inf"), params, 0
    history = {"step": [], "train_loss": [], "validation_mse": []}
    for update in range(1, int(values["updates"]) + 1):
        active_horizons, probabilities = _schedule(values, update)
        key, horizon_key, batch_key = jax.random.split(key, 3)
        horizon = active_horizons[int(jax.random.choice(horizon_key, len(active_horizons), p=jnp.asarray(probabilities)))]
        arrays, count = train_arrays[horizon], len(train_windows[horizon])
        batch_size = 32 if values["dataset"] == "silverbox" and horizon in (384, 768) else int(values["batch_size"])
        indices = jax.random.randint(batch_key, (batch_size,), 0, count)
        host_indices = np.asarray(indices)
        params, opt_state, loss, _ = step(params, opt_state, tuple(jnp.asarray(item[host_indices]) for item in arrays))
        if update % int(values["validation_interval"]) == 0 or update == int(values["updates"]):
            validation = _validate_prediction_only(
                params, validation_arrays, bundle.metadata["dt"], spec, bundle.normalization
            )
            validation_mse = validation["validation_mse"]
            history["step"].append(update); history["train_loss"].append(float(loss)); history["validation_mse"].append(validation_mse)
            if np.isfinite(validation_mse) and validation_mse < best_loss:
                best_loss, best_params = validation_mse, jax.tree.map(lambda item: item.copy(), params)
                last_improvement = update
            if update >= 6000 and update - last_improvement >= int(values.get("early_stopping_patience", values["updates"])):
                break

    damping = np.asarray(damping_vector(best_params, spec))
    diagnostics = _validate_prediction_only(
        best_params, validation_arrays, bundle.metadata["dt"], spec, bundle.normalization
    )
    if values["dataset"] == "silverbox":
        diagnostics["full_validation"] = _sequential_validation(
            best_params, bundle.validation, bundle.metadata["dt"], spec, bundle.normalization
        )
    diagnostics.update({
        "damping": damping.tolist(),
        "damping_strictly_negative": bool(np.all(damping < 0)),
        "rejected": bool(not np.isfinite(best_loss) or diagnostics["nonfinite_count"] or np.any(damping >= 0)),
    })
    save_parameters(output / "parameters.npz", best_params)
    save_parameters(output / "best_validation_parameters.npz", best_params)
    save_parameters(output / "final_parameters.npz", params)
    np.savez(output / "training_history.npz", **{name: np.asarray(item) for name, item in history.items()})
    (output / "metrics_validation.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    (output / "diagnostics" / "stability.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output / "COMPLETED").write_text("completed\n")
    return diagnostics
