"""Open-loop latent PortHNN-u training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .artifacts import save_parameters
from .config import ExperimentConfig
from .data import DatasetBundle, Realization, WindowBatch
from .model import ModelSpec, damping_vector, decode, encode_initial_state, forcing, init_params, rollout


@dataclass(frozen=True)
class TrainingResult:
    params: dict
    history: dict[str, np.ndarray]
    best_validation_loss: float


def _model_spec(config: ExperimentConfig) -> ModelSpec:
    return ModelSpec(
        state_dim=config.state_dim,
        input_dim=1,
        output_dim=1,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        damping=config.damping,
        encoder_hidden_dims=config.encoder_hidden_dims,
    )


def _predict_batch(params, u_init, y_init, u_rollout, dt: float, spec: ModelSpec):
    def predict(one_u_init, one_y_init, one_u_rollout):
        state = encode_initial_state(params, one_u_init, one_y_init, spec)
        return decode(params, rollout(params, state, one_u_rollout, dt, spec))

    return jax.vmap(predict)(u_init, y_init, u_rollout)


def _l2_tree(tree) -> jax.Array:
    return sum(jnp.sum(value**2) for value in jax.tree.leaves(tree))


def train(config: ExperimentConfig, bundle: DatasetBundle, train_windows: WindowBatch, validation_windows: WindowBatch) -> TrainingResult:
    spec = _model_spec(config)
    dt = bundle.benchmark_metadata["dt"]
    params = init_params(jax.random.key(config.seed), spec, config.init_window)

    def loss_fn(params, u_init, y_init, u_rollout, y_target):
        prediction = _predict_batch(params, u_init, y_init, u_rollout, dt, spec)
        mse = jnp.mean((prediction - y_target) ** 2)
        force_l1 = jnp.mean(jnp.abs(jax.vmap(lambda values: forcing(params, values, spec))(u_rollout)))
        regularization = (
            config.lambda_force_l1 * force_l1
            + config.lambda_damping_l1 * jnp.mean(jnp.abs(damping_vector(params, spec)))
            + config.lambda_encoder_l2 * _l2_tree(params["encoder"])
            + config.lambda_readout_l2 * _l2_tree(params["readout"])
        )
        return mse + regularization, (mse, force_l1)

    schedule = lambda step: config.initial_learning_rate * 0.1 ** jnp.minimum(3, (step * 4) // config.iterations)
    optimizer = optax.chain(optax.clip_by_global_norm(config.gradient_clip_norm), optax.adam(schedule))
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state, u_init, y_init, u_rollout, y_target):
        (loss, details), gradient = jax.value_and_grad(loss_fn, has_aux=True)(params, u_init, y_init, u_rollout, y_target)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, params)
        return optax.apply_updates(params, updates), optimizer_state, loss, details[0]

    @jax.jit
    def validation_step(params, u_init, y_init, u_rollout, y_target):
        return loss_fn(params, u_init, y_init, u_rollout, y_target)[0]

    train_arrays = tuple(jnp.asarray(values) for values in (
        train_windows.u_init, train_windows.y_init, train_windows.u_rollout, train_windows.y_target
    ))
    validation_arrays = tuple(jnp.asarray(values) for values in (
        validation_windows.u_init, validation_windows.y_init, validation_windows.u_rollout, validation_windows.y_target
    ))
    key = jax.random.key(config.seed + 1)
    history = {"step": [], "train_loss": [], "train_mse": [], "validation_loss": []}
    best_validation = float("inf")
    best_params = params
    for step in range(1, config.iterations + 1):
        key, batch_key = jax.random.split(key)
        indices = jax.random.randint(batch_key, (config.batch_size,), 0, len(train_windows))
        batch = tuple(values[indices] for values in train_arrays)
        params, optimizer_state, loss, mse = train_step(params, optimizer_state, *batch)
        if step % config.validation_every == 0 or step == config.iterations:
            validation_losses = []
            for start in range(0, len(validation_windows), config.batch_size):
                end = min(start + config.batch_size, len(validation_windows))
                validation_losses.append(float(validation_step(params, *(value[start:end] for value in validation_arrays))))
            validation_loss = float(np.mean(validation_losses))
            history["step"].append(step)
            history["train_loss"].append(float(loss))
            history["train_mse"].append(float(mse))
            history["validation_loss"].append(validation_loss)
            if np.isfinite(validation_loss) and validation_loss < best_validation:
                best_validation = validation_loss
                best_params = jax.tree.map(lambda value: value.copy(), params)
    return TrainingResult(
        best_params,
        {name: np.asarray(values) for name, values in history.items()},
        best_validation,
    )


def save_training_result(path: Path, result: TrainingResult) -> None:
    save_parameters(path / "parameters.npz", result.params)
    np.savez(path / "training_history.npz", **result.history)
    (path / "metrics_validation.json").write_text(json.dumps({
        "selection_metric": "validation_rollout_mse",
        "best_validation_loss": result.best_validation_loss,
    }, indent=2, sort_keys=True))


def evaluate(params, config: ExperimentConfig, bundle: DatasetBundle, output_path: Path) -> dict:
    spec = _model_spec(config)
    metrics = {}
    for realization in bundle.test:
        if len(realization.inputs) <= config.init_window:
            raise ValueError(f"{realization.name}: no samples remain after initialization")
        u_init = jnp.asarray(realization.inputs[:config.init_window])
        y_init = jnp.asarray(realization.outputs[:config.init_window])
        u_scored = jnp.asarray(realization.inputs[config.init_window:])
        y_scored = realization.outputs[config.init_window:]
        initial_state = encode_initial_state(params, u_init, y_init, spec)
        prediction = np.asarray(decode(params, rollout(params, initial_state, u_scored, realization.dt, spec)))
        physical_prediction = prediction * bundle.normalization.output_scale + bundle.normalization.output_mean
        physical_target = y_scored * bundle.normalization.output_scale + bundle.normalization.output_mean
        rmse = float(np.sqrt(np.mean((physical_prediction - physical_target) ** 2)))
        normalized_rmse = float(np.sqrt(np.mean((prediction - y_scored) ** 2)))
        nrmse_std = float(normalized_rmse / np.std(y_scored))
        metrics[realization.name] = {
            "rmse_physical": rmse,
            "normalized_rmse": normalized_rmse,
            "nrmse_std": nrmse_std,
            "scored_samples": len(y_scored),
        }
        np.savez(
            output_path / "predictions" / f"{realization.name}.npz",
            y_prediction_normalized=prediction,
            y_target_normalized=y_scored,
            y_prediction_physical=physical_prediction,
            y_target_physical=physical_target,
            input_normalized=np.asarray(u_scored),
        )
    metrics["aggregate_rmse_physical"] = float(np.mean([value["rmse_physical"] for value in metrics.values()]))
    (output_path / "metrics_test.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics