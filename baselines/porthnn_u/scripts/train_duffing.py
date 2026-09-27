"""Train and export the matched PortHNN-u Duffing baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from porthnn_u.duffing import (
    forcing,
    hamiltonian,
    initialize,
    parameter_count,
    rk4_step,
    rollout,
    save_parameters,
)


DATASET = Path(os.environ.get("PORTHNN_U_DUFFING_DATASET", "datasets/duffing_doublewell/parameters/duffing_v1.npz"))
OUTPUT = ROOT / "results" / "porthnn_u" / "duffing"
EXTERNAL_COMMIT = "dc5e2088382c1db72bc2ea64d9f0c8e00ae038d6"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _flatten_transitions(states: np.ndarray, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        states[:, :-1].reshape(-1, 2),
        inputs.reshape(-1),
        states[:, 1:].reshape(-1, 2),
    )


def train(
    output_dir: Path,
    *,
    iterations: int = 10_000,
    batch_size: int = 200,
    learning_rate: float = 1e-3,
    seed: int = 0,
    hidden_dims: tuple[int, ...] = (200, 200, 200),
    smoke: bool = False,
) -> dict:
    data = np.load(DATASET)
    train_now, train_input, train_next = _flatten_transitions(data["z_train"], data["u_train"])
    val_now, val_input, val_next = _flatten_transitions(data["z_val"], data["u_val"])
    if smoke:
        iterations = 2
        batch_size = 16
        train_now, train_input, train_next = train_now[:256], train_input[:256], train_next[:256]
        val_now, val_input, val_next = val_now[:256], val_input[:256], val_next[:256]

    dt = 0.02
    lambda_force = 1e-6
    lambda_damping = 1e-6
    params = initialize(seed=seed, hidden_dims=hidden_dims)
    boundaries_and_scales = {
        max(iterations // 4, 1): 0.1,
        max(iterations // 2, 2): 0.1,
        max(3 * iterations // 4, 3): 0.1,
    }
    schedule = optax.piecewise_constant_schedule(learning_rate, boundaries_and_scales)
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(params)

    def loss_function(model_params, now, measured_input, target):
        prediction = jax.vmap(lambda state, value: rk4_step(model_params, state, value, dt))(now, measured_input)
        state_loss = jnp.mean((prediction - target) ** 2)
        force_values = jax.vmap(lambda value: forcing(model_params, value))(measured_input)
        regularization = lambda_force * jnp.mean(jnp.abs(force_values)) + lambda_damping * jnp.abs(model_params["damping"])
        return state_loss + regularization, (state_loss, regularization)

    @jax.jit
    def update(model_params, state, now, measured_input, target):
        (loss, auxiliary), gradients = jax.value_and_grad(loss_function, has_aux=True)(
            model_params, now, measured_input, target
        )
        updates, state = optimizer.update(gradients, state, model_params)
        return optax.apply_updates(model_params, updates), state, loss, auxiliary

    validation_size = min(4096, len(val_now))
    validation_indices = np.random.default_rng(17).choice(len(val_now), validation_size, replace=False)
    val_batch = tuple(jnp.asarray(array[validation_indices]) for array in (val_now, val_input, val_next))

    @jax.jit
    def validation_loss(model_params):
        return loss_function(model_params, *val_batch)[0]

    rng = np.random.default_rng(seed)
    history_steps, train_losses, validation_losses = [], [], []
    best_validation = float("inf")
    best_params = params
    validation_every = 1 if smoke else 100
    started = time.time()
    for iteration in range(iterations):
        indices = rng.integers(0, len(train_now), size=batch_size)
        batch = tuple(jnp.asarray(array[indices]) for array in (train_now, train_input, train_next))
        params, optimizer_state, loss, auxiliary = update(params, optimizer_state, *batch)
        if iteration == 0 or (iteration + 1) % validation_every == 0 or iteration + 1 == iterations:
            current_validation = float(validation_loss(params))
            history_steps.append(iteration + 1)
            train_losses.append(float(loss))
            validation_losses.append(current_validation)
            if current_validation < best_validation:
                best_validation = current_validation
                best_params = jax.tree_util.tree_map(lambda value: value.copy(), params)
            print(
                f"iteration={iteration + 1} train={float(auxiliary[0]):.8e} "
                f"validation={current_validation:.8e} damping={float(params['damping']):.6f}",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_parameters(output_dir / "parameters.npz", best_params)
    np.savez_compressed(
        output_dir / "training_history.npz", steps=np.asarray(history_steps),
        train_loss=np.asarray(train_losses), validation_loss=np.asarray(validation_losses),
    )
    config = {
        "label": "PortHNN-u (input-conditioned reimplementation)",
        "architecture_source": "Desai et al., PRE 104, 034312 (2021)",
        "external_commit": EXTERNAL_COMMIT,
        "adaptation": "F_theta(t) replaced by F_theta(u); no state dependence added",
        "activation": "tanh",
        "official_source_activation": "sine",
        "hidden_layers_per_branch": len(hidden_dims),
        "hidden_dims": list(hidden_dims),
        "parameter_count": parameter_count(best_params),
        "training_mode": "embedded single-step RK4 from state/input observations",
        "dt": dt, "iterations": iterations, "batch_size": batch_size,
        "optimizer": "Adam", "initial_learning_rate": learning_rate,
        "learning_rate_schedule": "multiply by 0.1 at each quarter budget",
        "lambda_force_l1": lambda_force, "lambda_damping_l1": lambda_damping,
        "selection": "minimum fixed validation-subsample one-step loss",
        "seed": seed, "smoke": smoke,
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "normalization": "identity physical coordinates",
        "elapsed_seconds": time.time() - started,
        "best_validation_loss": best_validation,
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "provenance.json", {
        "status": "adapted reimplementation, not official reproduction",
        "official_repository": "https://github.com/shaandesai1/PortHNN",
        "external_commit": EXTERNAL_COMMIT,
        "official_runtime_available": False,
        "reason": "PyTorch unavailable in host and project container",
    })

    if not smoke:
        test_states = np.asarray(data["z_test"])
        test_inputs = np.asarray(data["u_test"])[..., 0]
        rollout_batch = jax.jit(jax.vmap(lambda initial, values: rollout(best_params, initial, values, dt)))
        predicted = np.asarray(rollout_batch(jnp.asarray(test_states[:, 0]), jnp.asarray(test_inputs)))
        error = predicted - test_states
        per_trajectory_rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
        rmse = float(np.sqrt(np.mean(error**2)))
        scale = float(np.sqrt(np.mean((test_states - np.mean(test_states, axis=(0, 1))) ** 2)))
        np.savez_compressed(
            output_dir / "predictions.npz", truth=test_states, predictions=predicted,
            inputs=test_inputs, per_trajectory_rmse=per_trajectory_rmse,
        )
        input_axis = np.linspace(-0.65, 0.65, 401, dtype=np.float32)
        force_axis = np.asarray(jax.vmap(lambda value: forcing(best_params, value))(jnp.asarray(input_axis)))
        np.savez_compressed(output_dir / "forcing_curve.npz", input=input_axis, learned_force=force_axis, true_force=input_axis)
        _write_json(output_dir / "metrics.json", {
            "test_rmse": rmse, "test_nrmse": rmse / scale,
            "median_trajectory_rmse": float(np.median(per_trajectory_rmse)),
            "maximum_trajectory_rmse": float(np.max(per_trajectory_rmse)),
            "learned_damping_parameter_N": float(best_params["damping"]),
            "physical_damping_parameter_N": -0.4,
            "forcing_rmse_on_input_grid": float(np.sqrt(np.mean((force_axis - input_axis) ** 2))),
        })
    return config


def main() -> None:
    global DATASET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DATASET,
                        help="Duffing .npz dataset; defaults to PORTHNN_U_DUFFING_DATASET or the documented relative path")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[200, 200, 200])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    DATASET = args.dataset
    result = train(
        args.output_dir, iterations=args.iterations, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed,
        hidden_dims=tuple(args.hidden_dims), smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
