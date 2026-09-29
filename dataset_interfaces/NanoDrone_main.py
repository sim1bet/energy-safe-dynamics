# Author: Simone Betteti
"""State-observed pH-EBM identification for the 2026 Nano-drone benchmark."""
from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "EBM_model", _ROOT / "Interface_code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
import optax

import EBM_param_fields as pf
from EBM_class import (
    LayerSpec,
    activation_polynomial_stable,
    activation_tanh,
    lagrangian_polynomial_stable,
    lagrangian_tanh,
)
from EBM_param_fields import init_params, make_grad_energy
from EBM_rollout import make_batch_rollout_fn
from EBM_training import make_train_step, tree_all_finite
from Duffing_DoubleWell_main import make_state_observed_loss_fn
from experiment_paths import get_experiment_paths
from nanodrone.data import (
    INPUT_COLUMNS,
    STATE_COLUMNS,
    AffineScaler,
    NanoDroneTrajectory,
    fit_scalers,
    load_nanodrone_dataset,
    make_multitrajectory_dataset,
    transform_inputs,
    write_dataset_manifest,
)
from nanodrone.evaluation import (
    aggregate_run_metrics,
    compute_run_metrics,
    save_benchmark_metrics,
)
from nanodrone.plotting import (
    plot_horizon_metrics,
    plot_motor_inputs,
    plot_orientation_error,
    plot_position_3d,
    plot_reconstruction_windows,
    plot_training_history,
)
from nanodrone.rotational import (
    make_chirp_frequency_loss_fn,
    robust_scale,
    torque_targets_for_windows,
)
from post_training_eval import (
    run_theory_aligned_certificates,
    save_checkpoint,
    save_json,
    write_run_status,
)


EXPERIMENT_NAME = "nanodrone"
DT = 0.01


def get_config() -> dict:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "d": 12, "m_ports": 4, "n_obs": 12,
        "state_observed_mode": True, "use_identity_readout": True,
        "hidden": 64, "layer_dims": [96, 48],
        "first_layer_type": "tanh", "second_layer_type": "polynomial_stable",
        "p_1": 1.0, "p_2": 4.0, "softmax_groups": 1,
        "b_init_scale": 1.0, "vf_scale": 1.0,
        "use_feedthrough": False, "use_nonlinear_readout": False,
        "use_saturating_readout": False, "use_state_damping": False,
        "use_state_interconnection": False,
        "use_quadratic_interconnection": False,
        "use_cubic_interconnection": False,
        "use_rich_interconnection": False, "use_rich_damping": False,
        "use_rich_input_matrix": False, "ph_field_width": 64,
        "input_matrix_structure": "unrestricted",
        "use_input_gain": False, "use_saturating_input": False,
        "use_input_aware_encoder": False, "damping_scale": 1.0,
        "chol_clip_exp": 2.0,
        "batch_size": 64, "seq_len": 25, "stride": 10, "init_win": 1,
        "n_epochs": 300, "lr": 3e-4, "full_phase_lr_scale": 0.25,
        "use_lr_decay": False, "rollout_integrator": "rk4",
        "rollout_stop_grad_epochs": 10,
        "observation_loss": "huber", "huber_delta": 1.0,
        "w_derivative": 0.0, "normalized_increment_loss": False,
        "rollout_increment_loss_weight": 0.0,
        "fast_transition_loss_weight": 1.0,
        "nanodrone_state_loss_weights": [1.0] * 12,
        "nanodrone_balance_families": False,
        "nanodrone_balance_runs": False,
        "nanodrone_metric_aligned_loss": False,
        "nanodrone_validation_selection": "weighted_start_nrmse",
        "weight_obs_by_magnitude": False, "amplitude_weight_power": 0.0,
        "rise_weight_power": 0.0, "w_rollout": 1.0, "w_passivity": 0.0,
        "w_reg": 1e-6, "trunk_reg_mult": 1.0, "w_reg_B": 0.0,
        "w_state": 0.0, "weight_decay": 1e-4,
        "grad_clip_norm": 1.0, "grad_norm_guard": 10000.0,
        "max_bad_updates": 200, "grad_e_clip": 100.0,
        "xdot_clip": 100.0, "state_norm_clip": 30.0, "x0_norm_max": 30.0,
        "input_noise_std": 0.0, "init_noise_std": 0.0,
        "rotational_momentum_loss_weight": 0.0,
        "rotational_shape_loss_weight": 0.0,
        "rotational_frequency_loss_weight": 0.0,
        "rotational_frequency_batch_size": 27,
        "rotational_auxiliary_ramp_epochs": 50,
        "use_iterative_init": False, "init_multistart": 1,
        "enable_controller": False, "save_plots": True, "plot_max_points": 10000,
        "nanodrone_dataset_root": str(_ROOT / ".external" / "nanodrone-sysid-benchmark"),
        "nanodrone_input_representation": "motor_speed",
        "nanodrone_state_representation": "so3_log_12d",
        "nanodrone_split": "development", "nanodrone_validation_every": 20,
        "nanodrone_benchmark_horizon": 50, "nanodrone_run_stress_tests": True,
        "nanodrone_energy_quantile": 0.90, "nanodrone_eval_stride": 1,
        "nanodrone_long_rollout_steps": 500,
        "nanodrone_max_train_windows": 0, "nanodrone_max_eval_starts": 0,
        "nanodrone_train_flight_limit": 0,
        "stress_test_mode": "theory_aligned_only", "stress_smoke": False,
        "print_every": 10,
    }


DEFAULT_CONFIG = get_config()


def _validate_config(config: dict) -> None:
    if config.get("experiment_name") != EXPERIMENT_NAME:
        raise ValueError(f"Expected experiment_name={EXPERIMENT_NAME!r}")
    expected_ports = {
        "motor_speed": 4, "motor_speed_squared": 4, "rotor_wrench": 4,
        "dynamic_physical_port": 6, "combined_dynamic_port": 10,
    }.get(str(config["nanodrone_input_representation"]))
    if expected_ports is None or (
        int(config["d"]), int(config["m_ports"]), int(config["n_obs"])
    ) != (12, expected_ports, 12):
        raise ValueError(
            "Nano-drone requires d=12, n_obs=12, and m_ports matching its input representation"
        )
    if not config.get("state_observed_mode") or not config.get("use_identity_readout"):
        raise ValueError("Primary Nano-drone runs require state-observed mode and identity readout")
    forbidden = (
        "use_feedthrough", "use_nonlinear_readout", "use_saturating_readout",
        "use_input_gain", "use_saturating_input", "enable_controller",
        "use_iterative_init", "use_input_aware_encoder",
    )
    enabled = [name for name in forbidden if config.get(name, False)]
    if enabled:
        raise ValueError(f"Nano-drone primary configuration enables forbidden flags: {enabled}")
    if config.get("first_layer_type") != "tanh" or config.get("second_layer_type") != "polynomial_stable":
        raise ValueError("Nano-drone v1 requires tanh then polynomial_stable energy layers")
    if config.get("nanodrone_split") not in {"development", "final"}:
        raise ValueError("nanodrone_split must be 'development' or 'final'")
    if config.get("nanodrone_input_representation") not in {
        "motor_speed", "motor_speed_squared", "rotor_wrench",
        "dynamic_physical_port", "combined_dynamic_port",
    }:
        raise ValueError(
            "Unknown nanodrone_input_representation"
        )
    if config.get("input_matrix_structure") != "unrestricted":
        raise ValueError("NanoDrone requires input_matrix_structure='unrestricted'")
    if config.get("stress_test_mode") != "theory_aligned_only":
        raise ValueError("Nano-drone requires stress_test_mode='theory_aligned_only'")
    if config.get("nanodrone_validation_selection") not in {
        "weighted_start_nrmse", "equal_family_nrmse"
    }:
        raise ValueError(
            "nanodrone_validation_selection must be 'weighted_start_nrmse' "
            "or 'equal_family_nrmse'"
        )
    state_weights = config.get("nanodrone_state_loss_weights", [1.0] * 12)
    if len(state_weights) != 12 or any(float(value) <= 0.0 for value in state_weights):
        raise ValueError("nanodrone_state_loss_weights must contain 12 positive values")
    if float(config.get("rollout_increment_loss_weight", 0.0)) < 0.0:
        raise ValueError("rollout_increment_loss_weight must be non-negative")
    for name in ("input_noise_std", "init_noise_std"):
        if float(config.get(name, 0.0)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in (
        "rotational_momentum_loss_weight", "rotational_shape_loss_weight",
        "rotational_frequency_loss_weight",
    ):
        if float(config.get(name, 0.0)) < 0.0:
            raise ValueError(f"{name} must be non-negative")


def _apply_runtime_flags(config: dict) -> None:
    values = {
        "CHOL_CLIP_EXP": config["chol_clip_exp"],
        "DAMPING_SCALE": config["damping_scale"],
        "B_INIT_SCALE": config["b_init_scale"],
        "VF_SCALE": config["vf_scale"],
        "USE_IDENTITY_READOUT": config["use_identity_readout"],
        "USE_FEEDTHROUGH": config["use_feedthrough"],
        "USE_NONLINEAR_READOUT": config["use_nonlinear_readout"],
        "USE_SATURATING_READOUT": config["use_saturating_readout"],
        "USE_STATE_DAMPING": config["use_state_damping"],
        "USE_INPUT_GAIN": config["use_input_gain"],
        "USE_SATURATING_INPUT": config["use_saturating_input"],
        "USE_INPUT_AWARE_ENCODER": config["use_input_aware_encoder"],
        "USE_STATE_INTERCONNECTION": config["use_state_interconnection"],
        "USE_QUADRATIC_INTERCONNECTION": config["use_quadratic_interconnection"],
        "USE_CUBIC_INTERCONNECTION": config["use_cubic_interconnection"],
        "USE_RICH_INTERCONNECTION": config["use_rich_interconnection"],
        "USE_RICH_DAMPING": config["use_rich_damping"],
        "USE_RICH_INPUT_MATRIX": config["use_rich_input_matrix"],
        "PH_FIELD_WIDTH": config["ph_field_width"],
        "INPUT_MATRIX_STRUCTURE": config["input_matrix_structure"],
        "GRAD_E_CLIP": config["grad_e_clip"],
        "XDOT_CLIP": config["xdot_clip"],
        "X0_NORM_MAX": config["x0_norm_max"],
        "STATE_NORM_CLIP": config["state_norm_clip"],
    }
    for name, value in values.items():
        setattr(pf, name, value)


def _build_layers(config: dict) -> tuple:
    return (
        LayerSpec(lagrangian_tanh, activation_tanh, (float(config["p_1"]),)),
        LayerSpec(
            lagrangian_polynomial_stable,
            activation_polynomial_stable,
            (float(config["p_2"]),),
        ),
    )


def _normalized_trajectory(
    trajectory: NanoDroneTrajectory,
    state_scaler: AffineScaler,
    input_scaler: AffineScaler,
    input_representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = transform_inputs(trajectory.u, input_representation)
    return state_scaler.transform(trajectory.y), input_scaler.transform(inputs)


def _weight_decay_mask(params):
    """Exclude unused observed-state encoder/readout parameters from AdamW."""
    mask = jax.tree_util.tree_map(lambda _: True, params)
    inert = {
        name: False
        for name in (
            "C_state", "c_bias", "D_feed", "C_ro1", "b_ro1", "C_ro2",
            "y_sat_pos_raw", "y_sat_neg_raw", "W_enc", "b_enc",
        )
    }
    return mask._replace(trunk=mask.trunk._replace(**inert))


def _rolling_predictions(
    params,
    grad_energy,
    trajectories: list[NanoDroneTrajectory],
    state_scaler: AffineScaler,
    input_scaler: AffineScaler,
    config: dict,
) -> tuple[dict[str, dict], dict[str, dict]]:
    horizon = int(config["nanodrone_benchmark_horizon"])
    stride = int(config.get("nanodrone_eval_stride", 1))
    maximum = int(config.get("nanodrone_max_eval_starts", 0))
    batch_size = max(int(config.get("evaluation_batch_size", 256)), 1)
    rollout = make_batch_rollout_fn(
        grad_energy, 12, int(config["m_ports"]), DT,
        integrator=str(config["rollout_integrator"])
    )
    predictions, metrics = {}, {}
    for trajectory in trajectories:
        windows = make_multitrajectory_dataset(
            [trajectory], horizon, stride, state_scaler, input_scaler,
            input_representation=config["nanodrone_input_representation"],
        )
        if maximum > 0:
            keep = slice(0, maximum)
        else:
            keep = slice(None)
        x0 = windows.x0[keep]
        control = windows.u_seq[keep]
        target = windows.target[keep]
        starts = windows.start_index[keep]
        chunks = []
        for begin in range(0, len(x0), batch_size):
            end = begin + batch_size
            states, _, observed = rollout(
                params,
                jnp.asarray(x0[begin:end]),
                jnp.asarray(control[begin:end]),
                stop_grad=True,
            )
            if states.shape != observed.shape:
                raise RuntimeError("Identity readout/state rollout shape mismatch")
            chunks.append(np.asarray(observed))
        predicted = np.concatenate(chunks, axis=0)
        if predicted.shape != target.shape:
            raise RuntimeError(f"Prediction/target mismatch: {predicted.shape} != {target.shape}")
        normalized_state, normalized_input = _normalized_trajectory(
            trajectory, state_scaler, input_scaler,
            config["nanodrone_input_representation"],
        )
        long_steps = min(int(config["nanodrone_long_rollout_steps"]), len(normalized_state) - 1)
        long_state, _, long_observed = rollout(
            params,
            jnp.asarray(normalized_state[:1]),
            jnp.asarray(normalized_input[None, :long_steps]),
            stop_grad=True,
        )
        if long_state.shape != long_observed.shape:
            raise RuntimeError("Long identity-readout/state rollout shape mismatch")
        metrics[trajectory.name] = compute_run_metrics(target, predicted, state_scaler)
        predictions[trajectory.name] = {
            "trajectory": trajectory,
            "starts": starts,
            "input_normalized": control,
            "true_normalized": target,
            "predicted_normalized": predicted,
            "long_true_normalized": normalized_state[1:long_steps + 1],
            "long_predicted_normalized": np.asarray(long_observed[0]),
        }
    return predictions, metrics


def _validation_log(metrics: dict[str, dict]) -> dict[str, float]:
    aggregate = aggregate_run_metrics(metrics)
    weighted_nrmse = np.average(
        [value["state_nrmse_normalized"] for value in metrics.values()],
        weights=[value["n_starts"] for value in metrics.values()],
    )
    equal_family_nrmse = np.mean([
        value["state_nrmse_normalized"] for value in metrics.values()
    ])
    return {
        "val/state_nrmse": float(weighted_nrmse),
        "val/equal_family_state_nrmse": float(equal_family_nrmse),
        "val/position_rmse": aggregate["aggregate_mae"]["position"],
        "val/velocity_rmse": aggregate["aggregate_mae"]["velocity"],
        "val/rotation_geodesic_deg": float(np.degrees(aggregate["aggregate_mae"]["rotation"])),
        "val/angular_velocity_rmse": aggregate["aggregate_mae"]["omega"],
        "val/h50_position_mae": float(aggregate["mae_by_horizon"]["position"][-1]),
        "val/h50_velocity_mae": float(aggregate["mae_by_horizon"]["velocity"][-1]),
        "val/h50_rotation_mae_deg": float(np.degrees(aggregate["mae_by_horizon"]["rotation"][-1])),
        "val/h50_omega_mae": float(aggregate["mae_by_horizon"]["omega"][-1]),
    }


def _sample_epoch_indices(
    trajectory_names: np.ndarray,
    train_count: int,
    used_count: int,
    rng: np.random.Generator,
    balance_families: bool,
    balance_runs: bool = False,
) -> np.ndarray:
    if not balance_families and not balance_runs:
        return rng.permutation(train_count)[:used_count]
    groups = np.asarray(trajectory_names[:train_count]) if balance_runs else np.asarray([
        str(name).split("_", 1)[0] for name in trajectory_names[:train_count]
    ])
    expected = np.unique(groups) if balance_runs else ("square", "random", "chirp")
    missing = [name for name in expected if not np.any(groups == name)]
    if missing:
        raise ValueError(f"Cannot balance absent NanoDrone families: {missing}")
    per_family = used_count // len(expected)
    sampled = [
        rng.choice(np.flatnonzero(groups == name), size=per_family, replace=True)
        for name in expected
    ]
    remainder = used_count - len(expected) * per_family
    if remainder:
        sampled.append(rng.choice(train_count, size=remainder, replace=False))
    return rng.permutation(np.concatenate(sampled))


def _augment_training_batch(
    x0: np.ndarray,
    u_seq: np.ndarray,
    rng: np.random.Generator,
    init_noise_std: float,
    input_noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    augmented_x0 = np.asarray(x0)
    augmented_u = np.asarray(u_seq)
    if init_noise_std > 0.0:
        augmented_x0 = augmented_x0 + rng.normal(
            0.0, init_noise_std, augmented_x0.shape
        ).astype(augmented_x0.dtype)
    if input_noise_std > 0.0:
        augmented_u = augmented_u + rng.normal(
            0.0, input_noise_std, augmented_u.shape
        ).astype(augmented_u.dtype)
    return augmented_x0, augmented_u


def _sample_chirp_spectral_indices(
    windows, trajectory_lengths: dict[str, int], batch_size: int, rng: np.random.Generator
) -> np.ndarray:
    strata = []
    for name in sorted(set(windows.trajectory_name)):
        length = trajectory_lengths[str(name)]
        starts = windows.start_index[windows.trajectory_name == name]
        source_indices = np.flatnonzero(windows.trajectory_name == name)
        for third in range(3):
            lower, upper = third * length / 3.0, (third + 1) * length / 3.0
            eligible = source_indices[(starts >= lower) & (starts < upper)]
            if len(eligible) == 0:
                raise RuntimeError(f"No Chirp spectral windows for {name} third {third}")
            strata.append(eligible)
    if batch_size % len(strata) != 0:
        raise ValueError(
            f"rotational_frequency_batch_size={batch_size} must be divisible by "
            f"the {len(strata)} run/third strata"
        )
    per_stratum = batch_size // len(strata)
    return rng.permutation(np.concatenate([
        rng.choice(indices, size=per_stratum, replace=len(indices) < per_stratum)
        for indices in strata
    ]))


def run_training(seed: int = 42, config: Optional[dict] = None, verbose: bool = True):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    _validate_config(cfg)
    _apply_runtime_flags(cfg)
    dataset = load_nanodrone_dataset(
        cfg["nanodrone_dataset_root"],
        include_official_test=cfg["nanodrone_split"] == "final",
    )
    training_key = "development_train" if cfg["nanodrone_split"] == "development" else "official_train"
    training_trajectories = dataset[training_key]
    flight_limit = int(cfg.get("nanodrone_train_flight_limit", 0))
    if flight_limit > 0:
        training_trajectories = training_trajectories[:flight_limit]
    input_representation = cfg["nanodrone_input_representation"]
    state_scaler, input_scaler = fit_scalers(training_trajectories, input_representation)
    train_windows = make_multitrajectory_dataset(
        training_trajectories,
        int(cfg["seq_len"]),
        int(cfg["stride"]),
        state_scaler,
        input_scaler,
        input_representation=input_representation,
    )
    maximum_windows = int(cfg.get("nanodrone_max_train_windows", 0))
    train_count = min(len(train_windows.x0), maximum_windows) if maximum_windows > 0 else len(train_windows.x0)
    if train_count <= 0:
        raise RuntimeError("No Nano-drone training windows")
    momentum_enabled = float(cfg["rotational_momentum_loss_weight"]) > 0.0
    shape_enabled = float(cfg["rotational_shape_loss_weight"]) > 0.0
    frequency_enabled = float(cfg["rotational_frequency_loss_weight"]) > 0.0
    torque_targets = (
        torque_targets_for_windows(training_trajectories, train_windows, int(cfg["seq_len"]))
        if momentum_enabled else None
    )
    omega_scale = robust_scale(np.concatenate([
        trajectory.y[:, 9:12] for trajectory in training_trajectories
    ])).tolist()
    if momentum_enabled:
        valid_torque = torque_targets[..., 3] > 0.0
        torque_scale = robust_scale(torque_targets[..., :3][valid_torque]).tolist()
    else:
        torque_scale = [1.0, 1.0, 1.0]
    if frequency_enabled:
        chirp_trajectories = [
            trajectory for trajectory in training_trajectories if trajectory.family == "chirp"
        ]
        frequency_windows = make_multitrajectory_dataset(
            chirp_trajectories, 128, 10, state_scaler, input_scaler,
            input_representation=input_representation,
        )
        chirp_lengths = {trajectory.name: len(trajectory.y) for trajectory in chirp_trajectories}
    else:
        frequency_windows = None
        chirp_lengths = {}

    layers = _build_layers(cfg)
    grad_energy = make_grad_energy(layers)
    params = init_params(
        jax.random.PRNGKey(seed), d=12, m=int(cfg["m_ports"]), n_obs=12, init_win=1,
        layer_dims=list(cfg["layer_dims"]), hidden=int(cfg["hidden"]), dt=DT,
    )
    batch_size = min(int(cfg["batch_size"]), train_count)
    n_batches = max(train_count // batch_size, 1)
    used_count = n_batches * batch_size
    stop_grad_epochs = int(cfg["rollout_stop_grad_epochs"])

    decay_mask = _weight_decay_mask(params)

    def optimizer(learning_rate):
        return optax.chain(
            optax.clip_by_global_norm(float(cfg["grad_clip_norm"])),
            optax.adamw(
                float(learning_rate), weight_decay=float(cfg["weight_decay"]),
                mask=decay_mask,
            ),
        )

    optimizer_stop = optimizer(cfg["lr"])
    optimizer_full = optimizer(float(cfg["lr"]) * float(cfg["full_phase_lr_scale"]))
    common_loss = {
        "grad_energy_fn": grad_energy, "d": 12, "m": int(cfg["m_ports"]), "dt": DT,
        "integrator": str(cfg["rollout_integrator"]), "w_reg": float(cfg["w_reg"]),
        "trunk_mult": float(cfg["trunk_reg_mult"]), "w_reg_B": float(cfg["w_reg_B"]),
        "observation_loss": str(cfg["observation_loss"]),
        "huber_delta": float(cfg["huber_delta"]),
        "w_derivative": float(cfg["w_derivative"]),
        "rollout_increment_loss_weight": float(
            cfg["rollout_increment_loss_weight"]
        ),
        "fast_transition_loss_weight": float(cfg["fast_transition_loss_weight"]),
        "normalized_increment_loss": bool(cfg["normalized_increment_loss"]),
        "coordinate_loss_weights": [
            float(value) for value in cfg["nanodrone_state_loss_weights"]
        ],
        "metric_aligned_observation": bool(cfg["nanodrone_metric_aligned_loss"]),
        "state_mean": state_scaler.mean.tolist(),
        "state_std": state_scaler.std.tolist(),
        "momentum_loss_weight": float(cfg["rotational_momentum_loss_weight"]),
        "angular_shape_loss_weight": float(cfg["rotational_shape_loss_weight"]),
        "torque_scale": torque_scale,
        "omega_scale": omega_scale,
    }
    loss_stop = make_state_observed_loss_fn(**common_loss, rollout_stop_grad=True)
    loss_full = make_state_observed_loss_fn(**common_loss, rollout_stop_grad=False)
    step_stop = make_train_step(
        loss_stop, optimizer_stop, float(cfg["grad_norm_guard"]), float(cfg["grad_clip_norm"])
    )
    step_full = make_train_step(
        loss_full, optimizer_full, float(cfg["grad_norm_guard"]), float(cfg["grad_clip_norm"])
    )
    if frequency_enabled:
        frequency_loss = make_chirp_frequency_loss_fn(
            grad_energy, state_scaler.mean, state_scaler.std,
            integrator=str(cfg["rollout_integrator"]),
        )

        def combined_loss(base_loss):
            def loss_fn(params, x0_batch, u_batch, packed_target):
                base_target, frequency_x0, frequency_u, frequency_target, ramp = packed_target
                base_total, aux = base_loss(params, x0_batch, u_batch, base_target)
                spectral_loss, spectral_aux = frequency_loss(
                    params, frequency_x0, frequency_u, frequency_target
                )
                aux = dict(aux)
                aux.update(spectral_aux)
                aux["frequency_loss"] = spectral_loss
                return (
                    base_total
                    + ramp * float(cfg["rotational_frequency_loss_weight"]) * spectral_loss,
                    aux,
                )
            return loss_fn

        step_frequency_stop = make_train_step(
            combined_loss(loss_stop), optimizer_stop, float(cfg["grad_norm_guard"]),
            float(cfg["grad_clip_norm"]),
        )
        step_frequency_full = make_train_step(
            combined_loss(loss_full), optimizer_full, float(cfg["grad_norm_guard"]),
            float(cfg["grad_clip_norm"]),
        )
    opt_state = optimizer_stop.init(params)
    validation_trajectories = dataset["development_validation"]
    rng = np.random.default_rng(seed)
    history = {name: [] for name in (
        "epoch", "loss_total", "loss_obs", "loss_derivative",
        "loss_rollout_increment", "loss_momentum", "loss_angular_shape",
        "loss_frequency", "loss_frequency_gain", "loss_frequency_coherence",
        "loss_frequency_phase",
        "train_nrmse", "grad_norm", "grad_norm_pre_clip", "grad_norm_post_clip",
        "grad_norm_H", "grad_norm_J", "grad_norm_R", "grad_norm_G",
        "grad_norm_shared_field", "clipping_events", "skipped_updates",
        "nonfinite_events",
    )}
    best_nrmse = np.inf
    best_params = params
    best_equal_epoch = None
    best_angular_metric = np.inf
    best_angular_params = params
    best_angular_epoch = None
    validation_trace = []
    total_skipped = 0
    total_nonfinite = 0
    total_clipping_events = 0
    start_time = time.perf_counter()

    for epoch in range(int(cfg["n_epochs"])):
        if epoch == stop_grad_epochs:
            opt_state = optimizer_full.init(params)
        indices = _sample_epoch_indices(
            train_windows.trajectory_name,
            train_count,
            used_count,
            rng,
            bool(cfg["nanodrone_balance_families"]),
            bool(cfg["nanodrone_balance_runs"]),
        )
        totals, observations, derivatives, rollout_increments = [], [], [], []
        momentum_losses, shape_losses = [], []
        frequency_losses, frequency_gain_losses = [], []
        frequency_coherence_losses, frequency_phase_losses = [], []
        nrmse_values, gradients = [], []
        gradients_post_clip = []
        grouped_gradients = {name: [] for name in ("H", "J", "R", "G", "shared_field")}
        skipped_epoch = 0
        nonfinite_epoch = 0
        clipping_epoch = 0
        for batch in range(n_batches):
            index = indices[batch * batch_size:(batch + 1) * batch_size]
            batch_x0, batch_u = _augment_training_batch(
                train_windows.x0[index],
                train_windows.u_seq[index],
                rng,
                float(cfg["init_noise_std"]),
                float(cfg["input_noise_std"]),
            )
            batch_target = jnp.asarray(train_windows.target[index])
            if momentum_enabled or shape_enabled:
                ramp_epochs = max(int(cfg["rotational_auxiliary_ramp_epochs"]), 1)
                auxiliary_ramp = jnp.asarray(min((epoch + 1) / ramp_epochs, 1.0))
                batch_torque = (
                    jnp.asarray(torque_targets[index])
                    if momentum_enabled else jnp.zeros((len(index), int(cfg["seq_len"]), 4))
                )
                batch_target = (batch_target, batch_torque, auxiliary_ramp)
            step = step_stop if epoch < stop_grad_epochs else step_full
            use_frequency = frequency_enabled and ((epoch * n_batches + batch + 1) % 4 == 0)
            if use_frequency:
                spectral_indices = _sample_chirp_spectral_indices(
                    frequency_windows, chirp_lengths,
                    int(cfg["rotational_frequency_batch_size"]), rng,
                )
                step = step_frequency_stop if epoch < stop_grad_epochs else step_frequency_full
                batch_target = (
                    batch_target,
                    jnp.asarray(frequency_windows.x0[spectral_indices]),
                    jnp.asarray(frequency_windows.u_seq[spectral_indices]),
                    jnp.asarray(frequency_windows.target[spectral_indices]),
                    jnp.asarray(min(
                        (epoch + 1) / max(int(cfg["rotational_auxiliary_ramp_epochs"]), 1), 1.0
                    )),
                )
            params, opt_state, loss, aux, stats = step(
                params,
                opt_state,
                jnp.asarray(batch_x0),
                jnp.asarray(batch_u),
                batch_target,
            )
            if not tree_all_finite(params):
                raise RuntimeError(f"Non-finite parameters at epoch {epoch}, batch {batch}")
            applied = bool(np.asarray(stats["update_applied"]))
            skipped_epoch += int(not applied)
            nonfinite_epoch += int(not (
                bool(np.asarray(stats["finite_grad"]))
                and bool(np.asarray(stats["finite_loss"]))
                and bool(np.asarray(stats["finite_updates"]))
                and bool(np.asarray(stats["finite_params"]))
            ))
            clipping_epoch += int(bool(np.asarray(stats["gradient_clipped"])))
            totals.append(float(loss)); observations.append(float(aux["obs_loss"]))
            derivatives.append(float(aux["derivative_loss"]))
            rollout_increments.append(float(aux["rollout_increment_loss"]))
            momentum_losses.append(float(aux["momentum_loss"]))
            shape_losses.append(float(aux["angular_shape_loss"]))
            frequency_losses.append(float(aux.get("frequency_loss", 0.0)))
            frequency_gain_losses.append(float(aux.get("frequency_gain_loss", 0.0)))
            frequency_coherence_losses.append(float(aux.get("frequency_coherence_loss", 0.0)))
            frequency_phase_losses.append(float(aux.get("frequency_phase_loss", 0.0)))
            nrmse_values.append(float(aux["normalized_nrmse"])); gradients.append(float(stats["grad_norm_pre_clip"]))
            gradients_post_clip.append(float(stats["grad_norm_post_clip"]))
            for name in grouped_gradients:
                grouped_gradients[name].append(float(stats[f"grad_norm_{name}"]))
        total_skipped += skipped_epoch
        total_nonfinite += nonfinite_epoch
        total_clipping_events += clipping_epoch
        if total_skipped > int(cfg["max_bad_updates"]):
            raise RuntimeError(f"Exceeded max_bad_updates={cfg['max_bad_updates']}")
        history["epoch"].append(epoch + 1)
        history["loss_total"].append(float(np.mean(totals)))
        history["loss_obs"].append(float(np.mean(observations)))
        history["loss_derivative"].append(float(np.mean(derivatives)))
        history["loss_rollout_increment"].append(float(np.mean(rollout_increments)))
        history["loss_momentum"].append(float(np.mean(momentum_losses)))
        history["loss_angular_shape"].append(float(np.mean(shape_losses)))
        history["loss_frequency"].append(float(np.mean(frequency_losses)))
        history["loss_frequency_gain"].append(float(np.mean(frequency_gain_losses)))
        history["loss_frequency_coherence"].append(float(np.mean(frequency_coherence_losses)))
        history["loss_frequency_phase"].append(float(np.mean(frequency_phase_losses)))
        history["train_nrmse"].append(float(np.mean(nrmse_values)))
        history["grad_norm"].append(float(gradients[-1]))
        history["grad_norm_pre_clip"].append(float(np.mean(gradients)))
        history["grad_norm_post_clip"].append(float(np.mean(gradients_post_clip)))
        for name, values in grouped_gradients.items():
            history[f"grad_norm_{name}"].append(float(np.mean(values)))
        history["clipping_events"].append(clipping_epoch)
        history["skipped_updates"].append(skipped_epoch)
        history["nonfinite_events"].append(nonfinite_epoch)

        log = {
            "epoch": epoch + 1, "loss/total": history["loss_total"][-1],
            "loss/obs": history["loss_obs"][-1], "loss/passivity": 0.0,
            "loss/derivative": history["loss_derivative"][-1],
            "loss/rollout_increment": history["loss_rollout_increment"][-1],
            "loss/rotational_momentum": history["loss_momentum"][-1],
            "loss/rotational_shape": history["loss_angular_shape"][-1],
            "loss/rotational_frequency": history["loss_frequency"][-1],
            "loss/frequency_gain": history["loss_frequency_gain"][-1],
            "loss/frequency_coherence": history["loss_frequency_coherence"][-1],
            "loss/frequency_phase": history["loss_frequency_phase"][-1],
            "loss/reg": float(aux["reg_loss"]), "loss/state": 0.0,
            "train/finite_rollout": float(aux["finite_rollout"]),
            "train/finite_penalty": float(not bool(np.asarray(aux["finite_rollout"]))),
            "train/skipped_updates_epoch": skipped_epoch,
            "train/total_skipped_updates": total_skipped,
            "train/grad_norm_last": gradients[-1],
            "train/grad_norm_pre_clip_mean": history["grad_norm_pre_clip"][-1],
            "train/grad_norm_post_clip_mean": history["grad_norm_post_clip"][-1],
            "train/clipping_events_epoch": clipping_epoch,
            "train/total_clipping_events": total_clipping_events,
            "train/nonfinite_events_epoch": nonfinite_epoch,
            "train/total_nonfinite_events": total_nonfinite,
            "train/update_applied_last": float(applied),
            "train/stop_grad_phase": float(epoch < stop_grad_epochs),
            "train/train_nrmse_ema": history["train_nrmse"][-1],
            "train/phase_local": 0.0,
        }
        validate = ((epoch + 1) % int(cfg["nanodrone_validation_every"]) == 0
                    or epoch == int(cfg["n_epochs"]) - 1)
        if validate:
            _, validation_metrics = _rolling_predictions(
                params, grad_energy, validation_trajectories,
                state_scaler, input_scaler, cfg,
            )
            val_log = _validation_log(validation_metrics)
            log.update(val_log)
            validation_trace.append({"epoch": epoch + 1, **val_log})
            selection_key = (
                "val/equal_family_state_nrmse"
                if cfg["nanodrone_validation_selection"] == "equal_family_nrmse"
                else "val/state_nrmse"
            )
            if val_log[selection_key] < best_nrmse:
                best_nrmse = val_log[selection_key]
                best_params = params
                best_equal_epoch = epoch + 1
            angular_key = "val/angular_velocity_rmse"
            if val_log[angular_key] < best_angular_metric:
                best_angular_metric = val_log[angular_key]
                best_angular_params = params
                best_angular_epoch = epoch + 1
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(log)
        except Exception as exc:
            if verbose:
                print(f"wandb.log() failed: {exc}")
        if verbose and (epoch % int(cfg["print_every"]) == 0 or validate):
            message = f"[nanodrone] epoch {epoch + 1:4d} loss={history['loss_total'][-1]:.6f} train_nrmse={history['train_nrmse'][-1]:.4f}"
            if validate:
                message += f" val_nrmse={log['val/state_nrmse']:.4f}"
            print(message)

    wall_time = time.perf_counter() - start_time
    metadata = {
        "config": cfg, "dataset": dataset, "training_trajectories": training_trajectories,
        "state_scaler": state_scaler, "input_scaler": input_scaler,
        "train_windows": train_windows, "history": history,
        "best_validation_nrmse": float(best_nrmse), "n_batches": n_batches,
        "best_equal_epoch": best_equal_epoch,
        "best_angular_metric": float(best_angular_metric),
        "best_angular_epoch": best_angular_epoch,
        "best_angular_params": best_angular_params,
        "validation_trace": validation_trace,
        "total_clipping_events": total_clipping_events,
        "total_skipped_updates": total_skipped,
        "total_nonfinite_events": total_nonfinite,
        "optimizer_updates": n_batches * int(cfg["n_epochs"]), "wall_time_seconds": wall_time,
        "seed": seed,
        "final_params": params,
        "rotational_auxiliary_scales": {
            "omega": omega_scale, "torque": torque_scale,
        },
    }
    return best_params, layers, grad_energy, metadata


def _save_scalers(paths, state_scaler: AffineScaler, input_scaler: AffineScaler) -> None:
    save_json(
        paths.config_dir / "scalers.json",
        state_mean=state_scaler.mean, state_std=state_scaler.std,
        input_mean=input_scaler.mean, input_std=input_scaler.std,
        normalization="(value - mean) / (std + 1e-8)",
        input_columns=list(input_scaler.columns),
        normalized_input_zero="mean training input operating point",
    )
    np.savez_compressed(
        paths.config_dir / "scalers.npz",
        state_mean=state_scaler.mean, state_std=state_scaler.std,
        input_mean=input_scaler.mean, input_std=input_scaler.std,
    )


def _save_prediction_plots(paths, split_name: str, predictions: dict, metrics: dict, state_scaler, input_scaler) -> None:
    split_dir = paths.signals_dir / split_name
    for name, prediction in predictions.items():
        trajectory = prediction["trajectory"]
        output_dir = split_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        true_physical = metrics[name]["true_physical"]
        predicted_physical = metrics[name]["predicted_physical"]
        long_true = state_scaler.inverse_transform(prediction["long_true_normalized"])
        long_predicted = state_scaler.inverse_transform(prediction["long_predicted_normalized"])
        starts = prediction["starts"]
        plot_motor_inputs(trajectory.time, trajectory.u, output_dir / "inputs_motors")
        plot_reconstruction_windows(true_physical, predicted_physical, starts, DT, slice(0, 3), ("x", "y", "z"), "m", "Position reconstruction", output_dir / "position_reconstruction")
        plot_reconstruction_windows(true_physical, predicted_physical, starts, DT, slice(3, 6), ("vx", "vy", "vz"), "m/s", "Velocity reconstruction", output_dir / "velocity_reconstruction")
        plot_reconstruction_windows(true_physical, predicted_physical, starts, DT, slice(6, 9), ("rot_x", "rot_y", "rot_z"), "rad", "SO(3)-log reconstruction", output_dir / "orientation_rotvec_reconstruction")
        plot_orientation_error(metrics[name]["errors"]["rotation"], starts, DT, output_dir / "orientation_geodesic_error")
        plot_reconstruction_windows(true_physical, predicted_physical, starts, DT, slice(9, 12), ("wx", "wy", "wz"), "rad/s", "Angular-velocity reconstruction", output_dir / "angular_velocity_reconstruction")
        plot_reconstruction_windows(long_true[None], long_predicted[None], np.array([0]), DT, slice(0, 3), ("x", "y", "z"), "m", "Long contiguous position rollout", output_dir / "position_long_rollout", max_windows=1)
        plot_reconstruction_windows(long_true[None], long_predicted[None], np.array([0]), DT, slice(3, 6), ("vx", "vy", "vz"), "m/s", "Long contiguous velocity rollout", output_dir / "velocity_long_rollout", max_windows=1)
        long_position = np.vstack([trajectory.y[0, :3], long_predicted[:, :3]])
        plot_position_3d(trajectory.y[:len(long_position), :3], predicted_physical[:, :, :3], output_dir / "position_trajectory_3d", long_prediction=long_position)
        trajectory_dir = paths.trajectories_dir / split_name
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_dir / f"{name}_rolling_predictions.npz",
            starts=starts, true_normalized=prediction["true_normalized"],
            predicted_normalized=prediction["predicted_normalized"],
            true_physical=true_physical, predicted_physical=predicted_physical,
            input_normalized=prediction["input_normalized"],
            long_true_normalized=prediction["long_true_normalized"],
            long_predicted_normalized=prediction["long_predicted_normalized"],
            long_true_physical=long_true, long_predicted_physical=long_predicted,
        )


def _make_stress_config(config: dict, seed: int):
    from Stress_tests.config import ProjectionConfig, StressTestConfig, UncertaintySet
    if config.get("stress_test_mode") != "theory_aligned_only":
        raise ValueError("Nano-drone requires stress_test_mode='theory_aligned_only'")
    uncertainty = UncertaintySet(input_norm="linf", input_radius=1.0, state_disturbance_radius=0.0)
    if not config.get("stress_smoke", False):
        return StressTestConfig(
            seed=seed, uncertainty=uncertainty,
            run_non_theoretical_diagnostics=False,
        )
    projection = ProjectionConfig(
        grid_size=21, n_minima_starts=16, minima_steps=30,
        profile_steps=10, profile_restarts=1, profile_chunk_size=256,
    )
    return StressTestConfig(
        seed=seed, uncertainty=uncertainty, projection=projection,
        run_non_theoretical_diagnostics=False,
        n_boundary_directions=32, boundary_bisection_steps=20,
        n_rollouts=4, horizon_steps=20, n_fidelity_samples=32,
        implemented_attack_steps=5, implemented_attack_restarts=1,
        frontier_grid_size=7,
    )


def _select_energy_shell(params, layers, reference_states: np.ndarray, config: dict, seed: int):
    from Stress_tests.geometry import discover_minima, filter_converged_wells
    from Stress_tests.model_adapter import EBMStressAdapter
    stress_config = _make_stress_config(config, seed)
    adapter = EBMStressAdapter(params, layers, d=12, m=int(config["m_ports"]), dt=DT)
    minima = discover_minima(adapter, reference_states, stress_config.projection, seed=seed)
    converged = filter_converged_wells(
        adapter, minima, grad_norm_tol=stress_config.projection.minima_grad_norm_tol
    )
    selected = converged if len(converged.states) else minima
    dominant = int(np.argmax(selected.multiplicities))
    minimum_energy = float(selected.energies[dominant])
    energies = []
    for begin in range(0, len(reference_states), 2048):
        energies.append(np.asarray(adapter.energy_batch(reference_states[begin:begin + 2048])))
    relative = np.concatenate(energies) - minimum_energy
    quantile = float(config["nanodrone_energy_quantile"])
    epsilon = minimum_energy + max(float(np.quantile(relative, quantile)), 1e-4)
    selection = {
        "rule": "dominant converged minimum plus fixed reference-state relative-energy quantile",
        "quantile": quantile,
        "dominant_minimum_index": dominant,
        "minimum_state": selected.states[dominant].tolist(),
        "minimum_energy": minimum_energy,
        "epsilon": epsilon,
    }
    return epsilon, stress_config, selection


def _run_stress_and_radii(params, layers, reference_states, reference_inputs, state_scaler, input_scaler, config, paths, seed):
    from Stress_tests.certificate import compute_uniform_radius_certificate
    from Stress_tests.config import UncertaintySet
    from Stress_tests.model_adapter import EBMStressAdapter
    from post_training_eval import run_stress_tests
    epsilon, stress_config, selection = _select_energy_shell(
        params, layers, reference_states, config, seed
    )
    port_count = int(config["m_ports"])
    summary = run_stress_tests(
        params=params, layers=layers, d=12, m=port_count, dt=DT, epsilon=epsilon, gamma=1.0,
        reference_states=reference_states, reference_inputs=reference_inputs,
        output_dir=paths.stress_test_dir, stress_config=stress_config, seed=seed,
    )
    arrays = np.load(paths.stress_test_dir / "arrays" / "stress_arrays.npz")
    boundary = arrays["boundary_states"]
    components = arrays["boundary_anchor_indices"]
    adapter = EBMStressAdapter(params, layers, d=12, m=port_count, dt=DT)
    certificates = {}
    for norm in ("l2", "linf"):
        certificate = compute_uniform_radius_certificate(
            adapter, boundary, components,
            UncertaintySet(input_norm=norm, input_radius=1.0, state_disturbance_radius=0.0),
        )
        finite = certificate.pointwise_radius[np.isfinite(certificate.pointwise_radius)]
        certificates[norm] = max(float(np.min(finite)) if len(finite) else 0.0, 0.0)
    alpha = certificates["linf"]
    physical_low = input_scaler.inverse_transform(-alpha * np.ones(port_count))
    physical_high = input_scaler.inverse_transform(alpha * np.ones(port_count))
    radius_summary = {
        "scope": "sampled learned-model energy boundary",
        "learned_model_only": True,
        "input_representation": config["nanodrone_input_representation"],
        "input_columns": list(input_scaler.columns),
        "rho_star_l2_normalized": certificates["l2"],
        "alpha_star_input_box_normalized": alpha,
        "normalized_input_center": [0.0] * port_count,
        "input_center_unscaled": input_scaler.mean.tolist(),
        "input_low_unscaled": physical_low.tolist(),
        "input_high_unscaled": physical_high.tolist(),
        "shell_selection": selection,
    }
    save_json(paths.stress_test_dir / "motor_certificate.json", **radius_summary)
    return summary, radius_summary


def train(seed: int = 42, config: Optional[dict] = None):
    params, layers, grad_energy, metadata = run_training(seed=seed, config=config)
    cfg = metadata["config"]
    run_id = str(cfg.get("wandb_source_run_id") or f"seed{seed}")
    paths = get_experiment_paths(EXPERIMENT_NAME, run_id)
    save_json(paths.config_dir / "config.json", **cfg)
    save_checkpoint(paths.checkpoint_dir / "params.pkl", params)
    save_checkpoint(paths.checkpoint_dir / "params_best_equal_family.pkl", params)
    save_checkpoint(
        paths.checkpoint_dir / "params_best_angular_diagnostic.pkl",
        metadata["best_angular_params"],
    )
    save_checkpoint(paths.checkpoint_dir / "params_final.pkl", metadata["final_params"])
    save_json(
        paths.checkpoint_dir / "checkpoint_metadata.json",
        official_selector="equal_family_nrmse",
        best_equal_epoch=metadata["best_equal_epoch"],
        best_equal_nrmse=metadata["best_validation_nrmse"],
        best_angular_diagnostic_selector="aggregate_physical_omega_mae_h1_h50",
        best_angular_diagnostic_epoch=metadata["best_angular_epoch"],
        best_angular_diagnostic_metric=metadata["best_angular_metric"],
        final_epoch=int(cfg["n_epochs"]),
    )
    _save_scalers(paths, metadata["state_scaler"], metadata["input_scaler"])
    write_dataset_manifest(
        paths.experiment_root / "dataset_manifest.json", metadata["dataset"],
        state_scaler=metadata["state_scaler"],
        input_scaler=metadata["input_scaler"],
        input_representation=cfg["nanodrone_input_representation"],
        root=cfg["nanodrone_dataset_root"],
    )
    history = {key: np.asarray(value) for key, value in metadata["history"].items()}
    np.savez_compressed(paths.metrics_dir / "training_history.npz", **history)
    save_json(paths.metrics_dir / "validation_trace.json", events=metadata["validation_trace"])
    if cfg["save_plots"]:
        plot_training_history(history, paths.signals_dir / "training_curves")

    evaluation_key = "development_validation" if cfg["nanodrone_split"] == "development" else "official_test"
    split_name = "validation" if cfg["nanodrone_split"] == "development" else "test"
    evaluation_trajectories = metadata["dataset"][evaluation_key]
    predictions, per_run = _rolling_predictions(
        params, grad_energy, evaluation_trajectories,
        metadata["state_scaler"], metadata["input_scaler"], cfg,
    )
    aggregate = aggregate_run_metrics(per_run)
    save_benchmark_metrics(paths.metrics_dir, per_run, aggregate)
    if cfg["save_plots"]:
        _save_prediction_plots(
            paths, split_name, predictions, per_run,
            metadata["state_scaler"], metadata["input_scaler"],
        )
        plot_horizon_metrics(
            aggregate["mae_by_horizon"], DT,
            paths.signals_dir / split_name / "prediction_horizon_metrics",
        )

    reference_trajectories = (
        metadata["dataset"]["development_validation"]
        if cfg["nanodrone_split"] == "development"
        else metadata["dataset"]["official_train"]
    )
    reference_states = np.concatenate([
        metadata["state_scaler"].transform(value.y) for value in reference_trajectories
    ])
    reference_inputs = np.concatenate([
        metadata["input_scaler"].transform(transform_inputs(
            value.u, cfg["nanodrone_input_representation"]
        )) for value in reference_trajectories
    ])
    stress_summary = None
    radius_summary = None
    stress_complete = False
    theory_aligned_summary = None
    theory_aligned_complete = False
    if cfg["nanodrone_run_stress_tests"]:
        stress_summary, radius_summary = _run_stress_and_radii(
            params, layers, reference_states, reference_inputs,
            metadata["state_scaler"], metadata["input_scaler"], cfg, paths, seed,
        )
        stress_complete = True
        theory_aligned_summary = run_theory_aligned_certificates(
            checkpoint_path=paths.checkpoint_dir / "params.pkl",
            config_path=paths.config_dir / "config.json",
            stress_arrays_npz=paths.stress_test_dir / "arrays" / "stress_arrays.npz",
            output_dir=paths.stress_test_dir / "theory_aligned",
        )
        theory_aligned_complete = not theory_aligned_summary["errors"]

    try:
        repo_commit = subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repo_commit = None
    try:
        dirty_diff = subprocess.check_output(
            ["git", "-C", str(_ROOT), "diff", "--binary", "--no-ext-diff"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        dirty_diff = b""
    (paths.audits_dir / "git_diff.patch").write_bytes(dirty_diff)
    device_memory = {}
    for index, device in enumerate(jax.devices()):
        try:
            device_memory[str(index)] = device.memory_stats() or {}
        except (AttributeError, RuntimeError):
            device_memory[str(index)] = {"status": "unavailable"}
    protected_source_hashes = {
        relative_path: hashlib.sha256((_ROOT / relative_path).read_bytes()).hexdigest()
        for relative_path in cfg.get("required_source_hashes", {})
    }
    resolved_config_bytes = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    audit = {
        "experiment_name": EXPERIMENT_NAME, "run_id": run_id, "seed": seed,
        "repository_commit": repo_commit,
        "external_dataset_commit": "2d921b57d166fe2debe08a5d39bd07297c5abc39",
        "runtime_environment": {
            "python": sys.version,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
            "devices": [str(device) for device in jax.devices()],
            "device_memory_stats": device_memory,
            "host": platform.node(),
            "container": os.environ.get("APPTAINER_CONTAINER", "NOT_RECORDED"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_RECORDED"),
        },
        "resolved_config_sha256": hashlib.sha256(resolved_config_bytes).hexdigest(),
        "protected_source_hashes": protected_source_hashes,
        "git_dirty_diff_sha256": hashlib.sha256(dirty_diff).hexdigest(),
        "git_dirty": bool(dirty_diff),
        "evaluation_split": evaluation_key,
        "official_test_evaluated": cfg["nanodrone_split"] == "final",
        "benchmark_metrics": {
            "aggregate_mae": aggregate["aggregate_mae"],
            "n_starts": aggregate["n_starts"],
        },
        "best_validation_nrmse": metadata["best_validation_nrmse"],
        "checkpoint_epochs": {
            "best_equal": metadata["best_equal_epoch"],
            "best_angular_diagnostic": metadata["best_angular_epoch"],
            "final": int(cfg["n_epochs"]),
        },
        "optimizer_diagnostics": {
            "clipping_events": metadata["total_clipping_events"],
            "skipped_updates": metadata["total_skipped_updates"],
            "nonfinite_events": metadata["total_nonfinite_events"],
        },
        "stress": stress_summary, "motor_certificate": radius_summary,
        "theory_aligned_certificates": theory_aligned_summary,
        "scientific_scope": "formal certificate applies to the learned pH Neural ODE, not directly to the physical Crazyflie",
    }
    save_json(paths.audits_dir / "audit.json", **audit)
    from Stress_tests.theory_aligned.fitting.fitting_report import build_fitting_report
    fitting_report = build_fitting_report(str(paths.run_root), target_threshold=None)
    save_json(
        paths.stress_test_dir / "theory_aligned" / "fitting_evidence.json",
        **fitting_report.__dict__,
    )
    write_run_status(
        paths.run_root / "run_status.json", training_complete=True,
        reconstruction_evaluation_complete=True, signal_plots_complete=bool(cfg["save_plots"]),
        trajectory_plots_complete=bool(cfg["save_plots"]),
        stress_tests_complete=stress_complete, stress_plots_complete=stress_complete,
        theory_aligned_certificates_complete=theory_aligned_complete, audit_complete=True,
    )

    parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(params))
    print("=" * 68)
    print(f"Nano-drone run: {run_id}")
    print("=" * 68)
    print(f"split / evaluated          : {cfg['nanodrone_split']} / {evaluation_key}")
    print(f"dt / dimensions            : {DT:.6f} s / state 12 / ports {cfg['m_ports']}")
    print(f"EBM layers / parameters    : {cfg['layer_dims']} / {parameter_count}")
    print(f"epochs / sequence / stride : {cfg['n_epochs']} / {cfg['seq_len']} / {cfg['stride']}")
    print(f"optimizer updates          : {metadata['optimizer_updates']}")
    print(f"wall time [s]              : {metadata['wall_time_seconds']:.1f}")
    print(f"validation state NRMSE     : {metadata['best_validation_nrmse']:.6g}")
    print(f"MAE p / v / R / omega      : {aggregate['aggregate_mae']}")
    if radius_summary:
        print(f"rho* L2 / alpha* box       : {radius_summary['rho_star_l2_normalized']:.6g} / {radius_summary['alpha_star_input_box_normalized']:.6g}")
    print(f"results root               : {paths.run_root}")
    print("Certificate scope: learned pH Neural ODE only; not the physical Crazyflie.")
    print("=" * 68)
    return params, layers, grad_energy, metadata


if __name__ == "__main__":
    train()
