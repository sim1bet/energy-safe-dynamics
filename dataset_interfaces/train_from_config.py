#!/usr/bin/env python3
# Author: Simone Betteti
"""
Offline compute environment training entry point for W&B-generated configs.

This script reads one JSON payload produced by `generate_wandb_config.py`,
merges it with Silverbox_main.DEFAULT_CONFIG, opens an OFFLINE W&B run with the
same W&B run id assigned by the online sweep agent, and calls Silverbox_main.train.

Usage inside the container on the compute environment:
    python train_from_config.py \
        --config-json configs_queue/processed/<run_id>.json \
        --project ebm-silverbox
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Dataset entry modules live in Interface_code/ (and import EBM_model/*
# directly); neither is on sys.path by default for `python
# train_from_config.py ...` outside whatever PYTHONPATH the caller set up.
_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "Interface_code", _ROOT / "EBM_model"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# Maps the --dataset selector to the entry-point module. Each module must expose
# `DEFAULT_CONFIG` and a `train(seed, config)` with the dataset's loader bound in.
DATASET_MODULES = {
    "silverbox": "Silverbox_main",
    "cascaded_tanks": "Cascaded_Tanks_main",
    "emps": "EMPS_main",
    "wiener_hammerstein": "WienerHammerstein_main",
    "ced": "CED_main",
    "duffing_doublewell": "Duffing_DoubleWell_main",
    "deep_dissipative_nlink": "DeepDissipative_NLink_main",
    "walker2d": "Walker2d_main",
    "nanodrone": "NanoDrone_main",
}


def _normalise_value(key: str, value: Any) -> Any:
    """Normalise values that may arrive as strings from YAML/JSON/bash."""
    if key == "layer_dims":
        if isinstance(value, str):
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        return [int(x) for x in value]
    if key in {"batch_size", "hidden", "d", "m_ports", "n_obs", "seq_len", "stride", "seq_len_stage1", "seq_len_stage2", "curriculum_switch_update", "curriculum_switch_update2", "lr_warmup_updates", "init_win", "n_epochs", "n_updates", "stop_grad_updates", "rollout_stop_grad_epochs", "validation_every", "validation_batches", "profile_updates", "max_train_episodes", "max_validation_episodes", "max_test_episodes", "ood_episode_limit", "terminal_margin_steps", "init_steps_test", "readout_hidden", "softmax_groups", "plot_max_points", "controller_min_energy_n_steps", "controller_n_ctrl_steps_per_epoch", "init_multistart", "init_multistart_seed", "early_window_steps", "nanodrone_validation_every", "nanodrone_benchmark_horizon", "nanodrone_eval_stride", "nanodrone_long_rollout_steps", "nanodrone_max_train_windows", "nanodrone_max_eval_starts", "nanodrone_train_flight_limit"}:
        return int(value)
    if key in {"use_lr_decay", "use_feedthrough", "use_identity_readout", "state_observed_mode", "use_nonlinear_readout", "use_state_damping", "use_state_interconnection", "use_quadratic_interconnection", "use_cubic_interconnection", "use_ema", "save_plots", "use_iterative_init", "enable_controller", "use_input_aware_encoder", "use_saturating_readout", "use_input_gain", "use_saturating_input", "use_rich_interconnection", "use_rich_damping", "use_rich_input_matrix", "requires_rich_ph_fields", "ph_field_bounded_outputs", "normalized_increment_loss", "include_terminal_windows", "run_stress_tests", "stress_smoke", "screening_only", "run_unconstrained_baseline", "nanodrone_run_stress_tests", "nanodrone_balance_families"}:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if key == "local_opt_switch":
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if key in {"lr", "lr_min", "w_rollout", "w_passivity", "w_reg", "trunk_reg_mult", "chol_clip_exp", "p_1", "p_2", "init_lr",
               "damping_scale", "b_init_scale", "vf_scale", "weight_decay",
               "input_noise_std", "init_noise_std", "ema_decay",
               "full_phase_lr_scale", "local_opt_nrmse_threshold", "local_opt_lr_scale",
               "local_opt_momentum", "local_opt_ema_beta",
               "controller_loss_threshold", "controller_ema_beta", "controller_gamma_init",
               "controller_margin_init", "controller_min_margin_floor", "controller_ctrl_lr",
               "controller_w_margin_reg", "controller_min_energy_lr", "init_multistart_noise",
               "early_window_weight", "amplitude_weight_power", "rise_weight_power",
               "sat_scale_init", "sat_scale_min", "sat_u_init", "sat_u_min",
               "w_derivative", "velocity_loss_weight", "fast_transition_loss_weight",
               "nanodrone_energy_quantile"}:
        return float(value)
    return value


def load_payload(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        payload = json.load(f)
    # Backward compatible: accept either {run_id, config} or a bare config dict.
    if "config" not in payload:
        payload = {"run_id": Path(path).stem, "config": payload}
    return payload


def check_module_experiment_name(dataset_module, dataset_key: str, module_name: str) -> None:
    """A registered module's own ``EXPERIMENT_NAME`` must match the key it is
    registered under in ``DATASET_MODULES`` (catches a typo'd registration)."""
    module_experiment_name = getattr(dataset_module, "EXPERIMENT_NAME", dataset_key)
    if module_experiment_name != dataset_key:
        raise SystemExit(
            f"Internal error: DATASET_MODULES[{dataset_key!r}] = {module_name!r}, but that "
            f"module's EXPERIMENT_NAME is {module_experiment_name!r}. Fix DATASET_MODULES."
        )


def check_config_experiment_name(train_config: Dict[str, Any], dataset_key: str, config_json: str) -> None:
    """A config's own declared ``experiment_name`` (written by the config
    generator) must match the module actually selected by --dataset/
    EBM_DATASET. This is the guard against the exact failure mode that
    misrouted an n-link Stage-B config through ``Duffing_DoubleWell_main``:
    only --dataset/EBM_DATASET picked the module, with nothing
    cross-checking the config's own declared identity. Fails loudly BEFORE
    wandb.init/training, not with a cryptic downstream shape error deep in
    training. See ``experiment_paths.py`` and ``DEBUGGING_REPORT.md``."""
    config_experiment_name = train_config.get("experiment_name")
    if config_experiment_name is not None and config_experiment_name != dataset_key:
        raise SystemExit(
            f"Experiment-identity mismatch for {config_json}: the config declares "
            f"experiment_name={config_experiment_name!r} but --dataset/EBM_DATASET="
            f"{dataset_key!r} was selected. Refusing to train -- this config almost "
            "certainly belongs under a different EBM_DATASET/CONFIG_DIR."
        )


def check_required_model_features(
    suggested_config: Dict[str, Any], default_config: Dict[str, Any], config_json: str
) -> None:
    """Reject configs whose architecture is not implemented by the dataset module."""
    if not suggested_config.get("requires_rich_ph_fields", False):
        return
    required_version = int(suggested_config.get("rich_ph_fields_version", 1))
    available_version = int(default_config.get("rich_ph_fields_version", 0))
    if available_version < required_version:
        raise SystemExit(
            f"Model-feature mismatch for {config_json}: this config requires rich pH fields "
            f"version {required_version}, but the selected dataset module provides version "
            f"{available_version}. Implement and register the rich J/R/G heads before submission."
        )


def check_frozen_campaign_contract(config: Dict[str, Any], config_json: str) -> None:
    """Enforce optional source and architecture fingerprints before training."""
    for relative_path, expected_hash in config.get("required_source_hashes", {}).items():
        path = (_ROOT / relative_path).resolve()
        if _ROOT.resolve() not in path.parents or not path.is_file():
            raise SystemExit(f"Invalid required source path for {config_json}: {relative_path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise SystemExit(
                f"Frozen source mismatch for {config_json}: {relative_path} has "
                f"sha256={actual_hash}, expected {expected_hash}. Refusing to train."
            )
    for key, expected_value in config.get("frozen_config_assertions", {}).items():
        if config.get(key) != expected_value:
            raise SystemExit(
                f"Frozen config mismatch for {config_json}: {key}={config.get(key)!r}, "
                f"expected {expected_value!r}. Refusing to train."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--run-id", default=None, help="Override run id. Defaults to payload run_id or JSON filename stem.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=os.environ.get("EBM_DATASET", "silverbox"),
                        choices=sorted(DATASET_MODULES),
                        help="Benchmark to train on; selects the entry-point module.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args()

    # Project defaults to ebm-<dataset> unless explicitly overridden.
    if not args.project:
        args.project = f"ebm-{args.dataset.replace('_', '-')}"

    # Must be set before importing/initialising wandb in this process.
    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_PROJECT", args.project)

    import wandb
    module_name = DATASET_MODULES[args.dataset]
    dataset_module = importlib.import_module(module_name)
    DEFAULT_CONFIG = dataset_module.DEFAULT_CONFIG
    train = dataset_module.train
    check_module_experiment_name(dataset_module, args.dataset, module_name)

    payload = load_payload(args.config_json)
    run_id = args.run_id or payload.get("run_id") or Path(args.config_json).stem
    suggested_config = payload.get("config", {})
    check_required_model_features(suggested_config, DEFAULT_CONFIG, args.config_json)
    # Per-config seed override (e.g. for seed-variance/ensemble batches): the JSON
    # payload's top-level "seed" wins over the --seed CLI default (42) used by
    # the batch launcher for every packed configuration.
    # NOTE (Jul8 bugfix): some generators write a BARE flat config dict (no
    # {run_id, config} wrapper) with "seed" mixed in as an ordinary config key
    # (e.g. generate_ced_v4.py's seed7 replicate). load_payload()'s back-compat
    # path then nests the WHOLE flat dict under "config", silently burying the
    # seed override -> payload.get("seed", ...) never finds it and falls back to
    # the CLI default (42), so the run trains with the wrong seed with no error.
    # Fix: also look for "seed" embedded inside the config dict, and pop it out
    # so it never leaks into train_config as a spurious hyperparameter.
    embedded_seed = suggested_config.pop("seed", None)
    seed_override = payload.get("seed", embedded_seed)
    seed = int(seed_override) if seed_override is not None else args.seed

    train_config = DEFAULT_CONFIG.copy()
    for k, v in suggested_config.items():
        train_config[k] = _normalise_value(k, v)

    check_config_experiment_name(train_config, args.dataset, args.config_json)
    check_frozen_campaign_contract(train_config, args.config_json)
    train_config.setdefault("experiment_name", args.dataset)

    # Optional but useful metadata.
    train_config["wandb_offline_pipeline"] = True
    train_config["wandb_source_run_id"] = run_id

    print("=" * 80)
    print("Launching offline W&B training")
    print(f"dataset     : {args.dataset} -> {module_name}")
    print(f"config_json : {args.config_json}")
    print(f"run_id      : {run_id}")
    print(f"project     : {args.project}")
    print(f"entity      : {args.entity}")
    print(f"seed        : {seed}")
    print("sweep config:")
    for k in sorted(suggested_config):
        print(f"  {k}: {suggested_config[k]}")
    print("=" * 80)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=run_id,
        name=args.run_name or run_id,
        config=train_config,
        mode="offline",
        resume="allow",
        reinit=True,
    )
    try:
        train(seed=seed, config=train_config)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
