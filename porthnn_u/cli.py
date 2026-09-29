# Author: Simone Betteti
"""Command-line entry points for latent PortHNN-u experiments."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import jax
import numpy as np
import yaml

from .artifacts import initialize_run_directory, load_parameters, run_manifest, update_manifest
from .campaign import generate_campaign
from .campaign_train import train_campaign
from .generic_ports_nrmse02 import generate_generic_ports_nrmse02
from .ced_followup import generate_followup
from .modular_campaign import generate_modular_campaign
from .modular16_campaign import generate_modular16
from .modular_train import train_modular_campaign
from .config import ExperimentConfig, load_config
from .data import load_dataset, make_windows
from .model import ModelSpec, init_params
from .train import evaluate, save_training_result, train


def _load_resolved(path: Path) -> ExperimentConfig:
    with (path / "resolved_config.yaml").open() as handle:
        values = yaml.safe_load(handle)
    values["hidden_dims"] = tuple(values["hidden_dims"])
    values["encoder_hidden_dims"] = tuple(values["encoder_hidden_dims"])
    return ExperimentConfig(**values)


def _bundle(config: ExperimentConfig):
    bundle = load_dataset(
        config.dataset,
        validation_fraction=config.validation_fraction,
        guard=config.init_window + config.train_horizon,
    )
    return bundle


def train_command(args) -> None:
    config = load_config(args.config)
    if args.seed is not None:
        config = ExperimentConfig(**{**config.to_dict(), "seed": args.seed})
    bundle_probe = load_dataset(config.dataset, validation_fraction=config.validation_fraction, guard=1)
    config = config.resolved(init_window=bundle_probe.benchmark_metadata["init_window"])
    bundle = _bundle(config)
    train_windows = make_windows(bundle.train, config.init_window, config.train_horizon, config.stride)
    validation_windows = make_windows(bundle.validation, config.init_window, config.train_horizon, config.stride)
    manifest = run_manifest(config, bundle.benchmark_metadata)
    output = initialize_run_directory(args.output, config, bundle.normalization.to_dict(), manifest)
    result = train(config, bundle, train_windows, validation_windows)
    save_training_result(output, result)
    update_manifest(output, manifest)
    print(f"best validation rollout MSE: {result.best_validation_loss:.8g}")


def evaluate_command(args) -> None:
    output = Path(args.run)
    config = _load_resolved(output)
    bundle = _bundle(config)
    spec = ModelSpec(config.state_dim, 1, 1, config.hidden_dims, config.activation, config.damping, config.encoder_hidden_dims)
    template = init_params(jax.random.key(config.seed), spec, config.init_window)
    params = load_parameters(output / "parameters.npz", template)
    metrics = evaluate(params, config, bundle, output)
    print(yaml.safe_dump(metrics, sort_keys=True))


def inspect_data_command(args) -> None:
    bundle = load_dataset(args.dataset, validation_fraction=args.validation_fraction, guard=args.guard)
    for split in ("train", "validation", "test"):
        realizations = getattr(bundle, split)
        print(f"{split}: " + ", ".join(f"{item.name}({len(item.inputs)})" for item in realizations))
    print(yaml.safe_dump(bundle.benchmark_metadata, sort_keys=True))


def generate_campaign_command(args) -> None:
    manifest = generate_campaign(args.root)
    print(f"generated {len(manifest['configs'])} unique campaign configurations")


def campaign_train_command(args) -> None:
    diagnostics = train_campaign(args.config, args.output, args.fold)
    print(yaml.safe_dump(diagnostics, sort_keys=True))


def generate_ced_followup_command(args) -> None:
    manifest = generate_followup(args.root)
    print(f"generated {len(manifest['configs'])} unique CED follow-up configurations")


def generate_modular_campaign_command(args) -> None:
    manifest = generate_modular_campaign(args.root)
    print(f"generated {len(manifest['configs'])} unique modular campaign configurations")


def generate_modular16_command(args) -> None:
    manifest = generate_modular16(args.root)
    print(f"generated {len(manifest['configs'])} unique modular16 configurations")


def generate_generic_ports_command(args) -> None:
    manifest = generate_generic_ports_nrmse02(args.root)
    print(f"generated {len(manifest['configs'])} unique generic-port configurations")


def modular_train_command(args) -> None:
    diagnostics = train_modular_campaign(args.config, args.output, args.fold)
    print(yaml.safe_dump(diagnostics, sort_keys=True))


def physical_benchmark_command(args) -> None:
    """Dispatch the Duffing/three-link scripts while preserving their CLI."""
    release_root = Path(__file__).resolve().parents[1]
    script_root = release_root / "baselines" / "porthnn_u" / "scripts"
    script = {
        "train-duffing": "train_duffing.py",
        "train-three-link": "train_three_link.py",
        "evaluate-three-link": "evaluate_three_link.py",
    }[args.command]
    sys.argv = [script, *args.arguments]
    runpy.run_path(str(script_root / script), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(prog="porthnn-u")
    commands = parser.add_subparsers(required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--seed", type=int)
    train_parser.set_defaults(func=train_command)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--run", required=True)
    evaluate_parser.set_defaults(func=evaluate_command)
    inspect_parser = commands.add_parser("inspect-data")
    inspect_parser.add_argument("--dataset", choices=("silverbox", "ced"), required=True)
    inspect_parser.add_argument("--validation-fraction", type=float, default=0.2)
    inspect_parser.add_argument("--guard", type=int, default=1)
    inspect_parser.set_defaults(func=inspect_data_command)
    generate_parser = commands.add_parser("generate-campaign")
    generate_parser.add_argument("--root", default=".")
    generate_parser.set_defaults(func=generate_campaign_command)
    followup_parser = commands.add_parser("generate-ced-followup")
    followup_parser.add_argument("--root", default=".")
    followup_parser.set_defaults(func=generate_ced_followup_command)
    modular_parser = commands.add_parser("generate-modular-campaign")
    modular_parser.add_argument("--root", default=".")
    modular_parser.set_defaults(func=generate_modular_campaign_command)
    modular16_parser = commands.add_parser("generate-modular16")
    modular16_parser.add_argument("--root", default=".")
    modular16_parser.set_defaults(func=generate_modular16_command)
    generic_ports_parser = commands.add_parser("generate-generic-ports")
    generic_ports_parser.add_argument("--root", default=".")
    generic_ports_parser.set_defaults(func=generate_generic_ports_command)
    modular_train_parser = commands.add_parser("modular-train")
    modular_train_parser.add_argument("--config", required=True)
    modular_train_parser.add_argument("--output", required=True)
    modular_train_parser.add_argument("--fold", type=int)
    modular_train_parser.set_defaults(func=modular_train_command)
    campaign_parser = commands.add_parser("campaign-train")
    campaign_parser.add_argument("--config", required=True)
    campaign_parser.add_argument("--output", required=True)
    campaign_parser.add_argument("--fold", type=int)
    campaign_parser.set_defaults(func=campaign_train_command)
    for command in ("train-duffing", "train-three-link", "evaluate-three-link"):
        physical = commands.add_parser(command)
        physical.add_argument("arguments", nargs=argparse.REMAINDER)
        physical.set_defaults(func=physical_benchmark_command, command=command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
