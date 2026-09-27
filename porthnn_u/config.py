"""Serializable configuration for PortHNN-u experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    experiment_name: str = "porthnn_u"
    seed: int = 0
    dataset: str = "silverbox"
    state_dim: int = 4
    hidden_dims: tuple[int, ...] = (200, 200, 200)
    activation: str = "tanh"
    damping: str = "scalar_unconstrained"
    encoder_hidden_dims: tuple[int, ...] = (128, 64)
    init_window: int | None = None
    train_horizon: int = 128
    stride: int = 8
    validation_fraction: float = 0.2
    batch_size: int = 64
    iterations: int = 10_000
    initial_learning_rate: float = 3e-4
    gradient_clip_norm: float = 1.0
    lambda_force_l1: float = 1e-6
    lambda_damping_l1: float = 1e-6
    lambda_encoder_l2: float = 1e-6
    lambda_readout_l2: float = 1e-6
    validation_every: int = 100
    precision: str = "float32"

    def __post_init__(self) -> None:
        if self.dataset not in {"silverbox", "ced"}:
            raise ValueError("dataset must be 'silverbox' or 'ced'")
        if self.state_dim <= 0 or self.state_dim % 2:
            raise ValueError("canonical PortHNN-u requires an even state_dim")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if any(width <= 0 for width in self.encoder_hidden_dims):
            raise ValueError("encoder_hidden_dims must contain positive widths")
        if self.init_window is not None and self.init_window <= 0:
            raise ValueError("init_window must be positive")
        if self.train_horizon <= 0 or self.stride <= 0 or self.batch_size <= 0:
            raise ValueError("horizon, stride, and batch_size must be positive")
        if self.iterations <= 0 or self.validation_every <= 0:
            raise ValueError("iterations and validation_every must be positive")
        if self.initial_learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("learning rate and gradient clip norm must be positive")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")

    def resolved(self, *, init_window: int) -> "ExperimentConfig":
        if self.init_window is None:
            return ExperimentConfig(**{**asdict(self), "init_window": init_window})
        if self.init_window != init_window:
            raise ValueError(
                f"configured init_window={self.init_window} disagrees with benchmark metadata={init_window}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open() as handle:
        raw = yaml.safe_load(handle) or {}
    model = raw.pop("model", {})
    data = raw.pop("data", {})
    raw.pop("integration", None)
    optimization = raw.pop("optimization", {})
    raw.pop("evaluation", None)
    values = {
        **raw,
        "dataset": data.get("dataset", raw.get("dataset", "silverbox")),
        "init_window": data.get("init_window"),
        "train_horizon": data.get("train_horizon", 128),
        "stride": data.get("stride", 8),
        "validation_fraction": data.get("validation_fraction", 0.2),
        "state_dim": model.get("state_dim", 4),
        "hidden_dims": tuple(model.get("hidden_dims", (200, 200, 200))),
        "activation": model.get("activation", "tanh"),
        "damping": model.get("damping", "scalar_unconstrained"),
        "encoder_hidden_dims": tuple(model.get("encoder_hidden_dims", (128, 64))),
        "iterations": optimization.get("iterations", 10_000),
        "batch_size": optimization.get("batch_size", 64),
        "initial_learning_rate": optimization.get("initial_learning_rate", 3e-4),
        "gradient_clip_norm": optimization.get("gradient_clip_norm", 1.0),
        "lambda_force_l1": optimization.get("lambda_force_l1", 1e-6),
        "lambda_damping_l1": optimization.get("lambda_damping_l1", 1e-6),
        "lambda_encoder_l2": optimization.get("lambda_encoder_l2", 1e-6),
        "lambda_readout_l2": optimization.get("lambda_readout_l2", 1e-6),
        "validation_every": optimization.get("validation_every", 100),
    }
    return ExperimentConfig(**values)