"""Immutable configuration generation for the constrained-damping campaign."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path

import yaml


CAMPAIGN_NAME = "porthnn_u_16x2_sota"
MODEL_LABEL = "Latent PortHNN-u (input-conditioned forcing, constrained damping)"
AXES = tuple(
    (index, latent_dim, damping, activation, forcing)
    for index, (latent_dim, damping, activation, forcing) in enumerate(
        (
            (latent_dim, damping, activation, forcing)
            for latent_dim in (2, 4)
            for damping in ("scalar_negative_softplus", "vector_negative_softplus")
            for activation in ("tanh", "sin")
            for forcing in ("direct", "bounded")
        ),
        start=1,
    )
)


def source_tree_hash(root: Path) -> str:
    digest = sha256()
    for path in sorted((root / "porthnn_u").glob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_hash(values: dict) -> str:
    encoded = yaml.safe_dump(values, sort_keys=True).encode()
    return sha256(encoded).hexdigest()


def _base(dataset: str) -> dict:
    if dataset == "silverbox":
        return {
            "dataset": dataset,
            "init_window": 50,
            "max_horizon": 768,
            "hidden_dims": [200, 200, 200],
            "forcing_hidden_dims": [64, 64],
            "encoder_hidden_dims": [128, 64],
            "force_cap": 10.0,
            "lambda_force_jacobian": 1e-4,
            "updates": 15000,
            "batch_size": 64,
            "validation_horizons": [192, 768, 3072],
            "validation_interval": 100,
            "training_schedule": [
                {"until": 3000, "horizons": [64], "probabilities": [1.0]},
                {"until": 7000, "horizons": [64, 192], "probabilities": [0.25, 0.75]},
                {"until": 15000, "horizons": [192, 384, 768], "probabilities": [0.20, 0.50, 0.30]},
            ],
        }
    if dataset == "ced":
        return {
            "dataset": dataset,
            "init_window": 10,
            "max_horizon": 90,
            "hidden_dims": [64, 64],
            "forcing_hidden_dims": [32, 32],
            "encoder_hidden_dims": [64, 32],
            "force_cap": 4.0,
            "lambda_force_jacobian": 3e-4,
            "updates": 6000,
            "batch_size": 16,
            "validation_horizons": [32, 64, 90],
            "validation_interval": 100,
            "training_schedule": [
                {"until": 6000, "horizons": [32, 64, 90], "probabilities": [1 / 3, 1 / 3, 1 / 3]},
            ],
        }
    raise ValueError(f"unsupported campaign dataset: {dataset}")


def _common() -> dict:
    return {
        "campaign": CAMPAIGN_NAME,
        "development_only": True,
        "official_test_access": "forbidden",
        "model_label": MODEL_LABEL,
        "seed": 0,
        "optimizer": "adam",
        "learning_rate": 3e-4,
        "learning_rate_end": 3e-6,
        "warmup_updates": 500,
        "gradient_clip_global_norm": 1.0,
        "rk_solver": "rk4",
        "input_hold": "zero_order_hold",
        "precision": "float32",
        "damping_floor": 1e-4,
        "damping_initial": -0.10,
        "lambda_force_l1": 1e-6,
        "lambda_encoder_l2": 1e-6,
        "lambda_readout_l2": 1e-6,
        "lambda_damping_l1": 0.0,
        "late_loss_weight_start": 1.0,
        "late_loss_weight_end": 2.0,
    }


def generate_campaign(root: str | Path) -> dict:
    root = Path(root).resolve()
    campaign_root = root / "configs" / "porthnn_u" / "campaign16"
    resolved_hashes = []
    for dataset, prefix in (("silverbox", "sb"), ("ced", "ced")):
        base = {**_common(), **_base(dataset)}
        base_path = campaign_root / f"base_{dataset}.yaml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(yaml.safe_dump(base, sort_keys=False))
        output_directory = campaign_root / dataset
        output_directory.mkdir(exist_ok=True)
        generated = []
        for index, latent_dim, damping, activation, forcing in AXES:
            config_id = f"{prefix}_{index:02d}"
            values = {
                **base,
                "config_id": config_id,
                "latent_dim": latent_dim,
                "damping_parameterization": damping,
                "hamiltonian_activation": activation,
                "forcing_parameterization": forcing,
            }
            if latent_dim % 2 or "negative_softplus" not in damping:
                raise ValueError(f"invalid campaign cell: {config_id}")
            path = output_directory / f"{config_id}.yaml"
            path.write_text(yaml.safe_dump(values, sort_keys=False))
            generated.append((str(path.relative_to(root)), config_hash(values)))
        hashes = [item[1] for item in generated]
        if len(generated) != 16 or len(set(hashes)) != 16:
            raise ValueError(f"{dataset} campaign must contain 16 unique configurations")
        resolved_hashes.extend(generated)
    manifest = {
        "campaign": CAMPAIGN_NAME,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_tree_hash": source_tree_hash(root),
        "configs": [{"path": path, "sha256": digest} for path, digest in resolved_hashes],
        "axis_order": ["latent_dim", "damping_parameterization", "hamiltonian_activation", "forcing_parameterization"],
        "axis_values": [
            {"index": index, "latent_dim": dim, "damping": damping, "activation": activation, "forcing": forcing}
            for index, dim, damping, activation, forcing in AXES
        ],
    }
    result_root = root / "results" / "porthnn_u" / "campaign16"
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (result_root / "generation_report.json").write_text(json.dumps({
        "status": "passed", "silverbox_config_count": 16, "ced_config_count": 16,
        "unique_config_hashes": len({digest for _, digest in resolved_hashes}),
    }, indent=2, sort_keys=True))
    return manifest