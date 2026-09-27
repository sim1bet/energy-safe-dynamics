"""Resolved configuration generation for the CED capacity and recipe follow-up."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .campaign import MODEL_LABEL, config_hash, source_tree_hash


CAMPAIGN_NAME = "porthnn_u_ced_followup16"


def _base() -> dict:
    return {
        "campaign": CAMPAIGN_NAME,
        "development_only": True,
        "official_test_access": "forbidden",
        "model_label": MODEL_LABEL,
        "dataset": "ced",
        "seed": 0,
        "init_window": 10,
        "max_horizon": 90,
        "forcing_hidden_dims": [32, 32],
        "forcing_parameterization": "direct",
        "hamiltonian_activation": "tanh",
        "damping_parameterization": "scalar_negative_softplus",
        "damping_floor": 1e-4,
        "damping_initial": -0.10,
        "force_cap": 4.0,
        "optimizer": "adam",
        "learning_rate": 3e-4,
        "learning_rate_end": 3e-6,
        "warmup_updates": 500,
        "gradient_clip_global_norm": 1.0,
        "lambda_force_l1": 1e-6,
        "lambda_force_jacobian": 3e-4,
        "lambda_encoder_l2": 1e-6,
        "lambda_readout_l2": 1e-6,
        "lambda_damping_l1": 0.0,
        "validation_horizons": [32, 64, 90],
        "validation_interval": 100,
        "precision": "float32",
        "rk_solver": "rk4",
        "input_hold": "zero_order_hold",
    }


def generate_followup(root: str | Path) -> dict:
    root = Path(root).resolve()
    output = root / "configs" / "porthnn_u" / "ced_followup16"
    output.mkdir(parents=True, exist_ok=True)
    configs = []
    for index, (latent_dim, hamiltonian, encoder, recipe) in enumerate(
        (
            (dimension, hamiltonian, encoder, recipe)
            for dimension in (2, 4)
            for hamiltonian in ([64, 64], [128, 128, 128])
            for encoder in ([64, 32], [128, 64])
            for recipe in ("reference", "extended_tail")
        ),
        start=1,
    ):
        if recipe == "reference":
            recipe_values = {
                "updates": 6000,
                "batch_size": 16,
                "late_loss_weight_start": 1.0,
                "late_loss_weight_end": 1.0,
                "terminal_loss_weight": 0.0,
                "training_schedule": [{"until": 6000, "horizons": [32, 64, 90], "probabilities": [1 / 3, 1 / 3, 1 / 3]}],
            }
        else:
            recipe_values = {
                "updates": 15000,
                "batch_size": 16,
                "late_loss_weight_start": 1.0,
                "late_loss_weight_end": 2.0,
                "terminal_loss_weight": 0.10,
                "early_stopping_patience": 3000,
                "training_schedule": [{"until": 15000, "horizons": [32, 64, 90], "probabilities": [0.20, 0.30, 0.50]}],
            }
        values = {
            **_base(), **recipe_values,
            "config_id": f"ced2_{index:02d}",
            "latent_dim": latent_dim,
            "hidden_dims": hamiltonian,
            "encoder_hidden_dims": encoder,
            "training_recipe": recipe,
        }
        if values["hamiltonian_activation"] != "tanh" or values["forcing_parameterization"] != "direct" or values["damping_parameterization"] != "scalar_negative_softplus" or latent_dim % 2:
            raise ValueError("follow-up config violates fixed CED follow-up factors")
        path = output / f"{values['config_id']}.yaml"
        path.write_text(yaml.safe_dump(values, sort_keys=False))
        configs.append({"path": str(path.relative_to(root)), "sha256": config_hash(values)})
    if len(configs) != 16 or len({item["sha256"] for item in configs}) != 16:
        raise ValueError("CED follow-up must contain exactly 16 unique configurations")
    base = _base()
    (output / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    result = root / "results" / "porthnn_u" / "ced_followup16"
    result.mkdir(parents=True, exist_ok=True)
    manifest = {"campaign": CAMPAIGN_NAME, "created_utc": datetime.now(UTC).isoformat(), "source_tree_hash": source_tree_hash(root), "configs": configs}
    (result / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest