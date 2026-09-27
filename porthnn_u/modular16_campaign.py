"""Physics-oriented 16-cell modular PortHNN-u campaign generation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .campaign import config_hash, source_tree_hash


CAMPAIGN = "modular_porthnn_u_physics16"
LABEL = "Generalized Latent PortHNN-u (modular pH-EBM matrices)"


def _common() -> dict:
    return {
        "campaign": CAMPAIGN, "model_schema_version": 3, "development_only": True,
        "official_test_access": "forbidden", "model_label": LABEL, "seed": 0,
        "input_dim": 1, "output_dim": 1, "storage_type": "existing_porthnn_u",
        "storage_hidden_dims": [200, 200, 200], "storage_activation": "tanh",
        "storage": {"type": "existing_porthnn_u", "hidden_widths": [200, 200, 200], "activation": "tanh"},
        "learning_rate": 3e-4, "learning_rate_end": 3e-6, "warmup_updates": 500,
        "gradient_clip_global_norm": 1.0, "storage_weight_decay": 1e-6,
        "matrix_weight_decay": 1e-6, "j_residual_l2": 1e-7, "precision": "float32",
        "rk_solver": "rk4", "input_hold": "zero_order_hold", "validation_interval": 100,
        "validation_batch_size": 8, "late_loss_weight_start": 1.0, "late_loss_weight_end": 2.0,
        "legacy_porthnn_u": False,
    }


def _base(dataset: str) -> dict:
    if dataset == "silverbox":
        return {
            "dataset": dataset, "init_window": 50, "max_horizon": 768,
            "encoder_hidden_dims": [128, 64], "matrix_hidden_dims": [128, 128],
            "r_diagonal": 0.05, "g_jacobian_weight": 1e-5, "updates": 15000,
            "batch_size": 64, "training_window_stride": 16, "window_start_sampling": "fixed_stride",
            "window_start_stride_samples": 16, "sample_every_point": False,
            "validation_horizons": [64, 192, 384, 768],
            "training_schedule": [
                {"until": 1000, "horizons": [64, 192], "probabilities": [0.40, 0.60]},
                {"until": 15000, "horizons": [64, 192, 384, 768], "probabilities": [0.15, 0.35, 0.30, 0.20]},
            ],
            "batch_size_h64": 64, "batch_size_h192": 64, "batch_size_h384": 32, "batch_size_h768": 16,
        }
    if dataset == "ced":
        return {
            "dataset": dataset, "init_window": 10, "max_horizon": 90,
            "encoder_hidden_dims": [64, 32], "matrix_hidden_dims": [32, 32],
            "r_diagonal": 0.02, "g_jacobian_weight": 3e-5, "updates": 12000,
            "batch_size": 16, "training_window_stride": 1, "terminal_loss_weight": 0.10,
            "validation_horizons": [32, 64, 90],
            "training_schedule": [
                {"until": 1000, "horizons": [32, 64], "probabilities": [0.40, 0.60]},
                {"until": 12000, "horizons": [32, 64, 90], "probabilities": [0.20, 0.30, 0.50]},
            ],
            "early_stopping_patience": 3000, "minimum_updates_before_stopping": 6000,
        }
    raise ValueError(f"unsupported dataset: {dataset}")


def _levels(dataset: str):
    dissipation = ("momentum_block_psd", "full_psd_state_dependent") if dataset == "silverbox" else ("full_psd_constant", "full_psd_state_dependent")
    return (
        (dimension, interconnection, dissipation_type, input_map)
        for dimension in (2, 4)
        for interconnection in ("canonical", "canonical_plus_learned_skew")
        for dissipation_type in dissipation
        for input_map in ("full_matrix_constant", "full_matrix_state_dependent")
    )


def generate_modular16(root: str | Path) -> dict:
    root = Path(root).resolve()
    config_root = root / "configs" / "porthnn_u" / "modular16"
    manifest_configs = []
    for dataset, prefix in (("silverbox", "sbjrg"), ("ced", "cedjrg")):
        base = {**_common(), **_base(dataset)}
        (config_root / f"base_{dataset}.yaml").parent.mkdir(parents=True, exist_ok=True)
        (config_root / f"base_{dataset}.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
        output = config_root / dataset; output.mkdir(exist_ok=True)
        configs = []
        for index, (dimension, interconnection, dissipation, input_map) in enumerate(_levels(dataset), start=1):
            values = {
                **base, "config_id": f"{prefix}_{index:02d}", "latent_dim": dimension,
                "interconnection": {"type": interconnection},
                "dissipation": {"type": dissipation, "state_dependent": dissipation.endswith("state_dependent") or dissipation == "momentum_block_psd"},
                "input_map": {"type": input_map, "state_dependent": input_map.endswith("state_dependent")},
            }
            if dimension not in (2, 4) or dimension % 2 or values["official_test_access"] != "forbidden":
                raise ValueError("invalid modular16 configuration")
            path = output / f"{values['config_id']}.yaml"
            path.write_text(yaml.safe_dump(values, sort_keys=False))
            configs.append({"path": str(path.relative_to(root)), "sha256": config_hash(values)})
        if len(configs) != 16 or len({item["sha256"] for item in configs}) != 16:
            raise ValueError(f"{dataset} modular16 configurations must be exactly 16 and unique")
        manifest_configs.extend(configs)
    (config_root / "base_common.yaml").write_text(yaml.safe_dump(_common(), sort_keys=False))
    result = root / "results" / "porthnn_u" / "modular16"; result.mkdir(parents=True, exist_ok=True)
    manifest = {"campaign": CAMPAIGN, "created_utc": datetime.now(UTC).isoformat(), "source_tree_hash": source_tree_hash(root), "factors": {"silverbox": ["latent_dim", "interconnection", "momentum_or_full_dissipation", "constant_or_state_dependent_input"], "ced": ["latent_dim", "interconnection", "constant_or_state_dependent_full_dissipation", "constant_or_state_dependent_input"]}, "configs": manifest_configs}
    (result / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest