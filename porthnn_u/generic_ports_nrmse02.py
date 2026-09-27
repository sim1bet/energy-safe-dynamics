"""Generate the continuous-rollout-aware generic-port NRMSE campaign."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .campaign import config_hash, source_tree_hash


CAMPAIGN = "generic_ports_nrmse02"
LABEL = "Generalized Latent PortHNN-u (generic pH ports)"


def _common() -> dict:
    return {
        "campaign": CAMPAIGN, "model_schema_version": 4,
        "development_only": True, "official_test_access": "forbidden",
        "verification_profile": "production",
        "allow_smoke_output_bound_override": False,
        "max_abs_prediction_limit": 10.0,
        "model_label": LABEL, "seed": 0, "input_dim": 1, "output_dim": 1,
        "storage_type": "existing_porthnn_u", "storage_hidden_dims": [200, 200, 200],
        "storage_activation": "tanh", "storage": {"type": "existing_porthnn_u", "hidden_widths": [200, 200, 200], "activation": "tanh"},
        "legacy_porthnn_u": False, "precision": "float32", "rk_solver": "rk4",
        "input_hold": "zero_order_hold", "gradient_clip_global_norm": 1.0,
        "learning_rate": 2e-4, "learning_rate_end": 2e-6, "warmup_updates": 500,
        "matrix_activation": "tanh", "matrix_weight_decay": 1e-6,
        "late_loss_weight_start": 1.0, "late_loss_weight_end": 3.0,
        "validation_batch_size": 8, "continuous_validation_required": True,
    }


def _base(dataset: str) -> dict:
    if dataset == "silverbox":
        return {
            "dataset": dataset, "init_window": 50, "max_horizon": 768,
            "encoder_hidden_dims": [128, 64], "matrix_hidden_dims": [64, 64], "r_diagonal": 0.05,
            "g_jacobian_weight": 1e-5, "updates": 10000, "minimum_updates_before_stopping": 3000,
            "early_stopping_patience": 2000, "validation_interval": 250, "batch_size": 64,
            "training_window_stride": 16, "window_start_sampling": "fixed_stride",
            "window_start_stride_samples": 16, "sample_every_point": False,
            "validation_horizons": [64, 192, 384, 768],
            "training_schedule": [{"until": 10000, "horizons": [64, 192, 384], "probabilities": [0.15, 0.45, 0.40]}],
            "batch_size_h64": 64, "batch_size_h192": 64, "batch_size_h384": 32,
            "terminal_loss_weight": 0.25, "chained_loss_probability": 0.5,
            "chained_horizon": 192, "chained_chunks": 4,
            "validation_score": {"type": "silverbox_continuous", "weights": {"192": 0.10, "384": 0.15, "768": 0.20, "full": 0.55}},
            "input_map": {"type": "full_matrix_state_dependent", "state_dependent": True, "hidden_widths": [64, 64]},
            "readout": {"type": "affine_normalized", "input_feedthrough": False},
        }
    if dataset == "ced":
        return {
            "dataset": dataset, "init_window": 10, "max_horizon": 90,
            "encoder_hidden_dims": [64, 32], "matrix_hidden_dims": [32, 32], "r_diagonal": 0.02,
            "g_jacobian_weight": 3e-5, "matrix_weight_decay": 3e-6, "updates": 12000,
            "minimum_updates_before_stopping": 4000, "early_stopping_patience": 2500,
            "validation_interval": 100, "batch_size": 16, "training_window_stride": 1,
            "validation_horizons": [32, 64, 90],
            "training_schedule": [{"until": 12000, "horizons": [32, 64, 90], "probabilities": [0.15, 0.25, 0.60]}],
            "terminal_loss_weight": 0.20,
            "validation_score": {"type": "ced_continuous", "weights": {"32": 0.15, "64": 0.20, "90": 0.20, "full": 0.45}, "aggregate": {"mean": 0.60, "max": 0.25, "std": 0.15}},
            "input_map": {"type": "full_matrix_state_dependent", "state_dependent": True, "hidden_widths": [32, 32]},
        }
    raise ValueError(f"unsupported dataset: {dataset}")


def _assert_config(values: dict) -> None:
    assert values["official_test_access"] == "forbidden" and values["development_only"]
    assert values["verification_profile"] == "production"
    assert values["allow_smoke_output_bound_override"] is False
    assert values["max_abs_prediction_limit"] == 10.0
    assert values["storage"]["type"] == "existing_porthnn_u"
    assert values["input_map"]["type"] == "full_matrix_state_dependent"
    assert values["readout"].get("input_feedthrough") is False
    assert values["latent_dim"] in {2, 4}
    assert values["interconnection"]["type"] in {"canonical", "canonical_plus_learned_skew"}


def generate_generic_ports_nrmse02(root: str | Path) -> dict:
    root = Path(root).resolve()
    config_root = root / "configs" / "porthnn_u" / CAMPAIGN
    manifest_configs = []
    for dataset, prefix in (("silverbox", "sb02"), ("ced", "ced02")):
        base = {**_common(), **_base(dataset)}
        config_root.mkdir(parents=True, exist_ok=True)
        (config_root / f"base_{dataset}.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
        output = config_root / dataset; output.mkdir(exist_ok=True)
        configs = []
        for index, (dimension, interconnection, dissipation, level) in enumerate(
            (item for dimension in (2, 4) for interconnection in ("canonical", "canonical_plus_learned_skew")
             for dissipation in (("momentum_block_psd", "full_psd_state_dependent") if dataset == "silverbox" else ("full_psd_constant", "full_psd_state_dependent"))
             for item in ((dimension, interconnection, dissipation, 0), (dimension, interconnection, dissipation, 1))), start=1
        ):
            values = {**base, "config_id": f"{prefix}_{index:02d}", "latent_dim": dimension,
                      "interconnection": {"type": interconnection},
                      "dissipation": {"type": dissipation, "state_dependent": dissipation != "full_psd_constant"}}
            if dataset == "silverbox":
                values["symmetry_augmentation"] = bool(level)
                values["symmetry_transform"] = "negate_input_output"
            else:
                values["readout"] = {"type": "norm2" if level else "softabs_linear", "physical_space": True, "epsilon_scale": 1e-3, "input_feedthrough": False}
            _assert_config(values)
            path = output / f"{values['config_id']}.yaml"
            path.write_text(yaml.safe_dump(values, sort_keys=False))
            configs.append({"path": str(path.relative_to(root)), "sha256": config_hash(values)})
        if len(configs) != 16 or len({item["sha256"] for item in configs}) != 16:
            raise ValueError(f"{dataset} must have exactly 16 unique configurations")
        manifest_configs.extend(configs)
    (config_root / "base_common.yaml").write_text(yaml.safe_dump(_common(), sort_keys=False))
    manifest = {"campaign": CAMPAIGN, "created_utc": datetime.now(UTC).isoformat(), "source_tree_hash": source_tree_hash(root), "factors": {"silverbox": ["latent_dim", "interconnection", "dissipation", "symmetry_augmentation"], "ced": ["latent_dim", "interconnection", "dissipation", "physical_magnitude_readout"]}, "configs": manifest_configs}
    (config_root / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest