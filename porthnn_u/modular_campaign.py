"""Configuration generation for the controlled modular $(H,J,R,G)$ campaign."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .campaign import config_hash, source_tree_hash


CAMPAIGN_NAME = "porthnn_u_modular_hjrg8"
MODEL_LABEL = "Generalized Latent PortHNN-u (modular pH-EBM matrices)"
MECHANISMS = (
    ("legacy", "canonical", "legacy_scalar", "legacy_force"),
    ("g_only", "canonical", "legacy_scalar", "learned_full_matrix"),
    ("rg", "canonical", "learned_full_psd", "learned_full_matrix"),
    ("jrg", "canonical_plus_learned_skew", "learned_full_psd", "learned_full_matrix"),
)


def _base(dataset: str) -> dict:
    if dataset == "silverbox":
        return {"dataset": dataset, "init_window": 50, "max_horizon": 768, "matrix_hidden_dims": [128, 128], "r_diagonal": 0.05, "updates": 15000, "batch_size": 64, "validation_horizons": [64, 192, 384, 768], "training_schedule": [{"until": 1000, "horizons": [64, 192], "probabilities": [0.5, 0.5]}, {"until": 15000, "horizons": [64, 192, 384, 768], "probabilities": [0.15, 0.30, 0.30, 0.25]}]}
    if dataset == "ced":
        return {"dataset": dataset, "init_window": 10, "max_horizon": 90, "matrix_hidden_dims": [32, 32], "r_diagonal": 0.02, "updates": 12000, "batch_size": 16, "validation_horizons": [32, 64, 90], "training_schedule": [{"until": 1000, "horizons": [32, 64], "probabilities": [0.5, 0.5]}, {"until": 12000, "horizons": [32, 64, 90], "probabilities": [0.2, 0.3, 0.5]}], "early_stopping_patience": 3000}
    raise ValueError(f"unsupported dataset: {dataset}")


def _common() -> dict:
    return {"campaign": CAMPAIGN_NAME, "model_schema_version": 2, "development_only": True, "official_test_access": "forbidden", "model_label": MODEL_LABEL, "seed": 0, "input_dim": 1, "output_dim": 1, "storage": {"type": "existing_porthnn_u"}, "storage_hidden_dims": [200, 200, 200], "storage_activation": "tanh", "encoder_hidden_dims": [128, 64], "learning_rate": 3e-4, "learning_rate_end": 3e-6, "warmup_updates": 500, "gradient_clip_global_norm": 1.0, "matrix_weight_decay": 1e-6, "j_residual_l2": 1e-7, "g_jacobian_weight": 1e-5, "validation_interval": 100, "validation_batch_size": 8, "precision": "float32", "rk_solver": "rk4", "input_hold": "zero_order_hold"}


def generate_modular_campaign(root: str | Path) -> dict:
    root = Path(root).resolve()
    generated = []
    for dataset, prefix in (("silverbox", "sb_mod"), ("ced", "ced_mod")):
        directory = root / "configs" / "porthnn_u" / "modular_hjrg8" / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for index, (dimension, mechanism) in enumerate(((d, mechanism) for d in (2, 4) for mechanism in MECHANISMS), start=1):
            name, interconnection, dissipation, input_map = mechanism
            values = {**_common(), **_base(dataset), "config_id": f"{prefix}_{index:02d}", "latent_dim": dimension, "mechanism": name, "interconnection": {"type": interconnection}, "dissipation": {"type": dissipation}, "input_map": {"type": input_map}, "legacy_porthnn_u": name == "legacy"}
            if dimension % 2 or values["input_dim"] != 1:
                raise ValueError("invalid modular campaign dimensions")
            path = directory / f"{values['config_id']}.yaml"
            path.write_text(yaml.safe_dump(values, sort_keys=False))
            generated.append({"path": str(path.relative_to(root)), "sha256": config_hash(values)})
    if len(generated) != 16 or len({entry["sha256"] for entry in generated}) != 16:
        raise ValueError("modular campaign must contain eight unique configurations per dataset")
    result_directory = root / "results" / "porthnn_u" / "modular_hjrg8"
    result_directory.mkdir(parents=True, exist_ok=True)
    manifest = {"campaign": CAMPAIGN_NAME, "created_utc": datetime.now(UTC).isoformat(), "source_tree_hash": source_tree_hash(root), "configs": generated, "mechanisms": [item[0] for item in MECHANISMS]}
    (result_directory / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest