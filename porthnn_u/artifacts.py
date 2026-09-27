"""Portable parameter, provenance, and result artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import subprocess

import jax
import numpy as np
import yaml


def run_manifest(config, metadata: dict) -> dict:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        revision = "unavailable"
    return {
        "status": "adapted reimplementation, not official reproduction",
        "variant": "latent_porthnn_u",
        "official_repository": "https://github.com/shaandesai1/PortHNN",
        "official_commit": "dc5e2088382c1db72bc2ea64d9f0c8e00ae038d6",
        "implementation_commit_or_hash": revision,
        "dataset_name": config.dataset,
        "dataset_hashes": {
            "train": metadata["train_hashes"],
            "test": metadata["test_hashes"],
        },
        "seed": config.seed,
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "device": str(jax.devices()[0]),
        "precision": config.precision,
        "started_utc": datetime.now(UTC).isoformat(),
    }


def save_parameters(path: Path, params) -> None:
    arrays = {"damping": np.asarray(params["damping"])}
    for branch in ("hamiltonian", "forcing", "encoder"):
        for index, layer in enumerate(params[branch]):
            arrays[f"{branch}_weight_{index}"] = np.asarray(layer["weight"])
            if "bias" in layer:
                arrays[f"{branch}_bias_{index}"] = np.asarray(layer["bias"])
    arrays["readout_weight"] = np.asarray(params["readout"]["weight"])
    arrays["readout_bias"] = np.asarray(params["readout"]["bias"])
    np.savez(path, **arrays)


def load_parameters(path: Path, template):
    archive = np.load(path)
    params = dict(template)
    params["damping"] = jax.numpy.asarray(archive["damping"])
    for branch in ("hamiltonian", "forcing", "encoder"):
        layers = []
        for index, layer in enumerate(template[branch]):
            loaded = {"weight": jax.numpy.asarray(archive[f"{branch}_weight_{index}"])}
            if "bias" in layer:
                loaded["bias"] = jax.numpy.asarray(archive[f"{branch}_bias_{index}"])
            layers.append(loaded)
        params[branch] = tuple(layers)
    params["readout"] = {
        "weight": jax.numpy.asarray(archive["readout_weight"]),
        "bias": jax.numpy.asarray(archive["readout_bias"]),
    }
    return params


def initialize_run_directory(path: str | Path, config, normalization: dict, manifest: dict) -> Path:
    run_path = Path(path)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "predictions").mkdir(exist_ok=True)
    (run_path / "diagnostics").mkdir(exist_ok=True)
    (run_path / "logs").mkdir(exist_ok=True)
    with (run_path / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    (run_path / "normalization.json").write_text(json.dumps(normalization, indent=2, sort_keys=True))
    (run_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return run_path


def update_manifest(path: Path, manifest: dict) -> None:
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))