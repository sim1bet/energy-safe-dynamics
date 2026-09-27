"""Release-relative loaders for the frozen modular PortHNN-u champions."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .modular import init_modular_params
from .modular_train import _spec

RELEASE_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_ROOT = RELEASE_ROOT / "baselines" / "porthnn_u" / "champions"
CHAMPIONS = {
    "silverbox": CHAMPION_ROOT / "silverbox" / "sb02_16",
    "ced": CHAMPION_ROOT / "ced" / "ced02_07" / "fold_1",
}


def restore_tree(path: Path, template):
    """Restore the named-array checkpoint format written by modular training."""
    with np.load(path) as archive:
        remaining = set(archive.files)

        def restore(value, name=""):
            if isinstance(value, dict):
                return {key: restore(item, f"{name}_{key}" if name else key)
                        for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(restore(item, f"{name}_{index}")
                             for index, item in enumerate(value))
            if name not in remaining:
                raise KeyError(f"checkpoint array is missing: {name}")
            remaining.remove(name)
            loaded = np.asarray(archive[name])
            if loaded.shape != np.asarray(value).shape:
                raise ValueError(
                    f"shape mismatch for {name}: {loaded.shape} != {np.asarray(value).shape}")
            return jnp.asarray(loaded)

        params = restore(template)
        if remaining:
            raise ValueError(f"unexpected checkpoint arrays: {sorted(remaining)}")
        return params


def load_modular_champion(dataset: str):
    """Return ``(run, config, spec, params, normalization)`` for CED/Silverbox."""
    if dataset not in CHAMPIONS:
        raise KeyError(f"unknown modular champion {dataset!r}; choose {sorted(CHAMPIONS)}")
    run = CHAMPIONS[dataset]
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    spec = _spec(config)
    template = init_modular_params(
        jax.random.key(int(config["seed"])), spec, int(config["init_window"]))
    params = restore_tree(run / "best_validation_parameters.npz", template)
    normalization = json.loads((run / "normalization.json").read_text())
    normalization = {key: np.asarray(value, dtype=np.float32)
                     for key, value in normalization.items()}
    return run, config, spec, params, normalization
