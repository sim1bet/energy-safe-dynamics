#!/usr/bin/env python3
"""Run a saved champion configuration from the standalone prerelease tree."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERFACES = ROOT / "dataset_interfaces"
for path in (ROOT, INTERFACES, ROOT / "EBM_model"):
    sys.path.insert(0, str(path))

SPECS = {
    "ced": ("CED_main", "datasets/CED/champion_configuration/config.json", {}),
    "duffing_doublewell": ("Duffing_DoubleWell_main", "datasets/duffing_doublewell/champion_configuration/config.json", {}),
    "deep_dissipative_nlink_n2": (
        "DeepDissipative_NLink_main", "datasets/deep_dissipative_nlink_n2/champion_configuration/config.json",
        {"dataset_dir": "datasets/deep_dissipative_nlink_n2/parameters", "prefix": "nlink2_100"},
    ),
    "deep_dissipative_nlink_n3": (
        "DeepDissipative_NLink_main", "datasets/deep_dissipative_nlink_n3/champion_configuration/config.json",
        {"dataset_dir": "datasets/deep_dissipative_nlink_n3/parameters", "prefix": "ex1_n3"},
    ),
    "nanodrone_S3": (
        "NanoDrone_main", "datasets/nanodrone_S3/champion_configuration/s3_ph_T2_seed47.json",
        {"nanodrone_dataset_root": "datasets/nanodrone_S3/parameters/raw_data"},
    ),
    "silverbox": ("Silverbox_main", "datasets/Silverbox/champion_configuration/config.json", {}),
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(SPECS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and imports without training.")
    args = parser.parse_args()
    module_name, config_path, overrides = SPECS[args.dataset]
    payload = json.loads((ROOT / config_path).read_text())
    config = dict(payload.get("config", payload))
    config.update(overrides)
    for key, value in list(config.items()):
        if key.endswith("_dir") or key.endswith("_root"):
            config[key] = str(ROOT / value)
    module = __import__(module_name)
    if args.dry_run:
        print(json.dumps({"status": "OK", "module": module_name, "config": config_path}, sort_keys=True))
        return
    if not hasattr(module, "train"):
        raise SystemExit(f"{module_name} has no train(seed, config) entry point")
    module.train(seed=config.get("seed", 0) if args.seed is None else args.seed, config=config)

if __name__ == "__main__":
    main()
