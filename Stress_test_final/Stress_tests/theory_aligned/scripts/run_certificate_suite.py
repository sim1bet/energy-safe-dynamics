#!/usr/bin/env python3
# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Run the canonical, theory-aligned certificate checks for a given config
(Stress_tests/theory_aligned/configs/certificates/*.yaml). numpy + pyyaml
only; no JAX (see certificates/checkpoint_io.py for why this is possible).

Usage:
    python3 run_certificate_suite.py configs/certificates/nlink_copilotA1.yaml

This script runs ONLY the tests actually registered as canonical in
test_traceability.yaml (structural_checks + the sampled unforced-
dissipativity probe). It does NOT invoke the legacy Stress_tests/suite.py
pipeline (wells/boundary/frontier/rollouts) -- that pipeline is unchanged,
shared production infrastructure, run separately by the training pipeline
itself, and its outputs are read (not recomputed) for diagnostic reporting.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_ROOT / "certificates"))
from structural_checks import run_all_structural_checks  # noqa: E402
from unforced_dissipativity_probe import run_probe as run_dissipativity_probe  # noqa: E402
from equilibrium_hessian_probe import run_probe as run_equilibrium_probe  # noqa: E402


def main(config_path: str) -> int:
    cfg = yaml.safe_load(Path(config_path).read_text())
    repo_root = SUITE_ROOT.parents[1]
    checkpoint = str(repo_root / cfg["checkpoint_path"])
    config_json = str(repo_root / cfg["config_path"])
    npz_path = str(repo_root / cfg["stress_arrays_npz"])
    audit_json = str(repo_root / cfg["audit_json"])

    structural = run_all_structural_checks(checkpoint, config_json)
    dissipativity = run_dissipativity_probe(checkpoint, config_json)
    equilibrium = run_equilibrium_probe(checkpoint, config_json, npz_path, audit_json)

    out = {
        "run_id": cfg["run_id"],
        "canonical_structural_checks": [asdict(r) for r in structural],
        "canonical_sampled_check_OB-T3-1": asdict(dissipativity),
        "diagnostic_only_OB-T5-1_OB-T12-1": asdict(equilibrium),
    }
    print(json.dumps(out, indent=2))

    all_canonical_passed = all(r.passed for r in structural) and all(
        r["drift_raw_nonneg"] and r["drift_clipped_nonneg"] for r in dissipativity.results
    )
    return 0 if all_canonical_passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
