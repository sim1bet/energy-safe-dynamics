#!/usr/bin/env python3
# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Run the canonical fitting suite for a given config (Stress_tests/
theory_aligned/configs/fitting/*.yaml). Stdlib + pyyaml only; no JAX.

Usage:
    python3 run_fitting_suite.py configs/fitting/nlink_copilotA1.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_ROOT / "fitting"))
from fitting_report import build_fitting_report  # noqa: E402


def main(config_path: str) -> int:
    cfg = yaml.safe_load(Path(config_path).read_text())
    repo_root = SUITE_ROOT.parents[1]
    run_dir = repo_root / cfg["run_dir"]
    report = build_fitting_report(str(run_dir), cfg.get("target_threshold"))
    print(json.dumps(report.__dict__, indent=2))
    if report.missing_evidence:
        print(f"\nWARNING: missing evidence: {report.missing_evidence}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
