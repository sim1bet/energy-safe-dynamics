#!/usr/bin/env python3
# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Run the shared stress suite from a dataset-specific adapter module."""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT, ROOT / "Stress_test_final"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Stress_test_final.case import StressTestCase


def _jsonable(value):
    if isinstance(value, dict): return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def load_case(module_name: str, smoke: bool) -> StressTestCase:
    build_case = getattr(importlib.import_module(module_name), "build_case", None)
    if build_case is None: raise TypeError(f"{module_name} must define build_case(smoke=...) -> StressTestCase")
    case = build_case(smoke=smoke)
    if not isinstance(case, StressTestCase): raise TypeError("build_case must return StressTestCase")
    if case.reference_states.ndim != 2 or case.reference_states.shape[1] != case.d:
        raise ValueError("reference_states must have shape [n_samples, d]")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", help="Import path implementing build_case(smoke=...).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    from Stress_tests.suite import run_stress_suite

    case = load_case(args.adapter, args.smoke)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_stress_suite(
        params=case.params, layers=case.layers, d=case.d, m=case.m, dt=case.dt,
        epsilon=case.epsilon, gamma=case.gamma, reference_states=case.reference_states,
        reference_inputs=case.reference_inputs, output_dir=args.output_dir, config=case.config,
    )
    provenance = {
        "schema": "dataset_agnostic_stress_test_v1", "mode": "smoke" if args.smoke else "full",
        "adapter": args.adapter, "epsilon": case.epsilon, "gamma": case.gamma,
        "dimensions": {"state": case.d, "input": case.m}, "dt": case.dt,
        "comparison_epsilon": case.comparison_epsilon, "stress_config": asdict(case.config),
        "adapter_metadata": _jsonable(case.metadata),
        "epistemic_status": "Post-hoc sampled diagnostics on a frozen learned model; not a continuous-shell proof.",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "run_report.json").write_text(json.dumps({"decision": "COMPLETED_STRESS_SUITE", "summary": _jsonable(summary)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": "COMPLETED_STRESS_SUITE", "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__": main()
