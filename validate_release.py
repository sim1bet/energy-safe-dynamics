#!/usr/bin/env python3
# Author: Simone Betteti
"""Validate the public release; use --structural without runtime dependencies."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ("duffing_doublewell", "deep_dissipative_nlink_n2",
            "deep_dissipative_nlink_n3", "nanodrone_S3", "CED", "Silverbox")
ATTRIBUTED_SCOPES = ("EBM_model", "Stress_test_final", "runtime_hner99")
AUTHOR_LINE = "# Author: Simone Betteti"
PAPER_LINE = "# Paper: Safe-by-design learning via energy-based neural network"

def structural() -> dict:
    errors = []
    for name in DATASETS:
        if not (ROOT / "datasets" / name).is_dir(): errors.append(f"missing dataset: {name}")
    required = (
        "datasets/duffing_doublewell/checkpoints/params.pkl",
        "datasets/deep_dissipative_nlink_n2/checkpoints/params.pkl",
        "datasets/deep_dissipative_nlink_n3/checkpoints/params.pkl",
        "datasets/nanodrone_S3/checkpoints/best_physical_mae.pkl",
        "datasets/CED/checkpoints/params.pkl",
        "datasets/Silverbox/champion_configuration/config.json",
        "baselines/porthnn_u/champions/duffing/seed0/parameters.npz",
        "baselines/porthnn_u/champions/three_link/full_jrg/seed0/checkpoint_best.npz",
        "baselines/porthnn_u/champions/silverbox/sb02_16/best_validation_parameters.npz",
        "baselines/porthnn_u/champions/ced/ced02_07/fold_1/best_validation_parameters.npz",
        "scripts/validate_porthnn_u_artifacts.py",
        "scripts/validate_porthnn_u_runtime.py",
        "scripts/compute_robustness_radii.py",
        "scripts/audit_hner99_runtime.py",
        "scripts/validate_hner99_artifacts.py",
        "runtime_hner99/__init__.py",
        "runtime_hner99/__main__.py",
        "runtime_hner99/runtime.py",
        "runtime_hner99/audit.py",
        "runtime_hner99/artifacts.py",
        "runtime_hner99/ARTIFACT_INDEX.json",
        "certificates/hner99/FROZEN_RESULTS_PROVENANCE.json",
    )
    for relative in required:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0: errors.append(f"missing or empty: {relative}")
    for scope in ATTRIBUTED_SCOPES:
        for path in (ROOT / scope).rglob("*.py"):
            header = path.read_text(encoding="utf-8", errors="replace").splitlines()[:6]
            if AUTHOR_LINE not in header or PAPER_LINE not in header:
                errors.append(f"missing source attribution: {path.relative_to(ROOT)}")
    for path in ROOT.rglob("*"):
        if path.name in {"__pycache__", ".pytest_cache", ".DS_Store"}: errors.append(f"unwanted: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml", ".txt"}:
            text = path.read_text(errors="replace").lower()
            tokens = tuple(bytes.fromhex(value).decode("ascii") for value in (
                "2f6870632f", "2f736372617463682f", "7362657474657469",
                "23736261746368", "736c75726d5f"))
            for token in tokens:
                if token in text: errors.append(f"private/compute token {token!r}: {path.relative_to(ROOT)}")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "datasets": list(DATASETS), "file_count": sum(p.is_file() for p in ROOT.rglob("*"))}


def run_json_script(relative: str) -> dict:
    # Runtime validators import release modules; prevent their generated
    # bytecode from becoming an unexpected release artifact on the next run.
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, str(ROOT / relative)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"status": "FAIL", "stdout": proc.stdout}
    if proc.returncode:
        result["stderr"] = proc.stderr
        result["status"] = "FAIL"
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural", action="store_true", help="skip JAX inference checks")
    args = parser.parse_args()
    report = {"structural": structural()}
    if report["structural"]["status"] == "PASS":
        report["porthnn_u_artifacts"] = run_json_script(
            "scripts/validate_porthnn_u_artifacts.py"
        )
        report["hner99_artifacts"] = run_json_script(
            "scripts/validate_hner99_artifacts.py"
        )
    if all(section.get("status") == "PASS" for section in report.values()) and not args.structural:
        report["inference"] = run_json_script("scripts/validate_inference.py")
        report["porthnn_u_runtime"] = run_json_script(
            "scripts/validate_porthnn_u_runtime.py"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if any(section.get("status") != "PASS" for section in report.values()): raise SystemExit(1)

if __name__ == "__main__": main()
