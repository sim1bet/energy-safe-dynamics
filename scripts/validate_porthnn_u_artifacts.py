#!/usr/bin/env python3
# Author: Simone Betteti
"""Aggregate integrity and stored-result audit for all PortHNN-u champions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHAMPIONS = ROOT / "baselines" / "porthnn_u" / "champions"
EXPECTED = {
    "duffing": (
        CHAMPIONS / "duffing" / "seed0" / "parameters.npz",
        "de9e1801ba42711f9b31d0e4c06f10c0bf7fd594cf9865a3ba110a3047e50bbd",
    ),
    "three_link": (
        CHAMPIONS / "three_link" / "full_jrg" / "seed0" / "checkpoint_best.npz",
        "e173de487951e16d58222a9a8de2b6eeb4c2b56168b2fc49cc4698131d197733",
    ),
    "silverbox": (
        CHAMPIONS / "silverbox" / "sb02_16" / "best_validation_parameters.npz",
        "51a09b538ed695647472320708090c34f9e76d712d3257bb99e6a61158f509dc",
    ),
    "ced": (
        CHAMPIONS / "ced" / "ced02_07" / "fold_1" / "best_validation_parameters.npz",
        "8fff60c5aa0de7ada96df677e4264bf97df7fc727bc6907c69e2ceb59c3f2129",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_archive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        empty = [name for name in archive.files if archive[name].size == 0]
        nonfinite = [name for name in archive.files
                     if np.issubdtype(archive[name].dtype, np.number)
                     and not np.isfinite(archive[name]).all()]
        return {"arrays": len(archive.files), "empty": empty, "nonfinite": nonfinite}


def close(actual: float, expected: float, label: str, *, rtol: float = 1e-9) -> None:
    if not np.isclose(actual, expected, rtol=rtol, atol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def audit_stored_predictions() -> dict:
    report = {}
    duffing = CHAMPIONS / "duffing" / "seed0"
    with np.load(duffing / "predictions.npz") as values:
        error = values["predictions"].astype(np.float64) - values["truth"].astype(np.float64)
        rmse = float(np.sqrt(np.mean(error ** 2)))
        per_trajectory = np.sqrt(np.mean(error ** 2, axis=(1, 2)))
    metrics = json.loads((duffing / "metrics.json").read_text())
    close(rmse, metrics["test_rmse"], "duffing test RMSE", rtol=1e-6)
    close(float(np.median(per_trajectory)), metrics["median_trajectory_rmse"],
          "duffing median trajectory RMSE", rtol=1e-6)
    report["duffing"] = {"test_rmse_recomputed": rmse, "trajectories": len(per_trajectory)}

    three = CHAMPIONS / "three_link" / "full_jrg" / "seed0"
    with np.load(three / "test_predictions.npz") as values:
        error = values["prediction"].astype(np.float64) - values["truth"].astype(np.float64)
        output_error = (values["output_prediction"].astype(np.float64)
                        - values["output_truth"].astype(np.float64))
        rmse = float(np.sqrt(np.mean(error ** 2)))
        output_rmse = float(np.sqrt(np.mean(output_error ** 2)))
        nonfinite = int(np.size(values["prediction"]) - np.isfinite(values["prediction"]).sum())
    metrics = json.loads((three / "test_evaluation.json").read_text())["metrics"]
    close(rmse, metrics["aggregate_rmse"], "three-link aggregate RMSE")
    close(output_rmse, metrics["ph_ebm_comparable_output_rmse"],
          "three-link output RMSE")
    if nonfinite != metrics["nonfinite_prediction_count"]:
        raise AssertionError("three-link nonfinite count mismatch")
    report["three_link"] = {"aggregate_rmse_recomputed": rmse,
                            "output_rmse_recomputed": output_rmse}

    for dataset, run in {
        "silverbox": CHAMPIONS / "silverbox" / "sb02_16",
        "ced": CHAMPIONS / "ced" / "ced02_07" / "fold_1",
    }.items():
        with np.load(run / "continuous_validation_predictions.npz") as values:
            count = sum(len(values[name]) for name in values.files)
            finite = all(np.isfinite(values[name]).all() for name in values.files)
        metrics = json.loads((run / "metrics_validation.json").read_text())
        if not finite or metrics["nonfinite_count"] != 0 or metrics["rejected"]:
            raise AssertionError(f"{dataset}: invalid stored validation result")
        report[dataset] = {"validation_records": count,
                           "selection_score": metrics["selection_score"]}
    return report


def main() -> None:
    report = {"status": "PASS", "checkpoints": {}, "stored_predictions": {}}
    try:
        for name, (path, expected_hash) in EXPECTED.items():
            observed = sha256(path)
            if observed != expected_hash:
                raise AssertionError(f"{name}: checkpoint hash mismatch")
            archive = finite_archive(path)
            if archive["empty"] or archive["nonfinite"]:
                raise AssertionError(f"{name}: invalid arrays {archive}")
            report["checkpoints"][name] = {
                "path": path.relative_to(ROOT).as_posix(), "sha256": observed, **archive}
        report["stored_predictions"] = audit_stored_predictions()
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(report["status"] != "PASS")


if __name__ == "__main__":
    main()
