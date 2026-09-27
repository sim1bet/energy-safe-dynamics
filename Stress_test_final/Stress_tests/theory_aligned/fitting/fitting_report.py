# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Application-agnostic fitting-evidence reader.

Reads the ALREADY-COMPUTED nominal reconstruction metrics from a run's
``audits/audit.json`` (written by ``post_training_eval.py``) and reports
them as model-fitting evidence, with full provenance (config identity,
checkpoint path, dataset identity).

This script does NOT retrain, re-simulate, or otherwise recompute anything;
it is a read-only report generator over stored artifacts, using only the
Python standard library (``json``), so it can run in any environment,
independent of JAX/numpy availability, and independent of which experiment
(``deep_dissipative_nlink``, ``duffing_doublewell``, or a future
``nonlinearbenchmark.org`` task) produced the run directory, as long as it
follows the shared ``experiment_paths.py``/``post_training_eval.py`` layout
(``<run>/config/config.json``, ``<run>/audits/audit.json``,
``<run>/run_status.json``).

It never contributes to, or is described as, a formal-certificate
pass/fail decision (see ``Stress_tests/theory_aligned/README.md``).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class FittingReport:
    run_id: str
    config_path: str
    checkpoint_path: str
    audit_path: str
    dataset_prefix: str | None
    mode: str | None
    metric_name: str
    metric_definition: str
    split: str
    test_rmse: float | None
    test_nrmse: float | None
    per_trajectory_rmse_min: float | None
    per_trajectory_rmse_max: float | None
    per_trajectory_rmse_mean: float | None
    n_test_trajectories: int | None
    target_threshold: float | None
    target_met: bool | None
    missing_evidence: list


def build_fitting_report(run_dir: str, target_threshold: float | None = 0.05) -> FittingReport:
    run_dir = Path(run_dir)
    config_path = run_dir / "config" / "config.json"
    checkpoint_path = run_dir / "checkpoints" / "params.pkl"
    audit_path = run_dir / "audits" / "audit.json"
    missing = []

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        missing.append(f"config not found: {config_path}")

    audit = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
    else:
        missing.append(f"audit.json not found: {audit_path}")

    recon = audit.get("metrics", {}).get("reconstruction", {})
    is_nanodrone = audit.get("experiment_name") == "nanodrone"
    ptr = recon.get("per_trajectory_rmse")

    test_nrmse = (
        audit.get("best_validation_nrmse") if is_nanodrone
        else recon.get("nrmse")
    )
    target_met = None
    if target_threshold is not None and test_nrmse is not None:
        target_met = bool(test_nrmse < target_threshold)

    if not checkpoint_path.exists():
        missing.append(f"checkpoint not found: {checkpoint_path}")
    if not recon and not is_nanodrone:
        missing.append("audits/audit.json has no metrics.reconstruction block")

    metric_name = "validation state NRMSE" if is_nanodrone else "test NRMSE"
    metric_definition = (
        "Best periodic open-loop validation NRMSE over flight-safe windows: "
        "sqrt(mean((x_pred - x_true)^2)) divided channelwise by the training-state "
        "scale, then averaged over state channels."
        if is_nanodrone else
        "RMSE = sqrt(mean((y_pred - y_true)^2)) over the full test-set array "
        "(all test trajectories x all open-loop rollout steps x all output "
        "channels, flattened); NRMSE = RMSE / max(std(y_true), 1e-12). "
        "Defined in Interface_code/DeepDissipative_NLink_main.py::_rmse_nrmse "
        "(or the corresponding function for other experiments sharing the "
        "same evaluation convention)."
    )
    split = (
        str(audit.get("evaluation_split", "development_validation"))
        if is_nanodrone else
        "test (held-out trajectories, never used for training or for epsilon-shell/certificate reference-state selection)"
    )

    return FittingReport(
        run_id=audit.get("run_id", run_dir.name),
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        audit_path=str(audit_path),
        dataset_prefix=config.get("prefix"),
        mode=config.get("mode"),
        metric_name=metric_name,
        metric_definition=metric_definition,
          split=split,
        test_rmse=recon.get("rmse"),
        test_nrmse=test_nrmse,
        per_trajectory_rmse_min=min(ptr) if ptr else None,
        per_trajectory_rmse_max=max(ptr) if ptr else None,
        per_trajectory_rmse_mean=(sum(ptr) / len(ptr)) if ptr else None,
        n_test_trajectories=len(ptr) if ptr else None,
        target_threshold=target_threshold,
        target_met=target_met,
        missing_evidence=missing,
    )


if __name__ == "__main__":
    run_dir = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    report = build_fitting_report(run_dir, threshold)
    print(json.dumps(asdict(report), indent=2))
