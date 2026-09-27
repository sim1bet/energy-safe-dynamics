"""experiment_paths.py — single source of truth for per-run result directories.

Root cause of the mis-routing bug (see DEBUGGING_REPORT.md §B): nothing in the
pipeline previously carried an explicit, checkable experiment identity. A
config JSON generated for one experiment could be dispatched through another
experiment's training module (selected only by the `EBM_DATASET`/`--dataset`
CLI value), silently merging an incompatible config and writing its outputs
under the wrong experiment's `results/` tree.

This module is the single place that turns `(experiment_name, run_id)` into a
concrete directory tree. Every entry point must derive paths from here rather
than constructing ad hoc `os.path.join(...)` calls, so a given experiment's
outputs can never end up under another experiment's directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Per-run subdirectories (prompt §7/§25). Kept underneath the repository's
# existing `results/<experiment_name>/` convention rather than inventing a
# new top-level layout.
_SUBDIR_NAMES = (
    "config", "checkpoints", "metrics", "signals",
    "trajectories", "hamiltonian", "stress_tests", "ood", "audits", "logs",
)


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_name: str
    run_id: str
    experiment_root: Path   # results/<experiment_name>/
    run_root: Path          # results/<experiment_name>/runs/<run_id>/
    config_dir: Path
    checkpoint_dir: Path
    metrics_dir: Path
    signals_dir: Path
    trajectories_dir: Path
    hamiltonian_dir: Path
    stress_test_dir: Path
    ood_dir: Path
    audits_dir: Path
    logs_dir: Path

    def as_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name, "run_id": self.run_id,
            "experiment_root": str(self.experiment_root), "run_root": str(self.run_root),
            "config_dir": str(self.config_dir), "checkpoint_dir": str(self.checkpoint_dir),
            "metrics_dir": str(self.metrics_dir), "signals_dir": str(self.signals_dir),
            "trajectories_dir": str(self.trajectories_dir), "hamiltonian_dir": str(self.hamiltonian_dir),
            "stress_test_dir": str(self.stress_test_dir), "ood_dir": str(self.ood_dir),
            "audits_dir": str(self.audits_dir),
            "logs_dir": str(self.logs_dir),
        }


def get_experiment_paths(experiment_name: str, run_id: str,
                         root: Path | str | None = None, create: bool = True) -> ExperimentPaths:
    """Resolve (and by default create) every output directory for one run.

    ``experiment_name`` must be passed explicitly by the caller (never
    inferred from cwd, `__name__`, or a global) — see prompt §8. Two
    different `experiment_name` values, even called back-to-back in the same
    process, always resolve to disjoint subtrees (covered by
    ``Stress_tests/Additional_tests/tests/test_experiment_paths.py``).
    """
    if not experiment_name or not isinstance(experiment_name, str):
        raise ValueError(f"experiment_name must be a non-empty string, got {experiment_name!r}")
    if not run_id or not isinstance(run_id, str):
        raise ValueError(f"run_id must be a non-empty string, got {run_id!r}")

    root = Path(root) if root is not None else REPO_ROOT
    experiment_root = root / "results" / experiment_name
    run_root = experiment_root / "runs" / run_id
    sub = {name: run_root / name for name in _SUBDIR_NAMES}

    paths = ExperimentPaths(
        experiment_name=experiment_name, run_id=run_id,
        experiment_root=experiment_root, run_root=run_root,
        config_dir=sub["config"], checkpoint_dir=sub["checkpoints"],
        metrics_dir=sub["metrics"], signals_dir=sub["signals"],
        trajectories_dir=sub["trajectories"], hamiltonian_dir=sub["hamiltonian"],
        stress_test_dir=sub["stress_tests"], ood_dir=sub["ood"], audits_dir=sub["audits"],
        logs_dir=sub["logs"],
    )
    if create:
        for d in (paths.run_root, *sub.values()):
            d.mkdir(parents=True, exist_ok=True)
    return paths
