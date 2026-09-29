# Author: Simone Betteti
"""Standalone inspection of official Nano-drone trajectories before training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import DEFAULT_DATASET_ROOT, load_nanodrone_dataset
from .plotting import plot_motor_inputs


def trajectory_regularity_ratio(state: np.ndarray, delta: float = 1e-12) -> float:
    first = np.diff(np.asarray(state), axis=0)
    second = np.diff(np.asarray(state), n=2, axis=0)
    return float(np.sum(second ** 2) / (np.sum(first ** 2) + delta))


def _plot_state_groups(trajectory, output_dir: Path) -> None:
    groups = (
        (slice(0, 3), ("x", "y", "z"), "position [m]"),
        (slice(3, 6), ("vx", "vy", "vz"), "velocity [m/s]"),
        (slice(6, 9), ("rot_x", "rot_y", "rot_z"), "SO(3) log [rad]"),
        (slice(9, 12), ("wx", "wy", "wz"), "angular velocity [rad/s]"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for axis, (channels, labels, ylabel) in zip(axes, groups):
        for offset, label in enumerate(labels):
            axis.plot(trajectory.time, trajectory.y[:, channels.start + offset], label=label, linewidth=0.9)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{trajectory.family.title()} representative state: run {trajectory.run_id}")
    fig.tight_layout()
    for suffix, dpi in ((".png", 150), (".pdf", None)):
        fig.savefig(output_dir / f"{trajectory.family}_state{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    plot_motor_inputs(
        trajectory.time,
        trajectory.u,
        output_dir / f"{trajectory.family}_inputs_motors",
    )


def run_dataset_diagnostics(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path = "results/nanodrone/dataset_diagnostics",
) -> dict:
    dataset = load_nanodrone_dataset(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = dataset["official_train"] + dataset["official_test"]
    results = {
        trajectory.name: {
            "family": trajectory.family,
            "run_id": trajectory.run_id,
            "samples": len(trajectory.y),
            "regularity_ratio_R2": trajectory_regularity_ratio(trajectory.y),
        }
        for trajectory in trajectories
    }
    for family in ("square", "random", "chirp", "melon"):
        representative = next(value for value in trajectories if value.family == family)
        _plot_state_groups(representative, output_dir)
    (output_dir / "trajectory_regularity.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default="results/nanodrone/dataset_diagnostics")
    args = parser.parse_args()
    run_dataset_diagnostics(args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()
