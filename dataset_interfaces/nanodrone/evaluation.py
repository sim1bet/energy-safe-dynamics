"""Official rolling-horizon metrics for the Nano-drone benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .data import AffineScaler, rotvec_to_quaternion_xyzw


GROUPS = {
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "rotation": slice(6, 9),
    "omega": slice(9, 12),
}


def quaternion_geodesic_distance(
    true_quaternion_xyzw: np.ndarray, predicted_quaternion_xyzw: np.ndarray
) -> np.ndarray:
    """Return the shortest relative-rotation angle in radians on SO(3)."""
    true = np.asarray(true_quaternion_xyzw, dtype=np.float64)
    predicted = np.asarray(predicted_quaternion_xyzw, dtype=np.float64)
    if true.shape != predicted.shape or true.shape[-1] != 4:
        raise ValueError(f"Quaternion shapes must match (...,4), got {true.shape}, {predicted.shape}")
    true = true / np.linalg.norm(true, axis=-1, keepdims=True)
    predicted = predicted / np.linalg.norm(predicted, axis=-1, keepdims=True)
    absolute_dot = np.abs(np.sum(true * predicted, axis=-1))
    return 2.0 * np.arccos(np.clip(absolute_dot, 0.0, 1.0))


def rotvec_geodesic_distance(true_rotvec: np.ndarray, predicted_rotvec: np.ndarray) -> np.ndarray:
    return quaternion_geodesic_distance(
        rotvec_to_quaternion_xyzw(true_rotvec),
        rotvec_to_quaternion_xyzw(predicted_rotvec),
    )


def compute_physical_errors(true_state: np.ndarray, predicted_state: np.ndarray) -> dict[str, np.ndarray]:
    """Compute one physical group error per sample/window/horizon."""
    true_state = np.asarray(true_state, dtype=np.float64)
    predicted_state = np.asarray(predicted_state, dtype=np.float64)
    if true_state.shape != predicted_state.shape or true_state.shape[-1] != 12:
        raise ValueError(f"Expected matching (...,12) states, got {true_state.shape}, {predicted_state.shape}")
    return {
        "position": np.linalg.norm(true_state[..., 0:3] - predicted_state[..., 0:3], axis=-1),
        "velocity": np.linalg.norm(true_state[..., 3:6] - predicted_state[..., 3:6], axis=-1),
        "rotation": rotvec_geodesic_distance(true_state[..., 6:9], predicted_state[..., 6:9]),
        "omega": np.linalg.norm(true_state[..., 9:12] - predicted_state[..., 9:12], axis=-1),
    }


def compute_run_metrics(
    true_normalized: np.ndarray,
    predicted_normalized: np.ndarray,
    state_scaler: AffineScaler,
) -> dict:
    """Compute official and diagnostic metrics for arrays shaped [starts,H,12]."""
    true_normalized = np.asarray(true_normalized)
    predicted_normalized = np.asarray(predicted_normalized)
    if true_normalized.shape != predicted_normalized.shape:
        raise ValueError(
            f"Prediction/target mismatch: {predicted_normalized.shape} != {true_normalized.shape}"
        )
    if true_normalized.ndim != 3 or true_normalized.shape[-1] != 12:
        raise ValueError(f"Expected [starts,H,12], got {true_normalized.shape}")
    true_physical = state_scaler.inverse_transform(true_normalized)
    predicted_physical = state_scaler.inverse_transform(predicted_normalized)
    errors = compute_physical_errors(true_physical, predicted_physical)
    difference = predicted_normalized - true_normalized
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    target_std = float(np.std(true_normalized))
    return {
        "horizon": int(true_normalized.shape[1]),
        "n_starts": int(true_normalized.shape[0]),
        "mae_by_horizon": {name: np.mean(value, axis=0) for name, value in errors.items()},
        "aggregate_mae": {name: float(np.mean(value)) for name, value in errors.items()},
        "state_rmse_normalized": rmse,
        "state_nrmse_normalized": rmse / max(target_std, 1e-12),
        "per_channel_rmse_normalized": np.sqrt(np.mean(difference ** 2, axis=(0, 1))),
        "true_physical": true_physical,
        "predicted_physical": predicted_physical,
        "errors": errors,
    }


def aggregate_run_metrics(per_run: Mapping[str, dict]) -> dict:
    if not per_run:
        raise ValueError("At least one run metric is required")
    names = tuple(per_run)
    horizons = {per_run[name]["horizon"] for name in names}
    if len(horizons) != 1:
        raise ValueError(f"Run horizons do not match: {horizons}")
    counts = np.asarray([per_run[name]["n_starts"] for name in names], dtype=np.float64)
    weights = counts / counts.sum()
    by_horizon = {}
    aggregate = {}
    for group in GROUPS:
        values = np.stack([per_run[name]["mae_by_horizon"][group] for name in names])
        by_horizon[group] = np.sum(values * weights[:, None], axis=0)
        aggregate[group] = float(np.sum(
            np.asarray([per_run[name]["aggregate_mae"][group] for name in names]) * weights
        ))
    return {
        "horizon": horizons.pop(),
        "n_starts": int(counts.sum()),
        "mae_by_horizon": by_horizon,
        "aggregate_mae": aggregate,
    }


def paper_mae_summary(metrics: Mapping[str, dict]) -> dict[str, dict[str, float]]:
    """Return benchmark-paper H=1/10/50 and cumulative H=1:50 vector MAEs."""
    summary = {}
    for group in GROUPS:
        values = np.asarray(metrics["mae_by_horizon"][group], dtype=np.float64)
        if values.size < 50:
            raise ValueError(f"Paper MAE summary requires at least 50 horizons, got {values.size}")
        summary[group] = {
            "h1": float(values[0]),
            "h10": float(values[9]),
            "h50": float(values[49]),
            "cumulative_h1_h50": float(np.sum(values[:50])),
        }
    return summary


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def save_benchmark_metrics(
    output_dir: str | Path,
    per_run: Mapping[str, dict],
    aggregate: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_runs = {
        name: {
            key: value
            for key, value in metrics.items()
            if key not in {"true_physical", "predicted_physical", "errors"}
        }
        for name, metrics in per_run.items()
    }
    payload = {
        "protocol": "true-state rolling-start open-loop prediction",
        "horizon_steps": int(aggregate["horizon"]),
        "dt_seconds": 0.01,
        "rotation_metric": "shortest SO(3) geodesic angle in radians",
        "per_run": compact_runs,
        "aggregate": aggregate,
    }
    (output_dir / "benchmark_metrics.json").write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n"
    )
    arrays = {}
    for group in GROUPS:
        arrays[f"aggregate_mae_{group}"] = aggregate["mae_by_horizon"][group]
        for name, metrics in per_run.items():
            arrays[f"{name}_mae_{group}"] = metrics["mae_by_horizon"][group]
    np.savez_compressed(output_dir / "benchmark_metrics.npz", **arrays)