"""Publication-oriented plots for Nano-drone identification runs."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"true": "#176B87", "predicted": "#D1495B", "input": "#4F5D2F"}


def _save_both(fig, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix, dpi in ((".png", 150), (".pdf", None)):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_motor_inputs(time: np.ndarray, inputs: np.ndarray, stem: str | Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for channel, axis in enumerate(axes):
        axis.plot(time, inputs[:, channel], color=COLORS["input"], linewidth=1.0)
        axis.set_ylabel(f"motor {channel + 1}\n[rad/s]")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Recorded motor angular velocities")
    _save_both(fig, stem)


def plot_reconstruction_windows(
    true_state: np.ndarray,
    predicted_state: np.ndarray,
    starts: np.ndarray,
    dt: float,
    channels: slice,
    labels: Sequence[str],
    unit: str,
    title: str,
    stem: str | Path,
    max_windows: int = 3,
) -> None:
    true_state = np.asarray(true_state)
    predicted_state = np.asarray(predicted_state)
    if true_state.shape != predicted_state.shape or true_state.ndim != 3:
        raise ValueError(f"Expected matching [starts,H,12] arrays, got {true_state.shape}, {predicted_state.shape}")
    selected = np.unique(np.linspace(0, len(true_state) - 1, min(max_windows, len(true_state)), dtype=int))
    fig, axes = plt.subplots(len(labels), len(selected), figsize=(4.2 * len(selected), 2.3 * len(labels)), squeeze=False)
    for column, window_index in enumerate(selected):
        horizon = true_state.shape[1]
        time = (starts[window_index] + np.arange(1, horizon + 1)) * dt
        for row, channel in enumerate(range(channels.start, channels.stop)):
            axis = axes[row, column]
            axis.plot(time, true_state[window_index, :, channel], color=COLORS["true"], label="true")
            axis.plot(time, predicted_state[window_index, :, channel], color=COLORS["predicted"], linestyle="--", label="predicted")
            axis.set_ylabel(f"{labels[row]} [{unit}]")
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_title(f"start {starts[window_index] * dt:.2f} s")
            if row == len(labels) - 1:
                axis.set_xlabel("flight time [s]")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    _save_both(fig, stem)


def plot_orientation_error(
    errors_radians: np.ndarray, starts: np.ndarray, dt: float, stem: str | Path
) -> None:
    selected = np.unique(np.linspace(0, len(errors_radians) - 1, min(3, len(errors_radians)), dtype=int))
    fig, axis = plt.subplots(figsize=(10, 4))
    for index in selected:
        time = (starts[index] + np.arange(1, errors_radians.shape[1] + 1)) * dt
        axis.plot(time, np.degrees(errors_radians[index]), label=f"start {starts[index] * dt:.2f} s")
    axis.set(xlabel="flight time [s]", ylabel="SO(3) geodesic error [deg]", title="Orientation prediction error")
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save_both(fig, stem)


def plot_position_3d(
    true_full_position: np.ndarray,
    predicted_windows_position: np.ndarray,
    stem: str | Path,
    long_prediction: np.ndarray | None = None,
) -> None:
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    true_full_position = np.asarray(true_full_position)
    axis.plot(*true_full_position.T, color=COLORS["true"], linewidth=1.5, label="true trajectory")
    if long_prediction is not None:
        axis.plot(*np.asarray(long_prediction).T, color=COLORS["predicted"], linestyle="--", linewidth=1.2, label="long prediction")
    else:
        selected = np.unique(np.linspace(0, len(predicted_windows_position) - 1, min(8, len(predicted_windows_position)), dtype=int))
        for count, index in enumerate(selected):
            axis.plot(*predicted_windows_position[index].T, color=COLORS["predicted"], alpha=0.75, linewidth=1.0, label="50-step predictions" if count == 0 else None)
    axis.scatter(*true_full_position[0], color="#2A9D8F", s=45, label="start")
    axis.scatter(*true_full_position[-1], color="#E9C46A", edgecolor="black", s=45, label="end")
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="Melon position trajectory")
    spans = np.ptp(true_full_position, axis=0)
    if np.all(spans > 0):
        axis.set_box_aspect(spans)
    axis.legend(fontsize=8)
    _save_both(fig, stem)


def plot_horizon_metrics(metrics: Mapping[str, np.ndarray], dt: float, stem: str | Path) -> None:
    labels = {
        "position": "position MAE [m]",
        "velocity": "velocity MAE [m/s]",
        "rotation": "orientation MAE [deg]",
        "omega": "angular-velocity MAE [rad/s]",
    }
    horizon = len(next(iter(metrics.values())))
    time = np.arange(1, horizon + 1) * dt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, group in zip(axes.ravel(), labels):
        values = np.asarray(metrics[group])
        if group == "rotation":
            values = np.degrees(values)
        axis.plot(time, values, color=COLORS["predicted"], linewidth=1.5)
        axis.set_ylabel(labels[group])
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("prediction horizon [s]")
    fig.suptitle("Official rolling-start prediction errors")
    _save_both(fig, stem)


def plot_training_history(history: Mapping[str, Sequence[float]], stem: str | Path) -> None:
    fields = (
        ("loss_total", "total loss"),
        ("loss_obs", "observation loss"),
        ("train_nrmse", "train NRMSE"),
        ("grad_norm", "gradient norm"),
        ("skipped_updates", "skipped updates"),
    )
    fig, axes = plt.subplots(len(fields), 1, figsize=(9, 10), sharex=True)
    epoch = np.asarray(history.get("epoch", np.arange(len(history.get("loss_total", [])))))
    for axis, (key, label) in zip(axes, fields):
        fallback = np.full(epoch.shape, np.nan, dtype=float)
        axis.plot(epoch, history.get(key, fallback), linewidth=1.1)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("epoch")
    _save_both(fig, stem)