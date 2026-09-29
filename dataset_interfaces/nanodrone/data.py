"""Strict loader and trajectory-aware preprocessing for the Nano-drone data."""
from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EXTERNAL_COMMIT = "2d921b57d166fe2debe08a5d39bd07297c5abc39"
DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parents[1]
    / ".external"
    / "nanodrone-sysid-benchmark"
)
INPUT_COLUMNS = ("m1_rads", "m2_rads", "m3_rads", "m4_rads")
DYNAMIC_PORT_COLUMNS = (
    "collective_omega2", "roll_arm_omega2", "pitch_arm_omega2", "yaw_omega2",
    "yaw_acceleration_proxy", "signed_rotor_momentum_proxy",
)
COMBINED_DYNAMIC_PORT_COLUMNS = INPUT_COLUMNS + DYNAMIC_PORT_COLUMNS
ROTOR_WRENCH_COLUMNS = (
    "collective_omega2", "roll_omega2", "pitch_omega2", "yaw_omega2",
)
MOTOR_SPEED_SQUARED_COLUMNS = (
    "m1_rads2", "m2_rads2", "m3_rads2", "m4_rads2",
)
INPUT_REPRESENTATIONS = (
    "motor_speed", "motor_speed_squared", "rotor_wrench",
    "dynamic_physical_port", "combined_dynamic_port",
)
QUATERNION_COLUMNS = ("qx", "qy", "qz", "qw")
RAW_STATE_COLUMNS = (
    "x", "y", "z", "vx", "vy", "vz", "qx", "qy", "qz", "qw", "wx", "wy", "wz"
)
STATE_COLUMNS = (
    "x", "y", "z", "vx", "vy", "vz",
    "rot_x", "rot_y", "rot_z", "wx", "wy", "wz",
)
TRAIN_FAMILIES = ("square", "random", "chirp")
TEST_FAMILY = "melon"
EXPECTED_TRAIN_FILES = tuple(
    f"{family}_20251017_run{run}.csv"
    for family in TRAIN_FAMILIES
    for run in range(1, 5)
)
EXPECTED_TEST_FILES = tuple(f"melon_20251017_run{run}.csv" for run in range(1, 4))
REQUIRED_COLUMNS = ("t",) + RAW_STATE_COLUMNS + INPUT_COLUMNS
_NAME_RE = re.compile(r"^(square|random|chirp|melon)_20251017_run([1-4])\.csv$")


def quaternion_xyzw_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized `xyzw` quaternions to upstream-compatible SO(3) logs.

    This is the NumPy equivalent of the official loader's
    `xyzw -> wxyz -> pytorch3d.transforms.quaternion_to_axis_angle` path.
    No temporal unwrapping or sign rewriting is applied.
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 1e-12):
        raise ValueError("Quaternion contains non-finite or near-zero norms")
    normalized = quaternion / norm
    vector = normalized[..., :3]
    real = normalized[..., 3]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, real)
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1e-12,
    )
    return vector * scale[..., None]


def rotvec_to_quaternion_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vectors to unit quaternions in `xyzw` convention."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Expected rotation-vector shape (..., 3), got {rotvec.shape}")
    angle = np.linalg.norm(rotvec, axis=-1)
    half = 0.5 * angle
    scale = np.divide(
        np.sin(half),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > 1e-12,
    )
    quaternion = np.concatenate(
        [rotvec * scale[..., None], np.cos(half)[..., None]], axis=-1
    )
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


@dataclass(frozen=True)
class NanoDroneTrajectory:
    name: str
    family: str
    run_id: int
    source_split: str
    time: np.ndarray
    u: np.ndarray
    y: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        if self.u.ndim != 2 or self.u.shape[1] != 4:
            raise ValueError(f"{self.name}: expected u [T,4], got {self.u.shape}")
        if self.y.ndim != 2 or self.y.shape[1] != 12:
            raise ValueError(f"{self.name}: expected y [T,12], got {self.y.shape}")
        if len(self.time) != len(self.u) or len(self.u) != len(self.y):
            raise ValueError(f"{self.name}: time/u/y lengths do not match")


def transform_inputs(inputs: np.ndarray, representation: str = "motor_speed") -> np.ndarray:
    """Map motor speeds to the selected four-dimensional input coordinates."""
    inputs = np.asarray(inputs)
    if inputs.shape[-1] != 4:
        raise ValueError(f"Expected inputs with four motor channels, got {inputs.shape}")
    if representation == "motor_speed":
        return inputs
    if representation in {"dynamic_physical_port", "combined_dynamic_port"}:
        omega = inputs.astype(np.float64)
        omega2 = np.square(omega)
        m1, m2, m3, m4 = np.moveaxis(omega2, -1, 0)
        static = np.stack((
            m1 + m2 + m3 + m4,
            0.0353 * (-m1 - m2 + m3 + m4),
            0.0353 * (-m1 + m2 + m3 - m4),
            m1 - m2 + m3 - m4,
        ), axis=-1)
        derivative = np.concatenate(
            [np.zeros_like(omega[:1]), np.diff(omega, axis=0) / 0.01], axis=0
        )
        spin = np.asarray((1.0, -1.0, 1.0, -1.0))
        transient = np.stack((
            -np.sum(derivative * spin, axis=-1),
            np.sum(omega * spin, axis=-1),
        ), axis=-1)
        dynamic = np.concatenate([static, transient], axis=-1)
        return dynamic if representation == "dynamic_physical_port" else np.concatenate(
            [omega, dynamic], axis=-1
        )
    if representation not in {"motor_speed_squared", "rotor_wrench"}:
        raise ValueError(
            f"Unknown NanoDrone input representation {representation!r}; "
            f"expected one of {INPUT_REPRESENTATIONS}"
        )
    omega2 = np.square(inputs.astype(np.float64))
    if representation == "motor_speed_squared":
        return omega2
    m1, m2, m3, m4 = np.moveaxis(omega2, -1, 0)
    return np.stack(
        (
            m1 + m2 + m3 + m4,
            -m1 - m2 + m3 + m4,
            -m1 + m2 + m3 - m4,
            m1 - m2 + m3 - m4,
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class AffineScaler:
    mean: np.ndarray
    std: np.ndarray
    columns: tuple[str, ...]
    epsilon: float = 1e-8

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray], columns: Sequence[str]) -> "AffineScaler":
        values = np.concatenate([np.asarray(value, dtype=np.float64) for value in arrays], axis=0)
        if values.ndim != 2 or values.shape[1] != len(columns):
            raise ValueError(f"Scaler values {values.shape} do not match {len(columns)} columns")
        if not np.all(np.isfinite(values)):
            raise ValueError("Cannot fit scaler to non-finite values")
        return cls(values.mean(axis=0), values.std(axis=0), tuple(columns))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values) - self.mean) / (self.std + self.epsilon)).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * (self.std + self.epsilon) + self.mean

    def as_dict(self) -> dict:
        return {
            "columns": list(self.columns),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class WindowedDataset:
    x0: np.ndarray
    u_seq: np.ndarray
    target: np.ndarray
    trajectory_name: np.ndarray
    run_id: np.ndarray
    start_index: np.ndarray

    def __post_init__(self) -> None:
        count = self.x0.shape[0]
        if self.x0.ndim != 2 or self.x0.shape[1] != 12:
            raise ValueError(f"Expected x0 [B,12], got {self.x0.shape}")
        if self.u_seq.ndim != 3 or self.u_seq.shape[0] != count or self.u_seq.shape[2] <= 0:
            raise ValueError(f"Expected u_seq [B,H,m] with m>0, got {self.u_seq.shape}")
        expected = self.u_seq.shape[:2] + (12,)
        if self.target.shape != expected:
            raise ValueError(f"Expected target {expected}, got {self.target.shape}")
        if any(len(value) != count for value in (self.trajectory_name, self.run_id, self.start_index)):
            raise ValueError("Window metadata length does not match batch dimension")


def _read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {path}") from exc
        rows = list(reader)
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}; got {header}")
    try:
        values = np.asarray(rows, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{path.name}: non-numeric CSV value") from exc
    if values.ndim != 2 or values.shape[1] != len(header):
        raise ValueError(f"{path.name}: malformed CSV shape {values.shape} for {len(header)} columns")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name}: data contain NaN or infinite values")
    return header, values


def load_trajectory(path: str | Path, *, expected_dt: float = 0.01) -> NanoDroneTrajectory:
    path = Path(path)
    match = _NAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected Nano-drone filename: {path.name}")
    family, run_text = match.groups()
    header, values = _read_numeric_csv(path)
    index = {name: position for position, name in enumerate(header)}
    time = values[:, index["t"]]
    increments = np.diff(time)
    if len(increments) == 0 or not np.allclose(increments, expected_dt, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{path.name}: irregular timestamps; median={np.median(increments):.9g}, "
            f"range=[{np.min(increments):.9g}, {np.max(increments):.9g}]"
        )
    quaternion = values[:, [index[name] for name in QUATERNION_COLUMNS]]
    quaternion_norm = np.linalg.norm(quaternion, axis=-1)
    if not np.allclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-3):
        raise ValueError(f"{path.name}: quaternion norms are not close to one")
    rotvec = quaternion_xyzw_to_rotvec(quaternion)
    state = np.column_stack(
        [
            values[:, [index[name] for name in ("x", "y", "z")]],
            values[:, [index[name] for name in ("vx", "vy", "vz")]],
            rotvec,
            values[:, [index[name] for name in ("wx", "wy", "wz")]],
        ]
    )
    inputs = values[:, [index[name] for name in INPUT_COLUMNS]]
    return NanoDroneTrajectory(
        name=path.stem,
        family=family,
        run_id=int(run_text),
        source_split=path.parent.name,
        time=time,
        u=inputs.astype(np.float32),
        y=state.astype(np.float32),
        quaternion_xyzw=quaternion,
    )


def _require_exact_files(directory: Path, expected: Sequence[str]) -> None:
    actual = {path.name for path in directory.glob("*.csv")}
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    if missing or unexpected:
        raise FileNotFoundError(
            f"Dataset mismatch in {directory}: missing={missing}, unexpected={unexpected}"
        )
    pointers = []
    for filename in expected:
        with (directory / filename).open("rb") as handle:
            if handle.read(64).startswith(b"version https://git-lfs.github.com/spec"):
                pointers.append(filename)
    if pointers:
        raise RuntimeError(f"Git LFS objects are not materialized: {pointers}")


def load_nanodrone_dataset(
    root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    include_official_test: bool = True,
) -> dict[str, list[NanoDroneTrajectory]]:
    root = Path(root)
    train_dir, test_dir = root / "data" / "train", root / "data" / "test"
    _require_exact_files(train_dir, EXPECTED_TRAIN_FILES)
    official_train = [load_trajectory(train_dir / name) for name in EXPECTED_TRAIN_FILES]
    official_test = []
    if include_official_test:
        _require_exact_files(test_dir, EXPECTED_TEST_FILES)
        official_test = [load_trajectory(test_dir / name) for name in EXPECTED_TEST_FILES]
    development_train = [trajectory for trajectory in official_train if trajectory.run_id <= 3]
    development_validation = [trajectory for trajectory in official_train if trajectory.run_id == 4]
    if any(trajectory.family == TEST_FAMILY for trajectory in development_train):
        raise RuntimeError("Melon leaked into development training")
    return {
        "development_train": development_train,
        "development_validation": development_validation,
        "official_train": official_train,
        "official_test": official_test,
    }


def fit_scalers(
    trajectories: Sequence[NanoDroneTrajectory],
    input_representation: str = "motor_speed",
) -> tuple[AffineScaler, AffineScaler]:
    if not trajectories:
        raise ValueError("At least one training trajectory is required")
    if any(trajectory.family == TEST_FAMILY for trajectory in trajectories):
        raise ValueError("Refusing to fit scalers with held-out Melon data")
    state_scaler = AffineScaler.fit((trajectory.y for trajectory in trajectories), STATE_COLUMNS)
    input_columns = {
        "motor_speed": INPUT_COLUMNS,
        "motor_speed_squared": MOTOR_SPEED_SQUARED_COLUMNS,
        "rotor_wrench": ROTOR_WRENCH_COLUMNS,
        "dynamic_physical_port": DYNAMIC_PORT_COLUMNS,
        "combined_dynamic_port": COMBINED_DYNAMIC_PORT_COLUMNS,
    }[input_representation]
    input_scaler = AffineScaler.fit(
        (transform_inputs(trajectory.u, input_representation) for trajectory in trajectories),
        input_columns,
    )
    return state_scaler, input_scaler


def make_multitrajectory_dataset(
    trajectories: Sequence[NanoDroneTrajectory],
    seq_len: int,
    stride: int,
    state_scaler: AffineScaler | None = None,
    input_scaler: AffineScaler | None = None,
    input_representation: str = "motor_speed",
) -> WindowedDataset:
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive")
    x0, inputs, targets, names, runs, starts = [], [], [], [], [], []
    for trajectory in trajectories:
        state = trajectory.y if state_scaler is None else state_scaler.transform(trajectory.y)
        control = transform_inputs(trajectory.u, input_representation)
        if input_scaler is not None:
            control = input_scaler.transform(control)
        for start in range(0, len(state) - seq_len, stride):
            x0.append(state[start])
            inputs.append(control[start:start + seq_len])
            targets.append(state[start + 1:start + seq_len + 1])
            names.append(trajectory.name)
            runs.append(trajectory.run_id)
            starts.append(start)
    if not x0:
        raise ValueError("No windows were generated")
    return WindowedDataset(
        x0=np.asarray(x0, dtype=np.float32),
        u_seq=np.asarray(inputs, dtype=np.float32),
        target=np.asarray(targets, dtype=np.float32),
        trajectory_name=np.asarray(names),
        run_id=np.asarray(runs, dtype=np.int32),
        start_index=np.asarray(starts, dtype=np.int32),
    )


def write_dataset_manifest(
    path: str | Path,
    dataset: dict[str, list[NanoDroneTrajectory]],
    state_scaler: AffineScaler | None = None,
    input_scaler: AffineScaler | None = None,
    input_representation: str = "motor_speed",
    root: str | Path = DEFAULT_DATASET_ROOT,
) -> None:
    root = Path(root)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = EXTERNAL_COMMIT
    if commit != EXTERNAL_COMMIT:
        raise RuntimeError(f"Unexpected external dataset commit {commit}; expected {EXTERNAL_COMMIT}")
    records = []
    for split in ("official_train", "official_test"):
        for trajectory in dataset[split]:
            increments = np.diff(trajectory.time)
            records.append(
                {
                    "filename": trajectory.name + ".csv",
                    "source_split": trajectory.source_split,
                    "family": trajectory.family,
                    "run_id": trajectory.run_id,
                    "samples": len(trajectory.time),
                    "median_dt": float(np.median(increments)),
                }
            )
    payload = {
        "external_repository": "idsia-robotics/nanodrone-sysid-benchmark",
        "external_commit": commit,
        "files": records,
        "sample_counts": {
            "official_train": sum(len(value.time) for value in dataset["official_train"]),
            "official_test": sum(len(value.time) for value in dataset["official_test"]),
        },
        "development_train": [value.name + ".csv" for value in dataset["development_train"]],
        "development_validation": [value.name + ".csv" for value in dataset["development_validation"]],
        "official_test": [value.name + ".csv" for value in dataset["official_test"]],
        "dt": 0.01,
        "input_columns": list(INPUT_COLUMNS),
        "raw_input_columns": list(INPUT_COLUMNS),
        "model_input_representation": input_representation,
        "model_input_columns": list({
            "motor_speed": INPUT_COLUMNS,
            "motor_speed_squared": MOTOR_SPEED_SQUARED_COLUMNS,
            "rotor_wrench": ROTOR_WRENCH_COLUMNS,
            "dynamic_physical_port": DYNAMIC_PORT_COLUMNS,
            "combined_dynamic_port": COMBINED_DYNAMIC_PORT_COLUMNS,
        }[input_representation]),
        "state_columns": list(STATE_COLUMNS),
        "rotation_representation": "SO(3) logarithm from normalized xyzw quaternion",
        "normalization": "per-channel affine standardization fit on selected training flights only",
    }
    if state_scaler is not None:
        payload["state_scaler"] = state_scaler.as_dict()
    if input_scaler is not None:
        payload["input_scaler"] = input_scaler.as_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")