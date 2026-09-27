"""Leakage-safe nonlinear-benchmark dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import nonlinear_benchmarks

DATA_ROOT = Path(__file__).resolve().parents[1] / "datasets"


@dataclass(frozen=True)
class Realization:
    inputs: np.ndarray
    outputs: np.ndarray
    dt: float
    name: str
    init_window: int


@dataclass(frozen=True)
class NormalizationStats:
    input_mean: np.ndarray
    input_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray

    def to_dict(self) -> dict:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_scale": self.input_scale.tolist(),
            "output_mean": self.output_mean.tolist(),
            "output_scale": self.output_scale.tolist(),
        }


@dataclass(frozen=True)
class DatasetBundle:
    train: tuple[Realization, ...]
    validation: tuple[Realization, ...]
    test: tuple[Realization, ...]
    normalization: NormalizationStats
    benchmark_metadata: dict


@dataclass(frozen=True)
class WindowBatch:
    u_init: np.ndarray
    y_init: np.ndarray
    u_rollout: np.ndarray
    y_target: np.ndarray
    realization: tuple[str, ...]
    start: np.ndarray

    def __len__(self) -> int:
        return len(self.start)


@dataclass(frozen=True)
class DevelopmentBundle:
    train: tuple[Realization, ...]
    validation: tuple[Realization, ...]
    normalization: NormalizationStats
    metadata: dict
    train_window_starts: dict[str, np.ndarray] | None = None
    validation_window_starts: dict[str, np.ndarray] | None = None


def _array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def _split_raw(raw: list[tuple[str, np.ndarray, np.ndarray]], dt: float, init_window: int, *, validation_fraction: float, guard: int):
    train, validation = [], []
    for name, inputs, outputs in raw:
        validation_length = max(1, int(len(inputs) * validation_fraction))
        validation_start = len(inputs) - validation_length
        train_stop = validation_start - guard
        if train_stop < init_window + 1:
            raise ValueError(f"{name}: split leaves too little training data after a {guard}-sample guard")
        train.append(Realization(inputs[:train_stop], outputs[:train_stop], dt, f"{name}_train", init_window))
        validation.append(Realization(inputs[validation_start:], outputs[validation_start:], dt, f"{name}_validation", init_window))
    return tuple(train), tuple(validation)


def _normalization(train: tuple[Realization, ...]) -> NormalizationStats:
    inputs = np.concatenate([item.inputs for item in train])
    outputs = np.concatenate([item.outputs for item in train])
    input_scale = inputs.std(axis=0)
    output_scale = outputs.std(axis=0)
    if np.any(~np.isfinite(input_scale)) or np.any(input_scale < 1e-8):
        raise ValueError("training inputs contain a constant or nonfinite channel")
    if np.any(~np.isfinite(output_scale)) or np.any(output_scale < 1e-8):
        raise ValueError("training outputs contain a constant or nonfinite channel")
    return NormalizationStats(inputs.mean(axis=0), input_scale, outputs.mean(axis=0), output_scale)


def _apply_normalization(realizations: tuple[Realization, ...], stats: NormalizationStats) -> tuple[Realization, ...]:
    return tuple(
        Realization(
            (item.inputs - stats.input_mean) / stats.input_scale,
            (item.outputs - stats.output_mean) / stats.output_scale,
            item.dt,
            item.name,
            item.init_window,
        )
        for item in realizations
    )


def load_dataset(dataset: str, *, validation_fraction: float, guard: int) -> DatasetBundle:
    if dataset == "silverbox":
        train_record, test_records = nonlinear_benchmarks.Silverbox(dir_placement=DATA_ROOT)
        dt = float(train_record.sampling_time)
        init_window = int(test_records[0].state_initialization_window_length)
        train_raw = [("silverbox", _array(train_record.u), _array(train_record.y))]
        test_raw = [(name, _array(record.u), _array(record.y)) for name, record in zip(
            ("multisine", "arrow_full", "arrow_no_extrapolation"), test_records
        )]
    elif dataset == "ced":
        train_records, test_records = nonlinear_benchmarks.CED(dir_placement=DATA_ROOT)
        dt = float(train_records[0].sampling_time)
        init_window = int(test_records[0].state_initialization_window_length)
        train_raw = [
            (f"ced_train_{index}", _array(inputs), _array(outputs))
            for index, (inputs, outputs) in enumerate(train_records, start=1)
        ]
        test_raw = [
            (f"ced_{label}", _array(record.u), _array(record.y))
            for label, record in zip(("low_amplitude", "high_amplitude"), test_records)
        ]
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    train, validation = _split_raw(
        train_raw, dt, init_window, validation_fraction=validation_fraction, guard=guard
    )
    stats = _normalization(train)
    test = tuple(Realization(inputs, outputs, dt, name, init_window) for name, inputs, outputs in test_raw)
    return DatasetBundle(
        _apply_normalization(train, stats),
        _apply_normalization(validation, stats),
        _apply_normalization(test, stats),
        stats,
        {
            "dataset": dataset,
            "dt": dt,
            "init_window": init_window,
            "train_hashes": {name: array_hash(inputs, outputs) for name, inputs, outputs in train_raw},
            "test_hashes": {name: array_hash(inputs, outputs) for name, inputs, outputs in test_raw},
        },
    )


def load_campaign_development(dataset: str, *, init_window: int, max_horizon: int, fold: int | None = None) -> DevelopmentBundle:
    """Load development data only; official test arrays are never converted or returned."""
    guard = init_window + max_horizon
    if dataset == "silverbox":
        train_record, _ = nonlinear_benchmarks.Silverbox(dir_placement=DATA_ROOT)
        dt = float(train_record.sampling_time)
        raw = [("silverbox", _array(train_record.u), _array(train_record.y))]
        train, validation = _split_raw(raw, dt, init_window, validation_fraction=0.2, guard=guard)
        metadata = {"dataset": dataset, "dt": dt, "init_window": init_window, "fold": None}
    elif dataset == "ced":
        if fold not in range(4):
            raise ValueError("CED campaign requires fold in {0, 1, 2, 3}")
        train_records, _ = nonlinear_benchmarks.CED(dir_placement=DATA_ROOT)
        dt = float(train_records[0].sampling_time)
        raw_sources, normalization_inputs, normalization_outputs = [], [], []
        train_starts, validation_starts = {}, {}
        for index, (inputs, outputs) in enumerate(train_records, start=1):
            input_values, output_values = _array(inputs), _array(outputs)
            total = init_window + max_horizon
            eligible_starts = np.arange(len(input_values) - total + 1, dtype=np.int64)
            blocks = np.array_split(eligible_starts, 4)
            validation_block = blocks[fold]
            if not len(validation_block):
                raise ValueError(f"CED fold {fold} has no eligible validation windows")
            held_first, held_last = int(validation_block[0]), int(validation_block[-1])
            safe = eligible_starts[(eligible_starts <= held_first - guard) | (eligible_starts >= held_last + 1 + guard)]
            if not len(safe):
                raise ValueError(f"CED fold {fold} has no guarded development training windows")
            name = f"ced_train_{index}_fold{fold}_source"
            raw_sources.append((name, input_values, output_values))
            train_starts[name] = safe
            validation_starts[name] = validation_block
            sample_mask = np.ones(len(input_values), dtype=bool)
            sample_mask[max(0, held_first - guard):min(len(input_values), held_last + 1 + total + guard)] = False
            normalization_inputs.append(input_values[sample_mask])
            normalization_outputs.append(output_values[sample_mask])
        input_values = np.concatenate(normalization_inputs)
        output_values = np.concatenate(normalization_outputs)
        stats = NormalizationStats(input_values.mean(axis=0), input_values.std(axis=0), output_values.mean(axis=0), output_values.std(axis=0))
        if np.any(stats.input_scale < 1e-8) or np.any(stats.output_scale < 1e-8):
            raise ValueError("CED fold has a constant training normalization channel")
        sources = tuple(Realization(inputs, outputs, dt, name, init_window) for name, inputs, outputs in raw_sources)
        train, validation = _apply_normalization(sources, stats), _apply_normalization(sources, stats)
        metadata = {"dataset": dataset, "dt": dt, "init_window": init_window, "fold": fold, "guard": guard}
        return DevelopmentBundle(train, validation, stats, metadata, train_starts, validation_starts)
    else:
        raise ValueError(f"unsupported campaign dataset: {dataset}")
    stats = _normalization(train)
    return DevelopmentBundle(_apply_normalization(train, stats), _apply_normalization(validation, stats), stats, metadata)


def make_windows(realizations: tuple[Realization, ...], init_window: int, horizon: int, stride: int) -> WindowBatch:
    windows = []
    total = init_window + horizon
    for realization in realizations:
        if len(realization.inputs) < total:
            continue
        for start in range(0, len(realization.inputs) - total + 1, stride):
            split = start + init_window
            stop = split + horizon
            windows.append((
                realization.inputs[start:split], realization.outputs[start:split],
                realization.inputs[split:stop], realization.outputs[split:stop], realization.name, start,
            ))
    if not windows:
        raise ValueError("no windows available; reduce init_window/horizon or adjust the split")
    u_init, y_init, u_rollout, y_target, names, starts = zip(*windows)
    return WindowBatch(
        np.stack(u_init), np.stack(y_init), np.stack(u_rollout), np.stack(y_target),
        tuple(names), np.asarray(starts, dtype=np.int64),
    )


def make_campaign_windows(bundle: DevelopmentBundle, *, split: str, init_window: int, horizon: int, stride: int) -> WindowBatch:
    """Create only the predeclared safe development windows for a campaign split."""
    realizations = bundle.train if split == "train" else bundle.validation
    selectors = bundle.train_window_starts if split == "train" else bundle.validation_window_starts
    if selectors is None:
        return make_windows(realizations, init_window, horizon, stride)
    windows = []
    total = init_window + horizon
    for realization in realizations:
        maximum_start = len(realization.inputs) - total
        starts = selectors[realization.name]
        starts = starts[starts <= maximum_start]
        if bundle.metadata["dataset"] == "ced":
            held = bundle.validation_window_starts[realization.name]
            held_first, held_last = int(held[0]), int(held[-1])
            guard = int(bundle.metadata["guard"])
            if split == "train":
                starts = starts[(starts <= held_first - guard) | (starts >= held_last + 1 + guard)]
        starts = starts[::stride]
        for start in starts:
            split_index, stop = int(start) + init_window, int(start) + total
            windows.append((
                realization.inputs[int(start):split_index], realization.outputs[int(start):split_index],
                realization.inputs[split_index:stop], realization.outputs[split_index:stop], realization.name, int(start),
            ))
    if not windows:
        raise ValueError(f"no {split} campaign windows available at horizon {horizon}")
    u_init, y_init, u_rollout, y_target, names, starts = zip(*windows)
    return WindowBatch(np.stack(u_init), np.stack(y_init), np.stack(u_rollout), np.stack(y_target), tuple(names), np.asarray(starts, dtype=np.int64))


def array_hash(inputs: np.ndarray, outputs: np.ndarray) -> str:
    digest = sha256()
    for values in (inputs, outputs):
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()
