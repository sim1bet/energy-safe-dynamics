# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Contract implemented by dataset-specific stress-test adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from Stress_tests.config import StressTestConfig


@dataclass(frozen=True)
class StressTestCase:
    """Inputs required by the shared suite, all in model-internal coordinates."""

    params: Any
    layers: tuple
    d: int
    m: int
    dt: float
    epsilon: float
    gamma: float
    reference_states: np.ndarray
    reference_inputs: np.ndarray | None
    config: StressTestConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    comparison_epsilon: float | None = None