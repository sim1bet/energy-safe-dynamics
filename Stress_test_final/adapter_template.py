# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Copy this module and implement a frozen-model adapter for one dataset."""
from __future__ import annotations

from Stress_test_final.case import StressTestCase


def build_case(*, smoke: bool) -> StressTestCase:
    """Return frozen parameters, internal-coordinate data, and a fixed shell."""
    del smoke
    raise NotImplementedError("Implement this adapter for the target dataset/model.")