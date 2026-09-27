# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Import-path helper for the repository's existing flat EBM_model imports.

The original code imports modules such as ``EBM_class`` directly even though
those files live in ``EBM_model/``.  Stress_tests may be imported from the
repository root or from a notebook, so we add that directory once here rather
than repeating sys.path mutations throughout the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EBM_MODEL_DIR = ROOT / "EBM_model"

for path in (ROOT, EBM_MODEL_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)
