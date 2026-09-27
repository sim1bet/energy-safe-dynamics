# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Compatibility loader for pre-feature-conditioned EBM checkpoints.

The preserved certificate checkpoints have the 20-field August 2026 trunk
schema. The current trunk appends 19 feature-conditioned fields. Those fields
are inactive for these champion configurations, so migration copies every
trained legacy leaf and takes only the inactive appended leaves from a freshly
initialized current-schema template.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import EBM_param_fields as pf


class LegacyTrunkParams(NamedTuple):
    L_M: object
    e_A: object
    B: object
    C_state: object
    c_bias: object
    D_feed: object
    C_ro1: object
    b_ro1: object
    C_ro2: object
    y_sat_pos_raw: object
    y_sat_neg_raw: object
    G_damp: object
    h_damp: object
    w_gain: object
    b_gain: object
    G_gain: object
    h_gain: object
    u_sat_raw: object
    W_enc: object
    b_enc: object


class LegacyEBMParams(NamedTuple):
    ebm_weights: object
    ebm_biases: object
    trunk: LegacyTrunkParams


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "EBM_param_fields" and name == "TrunkParams":
            return LegacyTrunkParams
        if module == "EBM_param_fields" and name == "EBMParams":
            return LegacyEBMParams
        return super().find_class(module, name)


def load_legacy_checkpoint(path: str | Path, current_template):
    """Load a legacy checkpoint and migrate it into the current named tuples."""
    with Path(path).open("rb") as stream:
        legacy = _LegacyUnpickler(stream).load()
    if not isinstance(legacy, LegacyEBMParams) or len(legacy.trunk) != 20:
        raise TypeError("checkpoint is not the supported 20-field legacy EBM schema")
    values = current_template.trunk._asdict()
    values.update(legacy.trunk._asdict())
    migrated = pf.EBMParams(
        ebm_weights=legacy.ebm_weights,
        ebm_biases=legacy.ebm_biases,
        trunk=pf.TrunkParams(**values),
    )
    return migrated
