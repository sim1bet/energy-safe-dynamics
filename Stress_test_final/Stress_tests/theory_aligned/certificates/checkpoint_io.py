# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Read-only, JAX-free loader for `EBMParams` checkpoints (params.pkl).

Purpose
-------
The trained-parameter pickle written by ``post_training_eval.save_checkpoint``
already contains a pytree of plain ``numpy`` arrays (the JAX arrays are
converted with ``jax.tree_util.tree_map(np.asarray, params)`` *before*
pickling). The only reason unpickling normally requires JAX installed is that
the pickle references the original ``EBMParams``/``TrunkParams`` NamedTuple
*classes* defined in ``EBM_model/EBM_param_fields.py``, which imports
``jax`` at module scope.

This module defines structurally-identical, JAX-free replicas of those two
NamedTuples (field order copied verbatim from
``EBM_model/EBM_param_fields.py``) and uses a custom
``pickle.Unpickler.find_class`` override to redirect class lookups to the
replicas. No training code is imported, no JAX dependency is required, and
the checkpoint file itself is never written to.

This module is application-agnostic: it only assumes the checkpoint was
produced by the shared ``EBM_model`` package used by every experiment in this
repository (``deep_dissipative_nlink``, ``duffing_doublewell``, and any
future ``nonlinearbenchmark.org``-style task built on the same model code).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple, Any


class TrunkParams(NamedTuple):
    L_M: Any
    e_A: Any
    B: Any
    C_state: Any
    c_bias: Any
    D_feed: Any
    C_ro1: Any
    b_ro1: Any
    C_ro2: Any
    y_sat_pos_raw: Any
    y_sat_neg_raw: Any
    G_damp: Any
    h_damp: Any
    w_gain: Any
    b_gain: Any
    G_gain: Any
    h_gain: Any
    u_sat_raw: Any
    W_enc: Any
    b_enc: Any
    G_A: Any = None
    h_A: Any = None
    G_A_quadratic: Any = None
    G_A_cubic: Any = None
    W_field1: Any = None
    b_field1: Any = None
    W_field2: Any = None
    b_field2: Any = None
    W_rich_A: Any = None
    W_rich_M: Any = None
    W_rich_B: Any = None


class EBMParams(NamedTuple):
    ebm_weights: Any
    ebm_biases: Any
    trunk: Any


class _JaxFreeUnpickler(pickle.Unpickler):
    """Redirects EBM_param_fields.{EBMParams,TrunkParams} to local replicas.

    All other classes/modules (numpy arrays, plain containers) are resolved
    normally by the base ``pickle.Unpickler``.
    """

    _REDIRECTS = {
        ("EBM_param_fields", "EBMParams"): EBMParams,
        ("EBM_param_fields", "TrunkParams"): TrunkParams,
    }

    def find_class(self, module: str, name: str):
        key = (module, name)
        if key in self._REDIRECTS:
            return self._REDIRECTS[key]
        return super().find_class(module, name)


def load_checkpoint_numpy_only(path: str | Path) -> EBMParams:
    """Load a `params.pkl` checkpoint using only `pickle` + `numpy`, no JAX.

    Returns the JAX-free `EBMParams` replica defined in this module. Field
    values are the exact `numpy.ndarray`s stored by `save_checkpoint`
    (`post_training_eval.py`); this function performs no numerical
    modification of them.
    """
    path = Path(path)
    with open(path, "rb") as f:
        return _JaxFreeUnpickler(f).load()
