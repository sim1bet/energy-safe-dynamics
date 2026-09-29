# Author: Simone Betteti
# CED_main.py — free-EBM port-Hamiltonian fit of the Coupled Electric Drives benchmark
# https://www.nonlinearbenchmark.org/benchmarks/coupled-electric-drives
#
# Why this benchmark suits a port-Hamiltonian EBM:
#   Two electric motors drive a pulley/belt that moves a flexible disc. The
#   rotational inertia is an energy-storage state and belt/bearing friction is the
#   dissipation -> natural port-Hamiltonian structure. The characteristic
#   nonlinearity is that the speed sensor loses the sign of the input (|u|-like
#   response), which a nonlinear state model must reconstruct.
#
# Data facts (loaded at runtime from `nonlinear_benchmarks`):
#   SISO but TWO realisations: low input amplitude (series 1) and high input
#   amplitude (series 2), each with its own train and test set. Short sequences,
#   state_initialization_window_length = 10. Both train sets may be used; the two
#   test sets must be reported separately, so we evaluate each and log the mean.
#
# Implementation note: the two train realisations are concatenated for the shared
# training loop (a handful of windows straddle the join — negligible, both halves
# are valid CED dynamics). The two test series are evaluated independently after
# training, with a per-series rollout (no cross-series state carry-over).

import numpy as np
import nonlinear_benchmarks
from scipy.io import loadmat
from dataset_paths import dataset_root

import Silverbox_main as sb
from Silverbox_main import get_config, evaluate_model, run_training as _sb_train

EVAL_UNIT = ('ticks/s', 1.0)

DATASET_DEFAULTS = dict(
    init_win=10,     # == benchmark state_initialization_window_length
    seq_len=64,
    stride=1,        # short series; keep all windows
    batch_size=16,
    n_epochs=500,
    vf_scale=4.0,    # start modest; sweep with seq_len
    damping_scale=1.0,
)

DEFAULT_CONFIG = {**get_config(), **DATASET_DEFAULTS}


def _normalise_pair(u, y, u_mu, u_std, y_mu, y_std):
    u = (np.array(u).reshape(-1, 1).astype(np.float32) - u_mu) / u_std
    y = (np.array(y).reshape(-1, 1).astype(np.float32) - y_mu) / y_std
    return u, y


def _load_ced_raw():
    """Load both CED realisations, normalise everything with combined-train stats.

    Returns dt, (u_tr_combined, y_tr_combined), [(u_te1,y_te1), (u_te2,y_te2)],
    init_len, stats.
    """
    bundled = dataset_root() / "CED" / "DATAUNIF.MAT"
    if bundled.is_file():
        # nonlinear_benchmarks uses its source ZIP as a cache sentinel and
        # otherwise downloads despite the bundled canonical MAT asset.
        raw = loadmat(bundled)
        u11, u12, y11, y12 = (np.asarray(raw[key]).reshape(-1, 1).astype(np.float32)
                                for key in ("u11", "u12", "z11", "z12"))
        u_tr1, u_tr2, y_tr1, y_tr2 = u11[:400], u12[:400], y11[:400], y12[:400]
        raw_tests, dt, init_len = ((u11[400:], y11[400:]), (u12[400:], y12[400:])), 0.02, 10
    else:
        train_val, test = nonlinear_benchmarks.CED(dir_placement=dataset_root())
        (u_tr1, y_tr1), (u_tr2, y_tr2) = train_val
        dt, init_len = float(train_val[0].sampling_time), int(test[0].state_initialization_window_length)
        u_tr1, u_tr2 = np.asarray(u_tr1).reshape(-1, 1).astype(np.float32), np.asarray(u_tr2).reshape(-1, 1).astype(np.float32)
        y_tr1, y_tr2 = np.asarray(y_tr1).reshape(-1, 1).astype(np.float32), np.asarray(y_tr2).reshape(-1, 1).astype(np.float32)
        raw_tests = tuple((np.asarray(ds.u).reshape(-1, 1).astype(np.float32), np.asarray(ds.y).reshape(-1, 1).astype(np.float32)) for ds in test)

    u_all = np.concatenate([u_tr1, u_tr2], axis=0)
    y_all = np.concatenate([y_tr1, y_tr2], axis=0)
    u_mu, u_std = float(u_all.mean()), float(u_all.std())
    y_mu, y_std = float(y_all.mean()), float(y_all.std())
    stats = dict(u_mu=u_mu, u_std=u_std, y_mu=y_mu, y_std=y_std)

    u_tr = (u_all - u_mu) / u_std
    y_tr = (y_all - y_mu) / y_std

    tests = []
    for u_raw, y_raw in raw_tests:
        u_te, y_te = _normalise_pair(u_raw, y_raw, u_mu, u_std, y_mu, y_std)
        tests.append((u_te, y_te))
    return dt, (u_tr, y_tr), tests, init_len, stats


def train(seed=0, config=None):
    """Train on both CED realisations, then evaluate each test series separately
    and log the mean NRMSE as the primary summary metric."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    dt, (u_tr, y_tr), tests, init_len, stats = _load_ced_raw()

    def load_fn():
        # The shared loop only consumes the train arrays + dt; the test slot is a
        # placeholder (test series 1) and is never evaluated here (eval_after=False).
        u_te0, y_te0 = tests[0]
        return u_tr, y_tr, dt, u_te0, y_te0, init_len, stats

    params, layers, grad_E, _, controller_info = _sb_train(
        seed=seed, config=cfg, load_fn=load_fn, eval_unit=EVAL_UNIT, eval_after=False,
    )

    D = int(cfg['d'])
    M_PORTS = int(cfg['m_ports'])
    INIT_WIN = int(cfg['init_win'])

    print("\n── CED per-series test evaluation ──")
    nrmses = []
    for i, (u_te, y_te) in enumerate(tests, start=1):
        rmse, nrmse = evaluate_model(
            params, layers, grad_E, cfg, D, M_PORTS, dt, INIT_WIN,
            init_len, u_te, y_te, stats,
            unit_label=EVAL_UNIT[0], unit_scale=EVAL_UNIT[1], log_prefix=f'test{i}',
            set_summary=False,
            controller=controller_info,
        )
        if nrmse is not None:
            nrmses.append(nrmse)

    if nrmses:
        mean_nrmse = float(np.mean(nrmses))
        print(f"[CED] mean test NRMSE over {len(nrmses)} series: {mean_nrmse:.6f}")
        try:
            import wandb
            wandb.log({'test/nrmse': mean_nrmse})
            wandb.summary['best_test_nrmse'] = mean_nrmse
        except Exception as e:
            print(f"✗ Failed to log CED mean metric: {e}")
    return params, layers, grad_E, stats


if __name__ == '__main__':
    train(seed=42, config=DEFAULT_CONFIG)
