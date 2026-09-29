# Author: Simone Betteti
# Silverbox_main.py — Robust training entry point for free-EBM pH model
# v2 update: observation loss defaults moved closer to the authors' supervised objective.

import os
import pickle
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import wandb
import nonlinear_benchmarks
from dataset_paths import dataset_root

from EBM_class import (
    LayerSpec,
    lagrangian_softmax_stable, activation_softmax_stable,
    lagrangian_grouped_softmax, activation_grouped_softmax,
    lagrangian_sigmoid, activation_sigmoid,
    lagrangian_tanh, activation_tanh,
    lagrangian_polynomial_stable, activation_polynomial_stable,
    lagrangian_polynomial_stable_mixed, activation_polynomial_stable_mixed,
    lagrangian_polynomial, activation_polynomial,
    lagrangian_epanichnikov, activation_epanichnikov,
)
from EBM_param_fields import init_params, make_grad_energy, encode_x0, encode_x0_batch
from EBM_rollout import make_rollout_fn, make_init_state_fn, make_batch_rollout_fn
from EBM_training import make_loss_fn, make_train_step
import EBM_controller as ctrl_mod


def get_config():
    config = {}
    try:
        if wandb.run is not None:
            config = dict(wandb.config)
    except Exception:
        pass
    return dict(
        # Architecture
        d=config.get('d', 4),           # issue 6: wider state gives more expressive EBM
        m_ports=config.get('m_ports', 1),
        n_obs=config.get('n_obs', 1),
        hidden=config.get('hidden', 64),
        p_1=config.get('p_1', 2.0),
        p_2=config.get('p_2', 4.0),
        p_2_hi=config.get('p_2_hi', None),  # mixed curvature: exponent for the hi-p unit group (None = off)
        p_2_hi_frac=config.get('p_2_hi_frac', 0.5),  # fraction of second-layer units at p_2_hi
        p_3=config.get('p_3', None),  # power for energy layers 3+ (deep stack); None => reuse p_2
        layer_dims=config.get('layer_dims', [32, 16]),
        first_layer_type=config.get('first_layer_type', 'polynomial'),  # issue 4: full-rank Jacobian
        second_layer_type=config.get('second_layer_type', 'polynomial_stable'),  # 'polynomial_simple' = hard
                                                                                 # ReLU-power gate (exact zero
                                                                                 # for negative pre-activations)
                                                                                 # instead of the smooth softplus tail
        softmax_groups=config.get('softmax_groups', 1),  # >1 => grouped_softmax: independent winner-take-all pools
        epan_beta=config.get('epan_beta', 1.0),
        epan_eps=config.get('epan_eps', 1e-6),
        # Training
        batch_size=config.get('batch_size', 32),
        seq_len=config.get('seq_len', 20),  # issue 7: balanced BPTT horizon
        stride=config.get('stride', 1),  # dataset window stride; >1 cuts redundant overlapping segments
        init_win=config.get('init_win', 50),
        n_epochs=config.get('n_epochs', 500),
        lr=config.get('lr', 5e-5),
        lr_end=config.get('lr_end', 1e-6),
        lr_schedule=config.get('lr_schedule', 'constant'),
        full_phase_lr_scale=config.get('full_phase_lr_scale', 0.25),
        use_lr_decay=config.get('use_lr_decay', False),
        # Mixed-optimizer (SWATS-style) local refinement. When local_opt_switch is
        # on, the FULL-BPTT phase starts on AdamW (good global search) and, once a
        # smoothed train-NRMSE proxy drops below local_opt_nrmse_threshold, hands off
        # to a local-refinement optimizer (default SGD+Nesterov, no weight decay) that
        # settles more cleanly into the minimum instead of bouncing on the plateau.
        local_opt_switch=config.get('local_opt_switch', False),
        local_optimizer=config.get('local_optimizer', 'sgd'),
        local_opt_nrmse_threshold=config.get('local_opt_nrmse_threshold', 0.05),
        local_opt_lr_scale=config.get('local_opt_lr_scale', 1.0),
        local_opt_momentum=config.get('local_opt_momentum', 0.9),
        local_opt_ema_beta=config.get('local_opt_ema_beta', 0.7),
        grad_clip_norm=config.get('grad_clip_norm', 1.0),
        grad_norm_guard=config.get('grad_norm_guard', 100.0),
        rollout_stop_grad_epochs=config.get('rollout_stop_grad_epochs', 50),  # issue 7: short warmup only
        max_bad_updates=config.get('max_bad_updates', 50),
        rollout_integrator=config.get('rollout_integrator', 'rk4'),
        rollout_substeps=config.get('rollout_substeps', 1),  # internal integrator sub-steps per dt sample
        rollout_substeps_local=config.get('rollout_substeps_local', None),  # LOCAL-phase + eval override (substeps curriculum)
        # Loss/regularisation
        w_rollout=config.get('w_rollout', 1.0),
        w_passivity=config.get('w_passivity', 0.0),
        w_reg=config.get('w_reg', 1e-6),
        trunk_reg_mult=config.get('trunk_reg_mult', 1.0),
        w_reg_B=config.get('w_reg_B', 0.1),    # separate, stronger reg for B
        w_state=config.get('w_state', 0.005),   # state trajectory norm penalty
        observation_loss=config.get('observation_loss', 'huber'),
        weight_obs_by_magnitude=config.get('weight_obs_by_magnitude', False),
        huber_delta=config.get('huber_delta', 1.0),
        huber_warmup_fraction=config.get('huber_warmup_fraction', 0.1),
        w_free_map=config.get('w_free_map', 0.0),
        w_sensor_curvature=config.get('w_sensor_curvature', 0.0),
        w_parameter_prior=config.get('w_parameter_prior', 0.0),
        full_rollout_weight_start=config.get('full_rollout_weight_start', 0.0),
        full_rollout_weight_end=config.get('full_rollout_weight_end', 0.0),
        select_best_validation=config.get('select_best_validation', False),
        validation_min_delta=config.get('validation_min_delta', 0.002),
        validation_patience=config.get('validation_patience', 250),
        # Amplitude-weighted observation loss (default 0.0 = exact no-op,
        # back-compatible): linearly upweights per-step loss by |y_true|
        # (mean-preserving), unlike the saturating weight_obs_by_magnitude
        # ratio - directly targets under/over-shoot at peaks/troughs.
        amplitude_weight_power=config.get('amplitude_weight_power', 0.0),
        # 'full' (default, back-compatible): amplitude weighting applies in
        # every training phase. 'local': only once the mixed-optimizer has
        # switched to the LOCAL/SGD phase, so it never perturbs the SWATS
        # switch-detector's proxy (SG/FULL phases always see power=0.0).
        amplitude_weight_scope=config.get('amplitude_weight_scope', 'full'),
        # Directional (rise-vs-fall) observation loss (default 0.0 = exact
        # no-op, back-compatible): up/downweights per-step loss by the sign
        # of the TRUE trajectory's local slope. Targets the cascaded_tanks
        # v16 asymmetry diagnosis - errors concentrate at downward-upward
        # (trough) inversions, where the recovery RISE lags, while upward-
        # downward (peak) transitions track fine. Positive values upweight
        # rising steps; negative upweight falling steps (contrast/control).
        rise_weight_power=config.get('rise_weight_power', 0.0),
        # scope mirrors amplitude_weight_scope (see above) for the same
        # switch-detector-safety reason.
        rise_weight_scope=config.get('rise_weight_scope', 'full'),
        # Numerical safety
        chol_clip_exp=config.get('chol_clip_exp', 1.0),
        damping_scale=config.get('damping_scale', 1.0),  # <1 forces light damping (resonant systems)
        b_init_scale=config.get('b_init_scale', 1.0),    # >0 reconnects input to state (0 = old zero-B)
        vf_scale=config.get('vf_scale', 1.0),            # >1 speeds dynamics to match resonance
        use_feedthrough=config.get('use_feedthrough', True),  # add D@u direct input->output feedthrough
        use_nonlinear_readout=config.get('use_nonlinear_readout', False),  # bounded sigmoid head on readout (kink/saturation)
        readout_hidden=config.get('readout_hidden', 16),     # width of the bounded readout head
        use_state_damping=config.get('use_state_damping', False),  # gate M by state: level-dependent dissipation
        # Saturating (bounded) output readout: learnable per-side soft-clamp
        # y=cap*tanh(y_lin/cap) modelling a genuine physical ceiling/floor
        # (e.g. cascaded-tanks overflow/empty-tank saturation) instead of the
        # unbounded linear(+NLRO) map, which cannot itself stop overshoot past
        # a saturation point. Default off/back-compatible; cap starts LARGE
        # (near-linear warm start) and can only be tightened by gradient descent.
        use_saturating_readout=config.get('use_saturating_readout', False),
        sat_scale_init=config.get('sat_scale_init', 6.0),
        sat_scale_min=config.get('sat_scale_min', 0.5),
        # State-gated INPUT GAIN: forced term g(x)*(B@u) with learnable gate in
        # (0,2), zero-init => g=1 => identical to constant B at start. 'scalar'
        # mode (default) = one shared gate (minimal capacity); 'vector' = per-
        # state-row gate (capacity-dose contrast). Targets the CT diagnosis that
        # a single constant input gain over-responds to sharp large inputs while
        # under-integrating sustained moderate ones (peak/trough error compounding).
        use_input_gain=config.get('use_input_gain', False),
        input_gain_mode=config.get('input_gain_mode', 'scalar'),
        # Saturating INPUT transform (Hammerstein-style pump/actuator saturation):
        # u_eff = cap*tanh(u/cap), learnable per-channel cap, warm-started LARGE.
        use_saturating_input=config.get('use_saturating_input', False),
        sat_u_init=config.get('sat_u_init', 4.0),
        sat_u_min=config.get('sat_u_min', 0.5),
        # Feature-conditioned pH topology used by the cascaded-tanks staged
        # campaign. The default constant mode is the legacy code path.
        ph_structure_mode=config.get('ph_structure_mode', 'constant'),
        ph_feature_hidden=config.get('ph_feature_hidden', 64),
        use_monotone_input=config.get('use_monotone_input', False),
        use_monotone_observation=config.get('use_monotone_observation', False),
        time_scale_mode=config.get('time_scale_mode', 'legacy'),
        alpha_fixed=config.get('alpha_fixed', 0.03),
        alpha_min=config.get('alpha_min', 0.01),
        alpha_max=config.get('alpha_max', 0.10),
        # x0 encoder also sees the burn-in INPUT window u (concatenated after y),
        # not just the output y - fixes the ROOT cause of poor x0 estimates on
        # input-forced systems (default False = old y-only encoder, back-compatible).
        use_input_aware_encoder=config.get('use_input_aware_encoder', False),
        # Stochastic regularisation / data augmentation (default OFF = 0.0). Small
        # Gaussian jitter on the training inputs and the x0 encoder window closes the
        # train->test gap on tiny datasets without adding model capacity. Targets
        # (y_seg) are never noised so supervision stays clean.
        input_noise_std=config.get('input_noise_std', 0.0),  # std added to input segments u each epoch
        init_noise_std=config.get('init_noise_std', 0.0),    # std added to the x0-encoder observation window
        # Polyak/EMA weight averaging (default OFF). Averages params over the post-
        # warmup epochs to cut the noisy-plateau bounce of the SGD/local phase; the
        # averaged weights are evaluated at test time alongside the raw weights.
        use_ema=config.get('use_ema', False),
        ema_decay=config.get('ema_decay', 0.99),  # per-epoch EMA decay; ~1/(1-decay) epoch window
        ema_scope=config.get('ema_scope', 'full'),  # 'full': accumulate from FULL-phase start (legacy);
                                                     # 'local': accumulate only once the SWATS local/SGD
                                                     # phase engages (targets just the noisy plateau).
        weight_decay=config.get('weight_decay', 1e-4),   # AdamW weight decay; 0 disables pull-to-zero
        grad_e_clip=config.get('grad_e_clip', 5.0),
        xdot_clip=config.get('xdot_clip', 5.0),
        x0_norm_max=config.get('x0_norm_max', 3.0),
        state_norm_clip=config.get('state_norm_clip', 10.0),
        # Logging/debug
        print_every=config.get('print_every', 5),
        # Reconstruction plotting (dataset-agnostic; default OFF).
        save_plots=config.get('save_plots', False),
        plot_dir=config.get('plot_dir', ''),          # empty => $EBM_PLOT_DIR or 'plots'
        plot_max_points=config.get('plot_max_points', 5000),  # subsample very long test trajectories
        # Persist the final trained parameter pytree in the native checkpoint
        # format. The path may be absolute or relative to the working directory.
        save_final_params=config.get('save_final_params', True),
        final_params_path=config.get(
            'final_params_path', 'results/silverbox/checkpoints/Silverbox_final_params.pkl'),
        # Test-time x0 refinement
        init_lr=config.get('init_lr', 5e-3),
        init_steps_test=config.get('init_steps_test', 300),
        # Force the gradient-refined x0 (see evaluate_model) even when
        # init_len == init_win (default False = old direct-encoder-only
        # behaviour, back-compatible).
        use_iterative_init=config.get('use_iterative_init', False),
        # Multi-start x0 refinement: retries from init_multistart-1 additional
        # Gaussian-perturbed starts (std init_multistart_noise) plus the plain
        # encode_x0 guess, keeping whichever reaches the lowest burn-in
        # reconstruction loss - guards against the reconstruction-loss surface's
        # local minima (default 1 = old single-start behaviour, back-compatible).
        init_multistart=config.get('init_multistart', 1),
        init_multistart_noise=config.get('init_multistart_noise', 0.1),
        init_multistart_seed=config.get('init_multistart_seed', 0),
        # Early-window observation-loss upweighting: multiplies the per-step
        # training loss by early_window_weight for the first early_window_steps
        # steps of each rollout window (default 0 steps/weight 1.0 = no-op,
        # back-compatible) - directly targets the initial-transient tracking
        # bottleneck by biasing gradient signal toward getting it right.
        early_window_steps=config.get('early_window_steps', 0),
        early_window_weight=config.get('early_window_weight', 1.0),
        # Minimally-invasive port-Hamiltonian CBF safety-filter controller
        # (EBM_controller.py). All default OFF/back-compatible. The controller
        # only starts being TUNED once an EMA of the train-NRMSE proxy drops
        # below controller_loss_threshold (i.e. once the un-controlled EBM is
        # already producing satisfying samples) and is only APPLIED at test
        # time (evaluate_model), never inside the EBM's own training gradient
        # path -- the plant model itself is identified exactly as before.
        enable_controller=config.get('enable_controller', False),
        controller_loss_threshold=config.get('controller_loss_threshold', 0.30),
        controller_ema_beta=config.get('controller_ema_beta', 0.9),
        controller_gamma_init=config.get('controller_gamma_init', 1.0),
        controller_margin_init=config.get('controller_margin_init', 1.0),
        controller_min_margin_floor=config.get('controller_min_margin_floor', 1e-3),
        controller_ctrl_lr=config.get('controller_ctrl_lr', 1e-3),
        controller_w_margin_reg=config.get('controller_w_margin_reg', 1e-3),
        controller_min_energy_n_steps=config.get('controller_min_energy_n_steps', 50),
        controller_min_energy_lr=config.get('controller_min_energy_lr', 1e-2),
        controller_n_ctrl_steps_per_epoch=config.get('controller_n_ctrl_steps_per_epoch', 1),
    )


DEFAULT_CONFIG = get_config()


def _load_bundled_silverbox():
    """Return the package's canonical splits from bundled MAT data, if present."""
    path = dataset_root() / "Silverbox" / "SNLS80mV.mat"
    if not path.is_file():
        return None
    raw = loadmat(path)
    u = np.asarray(raw["V1"]).reshape(-1, 1).astype(np.float32)
    y = np.asarray(raw["V2"]).reshape(-1, 1).astype(np.float32)
    start, stop = 40650, 127400
    split = start + int(0.75 * (stop - start))
    return ((u[start:split], y[start:split]),
            ((u[split:stop], y[split:stop]), (u[100:40575], y[100:40575]),
             (u[100:32100], y[100:32100])), 1.0 / 610.35, 50)


def load_silverbox():
    bundled = _load_bundled_silverbox()
    if bundled is not None:
        (u_tr, y_tr), tests, dt, init_len = bundled
        u_te, y_te = tests[1]
    else:
        train_val, test = nonlinear_benchmarks.Silverbox(dir_placement=dataset_root())
        u_tr, y_tr = np.array(train_val.u).reshape(-1, 1).astype(np.float32), np.array(train_val.y).reshape(-1, 1).astype(np.float32)
        dt = float(train_val.sampling_time)
        test_arrow = test[1]
        u_te, y_te = np.array(test_arrow.u).reshape(-1, 1).astype(np.float32), np.array(test_arrow.y).reshape(-1, 1).astype(np.float32)
        init_len = test_arrow.state_initialization_window_length
    u_mu, u_std = float(u_tr.mean()), float(u_tr.std())
    y_mu, y_std = float(y_tr.mean()), float(y_tr.std())
    u_tr = (u_tr - u_mu) / u_std
    y_tr = (y_tr - y_mu) / y_std
    u_te = (u_te - u_mu) / u_std
    y_te = (y_te - y_mu) / y_std
    stats = dict(u_mu=u_mu, u_std=u_std, y_mu=y_mu, y_std=y_std)
    return u_tr, y_tr, dt, u_te, y_te, init_len, stats


# Benchmark test-set labels, in the fixed order returned by
# nonlinear_benchmarks.Silverbox(dir_placement=dataset_root()): (multisine, arrow_full, arrow_no_extrapolation).
SILVERBOX_TEST_NAMES = ('multisine', 'arrow_full', 'arrow_no_extrapolation')


def load_silverbox_all_tests():
    """Return (dt, [(name, u_te, y_te), ...], init_len, stats) covering all three
    Silverbox leaderboard test sets, normalised with TRAIN statistics.

    The benchmark reports RMSE (in mV) on every one of the three test series, so
    a leaderboard-consistent evaluation must score all of them, not just
    arrow_full."""
    bundled = _load_bundled_silverbox()
    if bundled is not None:
        (u_tr, y_tr), raw_tests, dt, init_len = bundled
    else:
        train_val, test = nonlinear_benchmarks.Silverbox(dir_placement=dataset_root())
        u_tr, y_tr = np.array(train_val.u).reshape(-1, 1).astype(np.float32), np.array(train_val.y).reshape(-1, 1).astype(np.float32)
        dt, init_len = float(train_val.sampling_time), int(test[0].state_initialization_window_length)
        raw_tests = tuple((np.array(ds.u).reshape(-1, 1).astype(np.float32), np.array(ds.y).reshape(-1, 1).astype(np.float32)) for ds in test)
    u_mu, u_std = float(u_tr.mean()), float(u_tr.std())
    y_mu, y_std = float(y_tr.mean()), float(y_tr.std())
    stats = dict(u_mu=u_mu, u_std=u_std, y_mu=y_mu, y_std=y_std)
    tests = []
    for name, (u_raw, y_raw) in zip(SILVERBOX_TEST_NAMES, raw_tests):
        u_te = (u_raw - u_mu) / u_std
        y_te = (y_raw - y_mu) / y_std
        tests.append((name, u_te, y_te))
    return dt, tests, init_len, stats


def make_dataset(u: np.ndarray, y: np.ndarray, init_win: int, seq_len: int,
                 stride: int = 1, starts: np.ndarray | None = None):
    seg_len = init_win + seq_len
    T = u.shape[0]
    if starts is None:
        starts = np.arange(0, T - seg_len + 1, stride)
    else:
        starts = np.asarray(starts, dtype=np.int64)
        if np.any(starts < 0) or np.any(starts + seg_len > T):
            raise ValueError("explicit dataset window starts exceed the available samples")
    y_win = np.stack([y[s:s + init_win] for s in starts])
    u_win = np.stack([u[s:s + init_win] for s in starts])  # burn-in INPUT window (used by
                                                            # use_input_aware_encoder; harmless/
                                                            # unused extra array otherwise)
    u_seg = np.stack([u[s + init_win:s + seg_len] for s in starts])
    y_seg = np.stack([y[s + init_win:s + seg_len] for s in starts])
    return y_win, u_win, u_seg, y_seg


def tree_all_finite(tree) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)


def print_bad_leaves(tree, label='params'):
    for i, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        if not bool(jnp.all(jnp.isfinite(leaf))):
            print(f"  {label} leaf {i}: shape={leaf.shape}, nan={bool(jnp.any(jnp.isnan(leaf)))}, inf={bool(jnp.any(jnp.isinf(leaf)))}")


def _build_layers(config):
    first = config['first_layer_type'].lower()
    if first in ('polynomial', 'softplus', 'poly'):
        # issue 4: strictly-monotone, full-rank Jacobian — recommended default
        first_layer = LayerSpec(lagrangian_polynomial_stable, activation_polynomial_stable, (float(config['p_1']),))
    elif first == 'softmax':
        first_layer = LayerSpec(lagrangian_softmax_stable, activation_softmax_stable, (float(config['p_1']),))
    elif first in ('grouped_softmax', 'group_softmax', 'gsoftmax'):
        # G independent softmax pools over equal-size blocks of the first hidden
        # layer; raises coupling rank while keeping the winner-take-all attractor
        # bias. n_groups=1 == plain softmax. Requires layer_dims[0] % n_groups == 0.
        g = int(config['softmax_groups'])
        h0 = int(list(config['layer_dims'])[0])
        if g < 1 or h0 % g != 0:
            raise ValueError(
                f"softmax_groups={g} must be >=1 and divide layer_dims[0]={h0} evenly.")
        first_layer = LayerSpec(lagrangian_grouped_softmax, activation_grouped_softmax,
                                (float(config['p_1']), g))
    elif first in ('sigmoid', 'logistic'):
        # bounded (0,1), strictly-monotone, full-rank diagonal Jacobian; p_1 = gain/beta
        first_layer = LayerSpec(lagrangian_sigmoid, activation_sigmoid, (float(config['p_1']),))
    elif first == 'tanh':
        # bounded (-1,1), odd, strictly-monotone, full-rank diagonal Jacobian; p_1 = gain/beta
        first_layer = LayerSpec(lagrangian_tanh, activation_tanh, (float(config['p_1']),))
    elif first in ('epanechnikov', 'epanichnikov', 'epa'):
        first_layer = LayerSpec(lagrangian_epanichnikov, activation_epanichnikov, (float(config['epan_beta']), float(config['epan_eps'])))
    else:
        raise ValueError(f"Unknown first_layer_type={config['first_layer_type']}; expected 'polynomial', 'softmax', 'sigmoid', 'tanh', or 'epanechnikov'.")

    second = str(config.get('second_layer_type', 'polynomial_stable')).lower()
    if second in ('polynomial_stable', 'polynomial', 'poly_stable'):
        second_lagrangian, second_activation = lagrangian_polynomial_stable, activation_polynomial_stable
    elif second in ('polynomial_simple', 'poly_simple', 'relu_power', 'simple'):
        # Hard ReLU^p gate: L(x)=relu(x)^p/p, grad=relu(x)^(p-1). Exact zero for
        # negative pre-activations (implicit sparsity) instead of the smooth,
        # never-quite-zero softplus tail of the '_stable' variant. Still convex/
        # monotone (legitimate passive Lagrangian), but with a genuine kink at 0.
        second_lagrangian, second_activation = lagrangian_polynomial, activation_polynomial
    else:
        raise ValueError(f"Unknown second_layer_type={config['second_layer_type']}; expected 'polynomial_stable' or 'polynomial_simple'.")
    second_layer = LayerSpec(second_lagrangian, second_activation, (float(config['p_2']),))

    # MIXED-CURVATURE second layer (back-compatible: p_2_hi=None => scalar path
    # untouched, bit-identical). A fraction p_2_hi_frac of the layer's units get
    # exponent p_2_hi, the rest keep p_2 — the v35/v36 dose-response showed a
    # single global exponent must compromise between the sharp overflow-clip
    # regime (wants high p) and the soft mid-band (wants p≈4).
    p_2_hi = config.get('p_2_hi', None)
    if p_2_hi is not None:
        if second not in ('polynomial_stable', 'polynomial', 'poly_stable'):
            raise ValueError("p_2_hi requires second_layer_type='polynomial_stable'.")
        h1 = int(list(config['layer_dims'])[1])
        frac = float(config.get('p_2_hi_frac', 0.5))
        n_hi = int(round(frac * h1))
        if not (1 <= n_hi < h1):
            raise ValueError(f"p_2_hi_frac={frac} gives {n_hi} hi-units; need 1 <= n_hi < {h1}.")
        p_vec = jnp.concatenate([jnp.full((h1 - n_hi,), float(config['p_2'])),
                                 jnp.full((n_hi,), float(p_2_hi))])
        second_layer = LayerSpec(lagrangian_polynomial_stable_mixed,
                                 activation_polynomial_stable_mixed, (p_vec,))

    # DEEP ENERGY STACK (back-compatible): init_ebm_params/energy_EBM are already
    # generic over len(layer_dims); only this spec tuple was hardcoded to 2. Any
    # layer_dims entries beyond the second get additional layers of the SAME
    # second_layer_type, with power p_3 (default = p_2) for layers 3+. This adds
    # capacity purely inside the Hamiltonian H(x) — the port structure
    # xdot = (A-M)@gE + B@u (constant B, affine-linear input port) is untouched.
    n_energy_layers = len(list(config['layer_dims']))
    if n_energy_layers < 2:
        raise ValueError(f"layer_dims must have >=2 entries, got {config['layer_dims']}")
    p_3 = float(config.get('p_3') or config['p_2'])
    deeper = tuple(LayerSpec(second_lagrangian, second_activation, (p_3,))
                   for _ in range(n_energy_layers - 2))
    return (first_layer, second_layer) + deeper


def _make_optimizer(config, n_batches, phase='sg'):
    lr = float(config['lr'])
    if phase in ('full', 'local'):
        lr *= float(config['full_phase_lr_scale'])
    grad_clip = float(config['grad_clip_norm'])
    if phase == 'local':
        # Local-refinement phase: drop AdamW's adaptive per-coordinate scaling and
        # weight decay in favour of a clean descent method that settles into the
        # local minimum (SWATS-style Adam->SGD hand-off). No pull-to-zero here.
        lr *= float(config['local_opt_lr_scale'])
        kind = str(config['local_optimizer']).lower()
        if kind == 'sgd':
            inner = optax.sgd(lr, momentum=float(config['local_opt_momentum']), nesterov=True)
        elif kind == 'adam':
            inner = optax.adam(lr)
        elif kind == 'rmsprop':
            inner = optax.rmsprop(lr, momentum=float(config['local_opt_momentum']))
        else:
            raise ValueError(
                f"Unknown local_optimizer={config['local_optimizer']}; expected 'sgd', 'adam', or 'rmsprop'.")
        return optax.chain(optax.clip_by_global_norm(grad_clip), inner)
    wd = float(config['weight_decay'])
    if str(config.get('lr_schedule', 'constant')).lower() == 'cosine':
        lr_end = float(config.get('lr_end', lr * 0.01))
        if not 0.0 <= lr_end <= lr:
            raise ValueError(f"cosine lr_end must lie in [0, lr], got lr_end={lr_end}, lr={lr}")
        schedule = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=max(int(config['n_epochs']) * n_batches, 1),
            alpha=lr_end / max(lr, 1e-12),
        )
        adam = optax.adamw(schedule, weight_decay=wd)
    elif bool(config['use_lr_decay']):
        schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=max(10 * n_batches, 1),
            decay_rate=0.98,
            end_value=max(lr * 0.1, 1e-6),
        )
        adam = optax.adamw(schedule, weight_decay=wd)
    else:
        adam = optax.adamw(lr, weight_decay=wd)
    return optax.chain(optax.clip_by_global_norm(grad_clip), adam)


def evaluate_model(params, layers, grad_E, config, D, M_PORTS, dt, INIT_WIN,
                   init_len, u_te, y_te, stats,
                   unit_label='mV', unit_scale=1e3, log_prefix='test',
                   set_summary=True, controller=None, log_metrics=True):
    """Roll out the trained model on a single test trajectory and report (rmse, nrmse).

    Dataset-agnostic: the calling entry point supplies the test arrays, the
    state-initialisation window length (``init_len``) and the physical output
    unit used purely for the printed RMSE (``unit_label``/``unit_scale``);
    NRMSE is scale invariant. ``log_prefix`` lets multi-trajectory datasets
    (e.g. CED) log several test series without key collisions.

    ``controller``: optional dict {grad_energy_fn, energy_value_fn, epsilon,
    gamma, d, m} produced by run_training when `enable_controller` activated
    during training (see EBM_controller.py). When supplied, the test rollout
    uses the tuned least-squares CBF safety filter to compensate the input
    at every step instead of the plain (uncontrolled) rollout; None (default)
    reproduces the exact old behaviour.
    """
    if not tree_all_finite(params):
        print("✗ Parameters are non-finite at test time; aborting evaluation.")
        return None, None

    u_te_j = jnp.array(u_te)
    y_te_j = jnp.array(y_te)
    import EBM_param_fields as pf
    # x0 estimation at test time: two paths.
    #  - encode_x0(): a ONE-SHOT LINEAR map from the burn-in window's OUTPUT
    #    y ONLY (ignores u entirely) - fast, and what training always uses
    #    internally (encode_x0 on yw_ep), but on a short init_win (e.g. CED's
    #    benchmark-mandated 10 samples) it is often an imprecise x0 guess for
    #    an input-forced system, producing a visible transient error spike at
    #    the very start of the free-running test rollout that decays away
    #    after a few steps. use_input_aware_encoder (config-gated) fixes this
    #    at the ROOT by also giving W_enc the burn-in INPUT window (see
    #    EBM_param_fields.init_trunk); when it's on, y_te_j/u_te_j are
    #    concatenated the SAME way training's enc_win_all was built.
    #  - batch_init() (make_init_state_fn): starts from that same encode_x0
    #    guess, then runs init_steps_test Adam steps REFINING x0 to minimise
    #    the actual rollout reconstruction error over the burn-in window
    #    using BOTH u and y with the trained dynamics - a strictly better
    #    (if slower) x0 estimate. Previously only used when init_len !=
    #    INIT_WIN (extra pre-burn-in data available); use_iterative_init
    #    forces it even when init_len == INIT_WIN (config-gated, default
    #    False = old behaviour, back-compatible). init_multistart>1 retries
    #    from several perturbed starting points (the reconstruction-loss
    #    surface w.r.t. x0, like the EBM's own energy, is non-convex and a
    #    single gradient-descent run can get stuck in a local minimum) and
    #    keeps whichever restart reaches the lowest burn-in reconstruction
    #    loss.
    use_iterative_init = bool(config.get('use_iterative_init', False))
    use_input_aware_encoder = bool(config.get('use_input_aware_encoder', False))
    # eval integrates at the LOCAL-phase substeps when the curriculum is on
    eval_substeps = int(config.get('rollout_substeps_local') or config.get('rollout_substeps', 1))
    def _enc_window(y_win, u_win):
        return jnp.concatenate([y_win, u_win], axis=-1) if use_input_aware_encoder else y_win
    if init_len == INIT_WIN and not use_iterative_init:
        x0_test = encode_x0(params, _enc_window(y_te_j[:init_len], u_te_j[:init_len]))
    else:
        n_restarts = int(config.get('init_multistart', 1))
        _, batch_init = make_init_state_fn(
            grad_E, D, M_PORTS, dt,
            n_steps=int(config['init_steps_test']),
            lr=float(config['init_lr']),
            integrator=str(config['rollout_integrator']),
            n_restarts=n_restarts,
            restart_noise_std=float(config.get('init_multistart_noise', 0.1)),
            substeps=eval_substeps,
        )
        # batch_init is vmapped over a batch axis and returns a
        # (x_final_batch, dummy_batch) tuple; we feed it a single-example
        # batch (via the [None]s below), so [0] unwraps the tuple giving
        # x_final_batch of shape (1, D) and a further [0] takes that single
        # batch row down to the (D,) vector `rollout()` expects. Dropping the
        # second [0] (as this pre-existing branch did before it was ever
        # actually exercised) leaves a stray leading batch axis, which later
        # crashes energy_EBM's `jnp.dot(x, x)` with a contracting-dimension
        # shape mismatch.
        key = jax.random.split(jax.random.PRNGKey(int(config.get('init_multistart_seed', 0))), 1)
        x0_test = batch_init(params, u_te_j[init_len - INIT_WIN:init_len][None], y_te_j[init_len - INIT_WIN:init_len][None], key)[0][0]

    controller_active_frac = None
    if controller is not None:
        controlled_rollout = ctrl_mod.make_controlled_rollout_fn(
            controller['grad_energy_fn'], controller['energy_value_fn'],
            controller['d'], controller['m'], dt,
            integrator=str(config['rollout_integrator']))
        x_traj, y_port, y_pred, u_applied, H_traj, active_traj = controlled_rollout(
            params, x0_test, u_te_j[init_len:],
            jnp.asarray(controller['epsilon']), jnp.asarray(controller['gamma']),
            stop_grad=True, apply_control=True)
        controller_active_frac = float(jnp.mean(active_traj.astype(jnp.float32)))
        print(f"[{log_prefix}] controller ACTIVE fraction of steps: {controller_active_frac:.4f} "
              f"(epsilon={controller['epsilon']:.4f}, gamma={controller['gamma']:.4f})")
    else:
        rollout = make_rollout_fn(grad_E, D, M_PORTS, dt, integrator=str(config['rollout_integrator']),
                                  substeps=eval_substeps)
        x_traj, y_port, y_pred = rollout(params, x0_test, u_te_j[init_len:], stop_grad=True)
    print(f"[{log_prefix}] x_traj mean/std:", float(jnp.mean(x_traj)), float(jnp.std(x_traj)))
    print(f"[{log_prefix}] y_pred mean/std:", float(jnp.mean(y_pred)), float(jnp.std(y_pred)))
    print(f"[{log_prefix}] y_true mean/std:", float(jnp.mean(y_te_j[init_len:])), float(jnp.std(y_te_j[init_len:])))
    print(f"[{log_prefix}] ||C_state||={float(jnp.linalg.norm(params.trunk.C_state)):.4e}  "
          f"||B||={float(jnp.linalg.norm(params.trunk.B)):.4e}  "
          f"||W_enc||={float(jnp.linalg.norm(params.trunk.W_enc)):.4e}")
    # Feedthrough-sensitivity diagnostic: share of the output coming straight
    # from u through D_feed (the physical system is strictly proper - true
    # feedthrough share should be ~0).
    try:
        d_contrib = np.array(u_te_j[init_len:]) @ np.array(params.trunk.D_feed).T
        print(f"[{log_prefix}] ||D_feed||={float(jnp.linalg.norm(params.trunk.D_feed)):.4e}  "
              f"std(D_feed@u)/std(y_pred)={float(np.std(d_contrib)) / max(float(jnp.std(y_pred)), 1e-12):.4f}")
    except Exception as e:
        print(f"  ⚠ feedthrough diagnostic failed: {e}")
    y_std = stats['y_std']
    p_u = np.array(y_pred[:, 0]) * y_std * unit_scale
    t_u = np.array(y_te_j[init_len:, 0]) * y_std * unit_scale
    rmse = float(np.sqrt(np.mean((p_u - t_u) ** 2)))
    nrmse = rmse / max(float(np.std(t_u)), 1e-12)
    print(f"[{log_prefix}] RMSE  : {rmse:.4f} {unit_label}")
    print(f"[{log_prefix}] NRMSE : {nrmse:.6f}  (competitive target < 0.05)")
    # Input-transient error split: does |error| concentrate where u moves fast?
    try:
        err = p_u - t_u
        du = np.abs(np.gradient(np.array(u_te_j[init_len:, 0])))
        du_thr = float(np.quantile(du, 0.9))
        fast = du >= du_thr
        print(f"[{log_prefix}] corr(|err|,|du/dt|)={float(np.corrcoef(np.abs(err), du)[0, 1]):.3f}  "
              f"RMSE@fast-u={float(np.sqrt(np.mean(err[fast] ** 2))):.4f}  "
              f"RMSE@slow-u={float(np.sqrt(np.mean(err[~fast] ** 2))):.4f} {unit_label} "
              f"(fast = top-10% |du|)")
    except Exception as e:
        print(f"  ⚠ input-transient diagnostic failed: {e}")
    if log_metrics:
        try:
            log = {f'{log_prefix}/rmse_{unit_label}': rmse, f'{log_prefix}/nrmse': nrmse}
            if controller_active_frac is not None:
                log[f'{log_prefix}/controller_active_frac'] = controller_active_frac
            wandb.log(log)
            if set_summary:
                wandb.summary['best_test_nrmse'] = nrmse
            print('✓ Test metrics logged to wandb successfully')
        except Exception as e:
            print(f'✗ Failed to log final metrics to wandb: {e}')
    # Optional dataset-agnostic reconstruction plot (true vs predicted test signal).
    if bool(config.get('save_plots', False)):
        try:
            from plot_utils import plot_true_vs_pred
            plot_dir = str(config.get('plot_dir', '') or os.environ.get('EBM_PLOT_DIR', 'plots'))
            run_tag = None
            try:
                run_tag = wandb.run.id if wandb.run is not None else None
            except Exception:
                run_tag = None
            fname = f"{log_prefix}_recon" + (f"_{run_tag}" if run_tag else "") + ".png"
            save_path = os.path.join(plot_dir, fname)
            # Input overlay panel: aligned sample-for-sample with t_u/p_u (both
            # slice from init_len onward). Plotted in its NORMALISED form (same
            # scale training/rollout use) - purely for visually correlating
            # reconstruction-error onset with rises/falls of the driving input,
            # not a physical-unit claim. u_te_j is defined earlier in this
            # function (the jnp test-input array); reuse it directly.
            u_plot = np.array(u_te_j[init_len:, 0])
            plot_true_vs_pred(
                t_u, p_u, save_path, dt=float(dt), unit_label=unit_label,
                rmse=rmse, nrmse=nrmse, max_points=int(config.get('plot_max_points', 5000)),
                title=f"{log_prefix}: true vs reconstructed",
                u_input=u_plot,
            )
            print(f"[{log_prefix}] reconstruction plot saved -> {save_path}")
            # Raw trajectory dump: enables offline error analysis and
            # test-time ensembling across runs without re-running eval.
            npz_path = save_path.replace('.png', '_traj.npz')
            np.savez(npz_path, y_true=t_u, y_pred=p_u, u=u_plot, dt=float(dt))
            print(f"[{log_prefix}] trajectory arrays saved -> {npz_path}")
            try:
                if wandb.run is not None:
                    wandb.log({f'{log_prefix}/reconstruction': wandb.Image(save_path)})
            except Exception as e:
                print(f"  ⚠ wandb image log failed: {e}")
        except Exception as e:
            print(f"  ⚠ reconstruction plot failed: {e}")
    return rmse, nrmse


def run_training(seed=0, config=None, load_fn=None, eval_unit=('mV', 1e3),
                 eval_after=True, window_starts=None, window_weights=None,
                 window_horizons=None, full_rollout_batch=None):
    if load_fn is None:
        load_fn = load_silverbox
    unit_label, unit_scale = eval_unit
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(config)
        config = cfg

    import EBM_param_fields as pf
    pf.CHOL_CLIP_EXP = float(config['chol_clip_exp'])
    pf.DAMPING_SCALE = float(config['damping_scale'])
    pf.B_INIT_SCALE = float(config['b_init_scale'])
    pf.VF_SCALE = float(config['vf_scale'])
    pf.USE_FEEDTHROUGH = bool(config['use_feedthrough'])
    pf.USE_NONLINEAR_READOUT = bool(config['use_nonlinear_readout'])
    pf.READOUT_HIDDEN = int(config['readout_hidden'])
    pf.USE_STATE_DAMPING = bool(config['use_state_damping'])
    pf.USE_SATURATING_READOUT = bool(config['use_saturating_readout'])
    pf.SAT_SCALE_INIT = float(config['sat_scale_init'])
    pf.SAT_SCALE_MIN = float(config['sat_scale_min'])
    pf.USE_INPUT_GAIN = bool(config['use_input_gain'])
    pf.INPUT_GAIN_MODE = str(config['input_gain_mode'])
    pf.USE_SATURATING_INPUT = bool(config['use_saturating_input'])
    pf.SAT_U_INIT = float(config['sat_u_init'])
    pf.SAT_U_MIN = float(config['sat_u_min'])
    pf.PH_STRUCTURE_MODE = str(config.get('ph_structure_mode', 'constant')).lower()
    if pf.PH_STRUCTURE_MODE not in {'constant', 'state', 'tank_features'}:
        raise ValueError(
            f"Unknown ph_structure_mode={pf.PH_STRUCTURE_MODE!r}; expected constant, state, or tank_features.")
    pf.PH_FEATURE_HIDDEN = int(config.get('ph_feature_hidden', 64))
    pf.USE_MONOTONE_INPUT = bool(config.get('use_monotone_input', False))
    pf.USE_MONOTONE_OBSERVATION = bool(config.get('use_monotone_observation', False))
    pf.TIME_SCALE_MODE = str(config.get('time_scale_mode', 'legacy')).lower()
    if pf.TIME_SCALE_MODE not in {'legacy', 'fixed', 'trainable'}:
        raise ValueError(
            f"Unknown time_scale_mode={pf.TIME_SCALE_MODE!r}; expected legacy, fixed, or trainable.")
    pf.ALPHA_FIXED = float(config.get('alpha_fixed', 0.03))
    pf.ALPHA_MIN = float(config.get('alpha_min', 0.01))
    pf.ALPHA_MAX = float(config.get('alpha_max', 0.10))
    if not pf.ALPHA_MIN < pf.ALPHA_MAX:
        raise ValueError("alpha_min must be smaller than alpha_max")
    pf.USE_INPUT_AWARE_ENCODER = bool(config['use_input_aware_encoder'])
    pf.GRAD_E_CLIP = float(config['grad_e_clip'])
    pf.XDOT_CLIP = float(config['xdot_clip'])
    pf.X0_NORM_MAX = float(config['x0_norm_max'])
    pf.STATE_NORM_CLIP = float(config['state_norm_clip'])

    D = int(config['d'])
    M_PORTS = int(config['m_ports'])
    N_OBS = int(config['n_obs'])
    # x0-encoder window width: normally n_obs (output burn-in only); when
    # use_input_aware_encoder is on, the burn-in INPUT u is concatenated too
    # (see EBM_param_fields.init_trunk / EBM_rollout._encoder_window).
    N_OBS_EFF = N_OBS + (M_PORTS if pf.USE_INPUT_AWARE_ENCODER else 0)
    HIDDEN = int(config['hidden'])
    LAYER_DIMS = list(config['layer_dims'])
    BATCH_SIZE = int(config['batch_size'])
    SEQ_LEN = int(config['seq_len'])
    INIT_WIN = int(config['init_win'])
    N_EPOCHS = int(config['n_epochs'])
    PRINT_EVERY = int(config['print_every'])

    STRIDE = int(config['stride'])
    u_tr, y_tr, dt, u_te, y_te, init_len, stats = load_fn()
    pf.SAMPLE_TIME = float(dt)
    print(f"dt={dt:.4e}s  train={u_tr.shape[0]}  test={u_te.shape[0]}  init_win={init_len}")
    y_win_all, u_win_all, u_seg_all, y_seg_all = make_dataset(
        u_tr, y_tr, INIT_WIN, SEQ_LEN, stride=STRIDE, starts=window_starts)
    # Encoder-window array actually fed to encode_x0_batch throughout training:
    # opaque y-only unless use_input_aware_encoder concatenates the burn-in u too.
    enc_win_all = np.concatenate([y_win_all, u_win_all], axis=-1) if pf.USE_INPUT_AWARE_ENCODER else y_win_all
    N = enc_win_all.shape[0]
    N_use = (N // BATCH_SIZE) * BATCH_SIZE
    if N_use == 0:
        # batch_size > number of available windows (common on short datasets
        # like cascaded_tanks when a hyperparameter sweep tuned for a much
        # larger dataset draws a big batch_size/stride combo): fail loudly and
        # early with an actionable message instead of an opaque IndexError
        # from indexing an empty batch array deep inside the training loop.
        msg = (f"No batches available: {N} windows from {u_tr.shape[0]} train samples "
               f"(init_win={INIT_WIN}, seq_len={SEQ_LEN}, stride={STRIDE}) is smaller than "
               f"batch_size={BATCH_SIZE}. Reduce batch_size/seq_len/stride for this dataset.")
        try:
            wandb.log({'train/nonfinite_params': 1, 'error': msg})
            wandb.summary['best_test_nrmse'] = float('nan')
        except Exception:
            pass
        raise ValueError(msg)
    enc_win_all = enc_win_all[:N_use]
    u_seg_all = u_seg_all[:N_use]
    y_seg_all = y_seg_all[:N_use]
    if window_weights is None:
        window_weights = np.ones(N, dtype=np.float32)
    if window_horizons is None:
        window_horizons = np.full(N, SEQ_LEN, dtype=np.int32)
    window_weights = np.asarray(window_weights, dtype=np.float32)[:N_use]
    window_horizons = np.asarray(window_horizons, dtype=np.int32)[:N_use]
    if len(window_weights) != N_use or len(window_horizons) != N_use:
        raise ValueError("window weights/horizons must align with explicit window starts")
    if np.any(window_horizons < 1) or np.any(window_horizons > SEQ_LEN):
        raise ValueError("window horizons must lie in [1, seq_len]")
    time_mask_all = np.arange(SEQ_LEN)[None, :] < window_horizons[:, None]
    n_batches = N_use // BATCH_SIZE
    print(f"Dataset: {N_use} segments -> {n_batches} batches/epoch")

    layers = _build_layers(config)
    grad_E = make_grad_energy(layers)
    params = init_params(jax.random.PRNGKey(seed), d=D, m=M_PORTS, n_obs=N_OBS,
                         init_win=INIT_WIN, layer_dims=LAYER_DIMS, hidden=HIDDEN, dt=dt)  # issue 9: pass dt for physics-informed encoder

    optimizer_sg = _make_optimizer(config, n_batches, phase='sg')
    optimizer_full = _make_optimizer(config, n_batches, phase='full')
    optimizer_local = _make_optimizer(config, n_batches, phase='local')
    opt_state = optimizer_sg.init(params)

    common_loss_kwargs = dict(
        grad_energy_fn=grad_E,
        layers=layers,
        d=D,
        m=M_PORTS,
        dt=dt,
        w_rollout=float(config['w_rollout']),
        w_passivity=float(config['w_passivity']),
        w_reg=float(config['w_reg']),
        trunk_mult=float(config['trunk_reg_mult']),
        w_reg_B=float(config['w_reg_B']),
        w_state=float(config['w_state']),
        w_free_map=float(config.get('w_free_map', 0.0)),
        w_sensor_curvature=float(config.get('w_sensor_curvature', 0.0)),
        w_parameter_prior=float(config.get('w_parameter_prior', 0.0)),
        rollout_integrator=str(config['rollout_integrator']),
        rollout_substeps=int(config.get('rollout_substeps', 1)),
        observation_loss=str(config['observation_loss']),
        weight_obs_by_magnitude=bool(config['weight_obs_by_magnitude']),
        huber_delta=float(config['huber_delta']),
        early_window_steps=int(config.get('early_window_steps', 0)),
        early_window_weight=float(config.get('early_window_weight', 1.0)),
        amplitude_weight_power=float(config.get('amplitude_weight_power', 0.0)),
        rise_weight_power=float(config.get('rise_weight_power', 0.0)),
    )
    # amplitude_weight_scope='full' (default, back-compatible): the amplitude
    # weighting above applies identically in every phase (SG/FULL/LOCAL) - this
    # is what v14 tested and found net-harmful, plausibly because it perturbs
    # the SWATS switch-detector's train-NRMSE proxy (calibrated on the plain
    # loss) from epoch 1, shifting switch timing and fighting the well-tuned
    # SG->FULL->LOCAL recipe throughout. 'local' instead applies amplitude
    # weighting ONLY once the mixed-optimizer has switched to the LOCAL/SGD
    # phase - i.e. as a final fit-quality polish on an already-converged model,
    # never touching the switch-detector's own calibration (SG/FULL always use
    # amplitude_weight_power=0.0 regardless of this flag). rise_weight_scope
    # is the exact same mechanism, independently, for rise_weight_power.
    amp_scope = str(config.get('amplitude_weight_scope', 'full')).lower()
    rise_scope = str(config.get('rise_weight_scope', 'full')).lower()
    common_loss_kwargs_sgfull = dict(common_loss_kwargs)
    if amp_scope == 'local':
        common_loss_kwargs_sgfull['amplitude_weight_power'] = 0.0
    if rise_scope == 'local':
        common_loss_kwargs_sgfull['rise_weight_power'] = 0.0
    # substeps curriculum: SG/FULL train at base substeps (switch-detector
    # calibration untouched); LOCAL phase and final eval integrate at
    # rollout_substeps_local (None => same as base).
    common_loss_kwargs_local = dict(common_loss_kwargs)
    local_substeps = config.get('rollout_substeps_local', None)
    if local_substeps is not None:
        common_loss_kwargs_local['rollout_substeps'] = int(local_substeps)
    loss_fn_sg = make_loss_fn(**common_loss_kwargs_sgfull, rollout_stop_grad=True)
    loss_fn_full = make_loss_fn(**common_loss_kwargs_sgfull, rollout_stop_grad=False)
    loss_fn_local = make_loss_fn(**common_loss_kwargs_local, rollout_stop_grad=False)
    train_step_sg = make_train_step(loss_fn_sg, optimizer_sg, grad_norm_guard=float(config['grad_norm_guard']))
    train_step_full = make_train_step(loss_fn_full, optimizer_full, grad_norm_guard=float(config['grad_norm_guard']))
    train_step_local = make_train_step(loss_fn_local, optimizer_local, grad_norm_guard=float(config['grad_norm_guard']))
    train_step_full_rollout = None
    if full_rollout_batch is not None:
        full_rollout_kwargs = dict(common_loss_kwargs)
        full_rollout_kwargs.update({
            'w_reg': 0.0,
            'w_reg_B': 0.0,
            'w_state': 0.0,
            'w_free_map': 0.0,
            'w_sensor_curvature': 0.0,
            'w_parameter_prior': 0.0,
        })
        full_rollout_loss = make_loss_fn(**full_rollout_kwargs, rollout_stop_grad=False)
        train_step_full_rollout = make_train_step(
            full_rollout_loss, optimizer_full,
            grad_norm_guard=float(config['grad_norm_guard']))

    # Minimally-invasive port-Hamiltonian CBF safety-filter controller (EBM_controller.py).
    # Default OFF/back-compatible: the EBM's own training gradients/optimizers above are
    # completely untouched by this block. When on, a SEPARATE (tiny, 2-scalar) controller
    # optimizer tunes gamma/epsilon-margin, gated by an EMA of the same train-NRMSE proxy
    # already used for the SWATS switch, and the tuned filter is only ever *applied* to
    # compensate the input at test time (evaluate_model), never inside the EBM's own
    # backprop path -- the plant identification itself is unaffected.
    enable_controller = bool(config.get('enable_controller', False))
    ctrl_cfg = ctrl_state = controller_energy_value_fn = controller_diag_rollout = controller_train_step = None
    if enable_controller:
        ctrl_cfg = ctrl_mod.ControllerConfig(
            enable_controller=True,
            activation_loss_threshold=float(config.get('controller_loss_threshold', 0.30)),
            ema_beta=float(config.get('controller_ema_beta', 0.9)),
            gamma_init=float(config.get('controller_gamma_init', 1.0)),
            margin_init=float(config.get('controller_margin_init', 1.0)),
            min_margin_floor=float(config.get('controller_min_margin_floor', 1e-3)),
            ctrl_lr=float(config.get('controller_ctrl_lr', 1e-3)),
            w_margin_reg=float(config.get('controller_w_margin_reg', 1e-3)),
            min_energy_n_steps=int(config.get('controller_min_energy_n_steps', 50)),
            min_energy_lr=float(config.get('controller_min_energy_lr', 1e-2)),
            n_ctrl_steps_per_epoch=int(config.get('controller_n_ctrl_steps_per_epoch', 1)),
        )
        ctrl_state = ctrl_mod.init_controller_state(ctrl_cfg)
        _, controller_energy_value_fn = ctrl_mod.make_energy_fns(layers)
        # Cheap diagnostic (stop-gradient) rollout on the last minibatch, used only to
        # feed (x_traj, u) into the controller-tuning step -- never backpropagated into
        # the EBM params, so it cannot influence plant identification.
        controller_diag_rollout = make_batch_rollout_fn(grad_E, D, M_PORTS, dt, integrator=str(config['rollout_integrator']))
        controller_loss_fn = ctrl_mod.make_controller_loss_fn(grad_E, controller_energy_value_fn, D, M_PORTS, ctrl_cfg)
        controller_train_step = ctrl_mod.make_controller_train_step(controller_loss_fn, optax.adam(ctrl_cfg.ctrl_lr))

    rng = np.random.default_rng(seed)
    total_skipped_updates = 0
    switched_to_full = False
    # Mixed-optimizer (SWATS-style) local-refinement switch state.
    local_switch_enabled = bool(config['local_opt_switch'])
    local_nrmse_thresh = float(config['local_opt_nrmse_threshold'])
    local_ema_beta = float(config['local_opt_ema_beta'])
    switched_to_local = False
    local_init_done = False
    ema_train_nrmse = None

    # Stochastic-regularisation + Polyak/EMA state (all default-off, architecture unchanged).
    input_noise_std = float(config['input_noise_std'])
    init_noise_std = float(config['init_noise_std'])
    use_ema = bool(config['use_ema'])
    ema_decay = float(config['ema_decay'])
    ema_scope = str(config.get('ema_scope', 'full')).lower()
    ema_params = None  # populated once past the SG warmup when use_ema is on
    select_best_validation = bool(config.get('select_best_validation', False))
    best_validation_params = None
    best_validation_nrmse = float('inf')
    best_validation_epoch = 0
    significant_validation_nrmse = float('inf')
    significant_validation_epoch = 0
    train_obs_at_significant = None
    early_stop_requested = False
    if select_best_validation:
        validation_rollout = make_rollout_fn(
            grad_E, D, M_PORTS, dt, integrator=str(config['rollout_integrator']),
            substeps=int(config.get('rollout_substeps', 1)))
        validation_y = jnp.asarray(y_te)
        validation_u = jnp.asarray(u_te)
        validation_encoder = (
            jnp.concatenate([validation_y[:INIT_WIN], validation_u[:INIT_WIN]], axis=-1)
            if pf.USE_INPUT_AWARE_ENCODER else validation_y[:INIT_WIN])

        def validation_nrmse(candidate):
            x0_validation = encode_x0(candidate, validation_encoder)
            _, _, prediction = validation_rollout(
                candidate, x0_validation, validation_u[INIT_WIN:], stop_grad=True)
            target = validation_y[INIT_WIN:]
            return float(jnp.sqrt(jnp.mean((prediction - target) ** 2))
                         / jnp.maximum(jnp.std(target), 1e-12))

    for epoch in tqdm(range(N_EPOCHS)):
        idx = rng.permutation(N_use)
        yw_ep = enc_win_all[idx].reshape(n_batches, BATCH_SIZE, INIT_WIN, N_OBS_EFF)
        us_ep = u_seg_all[idx].reshape(n_batches, BATCH_SIZE, SEQ_LEN, M_PORTS)
        ys_ep = y_seg_all[idx].reshape(n_batches, BATCH_SIZE, SEQ_LEN, N_OBS)
        weights_ep = window_weights[idx].reshape(n_batches, BATCH_SIZE)
        masks_ep = time_mask_all[idx].reshape(n_batches, BATCH_SIZE, SEQ_LEN)

        # Stochastic regularisation: fresh Gaussian jitter every epoch on the inputs
        # and the x0-encoder window (targets ys_ep stay clean). No-ops when std == 0.
        if input_noise_std > 0.0:
            us_ep = us_ep + rng.normal(0.0, input_noise_std, us_ep.shape).astype(us_ep.dtype)
        if init_noise_std > 0.0:
            yw_ep = yw_ep + rng.normal(0.0, init_noise_std, yw_ep.shape).astype(yw_ep.dtype)

        use_stop_grad = epoch < int(config['rollout_stop_grad_epochs'])
        if (not use_stop_grad) and (not switched_to_full):
            opt_state = optimizer_full.init(params)
            switched_to_full = True
        if switched_to_local and (not local_init_done):
            opt_state = optimizer_local.init(params)
            local_init_done = True

        if use_stop_grad:
            train_step, loss_eval_fn = train_step_sg, loss_fn_sg
        elif switched_to_local:
            train_step, loss_eval_fn = train_step_local, loss_fn_local
        else:
            train_step, loss_eval_fn = train_step_full, loss_fn_full
        skipped_updates_epoch = 0
        last_stats = None
        if str(config['observation_loss']) == 'huber_to_mse':
            warmup_epochs = max(1, int(round(float(config.get('huber_warmup_fraction', 0.1)) * N_EPOCHS)))
            mse_blend = jnp.asarray(min(float(epoch) / warmup_epochs, 1.0), dtype=jnp.float32)
        else:
            mse_blend = jnp.asarray(0.0, dtype=jnp.float32)

        for b in range(n_batches):
            params, opt_state, _, _, step_stats = train_step(
                params, opt_state,
                jnp.array(yw_ep[b]),
                jnp.array(us_ep[b]),
                jnp.array(ys_ep[b]),
                mse_blend,
                jnp.array(weights_ep[b]),
                jnp.array(masks_ep[b]),
            )
            last_stats = step_stats
            if not bool(step_stats['update_applied']):
                skipped_updates_epoch += 1
                total_skipped_updates += 1
            if not tree_all_finite(params):
                print(f"\n✗ Non-finite parameters detected at epoch {epoch+1}, batch {b+1}/{n_batches}")
                print_bad_leaves(params, 'params')
                try:
                    wandb.log({'epoch': epoch + 1, 'train/nonfinite_params': 1, 'train/nonfinite_batch': b + 1})
                except Exception:
                    pass
                return params, layers, grad_E, stats, None

        if train_step_full_rollout is not None and not use_stop_grad:
            full_encoder, full_inputs, full_targets = full_rollout_batch
            full_weight = (float(config.get('full_rollout_weight_start', 0.1))
                           if epoch < N_EPOCHS // 2
                           else float(config.get('full_rollout_weight_end', 0.3)))
            params, opt_state, _, _, full_stats = train_step_full_rollout(
                params, opt_state,
                jnp.asarray(full_encoder)[None],
                jnp.asarray(full_inputs)[None],
                jnp.asarray(full_targets)[None],
                mse_blend,
                jnp.asarray([full_weight], dtype=jnp.float32),
                jnp.ones((1, len(full_inputs)), dtype=bool),
            )
            if not bool(full_stats['update_applied']):
                skipped_updates_epoch += 1
                total_skipped_updates += 1

        # Polyak/EMA weight averaging: accumulate only once past the SG warmup, where
        # training is meaningful (captures both FULL and local phases). Seeded from the
        # first eligible epoch's params, then decayed toward the running weights.
        # ema_scope='local' narrows this further to just the SWATS local/SGD phase,
        # which targets the noisy plateau instead of diluting the average with the
        # earlier (still improving) FULL-AdamW epochs.
        ema_eligible = (not use_stop_grad) and (ema_scope != 'local' or switched_to_local)
        if use_ema and ema_eligible:
            if ema_params is None:
                ema_params = params
            else:
                ema_params = jax.tree_util.tree_map(
                    lambda e, p: ema_decay * e + (1.0 - ema_decay) * p, ema_params, params)

        if (epoch + 1) % PRINT_EVERY == 0:
            total_loss, aux = loss_eval_fn(
                params, jnp.array(yw_ep[-1]), jnp.array(us_ep[-1]),
                jnp.array(ys_ep[-1]), mse_blend,
                jnp.array(weights_ep[-1]), jnp.array(masks_ep[-1]))
            # Smoothed train-NRMSE proxy: data is unit-variance normalised and the
            # huber obs loss ~= 0.5*MSE for in-delta residuals, so NRMSE ~= sqrt(2*obs).
            inst_nrmse = float(np.sqrt(max(2.0 * float(aux['obs_loss']), 0.0)))
            ema_train_nrmse = inst_nrmse if ema_train_nrmse is None else (
                local_ema_beta * ema_train_nrmse + (1.0 - local_ema_beta) * inst_nrmse)
            if select_best_validation:
                validation_score = validation_nrmse(params)
                if validation_score < best_validation_nrmse:
                    best_validation_nrmse = validation_score
                    best_validation_epoch = epoch + 1
                    best_validation_params = params
                if validation_score <= significant_validation_nrmse - float(
                        config.get('validation_min_delta', 0.002)):
                    significant_validation_nrmse = validation_score
                    significant_validation_epoch = epoch + 1
                    train_obs_at_significant = float(aux['obs_loss'])
                patience = int(config.get('validation_patience', 250))
                if ((epoch + 1 - significant_validation_epoch) >= patience
                        and train_obs_at_significant is not None):
                    train_change = abs(float(aux['obs_loss']) - train_obs_at_significant) / max(
                        abs(train_obs_at_significant), 1e-12)
                    early_stop_requested = train_change < 0.01
            if (local_switch_enabled and (not switched_to_local) and (not use_stop_grad)
                    and ema_train_nrmse < local_nrmse_thresh):
                switched_to_local = True
                print(f"\n→ Mixed-optimizer switch at epoch {epoch+1}: smoothed train NRMSE "
                      f"{ema_train_nrmse:.4f} < {local_nrmse_thresh:.4f}; handing FULL phase off "
                      f"to local optimizer '{config['local_optimizer']}'.")
            if enable_controller:
                was_active = bool(ctrl_state.active)
                x0_diag = encode_x0_batch(params, jnp.array(yw_ep[-1]))
                x_traj_diag, _, _ = controller_diag_rollout(params, x0_diag, jnp.array(us_ep[-1]), stop_grad=True)
                ctrl_state, ctrl_aux = ctrl_mod.controller_epoch_update(
                    ctrl_state, ctrl_cfg, params, x_traj_diag, jnp.array(us_ep[-1]),
                    inst_nrmse, controller_train_step, energy_value_fn=controller_energy_value_fn,
                )
                if bool(ctrl_state.active) and not was_active:
                    print(f"\n→ Controller tuning activated at epoch {epoch+1}: smoothed train NRMSE "
                          f"proxy {float(ctrl_state.loss_ema):.4f} < "
                          f"{ctrl_cfg.activation_loss_threshold:.4f}.")
                ctrl_log = {
                    'controller/active': float(ctrl_state.active),
                    'controller/loss_ema': float(ctrl_state.loss_ema),
                }
                if ctrl_aux is not None:
                    ctrl_log.update({f'controller/{k}': float(v) for k, v in ctrl_aux.items()})
                try:
                    wandb.log(ctrl_log)
                except Exception:
                    pass
            print(
                f"Epoch {epoch+1:4d}/{N_EPOCHS}  "
                f"phase={'SG' if use_stop_grad else ('LOCAL' if switched_to_local else 'FULL')}  "
                f"Total loss: {float(total_loss):.4e}  "
                f"obs={float(aux['obs_loss']):.4e}  "
                f"pass={float(aux['passivity_loss']):.4e}  "
                f"reg={float(aux['reg_loss']):.4e}  "
                f"skipped={skipped_updates_epoch}"
            )
            try:
                wandb.log({
                    'epoch': epoch + 1,
                    'loss/total': float(total_loss),
                    'loss/obs': float(aux['obs_loss']),
                    'loss/passivity': float(aux['passivity_loss']),
                    'loss/reg': float(aux['reg_loss']),
                    'loss/state': float(aux['state_loss']),
                    'loss/free_map': float(aux['free_map_loss']),
                    'loss/sensor_curvature': float(aux['sensor_curvature_loss']),
                    'loss/parameter_prior': float(aux['parameter_prior_loss']),
                    'loss/mse_blend': float(mse_blend),
                    'train/finite_rollout': float(aux['finite_rollout']),
                    'train/finite_penalty': float(aux['finite_penalty']),
                    'train/skipped_updates_epoch': skipped_updates_epoch,
                    'train/total_skipped_updates': total_skipped_updates,
                    'train/grad_norm_last': float(last_stats['grad_norm']) if last_stats is not None else np.nan,
                    'train/update_applied_last': float(last_stats['update_applied']) if last_stats is not None else 0.0,
                    'train/stop_grad_phase': float(use_stop_grad),
                    'train/train_nrmse_ema': float(ema_train_nrmse),
                    'train/phase_local': float(switched_to_local),
                    'validation/nrmse': validation_score if select_best_validation else np.nan,
                })
            except Exception as e:
                print(f"  ⚠ wandb.log() failed: {e}")

        if skipped_updates_epoch > int(config['max_bad_updates']):
            print(f"⚠ Too many skipped updates at epoch {epoch+1}: {skipped_updates_epoch}. Early stopping.")
            break
        if early_stop_requested:
            print(f"Early stopping at epoch {epoch+1}: validation improvement < "
                  f"{float(config.get('validation_min_delta', 0.002)):.4f} for "
                  f"{int(config.get('validation_patience', 250))} epochs and train loss changed <1%.")
            break

    if select_best_validation and best_validation_params is not None:
        params = best_validation_params
        stats['training_best_epoch'] = int(best_validation_epoch)
        stats['training_best_validation_nrmse'] = float(best_validation_nrmse)

    # Build the test-time controller descriptor, if it ever activated during training.
    # None whenever enable_controller is off or the loss-threshold gate never fired --
    # evaluate_model then falls back to the plain (uncontrolled) rollout, unchanged.
    controller_info = None
    if enable_controller:
        if bool(ctrl_state.active):
            gamma_final = float(ctrl_mod.gamma_from(ctrl_state.ctrl_params))
            margin_final = float(ctrl_mod.margin_from(ctrl_state.ctrl_params, ctrl_cfg.min_margin_floor))
            epsilon_final = float(ctrl_state.min_energy_est) + margin_final
            controller_info = {
                'grad_energy_fn': grad_E,
                'energy_value_fn': controller_energy_value_fn,
                'epsilon': epsilon_final,
                'gamma': gamma_final,
                'd': D,
                'm': M_PORTS,
            }
            print(f"\n→ Controller tuned: epsilon={epsilon_final:.4f} "
                  f"(min_E_est={float(ctrl_state.min_energy_est):.4f} + margin={margin_final:.4f}), "
                  f"gamma={gamma_final:.4f}")
        else:
            print("\n⚠ Controller enabled but never activated (train-NRMSE proxy never "
                  "dropped below controller_loss_threshold); test-time evaluation will use "
                  "the UNCONTROLLED rollout.")

    if eval_after:
        print("\n── Test evaluation ──")
        rmse, nrmse = evaluate_model(
            params, layers, grad_E, config, D, M_PORTS, dt, INIT_WIN,
            init_len, u_te, y_te, stats,
            unit_label=unit_label, unit_scale=unit_scale, log_prefix='test',
            set_summary=not (use_ema and ema_params is not None),
            controller=controller_info,
        )
        if use_ema and ema_params is not None:
            print("\n── Test evaluation (EMA weights) ──")
            rmse_e, nrmse_e = evaluate_model(
                ema_params, layers, grad_E, config, D, M_PORTS, dt, INIT_WIN,
                init_len, u_te, y_te, stats,
                unit_label=unit_label, unit_scale=unit_scale, log_prefix='test_ema',
                set_summary=False,
            )
            # Report the better of raw vs EMA as best_test_nrmse.
            cands = [v for v in (nrmse, nrmse_e) if v is not None]
            if cands:
                best = min(cands)
                which = 'ema' if (nrmse_e is not None and best == nrmse_e) else 'raw'
                print(f"[test] raw NRMSE={nrmse}  EMA NRMSE={nrmse_e}  -> best={best:.6f} ({which})")
                try:
                    wandb.summary['best_test_nrmse'] = best
                except Exception:
                    pass
    return params, layers, grad_E, stats, controller_info


def train(seed=0, config=None):
    """Silverbox entry point that scores all three benchmark test sets.

    Trains once (shared loop, eval deferred), then rolls out and reports RMSE
    (mV) + NRMSE on multisine, arrow_full and arrow_no_extrapolation — the exact
    trio the nonlinear-benchmark leaderboard expects for Silverbox."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    params, layers, grad_E, stats, controller_info = run_training(
        seed=seed, config=cfg, load_fn=load_silverbox,
        eval_unit=('mV', 1e3), eval_after=False,
    )

    if bool(cfg.get('save_final_params', True)):
        destination = Path(str(cfg.get(
            'final_params_path', 'results/silverbox/checkpoints/Silverbox_final_params.pkl')))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(jax.device_get(params), stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
        print(f"Saved final model parameters to {destination}")

    D = int(cfg['d'])
    M_PORTS = int(cfg['m_ports'])
    INIT_WIN = int(cfg['init_win'])
    dt, tests, init_len, stats = load_silverbox_all_tests()

    print("\n── Silverbox per-test-set evaluation ──")
    nrmses = []
    for name, u_te, y_te in tests:
        rmse, nrmse = evaluate_model(
            params, layers, grad_E, cfg, D, M_PORTS, dt, INIT_WIN,
            init_len, u_te, y_te, stats,
            unit_label='mV', unit_scale=1e3, log_prefix=f'test_{name}',
            set_summary=False,
            controller=controller_info,
        )
        if nrmse is not None:
            nrmses.append(nrmse)
    if nrmses:
        mean_nrmse = float(np.mean(nrmses))
        print(f"[silverbox] mean NRMSE over {len(nrmses)} test sets: {mean_nrmse:.6f}")
        try:
            wandb.summary['best_test_nrmse'] = mean_nrmse
        except Exception:
            pass
    return params, layers, grad_E, stats


if __name__ == '__main__':
    train(seed=42, config=DEFAULT_CONFIG)
