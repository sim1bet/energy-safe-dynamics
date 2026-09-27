"""DeepDissipative_NLink_main.py — import the EXACT upstream Deep Dissipative
Dynamics n-link dataset into our JAX port-Hamiltonian pipeline (prompt §17-22).

This module does NOT regenerate "equivalent" data. It reads the ``.npy``
files written by the upstream ``ddm-dataset nlink`` CLI
(``.external/DeepDissipativeModel``) verbatim, so any comparison uses
identical train/test trajectories for both methods (prompt §19).

Two labeled observation modes (never mixed in one comparison table, §20):

* ``output_only`` (primary/official): observation = upstream ``.obs.npy``
  (partial: only the first joint's angle+velocity, ``--nlink_q0u0``).  Uses the
  learned burn-in encoder + ``use_input_aware_encoder`` exactly like the other
  ``Interface_code`` benchmarks, reusing ``EBM_training.make_loss_fn`` /
  ``EBM_rollout.make_init_state_fn`` unchanged.
* ``state_observed`` (secondary/gray-box): observation = upstream
  ``.state.npy`` (the FULL physical state the upstream generator also saves).
  Reuses ``Duffing_DoubleWell_main.make_state_observed_loss_fn`` /
  ``make_windows`` unchanged (same bespoke true-x0/identity-readout trainer).

``d`` (our model's *latent* state order) is a free hyperparameter independent
of the true physical order, exactly as in the upstream config (confirmed by
reading ``ddm/util.py::init_dim_from_data`` — only ``obs_dim``/``in_dim`` are
derived from data; ``state_dim`` is not).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "EBM_model", _ROOT / "Interface_code"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import hashlib
import json
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from EBM_class import LayerSpec, lagrangian_tanh, activation_tanh, lagrangian_polynomial_stable, activation_polynomial_stable
from EBM_param_fields import init_params, make_grad_energy, encode_x0_batch
from EBM_rollout import make_batch_rollout_fn
from EBM_training import make_loss_fn, make_train_step, tree_all_finite
import EBM_param_fields as pf

from windowing_utils import make_encoder_windows
from Duffing_DoubleWell_main import make_windows as make_state_windows
from Duffing_DoubleWell_main import make_state_observed_loss_fn


# ══════════════════════════════════════════════════════════════════════════
# 1. Upstream dataset loader (verbatim; no regeneration)
# ══════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "deep_dissipative_nlink"


def load_nlink_dataset(dataset_dir: str | Path, prefix: str) -> dict:
    """Read ``{prefix}nlink.{train,test}.{obs,input,state}.npy`` + info.json.

    Returns raw arrays plus ``dt``, physical/observed dims, and a SHA-256 hash
    over the exact bytes of every file read (for the run manifest).

    Canonical shape convention (documented once, here -- see prompt §2): every
    array is ``[N_trajectories, T, features]`` with ``T`` identical across
    ``y``/``u``/``z`` (upstream's synchronous storage; unlike Duffing's own
    ``T+1``/``T`` convention, see the note in ``_train_state_observed``).
    """
    out_dir = Path(dataset_dir)
    stem = out_dir / f"{prefix}nlink"
    info = json.loads(Path(f"{stem}.info.json").read_text())

    def _load(name):
        return np.load(f"{stem}.{name}.npy").astype(np.float32)

    arrays = dict(
        y_train=_load("train.obs"), u_train=_load("train.input"), z_train=_load("train.state"),
        y_test=_load("test.obs"), u_test=_load("test.input"), z_test=_load("test.state"),
    )
    for name, arr in arrays.items():
        assert arr.ndim == 3, f"{name}: expected [N,T,features], got shape {arr.shape}"
    n_traj_train = arrays["y_train"].shape[0]
    T_train = arrays["y_train"].shape[1]
    for name in ("u_train", "z_train"):
        assert arrays[name].shape[0] == n_traj_train, (
            f"{name} has {arrays[name].shape[0]} trajectories, y_train has {n_traj_train}"
        )
        assert arrays[name].shape[1] == T_train, (
            f"{name} has T={arrays[name].shape[1]}, y_train has T={T_train} "
            "(upstream stores y/u/z with the SAME trajectory length)"
        )

    hasher = hashlib.sha256()
    for name in ("train.obs", "train.input", "train.state", "test.obs", "test.input", "test.state"):
        hasher.update(Path(f"{stem}.{name}.npy").read_bytes())
    return dict(
        **arrays, dt=float(info["dt"]), T=float(info["T"]),
        n_obs=arrays["y_train"].shape[-1], in_dim=arrays["u_train"].shape[-1],
        state_dim_physical=arrays["z_train"].shape[-1],
        dataset_hash=hasher.hexdigest(), info=info, prefix=prefix,
    )


# ══════════════════════════════════════════════════════════════════════════
# 2. Model configuration (prompt §21-22)
# ══════════════════════════════════════════════════════════════════════════

def get_config() -> dict:
    return dict(
        experiment_name=EXPERIMENT_NAME,
        mode="output_only",  # or "state_observed" -- never mixed in one table
        d=8,                 # our model's LATENT order; a free hyperparameter
        layer_dims=[128, 64],
        first_layer_type="tanh", p_1=2.0,
        second_layer_type="polynomial_stable", p_2=4.0,
        vf_scale=1.0, damping_scale=1.0,
        use_feedthrough=False, use_nonlinear_readout=False,
        use_saturating_readout=False, use_state_damping=False,
        use_input_gain=False, use_saturating_input=False,
        use_input_aware_encoder=True,  # output_only mode: partial obs, prompt §20
        rollout_integrator="rk4",
        init_win=10, batch_size=32, lr=3e-4, seq_len=100, stride=10,
        n_epochs=500, stop_grad_epochs=20,
        grad_clip_norm=1.0, grad_norm_guard=1000.0,
        w_reg=1e-6, trunk_reg_mult=1.0, w_reg_B=0.0, w_state=0.0, w_passivity=0.0,
        observation_loss="huber", huber_delta=1.0,
        chol_clip_exp=1.5, grad_e_clip=200.0, xdot_clip=200.0,
        state_norm_clip=50.0, x0_norm_max=50.0, b_init_scale=1.0,
        print_every=10,
    )


DEFAULT_CONFIG = get_config()


def _build_layers(config: dict) -> tuple:
    first = LayerSpec(lagrangian_tanh, activation_tanh, (float(config["p_1"]),))
    second = LayerSpec(lagrangian_polynomial_stable, activation_polynomial_stable, (float(config["p_2"]),))
    return (first, second)


def _apply_runtime_flags(cfg: dict) -> None:
    pf.CHOL_CLIP_EXP = float(cfg["chol_clip_exp"])
    pf.DAMPING_SCALE = float(cfg["damping_scale"])
    pf.B_INIT_SCALE = float(cfg["b_init_scale"])
    pf.VF_SCALE = float(cfg["vf_scale"])
    pf.USE_FEEDTHROUGH = bool(cfg["use_feedthrough"])
    pf.USE_NONLINEAR_READOUT = bool(cfg["use_nonlinear_readout"])
    pf.USE_STATE_DAMPING = bool(cfg["use_state_damping"])
    pf.USE_SATURATING_READOUT = bool(cfg["use_saturating_readout"])
    pf.USE_INPUT_GAIN = bool(cfg["use_input_gain"])
    pf.USE_SATURATING_INPUT = bool(cfg["use_saturating_input"])
    pf.USE_INPUT_AWARE_ENCODER = bool(cfg["use_input_aware_encoder"])
    pf.GRAD_E_CLIP = float(cfg["grad_e_clip"])
    pf.XDOT_CLIP = float(cfg["xdot_clip"])
    pf.X0_NORM_MAX = float(cfg["x0_norm_max"])
    pf.STATE_NORM_CLIP = float(cfg["state_norm_clip"])


def _rmse_nrmse(pred: np.ndarray, true: np.ndarray) -> tuple:
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    nrmse = rmse / max(float(np.std(true)), 1e-12)
    return rmse, nrmse


# ══════════════════════════════════════════════════════════════════════════
# 3. State-observed (gray-box) training -- reuses Duffing's bespoke trainer
# ══════════════════════════════════════════════════════════════════════════

def _train_state_observed(seed: int, cfg: dict, data: dict, verbose: bool) -> tuple:
    D = int(cfg["d"])
    M = int(data["in_dim"])
    dt = data["dt"]
    x0_tr, u_tr, z_tr = make_state_windows(data["z_train"], data["u_train"], int(cfg["seq_len"]), int(cfg["stride"]))
    assert x0_tr.shape[-1] == D, (
        f"x0_tr shape {x0_tr.shape} last dim != d={D}. In state_observed mode, d must equal "
        f"the upstream physical state dimension ({data['state_dim_physical']}); "
        "run_training() sets this automatically -- if it's wrong here, cfg['d'] was "
        "overridden after that point."
    )
    assert u_tr.shape[-1] == M, f"u_tr shape {u_tr.shape} last dim != in_dim={M}"

    layers = _build_layers(cfg)
    grad_E = make_grad_energy(layers)
    params = init_params(jax.random.PRNGKey(seed), d=D, m=M, n_obs=D, init_win=1,
                         layer_dims=list(cfg["layer_dims"]), dt=dt)

    n_batches = max(x0_tr.shape[0] // int(cfg["batch_size"]), 1)
    optimizer = optax.chain(optax.clip_by_global_norm(float(cfg["grad_clip_norm"])), optax.adamw(float(cfg["lr"])))
    opt_state = optimizer.init(params)

    loss_kwargs = dict(grad_energy_fn=grad_E, d=D, m=M, dt=dt, integrator=str(cfg["rollout_integrator"]),
                       w_reg=float(cfg["w_reg"]), trunk_mult=float(cfg["trunk_reg_mult"]),
                       w_reg_B=float(cfg["w_reg_B"]), observation_loss=str(cfg["observation_loss"]),
                       huber_delta=float(cfg["huber_delta"]))
    loss_sg = make_state_observed_loss_fn(**loss_kwargs, rollout_stop_grad=True)
    loss_full = make_state_observed_loss_fn(**loss_kwargs, rollout_stop_grad=False)
    step_sg = make_train_step(loss_sg, optimizer, grad_norm_guard=float(cfg["grad_norm_guard"]))
    step_full = make_train_step(loss_full, optimizer, grad_norm_guard=float(cfg["grad_norm_guard"]))

    rng = np.random.default_rng(seed)
    batch_size = int(cfg["batch_size"])
    N_use = n_batches * batch_size
    for epoch in range(int(cfg["n_epochs"])):
        idx = rng.permutation(x0_tr.shape[0])[:N_use]
        x0_ep = x0_tr[idx].reshape(n_batches, batch_size, D)
        u_ep = u_tr[idx].reshape(n_batches, batch_size, int(cfg["seq_len"]), M)
        z_ep = z_tr[idx].reshape(n_batches, batch_size, int(cfg["seq_len"]), D)
        step = step_sg if epoch < int(cfg["stop_grad_epochs"]) else step_full
        if epoch == int(cfg["stop_grad_epochs"]):
            opt_state = optimizer.init(params)
        last_loss = None
        for b in range(n_batches):
            params, opt_state, loss, aux, stats = step(
                params, opt_state, jnp.array(x0_ep[b]), jnp.array(u_ep[b]), jnp.array(z_ep[b]))
            last_loss = float(loss)
            if not tree_all_finite(params):
                raise RuntimeError(f"Non-finite parameters at epoch {epoch}, batch {b}")
        if verbose and (epoch % int(cfg["print_every"]) == 0 or epoch == int(cfg["n_epochs"]) - 1):
            print(f"[nlink/state_observed] epoch {epoch:4d}  loss={last_loss:.6f}")

    val_rollout = make_batch_rollout_fn(grad_E, D, M, dt, integrator=str(cfg["rollout_integrator"]))
    x0_full = jnp.array(data["z_test"][:, 0, :])
    # Upstream stores z/u with the SAME trajectory length (synchronous
    # convention), unlike Duffing's z=T+1/u=T; drop the last input sample so
    # the rollout length matches the number of states being predicted.
    u_full = jnp.array(data["u_test"][:, :-1, :])
    x_pred, _, _ = val_rollout(params, x0_full, u_full, stop_grad=True)
    z_true_full = data["z_test"][:, 1:, :]
    assert x_pred.shape == z_true_full.shape, (
        f"Prediction/target mismatch: pred={x_pred.shape}, target={z_true_full.shape}"
    )
    x_pred_np = np.asarray(x_pred)
    rmse, nrmse = _rmse_nrmse(x_pred_np, z_true_full)
    if verbose:
        print(f"[nlink/state_observed] test long-horizon RMSE={rmse:.4f}  NRMSE={nrmse:.4f}")
    extras = dict(test_true=z_true_full, test_pred=x_pred_np, test_u=np.asarray(u_full),
                 reference_states=x_pred_np.reshape(-1, D), reference_inputs=np.asarray(u_full).reshape(-1, M))
    return params, layers, grad_E, rmse, nrmse, extras


# ══════════════════════════════════════════════════════════════════════════
# 4. Output-only (latent/official) training -- reuses EBM_training.make_loss_fn
# ══════════════════════════════════════════════════════════════════════════

def _train_output_only(seed: int, cfg: dict, data: dict, verbose: bool) -> tuple:
    D = int(cfg["d"])
    M = int(data["in_dim"])
    N_OBS = int(data["n_obs"])
    dt = data["dt"]
    init_win = int(cfg["init_win"])

    y_win, u_win, u_seg, y_seg = make_encoder_windows(
        data["y_train"], data["u_train"], init_win, int(cfg["seq_len"]), int(cfg["stride"]))
    enc_win = np.concatenate([y_win, u_win], axis=-1) if pf.USE_INPUT_AWARE_ENCODER else y_win
    n_obs_eff = N_OBS + (M if pf.USE_INPUT_AWARE_ENCODER else 0)
    assert enc_win.shape[-1] == n_obs_eff, (
        f"encoder window last dim {enc_win.shape[-1]} != n_obs_eff={n_obs_eff} "
        f"(n_obs={N_OBS} + m_ports={M} if use_input_aware_encoder else n_obs alone) "
        "-- the input-aware encoder expects n_obs+m_ports features per step."
    )
    assert u_seg.shape[-1] == M, f"u_seg shape {u_seg.shape} last dim != in_dim={M}"
    assert y_seg.shape[-1] == N_OBS, f"y_seg shape {y_seg.shape} last dim != n_obs={N_OBS}"

    layers = _build_layers(cfg)
    grad_E = make_grad_energy(layers)
    params = init_params(jax.random.PRNGKey(seed), d=D, m=M, n_obs=N_OBS, init_win=init_win,
                         layer_dims=list(cfg["layer_dims"]), dt=dt)

    n_batches = max(enc_win.shape[0] // int(cfg["batch_size"]), 1)
    optimizer = optax.chain(optax.clip_by_global_norm(float(cfg["grad_clip_norm"])), optax.adamw(float(cfg["lr"])))
    opt_state = optimizer.init(params)

    common = dict(grad_energy_fn=grad_E, layers=layers, d=D, m=M, dt=dt,
                 w_rollout=1.0, w_passivity=float(cfg["w_passivity"]), w_reg=float(cfg["w_reg"]),
                 trunk_mult=float(cfg["trunk_reg_mult"]), w_reg_B=float(cfg["w_reg_B"]),
                 w_state=float(cfg["w_state"]), rollout_integrator=str(cfg["rollout_integrator"]),
                 observation_loss=str(cfg["observation_loss"]), huber_delta=float(cfg["huber_delta"]))
    loss_sg = make_loss_fn(**common, rollout_stop_grad=True)
    loss_full = make_loss_fn(**common, rollout_stop_grad=False)
    step_sg = make_train_step(loss_sg, optimizer, grad_norm_guard=float(cfg["grad_norm_guard"]))
    step_full = make_train_step(loss_full, optimizer, grad_norm_guard=float(cfg["grad_norm_guard"]))

    rng = np.random.default_rng(seed)
    batch_size = int(cfg["batch_size"])
    N_use = n_batches * batch_size
    for epoch in range(int(cfg["n_epochs"])):
        idx = rng.permutation(enc_win.shape[0])[:N_use]
        yw_ep = enc_win[idx].reshape(n_batches, batch_size, init_win, n_obs_eff)
        us_ep = u_seg[idx].reshape(n_batches, batch_size, int(cfg["seq_len"]), M)
        ys_ep = y_seg[idx].reshape(n_batches, batch_size, int(cfg["seq_len"]), N_OBS)
        step = step_sg if epoch < int(cfg["stop_grad_epochs"]) else step_full
        if epoch == int(cfg["stop_grad_epochs"]):
            opt_state = optimizer.init(params)
        last_loss = None
        for b in range(n_batches):
            params, opt_state, loss, aux, stats = step(
                params, opt_state, jnp.array(yw_ep[b]), jnp.array(us_ep[b]), jnp.array(ys_ep[b]))
            last_loss = float(loss)
            if not tree_all_finite(params):
                raise RuntimeError(f"Non-finite parameters at epoch {epoch}, batch {b}")
        if verbose and (epoch % int(cfg["print_every"]) == 0 or epoch == int(cfg["n_epochs"]) - 1):
            print(f"[nlink/output_only] epoch {epoch:4d}  loss={last_loss:.6f}")

    # Test evaluation: encode x0 from the burn-in window, roll out the remainder.
    y_win_test = data["y_test"][:, :init_win, :]
    u_win_test = data["u_test"][:, :init_win, :]
    enc_win_test = np.concatenate([y_win_test, u_win_test], axis=-1) if pf.USE_INPUT_AWARE_ENCODER else y_win_test
    x0_test = encode_x0_batch(params, jnp.array(enc_win_test))
    val_rollout = make_batch_rollout_fn(grad_E, D, M, dt, integrator=str(cfg["rollout_integrator"]))
    _, _, y_pred = val_rollout(params, x0_test, jnp.array(data["u_test"][:, init_win:, :]), stop_grad=True)
    y_true = data["y_test"][:, init_win:, :]
    assert y_pred.shape == y_true.shape, (
        f"Prediction/target mismatch: pred={y_pred.shape}, target={y_true.shape}"
    )
    y_pred_np = np.asarray(y_pred)
    rmse, nrmse = _rmse_nrmse(y_pred_np, y_true)
    if verbose:
        print(f"[nlink/output_only] test RMSE={rmse:.4f}  NRMSE={nrmse:.4f}")
    # Reference latent states for Stress_tests well-discovery/epsilon-selection:
    # encoded TRAIN latent states (no separate validation split exists upstream;
    # using train here, never the test-set adversarial results, keeps epsilon
    # selection leakage-free per prompt §14/§25).
    x0_train_sample = encode_x0_batch(params, jnp.array(enc_win[:256]))
    extras = dict(test_true=y_true, test_pred=y_pred_np, test_u=np.asarray(data["u_test"][:, init_win:, :]),
                 reference_states=np.asarray(x0_train_sample), reference_inputs=np.asarray(u_seg[:256, 0, :]))
    return params, layers, grad_E, rmse, nrmse, extras


# ══════════════════════════════════════════════════════════════════════════
# 5. Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_training(seed: int = 0, config: Optional[dict] = None,
                 dataset_dir: str | Path = "results/deep_dissipative_nlink/upstream_baselines/dataset",
                 prefix: str = "smoke012", verbose: bool = True) -> tuple:
    """Returns (params, layers, grad_E, meta). ``meta['mode']`` records which
    of the two labeled observation modes (state_observed/output_only) was used
    -- never compare across modes without that label (prompt §20)."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    if cfg.get("experiment_name", EXPERIMENT_NAME) != EXPERIMENT_NAME:
        raise ValueError(
            f"DeepDissipative_NLink_main.run_training received a config with "
            f"experiment_name={cfg.get('experiment_name')!r}, expected {EXPERIMENT_NAME!r}. "
            "This config was generated for a different experiment; refusing to train "
            "(see DEBUGGING_REPORT.md for the routing bug this guards against)."
        )
    _apply_runtime_flags(cfg)

    data = load_nlink_dataset(dataset_dir, prefix)
    mode = str(cfg["mode"])
    if mode == "state_observed":
        cfg["d"] = int(data["state_dim_physical"])  # gray-box: latent order = true physical order
        params, layers, grad_E, rmse, nrmse, extras = _train_state_observed(seed, cfg, data, verbose)
    elif mode == "output_only":
        params, layers, grad_E, rmse, nrmse, extras = _train_output_only(seed, cfg, data, verbose)
    else:
        raise ValueError(f"Unknown mode={mode!r}; expected 'state_observed' or 'output_only'.")

    meta = dict(mode=mode, dt=data["dt"], dataset_hash=data["dataset_hash"], prefix=prefix,
               n_obs=data["n_obs"], in_dim=data["in_dim"], state_dim_physical=data["state_dim_physical"],
               test_rmse=rmse, test_nrmse=nrmse, config=cfg, seed=seed, extras=extras)
    return params, layers, grad_E, meta


def train(seed: int = 0, config: Optional[dict] = None) -> tuple:
    """Uniform compute environment-pipeline entry point, see
    ``Duffing_DoubleWell_main.train`` / ``train_from_config.py``.
    ``config`` may set ``dataset_dir``/``prefix`` to point at a specific
    frozen upstream dataset; both default to the Priority-5/6 smoke dataset.
    After training, runs the full reconstruction->Stress_tests->plots->audit
    pipeline (prompt §11/§13/§24) via
    ``post_training_eval.run_post_training_evaluation``.
    """
    cfg = dict(config or {})
    dataset_dir = cfg.pop("dataset_dir", "results/deep_dissipative_nlink/upstream_baselines/dataset")
    prefix = cfg.pop("prefix", "smoke012")
    params, layers, grad_E, meta = run_training(seed=seed, config=cfg, dataset_dir=dataset_dir, prefix=prefix)
    full_cfg = meta["config"]
    extras = meta["extras"]
    dt = meta["dt"]
    D = int(full_cfg["d"])
    M = int(meta["in_dim"])
    run_id = str(full_cfg.get("wandb_source_run_id") or f"seed{seed}")

    from experiment_paths import get_experiment_paths
    from post_training_eval import run_post_training_evaluation, save_checkpoint, save_json
    paths = get_experiment_paths(EXPERIMENT_NAME, run_id)
    save_json(paths.config_dir / "config.json", **full_cfg)
    save_checkpoint(paths.checkpoint_dir / "params.pkl", params)

    from Stress_tests.integration import apply_runtime_config
    from Stress_tests.model_adapter import EBMStressAdapter
    from Stress_tests.experiments.nlink_dissipative import select_epsilon_shell
    apply_runtime_config(full_cfg)
    adapter = EBMStressAdapter(params, layers, d=D, m=M, dt=dt)

    reference_states = extras["reference_states"]
    reference_inputs = extras["reference_inputs"]
    # Energy-shell selection rule (prompt §25/§13): 95th percentile of
    # reference (never test-adversarial) latent-state energy + 10% margin,
    # frozen BEFORE any OOD/adversarial test.
    shell = select_epsilon_shell(adapter, reference_states, quantile=0.95, margin_fraction=0.10)
    epsilon = shell.epsilon
    gamma = 1.0

    result = run_post_training_evaluation(
        experiment_name=EXPERIMENT_NAME, run_id=run_id,
        params=params, layers=layers, grad_E=grad_E, d=D, m=M, dt=dt,
        test_true=extras["test_true"], test_pred=extras["test_pred"], test_u=extras["test_u"],
        unit_label="", reference_states=reference_states, reference_inputs=reference_inputs,
        epsilon=epsilon, gamma=gamma, paths=paths,
        extra_metrics=dict(epsilon_shell_selection=vars(shell), mode=meta["mode"]),
        seed=seed,
    )

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"test/rmse": meta["test_rmse"], "test/nrmse": meta["test_nrmse"]})
            wandb.summary["best_test_nrmse"] = meta["test_nrmse"]
    except Exception as e:
        print(f"wandb.log() failed: {e}")
    return params, layers, grad_E, meta


if __name__ == "__main__":
    run_training(seed=0, config=dict(n_epochs=5, seq_len=40, batch_size=8, stop_grad_epochs=2, print_every=1))
