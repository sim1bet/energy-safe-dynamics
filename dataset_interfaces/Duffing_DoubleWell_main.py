# Author: Simone Betteti
"""Duffing_DoubleWell_main.py — controlled double-well Duffing ground truth,
dataset generation, and a *gray-box / state-observed* trainer for the free-EBM
port-Hamiltonian model.

Scientific role
---------------
This reference experiment validates the multi-well
Hamiltonian geometry, connected-component energy-CBF certificates, saddle-
induced robustness collapse and plant/model-mismatch transfer described in
``Stress_tests/experiments/duffing_doublewell.py``.

Ground-truth system (physical units, NOT normalised)
-----------------------------------------------------
    H*(q, p) = p^2 / 2 + (q^2 - 1)^2 / 4
    grad H*(q, p) = (q^3 - q, p)
    J = [[0, 1], [-1, 0]]   (skew, lossless)
    R = diag(0, r)          (r >= 0, mechanical/viscous damping)
    G = [0, 1]^T            (force acts through the momentum channel only)

    qdot = p
    pdot = q - q^3 - r*p + u + d

with two stable minima at (-1, 0) and (1, 0) (H*=0) and a saddle at (0, 0)
(H*=0.25).  ``d`` is an external disturbance entering through the same channel
as the control input (``w = E d``, ``E = [0, 1]^T``).

Why this module does NOT reuse ``Silverbox_main.run_training`` verbatim
------------------------------------------------------------------------
Every existing benchmark module estimates the initial latent state ``x0``
from a *learned linear encoder* applied to a burn-in output window, and reads
the observation through a *learned* ``C_state`` readout — appropriate for
input-output benchmarks with no access to the true state.  For a plant-level
certificate we must instead train directly in the *physical* coordinates
``z = (q, p)``: the true initial state is used as ``x0`` (no encoder), and the
model's own state trajectory (``d = n_obs = 2``) *is* the observation (no
learned decoder).  This module builds a bespoke, minimal trainer for that
mode, reusing the model core unchanged (``EBM_class``, ``EBM_param_fields``,
``EBM_rollout.make_batch_rollout_fn``, ``EBM_training.make_train_step``) —
only the x0/observation wiring is new, exactly as recommended in the
experiment-implementation plan.

Physical vs. normalized units
------------------------------
Unlike the other ``Interface_code`` modules, arrays returned/consumed here are
in raw physical units (``stats`` is the identity map) because the state-
observed certificate must live in the same coordinates as the physical plant.
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
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import jax
import jax.numpy as jnp
import optax

from EBM_class import (
    LayerSpec,
    lagrangian_tanh, activation_tanh,
    lagrangian_polynomial_stable, activation_polynomial_stable,
)
from EBM_param_fields import init_params, make_grad_energy
from EBM_rollout import make_batch_rollout_fn
from EBM_training import (
    make_train_step, weight_regularisation, tree_all_finite, compute_gradient_norm,
)
import EBM_param_fields as pf
from nanodrone.rotational import omega_sign_mask


# ══════════════════════════════════════════════════════════════════════════
# 1. Ground-truth physics (pure NumPy — no JAX dependency for data generation)
# ══════════════════════════════════════════════════════════════════════════

def duffing_hamiltonian(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """H*(q, p) = p^2/2 + (q^2-1)^2/4.  Verified: H*(+-1,0)=0, H*(0,0)=0.25."""
    return 0.5 * p ** 2 + 0.25 * (q ** 2 - 1.0) ** 2


def duffing_grad_hamiltonian(q: np.ndarray, p: np.ndarray) -> tuple:
    """grad H*(q,p) = (q^3 - q, p)."""
    return q ** 3 - q, p


def duffing_vector_field(q, p, u, d, r=0.4, delta_r: float = 0.0, delta_k: float = 0.0):
    """(J-R) grad H* + G u + G d, with optional controlled plant mismatch.

    ``delta_r``: damping mismatch, plant uses ``r + delta_r``.
    ``delta_k``: cubic-stiffness mismatch, plant uses ``(1+delta_k) q^3``.
    Both default to 0, reproducing the exact nominal ground truth.
    """
    r_eff = r + delta_r
    dHq = q ** 3 - q if delta_k == 0.0 else (1.0 + delta_k) * q ** 3 - q
    qdot = p
    pdot = -dHq - r_eff * p + u + d
    return qdot, pdot


def _rk4_step(z: np.ndarray, u: float, d: float, dt: float, r: float,
              delta_r: float = 0.0, delta_k: float = 0.0) -> np.ndarray:
    def f(zz):
        qdot, pdot = duffing_vector_field(zz[0], zz[1], u, d, r=r, delta_r=delta_r, delta_k=delta_k)
        return np.array([qdot, pdot], dtype=np.float64)
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_duffing_trajectory(z0: np.ndarray, u_traj: np.ndarray, d_traj: Optional[np.ndarray] = None,
                                dt: float = 0.02, r: float = 0.4,
                                delta_r: float = 0.0, delta_k: float = 0.0) -> np.ndarray:
    """Zero-order-hold RK4 rollout.  Returns z_traj of shape [T+1, 2]; z_traj[0]=z0."""
    T = int(u_traj.shape[0])
    if d_traj is None:
        d_traj = np.zeros(T, dtype=np.float64)
    z = np.asarray(z0, dtype=np.float64).copy()
    out = np.empty((T + 1, 2), dtype=np.float64)
    out[0] = z
    for k in range(T):
        z = _rk4_step(z, float(u_traj[k]), float(d_traj[k]), dt, r, delta_r=delta_r, delta_k=delta_k)
        out[k + 1] = z
    return out


# ══════════════════════════════════════════════════════════════════════════
# 2. Forcing families
# ══════════════════════════════════════════════════════════════════════════

def forcing_piecewise_constant(rng: np.random.Generator, n: int, hold_steps: int, umax: float) -> np.ndarray:
    n_blocks = int(np.ceil(n / max(hold_steps, 1)))
    vals = rng.uniform(-umax, umax, size=n_blocks)
    return np.repeat(vals, hold_steps)[:n]


def forcing_prbs(rng: np.random.Generator, n: int, hold_steps: int, umax: float) -> np.ndarray:
    n_blocks = int(np.ceil(n / max(hold_steps, 1)))
    signs = rng.choice([-1.0, 1.0], size=n_blocks)
    return (umax * np.repeat(signs, hold_steps))[:n]


def forcing_multisine(rng: np.random.Generator, n: int, dt: float, umax: float,
                      n_tones: int = 3, f_max: float = 2.0) -> np.ndarray:
    t = np.arange(n) * dt
    freqs = rng.uniform(0.05, f_max, size=n_tones)
    phases = rng.uniform(0, 2 * np.pi, size=n_tones)
    sig = sum(np.sin(2 * np.pi * f * t + ph) for f, ph in zip(freqs, phases))
    sig = sig / max(np.max(np.abs(sig)), 1e-8)
    return umax * sig


def forcing_chirp(n: int, dt: float, f0: float, f1: float, umax: float) -> np.ndarray:
    t = np.arange(n) * dt
    Tspan = max(t[-1], 1e-6)
    inst_f = f0 + (f1 - f0) * (t / Tspan)
    phase = 2 * np.pi * np.cumsum(inst_f) * dt
    return umax * np.sin(phase)


def forcing_sinusoid(n: int, dt: float, freq: float, umax: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) * dt
    return umax * np.sin(2 * np.pi * freq * t + phase)


def forcing_step(n: int, umax: float, step_frac: float = 0.1) -> np.ndarray:
    out = np.zeros(n)
    out[int(step_frac * n):] = umax
    return out


def forcing_long_constant(n: int, umax: float) -> np.ndarray:
    return np.full(n, umax)


# ══════════════════════════════════════════════════════════════════════════
# 3. Dataset generation
# ══════════════════════════════════════════════════════════════════════════

TRAIN_FAMILIES = ("piecewise_constant", "prbs", "multisine")
OOD_FAMILIES = ("step", "chirp", "sinusoid_unseen", "long_constant")


def _sample_ic(rng: np.random.Generator, well: str, high_energy: bool) -> np.ndarray:
    if well == "left":
        q0 = rng.uniform(-1.4, -0.6)
    else:
        q0 = rng.uniform(0.6, 1.4)
    p0 = rng.uniform(-0.5, 0.5)
    if high_energy:
        p0 = rng.uniform(-1.2, 1.2)
    return np.array([q0, p0], dtype=np.float64)


def _make_train_forcing(rng: np.random.Generator, n: int, dt: float, high_energy: bool) -> tuple:
    family = rng.choice(TRAIN_FAMILIES)
    umax = 0.6 if high_energy else rng.uniform(0.15, 0.3)
    if family == "piecewise_constant":
        hold = int(rng.uniform(0.2, 0.8) / dt)
        u = forcing_piecewise_constant(rng, n, max(hold, 1), umax)
    elif family == "prbs":
        hold = int(rng.uniform(0.1, 0.4) / dt)
        u = forcing_prbs(rng, n, max(hold, 1), umax)
    else:
        u = forcing_multisine(rng, n, dt, umax, n_tones=rng.integers(2, 5))
    return u, family, umax


def generate_dataset(seed: int = 0, out_dir: str | Path = "results/duffing_doublewell/datasets",
                     n_train: int = 200, n_val: int = 40, n_test: int = 100,
                     T: float = 20.0, dt: float = 0.02, r: float = 0.4,
                     high_energy_frac: float = 0.2, save: bool = True) -> dict:
    """Generate train/val/nominal-test/OOD-test Duffing trajectories.

    Returns a dict of NumPy arrays plus metadata; also writes ``duffing_v1.npz``
    and a companion ``duffing_v1_meta.json`` (seed, generation config, per-
    trajectory family/well/energy labels, dataset SHA-256 hash) to ``out_dir``
    when ``save=True``.  The split is frozen once generated: OOD families are
    never used for training or model selection (see acceptance tests).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))

    def _build_split(n, high_energy_frac, tag):
        Z, U, meta = [], [], []
        for i in range(n):
            well = "left" if i % 2 == 0 else "right"
            high_energy = rng.random() < high_energy_frac
            z0 = _sample_ic(rng, well, high_energy)
            u, family, umax = _make_train_forcing(rng, n_steps, dt, high_energy)
            z_traj = simulate_duffing_trajectory(z0, u, dt=dt, r=r)
            Z.append(z_traj)
            U.append(u)
            meta.append(dict(well=well, high_energy=bool(high_energy), family=family, umax=float(umax)))
        return np.stack(Z).astype(np.float32), np.stack(U)[..., None].astype(np.float32), meta

    z_train, u_train, meta_train = _build_split(n_train, high_energy_frac, "train")
    z_val, u_val, meta_val = _build_split(n_val, high_energy_frac, "val")
    z_test, u_test, meta_test = _build_split(n_test, high_energy_frac, "test_nominal")

    # OOD test families: frozen, withheld from training/model-selection.
    ood = {}
    ood_meta = []
    n_per_family = max(n_test // len(OOD_FAMILIES), 4)
    for family in OOD_FAMILIES:
        Zf, Uf = [], []
        for i in range(n_per_family):
            well = "left" if i % 2 == 0 else "right"
            z0 = _sample_ic(rng, well, high_energy=False)
            if family == "step":
                u = forcing_step(n_steps, umax=rng.uniform(0.3, 0.7))
            elif family == "chirp":
                u = forcing_chirp(n_steps, dt, f0=0.05, f1=rng.uniform(2.0, 4.0), umax=rng.uniform(0.2, 0.5))
            elif family == "sinusoid_unseen":
                u = forcing_sinusoid(n_steps, dt, freq=rng.uniform(2.5, 5.0), umax=rng.uniform(0.2, 0.5))
            else:  # long_constant
                u = forcing_long_constant(n_steps, umax=rng.uniform(0.3, 0.6))
            z_traj = simulate_duffing_trajectory(z0, u, dt=dt, r=r)
            Zf.append(z_traj)
            Uf.append(u)
            ood_meta.append(dict(family=family, well=well))
        ood[f"z_{family}"] = np.stack(Zf).astype(np.float32)
        ood[f"u_{family}"] = np.stack(Uf)[..., None].astype(np.float32)

    data = dict(
        z_train=z_train, u_train=u_train,
        z_val=z_val, u_val=u_val,
        z_test=z_test, u_test=u_test,
        **ood,
    )
    gen_config = dict(seed=seed, n_train=n_train, n_val=n_val, n_test=n_test,
                      T=T, dt=dt, r=r, high_energy_frac=high_energy_frac,
                      train_families=list(TRAIN_FAMILIES), ood_families=list(OOD_FAMILIES))

    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "duffing_v1.npz"
        np.savez_compressed(npz_path, **data)
        dataset_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        meta_path = out_dir / "duffing_v1_meta.json"
        meta_path.write_text(json.dumps(dict(
            generation_config=gen_config,
            dataset_sha256=dataset_hash,
            meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
            meta_ood=ood_meta,
        ), indent=2))
        data["_npz_path"] = str(npz_path)
        data["_dataset_hash"] = dataset_hash
    data["_gen_config"] = gen_config
    return data


def load_duffing_doublewell(dataset_dir: str | Path = "results/duffing_doublewell/datasets") -> dict:
    """Load a previously generated dataset (or generate it with defaults if absent)."""
    out_dir = Path(dataset_dir)
    npz_path = out_dir / "duffing_v1.npz"
    meta_path = out_dir / "duffing_v1_meta.json"
    if not npz_path.exists():
        return generate_dataset(out_dir=out_dir)
    npz = np.load(npz_path)
    data = {k: npz[k] for k in npz.files}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        data["_gen_config"] = meta["generation_config"]
        data["_dataset_hash"] = meta["dataset_sha256"]
    data["_npz_path"] = str(npz_path)
    return data


# ══════════════════════════════════════════════════════════════════════════
# 4. Model configuration (prompt §10/§11) and windowed training data
# ══════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "duffing_doublewell"


def get_config() -> dict:
    return dict(
        experiment_name=EXPERIMENT_NAME,
        d=2, m_ports=1, n_obs=2,
        layer_dims=[64, 32],
        first_layer_type="tanh", p_1=2.0,
        second_layer_type="polynomial_stable", p_2=4.0,
        vf_scale=1.0, damping_scale=1.0,
        use_feedthrough=False, use_nonlinear_readout=False,
        use_saturating_readout=False, use_state_damping=False,
        use_input_gain=False, use_saturating_input=False,
        use_input_aware_encoder=False,
        rollout_integrator="rk4",
        # Training (staged per prompt §12; Stage-A/smoke defaults below,
        # override with Stage-B/final values via the config generator).
        batch_size=64, lr=3e-4, seq_len=250, stride=25,
        n_epochs=1000, stop_grad_epochs=20,
        grad_clip_norm=1.0, grad_norm_guard=1000.0,
        w_reg=1e-6, trunk_reg_mult=1.0, w_reg_B=0.0,
        observation_loss="huber", huber_delta=1.0,
        # Numerical safeguards deliberately loose (prompt §11): must stay
        # inactive throughout the certified region; verified by
        # Stress_tests.certificate_fidelity, not merely assumed.
        chol_clip_exp=1.5, grad_e_clip=200.0, xdot_clip=200.0,
        state_norm_clip=50.0, x0_norm_max=50.0,
        b_init_scale=1.0,
        print_every=10,
    )


DEFAULT_CONFIG = get_config()


def _build_layers(config: dict) -> tuple:
    first = LayerSpec(lagrangian_tanh, activation_tanh, (float(config["p_1"]),))
    second = LayerSpec(lagrangian_polynomial_stable, activation_polynomial_stable, (float(config["p_2"]),))
    return (first, second)


def make_windows(z_traj: np.ndarray, u_traj: np.ndarray, seq_len: int, stride: int) -> tuple:
    """Slice many independent trajectories into (x0, u_seg, z_seg_true) BPTT windows.

    ``x0`` is the TRUE physical state at the window start (no learned encoder).
    ``z_seg_true[k]`` is the true state AFTER applying ``u_seg[k]``, matching the
    indexing convention of ``EBM_rollout.make_rollout_fn`` (``x_traj[k]`` is the
    state following ``u_traj[k]``).
    """
    n_traj, T_plus_1, d = z_traj.shape
    T = T_plus_1 - 1
    x0s, u_segs, z_segs = [], [], []
    for i in range(n_traj):
        for s in range(0, T - seq_len + 1, stride):
            x0s.append(z_traj[i, s])
            u_segs.append(u_traj[i, s:s + seq_len])
            z_segs.append(z_traj[i, s + 1:s + 1 + seq_len])
    return (np.stack(x0s).astype(np.float32),
            np.stack(u_segs).astype(np.float32),
            np.stack(z_segs).astype(np.float32))


# ══════════════════════════════════════════════════════════════════════════
# 5. State-observed loss / train step (reuses EBM_training.make_train_step)
# ══════════════════════════════════════════════════════════════════════════

def make_state_observed_loss_fn(grad_energy_fn: Callable, d: int, m: int, dt: float,
                                integrator: str, w_reg: float, trunk_mult: float,
                                w_reg_B: float, observation_loss: str, huber_delta: float,
                                rollout_stop_grad: bool,
                                w_derivative: float = 0.0,
                                velocity_loss_weight: float = 1.0,
                                velocity_start_index: int | None = None,
                                fast_transition_loss_weight: float = 1.0,
                                normalized_increment_loss: bool = False,
                                rollout_increment_loss_weight: float = 0.0,
                                one_step_anchor_weight: float = 0.0,
                                coordinate_loss_weights: Optional[list[float]] = None,
                                metric_aligned_observation: bool = False,
                                state_mean: Optional[list[float]] = None,
                                state_std: Optional[list[float]] = None,
                                momentum_loss_weight: float = 0.0,
                                angular_shape_loss_weight: float = 0.0,
                                torque_scale: Optional[list[float]] = None,
                                omega_scale: Optional[list[float]] = None) -> Callable:
    batch_rollout = make_batch_rollout_fn(grad_energy_fn, d, m, dt, integrator=integrator)
    if coordinate_loss_weights is not None:
        if len(coordinate_loss_weights) != d:
            raise ValueError(
                f"coordinate_loss_weights must contain {d} values, got "
                f"{len(coordinate_loss_weights)}"
            )
        coordinate_weights = jnp.asarray(coordinate_loss_weights, dtype=jnp.float32)
        if bool(np.any(np.asarray(coordinate_loss_weights) <= 0.0)):
            raise ValueError("coordinate_loss_weights must be strictly positive")
    else:
        coordinate_weights = jnp.ones((d,))
    if coordinate_loss_weights is None and velocity_start_index is not None:
        coordinate_weights = coordinate_weights.at[int(velocity_start_index):].set(
            float(velocity_loss_weight)
        )
    coordinate_weights = coordinate_weights / jnp.mean(coordinate_weights)
    if metric_aligned_observation:
        if d != 12 or state_mean is None or state_std is None:
            raise ValueError("Metric-aligned observation requires 12-D state mean/std")
        physical_mean = jnp.asarray(state_mean)
        physical_std = jnp.asarray(state_std)
    if momentum_loss_weight > 0.0 or angular_shape_loss_weight > 0.0:
        if d != 12 or state_mean is None or state_std is None:
            raise ValueError("Rotational auxiliary losses require 12-D state mean/std")
        physical_mean = jnp.asarray(state_mean)
        physical_std = jnp.asarray(state_std)
    if momentum_loss_weight > 0.0:
        if torque_scale is None or len(torque_scale) != 3:
            raise ValueError("Momentum loss requires three torque scales")
        physical_torque_scale = jnp.asarray(torque_scale)
    if angular_shape_loss_weight > 0.0:
        if omega_scale is None or len(omega_scale) != 3:
            raise ValueError("Angular shape loss requires three omega scales")
        physical_omega_scale = jnp.asarray(omega_scale)
    inertia = jnp.asarray([2.3951e-5, 2.3951e-5, 3.2347e-5])

    def rotvec_quaternion(rotvec):
        angle = jnp.linalg.norm(rotvec, axis=-1, keepdims=True)
        scale = jnp.where(angle > 1e-8, jnp.sin(0.5 * angle) / angle, 0.5)
        return jnp.concatenate([rotvec * scale, jnp.cos(0.5 * angle)], axis=-1)

    def loss_fn(params, x0_batch, u_batch, z_batch_true):
        torque_batch = None
        auxiliary_ramp = jnp.asarray(1.0)
        if isinstance(z_batch_true, tuple):
            z_batch_true, torque_batch, auxiliary_ramp = z_batch_true
        x_trajs, _, _ = batch_rollout(params, x0_batch, u_batch, stop_grad=rollout_stop_grad)
        finite_rollout = tree_all_finite((x_trajs, x0_batch))
        x_trajs = jnp.nan_to_num(x_trajs, nan=0.0, posinf=0.0, neginf=0.0)
        diff = x_trajs - z_batch_true
        true_current = jnp.concatenate([x0_batch[:, None, :], z_batch_true[:, :-1, :]], axis=1)
        target_increments = z_batch_true - true_current
        transition_magnitude = jnp.linalg.norm(target_increments, axis=-1)
        relative_magnitude = transition_magnitude / (
            jnp.mean(transition_magnitude) + 1e-8
        )
        transition_weights = 1.0 + (float(fast_transition_loss_weight) - 1.0) * jnp.clip(
            relative_magnitude / 3.0, 0.0, 1.0
        )
        transition_weights = transition_weights / jnp.mean(transition_weights)
        if metric_aligned_observation:
            channel_rmse = jnp.sqrt(jnp.mean(diff ** 2, axis=(0, 1)) + 1e-8)
            true_rotvec = z_batch_true[..., 6:9] * physical_std[6:9] + physical_mean[6:9]
            predicted_rotvec = x_trajs[..., 6:9] * physical_std[6:9] + physical_mean[6:9]
            true_quaternion = rotvec_quaternion(true_rotvec)
            predicted_quaternion = rotvec_quaternion(predicted_rotvec)
            quaternion_dot = jnp.sum(true_quaternion * predicted_quaternion, axis=-1)
            aligned_prediction = jnp.where(
                (quaternion_dot < 0.0)[..., None],
                -predicted_quaternion,
                predicted_quaternion,
            )
            quaternion_chord = jnp.sqrt(
                jnp.sum((aligned_prediction - true_quaternion) ** 2, axis=-1) + 1e-8
            )
            rotation_error = 2.0 * quaternion_chord
            rotation_scale = jnp.sqrt(jnp.mean(physical_std[6:9] ** 2))
            rotation_nrmse = jnp.sqrt(jnp.mean(rotation_error ** 2) + 1e-8) / rotation_scale
            metric_channels = jnp.concatenate([
                channel_rmse[:6], jnp.repeat(rotation_nrmse, 3), channel_rmse[9:12]
            ])
            obs_loss = jnp.mean(metric_channels)
        elif observation_loss == "mse":
            per_coord = diff ** 2
            obs_loss = jnp.mean(
                per_coord * coordinate_weights * transition_weights[..., None]
            )
        else:
            per_coord = optax.losses.huber_loss(diff, jnp.zeros_like(diff), delta=huber_delta)
            obs_loss = jnp.mean(
                per_coord * coordinate_weights * transition_weights[..., None]
            )
        error_squared_sum = jnp.sum(diff ** 2)
        target_sum = jnp.sum(z_batch_true)
        target_squared_sum = jnp.sum(z_batch_true ** 2)
        sample_count = jnp.asarray(z_batch_true.size, dtype=diff.dtype)
        normalized_rmse = jnp.sqrt(error_squared_sum / sample_count)
        target_variance = jnp.maximum(
            target_squared_sum / sample_count - (target_sum / sample_count) ** 2,
            1e-12,
        )
        normalized_nrmse = normalized_rmse / jnp.sqrt(target_variance)
        def trajectory_increments(states, inputs):
            def one(state, control):
                field, _, _ = pf.vector_field_and_output(
                    params, state, control, grad_energy_fn, d, m
                )
                return dt * field
            return jax.vmap(one)(states, inputs)

        predicted_increments = jax.vmap(trajectory_increments)(true_current, u_batch)
        increment_error = predicted_increments - target_increments
        predicted_current = jnp.concatenate(
            [x0_batch[:, None, :], x_trajs[:, :-1, :]], axis=1
        )
        rollout_increment_error = (
            x_trajs - predicted_current - target_increments
        )
        if normalized_increment_loss:
            increment_scale = jnp.sqrt(
                jnp.mean(target_increments ** 2, axis=(0, 1)) + 1e-8
            )
            increment_scale = jnp.maximum(
                increment_scale, 0.1 * jnp.mean(increment_scale)
            )
            increment_error = increment_error / increment_scale
            rollout_increment_error = rollout_increment_error / increment_scale
        derivative_loss = jnp.mean(
            increment_error ** 2
            * coordinate_weights * transition_weights[..., None]
        )
        rollout_increment_loss = jnp.mean(
            rollout_increment_error ** 2
            * coordinate_weights * transition_weights[..., None]
        )
        momentum_loss = jnp.asarray(0.0)
        if momentum_loss_weight > 0.0:
            def implied_torque(state, control):
                field, _, _ = pf.vector_field_and_output(
                    params, state, control, grad_energy_fn, d, m
                )
                omega = state[9:12] * physical_std[9:12] + physical_mean[9:12]
                omega_dot = field[9:12] * physical_std[9:12]
                angular_momentum = inertia * omega
                return inertia * omega_dot + jnp.cross(omega, angular_momentum)

            teacher_torque = jax.vmap(jax.vmap(implied_torque))(true_current, u_batch)
            teacher_residual = (
                teacher_torque - torque_batch[..., :3]
            ) / physical_torque_scale
            teacher_mask = torque_batch[..., 3:4]
            teacher_huber = optax.losses.huber_loss(
                teacher_residual, jnp.zeros_like(teacher_residual), delta=1.0
            )
            teacher_loss = jnp.sum(teacher_huber * teacher_mask) / jnp.maximum(
                3.0 * jnp.sum(teacher_mask), 1.0
            )
            rollout_torque = jax.vmap(jax.vmap(implied_torque))(
                x_trajs[:, :10], u_batch[:, 1:11]
            )
            rollout_target = torque_batch[:, 1:11]
            rollout_residual = (
                rollout_torque - rollout_target[..., :3]
            ) / physical_torque_scale
            rollout_mask = rollout_target[..., 3:4]
            rollout_huber = optax.losses.huber_loss(
                rollout_residual, jnp.zeros_like(rollout_residual), delta=1.0
            )
            rollout_loss = jnp.sum(rollout_huber * rollout_mask) / jnp.maximum(
                3.0 * jnp.sum(rollout_mask), 1.0
            )
            momentum_loss = 0.7 * teacher_loss + 0.3 * rollout_loss

        angular_shape_loss = jnp.asarray(0.0)
        if angular_shape_loss_weight > 0.0:
            true_omega = (
                z_batch_true[:, :10, 9:12] * physical_std[9:12]
                + physical_mean[9:12]
            )
            predicted_omega = (
                x_trajs[:, :10, 9:12] * physical_std[9:12]
                + physical_mean[9:12]
            )
            sign_mask = omega_sign_mask(true_omega, physical_omega_scale).astype(true_omega.dtype)
            sign_loss_values = optax.sigmoid_binary_cross_entropy(
                predicted_omega / (0.15 * physical_omega_scale),
                (true_omega > 0.0).astype(true_omega.dtype),
            )
            sign_loss = jnp.sum(sign_loss_values * sign_mask) / jnp.maximum(
                jnp.sum(sign_mask), 1.0
            )
            true_mean = jnp.mean(true_omega, axis=(0, 1))
            predicted_mean = jnp.mean(predicted_omega, axis=(0, 1))
            bias_loss = jnp.sum(((predicted_mean - true_mean) / physical_omega_scale) ** 2)
            true_centered = true_omega - true_mean
            predicted_centered = predicted_omega - predicted_mean
            true_rms = jnp.sqrt(jnp.mean(true_centered ** 2, axis=(0, 1)) + 1e-12)
            predicted_rms = jnp.sqrt(jnp.mean(predicted_centered ** 2, axis=(0, 1)) + 1e-12)
            log_gain = jnp.clip(
                jnp.log((predicted_rms + 1e-8) / (true_rms + 1e-8)),
                -jnp.log(4.0), jnp.log(4.0),
            )
            gain_loss = jnp.sum(log_gain ** 2)
            angular_shape_loss = 0.5 * sign_loss + 0.25 * bias_loss + 0.25 * gain_loss
        reg_loss = weight_regularisation(params, w_reg, trunk_mult, w_reg_B=w_reg_B, include_output_heads=False)
        obs_loss = jnp.where(jnp.isfinite(obs_loss), obs_loss, jnp.array(1e3))
        derivative_loss = jnp.where(jnp.isfinite(derivative_loss), derivative_loss, jnp.array(1e3))
        rollout_increment_loss = jnp.where(
            jnp.isfinite(rollout_increment_loss), rollout_increment_loss, jnp.array(1e3)
        )
        reg_loss = jnp.where(jnp.isfinite(reg_loss), reg_loss, jnp.array(1e3))
        finite_penalty = jnp.where(finite_rollout, 0.0, 1e2)
        derivative_weight = float(w_derivative) + float(one_step_anchor_weight)
        total = (
            obs_loss
            + derivative_weight * derivative_loss
            + float(rollout_increment_loss_weight) * rollout_increment_loss
            + auxiliary_ramp * float(momentum_loss_weight) * momentum_loss
            + auxiliary_ramp * float(angular_shape_loss_weight) * angular_shape_loss
            + reg_loss
            + finite_penalty
        )
        aux = dict(
            obs_loss=obs_loss,
            derivative_loss=derivative_loss,
            rollout_increment_loss=rollout_increment_loss,
            momentum_loss=momentum_loss,
            angular_shape_loss=angular_shape_loss,
            reg_loss=reg_loss,
            finite_rollout=finite_rollout,
            normalized_rmse=normalized_rmse,
            normalized_nrmse=normalized_nrmse,
            error_squared_sum=error_squared_sum,
            target_sum=target_sum,
            target_squared_sum=target_squared_sum,
            sample_count=sample_count,
        )
        return total, aux
    return loss_fn


def _rmse_nrmse(pred: np.ndarray, true: np.ndarray) -> tuple:
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    nrmse = rmse / max(float(np.std(true)), 1e-12)
    return rmse, nrmse


# ══════════════════════════════════════════════════════════════════════════
# 6. Training entry point
# ══════════════════════════════════════════════════════════════════════════

def run_training(seed: int = 0, config: Optional[dict] = None,
                 dataset_dir: str | Path = "results/duffing_doublewell/datasets",
                 verbose: bool = True) -> tuple:
    """State-observed training loop.  Returns (params, layers, grad_E, meta).

    ``meta`` contains ``dt, r, dataset_hash, stats(identity), val_rmse,
    val_nrmse, config`` — enough to reconstruct any downstream certificate
    computation, mirroring what ``Silverbox_main.run_training`` returns
    (``params, layers, grad_E, stats, controller_info``) but for the
    gray-box/state-observed mode (``controller_info`` is always ``None``:
    ``enable_controller=False`` per prompt §27).
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    if cfg.get("experiment_name", EXPERIMENT_NAME) != EXPERIMENT_NAME:
        raise ValueError(
            f"Duffing_DoubleWell_main.run_training received a config with "
            f"experiment_name={cfg.get('experiment_name')!r}, expected {EXPERIMENT_NAME!r}. "
            "This config was generated for a different experiment; refusing to train "
            "(see DEBUGGING_REPORT.md for the routing bug this guards against)."
        )

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

    data = load_duffing_doublewell(dataset_dir)
    dt = float(data["_gen_config"]["dt"])
    r = float(data["_gen_config"]["r"])

    D, M = int(cfg["d"]), int(cfg["m_ports"])
    x0_tr, u_tr, z_tr = make_windows(data["z_train"], data["u_train"], int(cfg["seq_len"]), int(cfg["stride"]))
    x0_va, u_va, z_va = make_windows(data["z_val"], data["u_val"], int(cfg["seq_len"]), int(cfg["stride"]))

    # Shape trace (prompt §4): the physical Duffing state is always 2-D, so a
    # mismatched `d` (e.g. from an accidentally-merged foreign config) must
    # fail HERE with an actionable message, not deep inside a reshape() call.
    assert x0_tr.ndim == 2 and x0_tr.shape[-1] == D, (
        f"x0_tr shape {x0_tr.shape} does not match d={D}. The Duffing dataset's true "
        f"state dimension is 2; if D!=2 here, `d` was overridden by a config that does "
        f"not belong to this experiment."
    )
    assert u_tr.shape[-1] == M, f"u_tr shape {u_tr.shape} last dim != m_ports={M}"
    assert z_tr.shape == u_tr.shape[:-1] + (D,), (
        f"z_tr shape {z_tr.shape} inconsistent with u_tr shape {u_tr.shape} and d={D}"
    )

    layers = _build_layers(cfg)
    grad_E = make_grad_energy(layers)
    params = init_params(jax.random.PRNGKey(seed), d=D, m=M, n_obs=int(cfg["n_obs"]),
                         init_win=1, layer_dims=list(cfg["layer_dims"]), dt=dt)

    n_batches = max(x0_tr.shape[0] // int(cfg["batch_size"]), 1)
    optimizer = optax.chain(optax.clip_by_global_norm(float(cfg["grad_clip_norm"])), optax.adamw(float(cfg["lr"])))
    opt_state = optimizer.init(params)

    loss_kwargs = dict(grad_energy_fn=grad_E, d=D, m=M, dt=dt,
                       integrator=str(cfg["rollout_integrator"]),
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
            print(f"[duffing] epoch {epoch:4d}  loss={last_loss:.6f}")

    # Validation rollout (long-horizon, full validation trajectories).
    val_rollout = make_batch_rollout_fn(grad_E, D, M, dt, integrator=str(cfg["rollout_integrator"]))
    x0_full = jnp.array(data["z_val"][:, 0, :].astype(np.float32))
    u_full = jnp.array(data["u_val"].astype(np.float32))
    x_pred, _, _ = val_rollout(params, x0_full, u_full, stop_grad=True)
    z_true_full = data["z_val"][:, 1:, :]
    assert x_pred.shape == z_true_full.shape, (
        f"Prediction/target mismatch: pred={x_pred.shape}, target={z_true_full.shape}"
    )
    rmse, nrmse = _rmse_nrmse(np.asarray(x_pred), z_true_full)
    if verbose:
        print(f"[duffing] validation long-horizon RMSE={rmse:.4f}  NRMSE={nrmse:.4f}")

    meta = dict(dt=dt, r=r, stats=dict(u_mu=0.0, u_std=1.0, y_mu=0.0, y_std=1.0),
               dataset_hash=data.get("_dataset_hash"), val_rmse=rmse, val_nrmse=nrmse,
               config=cfg, seed=seed, data=data)
    return params, layers, grad_E, meta


def _make_duffing_stress_config(config: dict, seed: int):
    from Stress_tests.config import ProjectionConfig, StressTestConfig

    mode = str(config.get("stress_test_mode", "theory_aligned_only"))
    if mode != "theory_aligned_only":
        raise ValueError(
            "Duffing supports only stress_test_mode='theory_aligned_only'; "
            f"got {mode!r}."
        )
    alpha = float(config.get("stress_relative_energy_alpha", 0.8))
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"stress_relative_energy_alpha must lie in (0, 1), got {alpha}")
    return alpha, StressTestConfig(
        seed=seed,
        run_non_theoretical_diagnostics=False,
        relative_energy_alpha=alpha,
        projection=ProjectionConfig(minima_grad_norm_tol=0.05),
    )


def train(seed: int = 0, config: Optional[dict] = None) -> tuple:
    """Uniform compute environment-pipeline entry point (matches every other Interface_code
    module's ``train(seed, config)`` signature so this experiment can be
    queued through the existing offline W&B pipeline, see
    ``train_from_config.py``). After training, runs the full
    dataset->training->reconstruction->Stress_tests->plots->audit pipeline
    (prompt §11/§24) via ``post_training_eval.run_post_training_evaluation``,
    with experiment-specific phase-portrait/Hamiltonian-overlay plots.
    """
    params, layers, grad_E, meta = run_training(seed=seed, config=config)
    cfg = meta["config"]
    data = meta["data"]
    dt = meta["dt"]
    D, M = int(cfg["d"]), int(cfg["m_ports"])
    run_id = str(cfg.get("wandb_source_run_id") or f"seed{seed}")

    from experiment_paths import get_experiment_paths
    from post_training_eval import run_post_training_evaluation, save_checkpoint, save_json
    paths = get_experiment_paths(EXPERIMENT_NAME, run_id)
    save_json(paths.config_dir / "config.json", **cfg)
    save_checkpoint(paths.checkpoint_dir / "params.pkl", params)

    # Held-out TEST reconstruction (val was only used above for model
    # selection, per prompt §14 -- final reported metrics use the test split).
    val_rollout = make_batch_rollout_fn(grad_E, D, M, dt, integrator=str(cfg["rollout_integrator"]))
    x0_test = jnp.array(data["z_test"][:, 0, :].astype(np.float32))
    u_test = jnp.array(data["u_test"].astype(np.float32))
    x_pred_test, _, _ = val_rollout(params, x0_test, u_test, stop_grad=True)
    z_true_test = data["z_test"][:, 1:, :]
    x_pred_test_np = np.asarray(x_pred_test)

    from Stress_tests.integration import apply_runtime_config
    from Stress_tests.model_adapter import EBMStressAdapter
    from Stress_tests.experiments.duffing_doublewell import (
        alpha_to_epsilon, discover_wells_and_saddle, plot_phase_portrait,
    )
    apply_runtime_config(cfg)
    adapter = EBMStressAdapter(params, layers, d=D, m=M, dt=dt)

    reference_states = x_pred_test_np.reshape(-1, D)
    reference_inputs = np.asarray(u_test).reshape(-1, M)

    discovery = discover_wells_and_saddle(adapter, reference_states=reference_states, seed=seed)
    minima, saddle = discovery["minima"], discovery["saddle"]
    H_min = float(minima.energies[0])
    H_saddle = float(saddle["energy"]) if saddle is not None else H_min + 1.0
    gamma = 1.0

    stress_alpha, stress_config = _make_duffing_stress_config(cfg, seed)
    epsilon = alpha_to_epsilon(stress_alpha, H_min, H_saddle)

    def extra_plots(paths_):
        z_true_traj = data["z_test"][0, 1:, :]
        z_pred_traj = x_pred_test_np[0]
        plot_phase_portrait(
            z_true_traj, z_pred_traj, adapter, minima, saddle,
            out_path=str(paths_.hamiltonian_dir / "phase_portrait.png"),
            true_hamiltonian_fn=lambda Q, P: duffing_hamiltonian(Q, P), epsilon=epsilon,
        )

    result = run_post_training_evaluation(
        experiment_name=EXPERIMENT_NAME, run_id=run_id,
        params=params, layers=layers, grad_E=grad_E, d=D, m=M, dt=dt,
        test_true=z_true_test, test_pred=x_pred_test_np, test_u=np.asarray(u_test),
        unit_label="", reference_states=reference_states, reference_inputs=reference_inputs,
        epsilon=epsilon, gamma=gamma, paths=paths, extra_plot_fn=extra_plots,
        stress_config=stress_config,
        seed=seed,
    )

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"val/rmse": meta["val_rmse"], "val/nrmse": meta["val_nrmse"],
                      "test/rmse": result["metrics"]["reconstruction"]["rmse"],
                      "test/nrmse": result["metrics"]["reconstruction"]["nrmse"]})
            wandb.summary["best_test_nrmse"] = meta["val_nrmse"]
    except Exception as e:
        print(f"wandb.log() failed: {e}")
    return params, layers, grad_E, meta


if __name__ == "__main__":
    generate_dataset()
    run_training(seed=0, config=dict(n_epochs=5, seq_len=50, batch_size=16, stop_grad_epochs=2, print_every=1))
