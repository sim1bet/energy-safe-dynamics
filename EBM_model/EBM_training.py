# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

# EBM_training.py — Loss functions and training utilities
# v2 update: adds author-style supervised losses (mse/huber) as first-class options.

import jax
import jax.numpy as jnp
import optax
from typing import Callable

from EBM_param_fields import EBMParams, encode_x0_batch
from EBM_rollout import make_batch_rollout_fn
from EBM_class import energy_EBM


def magnitude_weights(y_true: jax.Array, eps: float = 1e-6) -> jax.Array:
    mag_sq = jnp.sum(y_true ** 2, axis=-1)
    return mag_sq / (mag_sq + eps)


def tree_all_finite(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == 0:
        return jnp.array(True)
    checks = [jnp.all(jnp.isfinite(x)) for x in leaves]
    return jnp.all(jnp.stack(checks))


def compute_gradient_norm(grads) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(grads)
    if len(leaves) == 0:
        return jnp.array(0.0)
    norm_sq = sum(jnp.sum(jnp.where(jnp.isfinite(g), g ** 2, 0.0)) for g in leaves)
    return jnp.sqrt(norm_sq)


def weight_regularisation(params: EBMParams,
                          reg: float,
                          trunk_mult: float = 1.0,
                          w_reg_B: float = 0.0,
                          include_output_heads: bool = False) -> jax.Array:
    ebm_leaves = list(params.ebm_weights)
    # L_M and e_A use the standard trunk_mult rate
    trunk_leaves = [
        params.trunk.L_M,
        params.trunk.e_A,
    ]
    if include_output_heads:
        trunk_leaves.extend([
            params.trunk.C_state, params.trunk.c_bias,
            params.trunk.W_enc, params.trunk.b_enc,
        ])
    def _mean_sq(leaves):
        if not leaves:
            return jnp.array(0.0)
        vals = jnp.concatenate([jnp.ravel(jnp.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) ** 2) for w in leaves])
        return jnp.mean(vals)
    # B gets its own, independently-tunable regularisation rate to prevent runaway growth
    B_reg = w_reg_B * _mean_sq([params.trunk.B])
    return reg * _mean_sq(ebm_leaves) + trunk_mult * reg * _mean_sq(trunk_leaves) + B_reg


def energy_consistency_loss(params: EBMParams,
                            x_traj: jax.Array,
                            u_traj: jax.Array,
                            y_port: jax.Array,
                            layers: tuple,
                            dt: float) -> jax.Array:
    def E_at(x):
        return energy_EBM(x, params.ebm_weights, params.ebm_biases, layers)
    x_traj = jnp.nan_to_num(x_traj, nan=0.0, posinf=0.0, neginf=0.0)
    E_traj = jax.vmap(E_at)(x_traj)
    E_traj = jnp.nan_to_num(E_traj, nan=0.0, posinf=0.0, neginf=0.0)
    dE_dt = jnp.diff(E_traj) / jnp.maximum(dt, 1e-8)
    dE_dt = jnp.clip(jnp.nan_to_num(dE_dt, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
    supply_rate = jnp.sum(jnp.nan_to_num(y_port[:-1]) * jnp.nan_to_num(u_traj[:-1]), axis=-1)
    residual = jax.nn.relu(dE_dt - supply_rate)
    return jnp.mean(residual ** 2)


def _author_style_supervised_loss(y_pred: jax.Array,
                                  y_true: jax.Array,
                                  observation_loss: str = 'huber',
                                  weight_obs_by_magnitude: bool = False,
                                  huber_delta: float = 1.0,
                                  early_window_steps: int = 0,
                                  early_window_weight: float = 1.0,
                                  amplitude_weight_power: float = 0.0,
                                  rise_weight_power: float = 0.0,
                                  mse_blend: float = 0.0,
                                  sample_weights: jax.Array = None,
                                  time_mask: jax.Array = None):
    diff = y_pred - y_true
    if observation_loss == 'mse':
        per_coord = diff ** 2
    elif observation_loss == 'huber':
        per_coord = optax.losses.huber_loss(diff, jnp.zeros_like(diff), delta=huber_delta)
    elif observation_loss == 'huber_to_mse':
        huber = optax.losses.huber_loss(diff, jnp.zeros_like(diff), delta=huber_delta)
        per_coord = (1.0 - mse_blend) * huber + mse_blend * diff ** 2
    else:
        raise ValueError(f'Unknown observation_loss={observation_loss}')
    per_t = jnp.mean(per_coord, axis=-1)
    # Early-window upweighting: directly targets the initial-transient
    # tracking bottleneck by biasing gradient signal toward the first
    # early_window_steps steps of each training rollout window (default
    # early_window_steps=0 is an exact no-op, back-compatible).
    if early_window_steps > 0 and early_window_weight != 1.0:
        T = per_t.shape[-1]
        step_idx = jnp.arange(T)
        time_w = jnp.where(step_idx < early_window_steps, early_window_weight, 1.0)
        per_t = per_t * time_w
    # Amplitude-weighted reweighting: unlike magnitude_weights (a SATURATING
    # ratio mag_sq/(mag_sq+eps) that is already ~1 almost everywhere on unit-
    # variance-normalised data, so it barely emphasises anything), this scales
    # the per-step weight LINEARLY with |y_true| (never saturates) so training
    # gradient signal is biased toward getting large-amplitude excursions
    # (peaks/troughs/overflow events) right, not just the bulk of the trace.
    # Mean-preserving (divides by mean(amp_w)) so the overall loss magnitude -
    # and therefore its balance against w_state/trunk_reg/passivity - stays on
    # the same scale as amplitude_weight_power=0.0 (exact no-op, back-compatible).
    #
    # NEGATIVE amplitude_weight_power (Jul22, ced_v8 follow-up): the ced_v8
    # batch found upweighting large amplitude (power>0, b1/b2 above) net-
    # HARMFUL for CED - the plots showed the model already tracks LARGE
    # excursions accurately and under-fits the small-amplitude bulk of the
    # trace instead, so upweighting large amplitude further only worsens that
    # imbalance. A negative power tests the OPPOSITE hypothesis: downweight
    # large-amplitude points (bounded in (0,1], never explodes near y=0),
    # relatively rebalancing gradient signal toward the small-amplitude
    # region. Same mean-preserving normalisation either way.
    if amplitude_weight_power > 0.0:
        amp_w = 1.0 + amplitude_weight_power * jnp.sqrt(jnp.sum(y_true ** 2, axis=-1) + 1e-12)
        per_t = amp_w * per_t / jnp.mean(amp_w)
    elif amplitude_weight_power < 0.0:
        amp_w = 1.0 / (1.0 + (-amplitude_weight_power) * jnp.sqrt(jnp.sum(y_true ** 2, axis=-1) + 1e-12))
        per_t = amp_w * per_t / jnp.mean(amp_w)
    # Directional (rise-vs-fall) reweighting (Jul22, cascaded_tanks v17 follow-
    # up): targets the NEW asymmetry diagnosis - reconstruction errors were
    # found to concentrate at DOWNWARD-UPWARD inversions (troughs), with the
    # subsequent RISE out of the trough lagging/undershooting the next peak,
    # while UPWARD-DOWNWARD (peak->fall) transitions track fine. Classifies
    # each step of the TRUE trajectory as rising or falling from its own local
    # slope (mean across output channels, since per_t already collapsed the
    # channel axis) and up/down-weights accordingly. Positive rise_weight_power
    # upweights RISING steps (tests "give the recovery-from-trough more
    # gradient signal"); negative upweights FALLING steps (control/contrast).
    # Mean-preserving (like amplitude_weight_power) so overall loss scale is
    # unaffected at power=0.0 (exact no-op, back-compatible). The first step of
    # each window repeats the second step's slope (no t-1 sample available).
    if rise_weight_power != 0.0:
        y_true_ch_mean = jnp.mean(y_true, axis=-1)  # [..., T]
        dy = jnp.diff(y_true_ch_mean, axis=-1)  # [..., T-1]
        dy = jnp.concatenate([dy[..., :1], dy], axis=-1)  # [..., T], pad front
        rising = dy > 0.0
        if rise_weight_power > 0.0:
            rise_w = jnp.where(rising, 1.0 + rise_weight_power, 1.0)
        else:
            rise_w = jnp.where(rising, 1.0, 1.0 + (-rise_weight_power))
        per_t = rise_w * per_t / jnp.mean(rise_w)
    if weight_obs_by_magnitude:
        w = magnitude_weights(y_true)
        per_t = w * per_t
    if time_mask is None:
        per_sample = jnp.mean(per_t, axis=-1)
    else:
        mask = jnp.asarray(time_mask, dtype=per_t.dtype)
        per_sample = jnp.sum(per_t * mask, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
    if sample_weights is not None:
        per_sample = per_sample * jnp.asarray(sample_weights, dtype=per_sample.dtype)
    return jnp.mean(per_sample)


def make_loss_fn(grad_energy_fn: Callable,
                 layers: tuple,
                 d: int,
                 m: int,
                 dt: float,
                 w_rollout: float = 1.0,
                 w_passivity: float = 0.0,
                 w_reg: float = 1e-6,
                 trunk_mult: float = 1.0,
                 w_reg_B: float = 0.0,
                 w_state: float = 0.0,
                 rollout_stop_grad: bool = False,
                 rollout_integrator: str = 'rk4',
                 observation_loss: str = 'huber',
                 weight_obs_by_magnitude: bool = False,
                 huber_delta: float = 1.0,
                 early_window_steps: int = 0,
                 early_window_weight: float = 1.0,
                 amplitude_weight_power: float = 0.0,
                 rise_weight_power: float = 0.0,
                 rollout_substeps: int = 1,
                 w_free_map: float = 0.0,
                 w_sensor_curvature: float = 0.0,
                 w_parameter_prior: float = 0.0):
    if any(float(v) != 0.0 for v in (w_free_map, w_sensor_curvature, w_parameter_prior)):
        raise ValueError("feature-map regularizers require the inactive experimental tank schema")
    batch_rollout = make_batch_rollout_fn(
        grad_energy_fn, d, m, dt, integrator=rollout_integrator, substeps=rollout_substeps)

    def loss_fn(params: EBMParams, y_win_batch: jax.Array, u_batch: jax.Array,
                y_batch_true: jax.Array, mse_blend: float = 0.0,
                sample_weights: jax.Array = None, time_mask: jax.Array = None):
        x0_batch = encode_x0_batch(params, y_win_batch)
        x_trajs, y_port_trajs, y_obs_trajs = batch_rollout(params, x0_batch, u_batch, stop_grad=rollout_stop_grad)
        finite_rollout = tree_all_finite((x_trajs, y_port_trajs, y_obs_trajs, x0_batch))
        x_trajs = jnp.nan_to_num(x_trajs, nan=0.0, posinf=0.0, neginf=0.0)
        y_port_trajs = jnp.nan_to_num(y_port_trajs, nan=0.0, posinf=0.0, neginf=0.0)
        y_obs_trajs = jnp.nan_to_num(y_obs_trajs, nan=0.0, posinf=0.0, neginf=0.0)

        obs_loss = _author_style_supervised_loss(
            y_obs_trajs, y_batch_true,
            observation_loss=observation_loss,
            weight_obs_by_magnitude=weight_obs_by_magnitude,
            huber_delta=huber_delta,
            early_window_steps=early_window_steps,
            early_window_weight=early_window_weight,
            amplitude_weight_power=amplitude_weight_power,
            rise_weight_power=rise_weight_power,
            mse_blend=mse_blend,
            sample_weights=sample_weights,
            time_mask=time_mask,
        )
        if w_passivity > 0.0:
            pass_loss = jnp.mean(jax.vmap(lambda x_t, u_t, y_p: energy_consistency_loss(params, x_t, u_t, y_p, layers, dt))(x_trajs, u_batch, y_port_trajs))
        else:
            pass_loss = jnp.array(0.0)
        # state trajectory norm penalty: keeps x bounded over long rollouts
        if w_state > 0.0:
            state_loss = jnp.mean(jnp.sum(x_trajs ** 2, axis=-1))
            state_loss = jnp.where(jnp.isfinite(state_loss), state_loss, jnp.array(1e3))
        else:
            state_loss = jnp.array(0.0)
        reg_loss = weight_regularisation(params, w_reg, trunk_mult, w_reg_B=w_reg_B, include_output_heads=False)
        obs_loss = jnp.where(jnp.isfinite(obs_loss), obs_loss, jnp.array(1e3))
        pass_loss = jnp.where(jnp.isfinite(pass_loss), pass_loss, jnp.array(1e3))
        reg_loss = jnp.where(jnp.isfinite(reg_loss), reg_loss, jnp.array(1e3))
        finite_penalty = jnp.where(finite_rollout, 0.0, 1e2)
        total = w_rollout * obs_loss + w_passivity * pass_loss + reg_loss + w_state * state_loss + finite_penalty
        aux = {
            'obs_loss': obs_loss,
            'passivity_loss': pass_loss,
            'reg_loss': reg_loss,
            'state_loss': state_loss,
            'finite_rollout': finite_rollout,
            'finite_penalty': finite_penalty,
            'free_map_loss': jnp.array(0.0),
            'sensor_curvature_loss': jnp.array(0.0),
            'parameter_prior_loss': jnp.array(0.0),
        }
        return total, aux
    return loss_fn


def make_train_step(loss_fn: Callable,
                    optimizer: optax.GradientTransformation,
                    grad_norm_guard: float = 100.0,
                    gradient_clip_norm: float | None = None):
    def group_norm(tree):
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if leaf is not None]
        return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))

    def parameter_group_norms(grads):
        trunk = grads.trunk._asdict()
        groups = {
            'H': (grads.ebm_weights, grads.ebm_biases),
            'J': tuple(trunk[name] for name in (
                'e_A', 'G_A', 'h_A', 'G_A_quadratic', 'G_A_cubic', 'W_rich_A'
            )),
            'R': tuple(trunk[name] for name in (
                'L_M', 'G_damp', 'h_damp', 'W_rich_M'
            )),
            'G': tuple(trunk[name] for name in (
                'B', 'w_gain', 'b_gain', 'G_gain', 'h_gain', 'u_sat_raw', 'W_rich_B'
            )),
            'shared_field': tuple(trunk[name] for name in (
                'W_field1', 'b_field1', 'W_field2', 'b_field2'
            )),
        }
        return {name: group_norm(group) for name, group in groups.items()}

    @jax.jit
    def train_step(params, opt_state, y_win_b, u_b, y_b, mse_blend=0.0,
                   sample_weights=None, time_mask=None):
        (loss, aux), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
            params, y_win_b, u_b, y_b, mse_blend, sample_weights, time_mask)
        grad_norm = compute_gradient_norm(grads)
        group_norms = parameter_group_norms(grads)
        clipped_grad_norm = (
            jnp.minimum(grad_norm, float(gradient_clip_norm))
            if gradient_clip_norm is not None else grad_norm
        )
        finite_grad = tree_all_finite(grads)
        finite_loss = jnp.isfinite(loss)
        can_try = finite_grad & finite_loss & (grad_norm <= grad_norm_guard)

        def do_update(args):
            params, opt_state, grads = args
            updates, opt_state2 = optimizer.update(grads, opt_state, params)
            finite_updates = tree_all_finite(updates)
            def apply(_):
                params2 = optax.apply_updates(params, updates)
                finite_params = tree_all_finite(params2)
                return jax.lax.cond(
                    finite_params,
                    lambda __: (params2, opt_state2, jnp.array(True), finite_updates, finite_params),
                    lambda __: (params, opt_state, jnp.array(False), finite_updates, finite_params),
                    operand=None,
                )
            return jax.lax.cond(
                finite_updates,
                apply,
                lambda _: (params, opt_state, jnp.array(False), finite_updates, jnp.array(True)),
                operand=None,
            )

        params_out, opt_out, update_applied, finite_updates, finite_params = jax.lax.cond(
            can_try,
            do_update,
            lambda args: (args[0], args[1], jnp.array(False), jnp.array(False), tree_all_finite(args[0])),
            operand=(params, opt_state, grads),
        )
        stats = {
            'grad_norm': grad_norm,
            'grad_norm_pre_clip': grad_norm,
            'grad_norm_post_clip': clipped_grad_norm,
            'gradient_clipped': (
                grad_norm > float(gradient_clip_norm)
                if gradient_clip_norm is not None else jnp.array(False)
            ),
            'grad_norm_H': group_norms['H'],
            'grad_norm_J': group_norms['J'],
            'grad_norm_R': group_norms['R'],
            'grad_norm_G': group_norms['G'],
            'grad_norm_shared_field': group_norms['shared_field'],
            'finite_grad': finite_grad,
            'finite_loss': finite_loss,
            'finite_updates': finite_updates,
            'finite_params': finite_params,
            'update_applied': update_applied,
        }
        return params_out, opt_out, loss, aux, stats
    return train_step


def make_epoch_fn(train_step: Callable, n_batches: int):
    @jax.jit
    def run_epoch(params, opt_state, y_win_epoch, u_epoch, y_epoch):
        def body(i, carry):
            params, opt_state = carry
            params, opt_state, _, _, _ = train_step(params, opt_state, y_win_epoch[i], u_epoch[i], y_epoch[i])
            return params, opt_state
        return jax.lax.fori_loop(0, n_batches, body, (params, opt_state))
    return run_epoch


def make_eval_fn(grad_energy_fn: Callable, d: int, m: int, dt: float, rollout_integrator: str = 'rk4'):
    batch_rollout = make_batch_rollout_fn(grad_energy_fn, d, m, dt, integrator=rollout_integrator)
    @jax.jit
    def eval_fn(params, x0_batch, u_traj_batch, y_traj_true):
        _, _, y_pred = batch_rollout(params, x0_batch, u_traj_batch, stop_grad=True)
        per_step = jnp.linalg.norm(y_pred - y_traj_true, axis=-1)
        per_traj = jnp.sqrt(jnp.mean(per_step ** 2, axis=-1))
        return jnp.mean(per_traj), per_traj
    return eval_fn
