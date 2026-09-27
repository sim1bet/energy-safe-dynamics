# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

# EBM_rollout.py — Rollout functions with safer RK4 integration
import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable
from functools import partial
import optax

from EBM_param_fields import EBMParams, vector_field_and_output, encode_x0, STATE_NORM_CLIP
import EBM_param_fields as pf  # module import so USE_INPUT_AWARE_ENCODER is re-read fresh at call time


def _clip_state(x: jax.Array) -> jax.Array:
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norm = jnp.linalg.norm(x)
    scale = jnp.minimum(1.0, STATE_NORM_CLIP / (norm + 1e-8))
    return x * scale


def make_rollout_fn(grad_energy_fn: Callable, d: int, m: int, dt: float, integrator: str = 'rk4',
                    substeps: int = 1):
    if int(substeps) < 1:
        raise ValueError("substeps must be at least one")
    dt_sub = dt / int(substeps)
    def step_dynamics(params, x, u):
        xdot, _, _ = vector_field_and_output(params, x, u, grad_energy_fn, d, m)
        return xdot

    def _one_step(params, x, u):
        if integrator == 'euler':
            xdot = step_dynamics(params, x, u)
            return _clip_state(x + dt_sub * xdot)
        elif integrator == 'rk4':
            k1 = step_dynamics(params, x, u)
            k2 = step_dynamics(params, _clip_state(x + 0.5 * dt_sub * k1), u)
            k3 = step_dynamics(params, _clip_state(x + 0.5 * dt_sub * k2), u)
            k4 = step_dynamics(params, _clip_state(x + dt_sub * k3), u)
            return _clip_state(x + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        else:
            raise ValueError(f'Unknown integrator: {integrator}')

    def integrate_step(params, x, u):
        x_next = _clip_state(x)
        for _ in range(int(substeps)):
            x_next = _one_step(params, x_next, u)
        _, y_port, y_obs = vector_field_and_output(params, x_next, u, grad_energy_fn, d, m)
        return x_next, y_port, y_obs

    @partial(jax.jit, static_argnames=('stop_grad',))
    def rollout(params: EBMParams, x0: jax.Array, u_traj: jax.Array, stop_grad: bool = False):
        def step_grad(x, u_k):
            x_next, yp, yo = integrate_step(params, x, u_k)
            return x_next, (x_next, yp, yo)
        def step_sg(x, u_k):
            xs = lax.stop_gradient(x)
            x_next, yp, yo = integrate_step(params, xs, u_k)
            return x_next, (x_next, yp, yo)
        fn = step_sg if stop_grad else step_grad
        _, (x_traj, y_port, y_obs) = lax.scan(fn, _clip_state(x0), u_traj)
        return x_traj, y_port, y_obs
    return rollout


def make_batch_rollout_fn(grad_energy_fn: Callable, d: int, m: int, dt: float, integrator: str = 'rk4',
                          substeps: int = 1):
    rollout = make_rollout_fn(grad_energy_fn, d, m, dt, integrator=integrator, substeps=substeps)
    @partial(jax.jit, static_argnames=('stop_grad',))
    def batch_rollout(params, x0_batch, u_batch, stop_grad=False):
        return jax.vmap(lambda x0, u_traj: rollout(params, x0, u_traj, stop_grad))(x0_batch, u_batch)
    return batch_rollout


def make_pmap_batch_rollout_fn(grad_energy_fn: Callable, d: int, m: int, dt: float, integrator: str = 'rk4',
                               substeps: int = 1):
    batch_rollout = make_batch_rollout_fn(grad_energy_fn, d, m, dt, integrator=integrator, substeps=substeps)
    @partial(jax.pmap, static_broadcasted_argnums=(3,))
    def pmap_batch_rollout(params_repl, x0_shard, u_shard, stop_grad=False):
        return batch_rollout(params_repl, x0_shard, u_shard, stop_grad)
    return pmap_batch_rollout


def make_encode_x0_batch(d: int):
    from EBM_param_fields import encode_x0_batch
    return jax.jit(lambda params, y_windows: encode_x0_batch(params, y_windows))


def _encoder_window(y_win: jax.Array, u_win: jax.Array) -> jax.Array:
    """Builds the (opaque) window passed to encode_x0: y-only unless
    USE_INPUT_AWARE_ENCODER is on, in which case the burn-in input u_win is
    concatenated after y_win along the last axis (matching how init_trunk
    sizes W_enc: n_obs channels first, then m channels, per timestep)."""
    if pf.USE_INPUT_AWARE_ENCODER:
        return jnp.concatenate([y_win, u_win], axis=-1)
    return y_win


def make_init_state_fn(grad_energy_fn: Callable, d: int, m: int, dt: float,
                       n_steps: int = 200, lr: float = 5e-3, integrator: str = 'rk4',
                       n_restarts: int = 1, restart_noise_std: float = 0.1,
                       substeps: int = 1):
    rollout = make_rollout_fn(grad_energy_fn, d, m, dt, integrator=integrator, substeps=substeps)
    def loss_x0(x0, params, u_win, y_win):
        # NOTE (Jul21 bugfix): stop_grad=True makes rollout() use step_sg for
        # EVERY step, including the very first one, whose incoming state IS
        # x0 - lax.stop_gradient there severs the gradient path back to x0
        # before integrate_step ever runs, so jax.grad(loss_x0) w.r.t. x0 was
        # IDENTICALLY ZERO and this whole refinement loop was a no-op (x0
        # never moved past the encode_x0 initial guess, confirmed by ced_v6
        # runs scoring bit-identical to their non-iterinit v5 counterparts).
        # The burn-in window is short (init_win, e.g. 10-50 steps) so full
        # BPTT through it to actually refine x0 is safe here.
        _, _, y_pred = rollout(params, x0, u_win, stop_grad=False)
        return jnp.mean((y_pred - y_win) ** 2)

    def _refine_from(x0_start, params, u_win, y_win):
        opt = optax.adam(lr)
        opt_state = opt.init(x0_start)
        def body(carry, _):
            x_curr, opt_state = carry
            _, grads = jax.value_and_grad(loss_x0)(x_curr, params, u_win, y_win)
            grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
            updates, opt_state = opt.update(grads, opt_state, x_curr)
            x_next = optax.apply_updates(x_curr, updates)
            x_next = _clip_state(x_next)
            return (x_next, opt_state), None
        (x_final, _), _ = lax.scan(body, (x0_start, opt_state), xs=None, length=n_steps)
        final_loss = loss_x0(x_final, params, u_win, y_win)
        return x_final, final_loss

    @jax.jit
    def init_state(params, u_win, y_win, key):
        x0_base = encode_x0(params, _encoder_window(y_win, u_win))
        if n_restarts <= 1:
            x_final, final_loss = _refine_from(x0_base, params, u_win, y_win)
            return x_final, final_loss
        # MULTI-START: the reconstruction-loss surface w.r.t. x0 (like the
        # EBM's own energy) is generically NON-CONVEX with many local minima,
        # so a single gradient-descent run from one starting point is not
        # guaranteed to find the best x0. Restart 0 is always the unperturbed
        # encode_x0 guess (so multistart can never do WORSE than the old
        # single-start refinement); restarts 1..n_restarts-1 are Gaussian-
        # perturbed starts. Keep whichever restart reaches the lowest final
        # reconstruction loss.
        noise = jax.random.normal(key, (n_restarts - 1,) + x0_base.shape) * restart_noise_std
        x0_starts = jnp.concatenate([x0_base[None], x0_base[None] + noise], axis=0)
        x_finals, final_losses = jax.vmap(
            lambda x0_try: _refine_from(x0_try, params, u_win, y_win)
        )(x0_starts)
        best = jnp.argmin(final_losses)
        return x_finals[best], final_losses[best]

    batch_init = jax.jit(jax.vmap(lambda params, uw, yw, k: init_state(params, uw, yw, k),
                                  in_axes=(None, 0, 0, 0)))
    return init_state, batch_init
