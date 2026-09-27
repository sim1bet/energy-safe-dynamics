# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

# EBM_controller.py — Minimally-invasive port-Hamiltonian safety-filter controllers
#
# Goal
# ----
# Add a *minimally invasive* controller on top of the trained free-EBM /
# port-Hamiltonian model (EBM_class.py / EBM_param_fields.py / EBM_rollout.py /
# EBM_training.py) that provably enforces a Control-Barrier-Function (CBF)
# condition on the energy sub-level set
#
#       H(x) = epsilon - E(x) ,        epsilon > min_x E(x)
#
# i.e. the "safe set" C = {x : E(x) <= epsilon} = {x : H(x) >= 0}. Enforcing
#
#       Hdot(x, u) >= -gamma * H(x) ,       gamma > 0
#
# (the standard CBF / Nagumo-type invariance condition) makes C forward
# invariant: if the trajectory starts inside C it can never leave it, i.e. the
# model's own energy can never blow up past epsilon. This is exactly the
# passivity / storage-function idea the port-Hamiltonian architecture is built
# on, turned into a hard *online* safety constraint instead of a soft training
# loss (c.f. EBM_training.energy_consistency_loss, which penalises passivity
# violations only as a training-time regulariser).
#
# Port-Hamiltonian structure exploited
# -------------------------------------
# The (uncontrolled, pre-outer-clip) vector field assembled in
# EBM_param_fields.vector_field_and_output is
#
#       xdot = VF_SCALE * ( (A - M) @ gE  +  B @ u )
#
# with A skew-symmetric (gyroscopic, energy-conserving) and M positive
# semi-definite (dissipative), gE = grad_x E(x) (soft-clipped). Because A is
# skew, gE . (A @ gE) == 0 identically, so along this field
#
#       Edot  = gE . xdot = VF_SCALE * ( -gE.M.gE + gE.B.u )
#       Hdot  = -Edot     = VF_SCALE * (  gE.M.gE - gE.B.u )
#             = drift(x)  -  a(x) . u
#
# with   drift(x) = VF_SCALE * gE^T M gE   (>= 0, M is PSD: damping alone can
#                                            only ever help H, never hurt it)
#        a(x)     = VF_SCALE * B^T gE      (m-vector: control sensitivity)
#
# so Hdot is an EXACT affine function of u (no linearisation needed) given the
# model's own assembled A/M/B matrices. This makes the one-sided CBF
# constraint
#
#       a(x) . u  <=  drift(x) + gamma * H(x)   =:  c(x)
#
# a single linear inequality in u, whose minimum-norm ("least squares")
# correction of a nominal input u_nom has the classical closed-form halfspace
# projection:
#
#       violation = a.u_nom - c
#       u_safe    = u_nom                          if violation <= 0
#                 = u_nom - a * violation / (a.a)   otherwise
#
# This IS the least-squares controller requested: it is the exact solution of
#       min_u  ||u - u_nom||^2   s.t.  a.u <= c
# solved analytically (no iterative QP solver, no approximation) — the
# correction is the smallest possible nudge that restores the CBF condition,
# i.e. "minimally invasive" by construction, and the resulting u_safe
# satisfies the Lie-derivative condition EXACTLY (see verify_condition below),
# not just approximately.
#
# Flag + loss-threshold gating
# -----------------------------
# ControllerConfig.enable_controller is a plain on/off switch (default False,
# fully back-compatible with every existing training script). Even when True,
# the controller only starts being *tuned* (and, if wired into a rollout, only
# starts *acting*) once the EBM's own training-loss proxy has fallen below
# ControllerConfig.activation_loss_threshold — see update_controller_gate,
# which keeps an EMA of the caller-supplied loss proxy (mirroring the
# SWATS/local_opt_switch EMA-gate pattern already used in Silverbox_main.py)
# and only flips `active=True` once EMA < threshold. Before that point the
# nominal (uncontrolled) EBM is still finding its footing and any correction
# would be fighting an unstable target; gating on the loss avoids that and
# guarantees the controller only ever has to apply a *minimal* correction on
# top of an already-good model, consistent with the "minimally invasive"
# design goal.
#
# epsilon > min_x E(x) guarantee
# --------------------------------
# epsilon is never a free/learnable scalar directly: epsilon = min_energy_est
# + margin, where min_energy_est is a (stop-gradient'd) numerical estimate of
# min_x E(x) obtained by short gradient-descent runs from sampled/visited
# states (estimate_min_energy), and margin = min_margin_floor +
# softplus(log_margin) + SAFE_EPS is REPARAMETRISED to be strictly positive by
# construction. So epsilon > min_energy_est always holds by construction,
# regardless of what the (tunable) log_margin parameter converges to.
#
# This module is deliberately standalone: it does not modify Silverbox_main.py
# / EBM_training.py / EBM_rollout.py. See the "Example integration" comment
# block at the bottom of this file for how a training script would wire it in.
#
# Public release

import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import NamedTuple, Callable, Optional, Any

from EBM_class import energy_EBM
from EBM_param_fields import EBMParams, vector_field_and_output
import EBM_param_fields as pf  # module import (not `from ... import X`) so mutable
                                # globals like VF_SCALE/GRAD_E_CLIP are re-read
                                # fresh at call time, exactly like every other
                                # file in this repo that respects runtime overrides.


__all__ = [
    'ControllerConfig', 'ControllerParams', 'ControllerState',
    'init_controller_params', 'init_controller_state',
    'gamma_from', 'margin_from',
    'make_energy_fns', 'make_controller_fns',
    'estimate_min_energy', 'update_controller_gate',
    'make_controller_loss_fn', 'make_controller_train_step',
    'controller_epoch_update',
    'make_controlled_rollout_fn', 'make_controlled_batch_rollout_fn',
]


# ══════════════════════════════════════════════════════════════════════════════
# Configuration / state containers
# ══════════════════════════════════════════════════════════════════════════════

class ControllerConfig(NamedTuple):
    """All knobs config-gated, default OFF => fully back-compatible.

    enable_controller         : master flag. False => controller code path is
                                 never engaged (behaviour identical to not
                                 having this module at all).
    activation_loss_threshold : controller tuning/action only starts once an
                                 EMA of the caller-supplied training-loss proxy
                                 (e.g. sqrt(2*obs_loss), same convention as the
                                 local_opt_switch EMA in Silverbox_main.py, or
                                 a held-out test NRMSE) drops below this value
                                 -- i.e. only once the un-controlled EBM is
                                 already producing satisfying samples.
    ema_beta                  : EMA decay for the loss-proxy gate.
    gamma_init / margin_init  : initial class-K rate and epsilon-margin.
    min_margin_floor          : hard positive floor added to margin, on top of
                                 softplus(log_margin), so epsilon-min_E can
                                 never be pushed arbitrarily close to 0 even if
                                 log_margin is driven very negative.
    ctrl_lr                   : learning rate for the (tiny, 2-scalar) gamma /
                                 margin controller-tuning optimizer.
    w_margin_reg              : regularises margin toward 0 (a tighter barrier)
                                 so the tuner does not "cheat" by inflating
                                 epsilon to avoid ever correcting anything --
                                 the minimal-invasiveness objective trades off
                                 correction magnitude against barrier tightness.
    min_energy_n_steps/lr     : gradient-descent estimate of min_x E(x).
    n_ctrl_steps_per_epoch    : how many controller-tuning gradient steps to
                                 take per call to controller_epoch_update.
    """
    enable_controller: bool = False
    activation_loss_threshold: float = 0.30
    ema_beta: float = 0.9
    gamma_init: float = 1.0
    margin_init: float = 1.0
    min_margin_floor: float = 1e-3
    ctrl_lr: float = 1e-3
    w_margin_reg: float = 1e-3
    min_energy_n_steps: int = 50
    min_energy_lr: float = 1e-2
    n_ctrl_steps_per_epoch: int = 1


class ControllerParams(NamedTuple):
    """Tunable controller parameters, reparametrised via softplus so gamma>0
    and margin>0 hold for ANY real-valued log_gamma/log_margin (i.e. for any
    point reachable by unconstrained gradient descent)."""
    log_gamma : jax.Array   # scalar
    log_margin: jax.Array   # scalar


class ControllerState(NamedTuple):
    ctrl_params    : ControllerParams
    opt_state      : Any
    loss_ema       : jax.Array   # EMA of the caller-supplied loss/NRMSE proxy
    active         : jax.Array   # bool: enable_controller AND loss_ema < threshold
    min_energy_est : jax.Array   # last estimate of min_x E(x) (stop-gradient'd)
    step_count     : int


def _inv_softplus(y: jax.Array) -> jax.Array:
    """Inverse of jax.nn.softplus, for initialising log_gamma/log_margin from
    a desired positive init value: softplus(inv_softplus(y)) == y."""
    y = jnp.maximum(jnp.asarray(y, dtype=jnp.float32), 1e-4)
    return jnp.log(jnp.expm1(y))


def init_controller_params(cfg: ControllerConfig) -> ControllerParams:
    log_gamma  = _inv_softplus(cfg.gamma_init)
    log_margin = _inv_softplus(max(cfg.margin_init - cfg.min_margin_floor, 1e-4))
    return ControllerParams(log_gamma=log_gamma, log_margin=log_margin)


def init_controller_state(cfg: ControllerConfig) -> ControllerState:
    ctrl_params = init_controller_params(cfg)
    optimizer = optax.adam(cfg.ctrl_lr)
    opt_state = optimizer.init(ctrl_params)
    return ControllerState(
        ctrl_params=ctrl_params,
        opt_state=opt_state,
        loss_ema=jnp.array(jnp.inf, dtype=jnp.float32),
        active=jnp.array(False),
        min_energy_est=jnp.array(0.0, dtype=jnp.float32),
        step_count=0,
    )


def gamma_from(ctrl_params: ControllerParams) -> jax.Array:
    """gamma > 0 for any real log_gamma."""
    return jax.nn.softplus(ctrl_params.log_gamma) + pf.SAFE_EPS


def margin_from(ctrl_params: ControllerParams, min_margin_floor: float = 0.0) -> jax.Array:
    """margin > min_margin_floor >= 0 for any real log_margin; epsilon =
    min_energy_est + margin therefore always satisfies epsilon > min_energy_est."""
    return min_margin_floor + jax.nn.softplus(ctrl_params.log_margin) + pf.SAFE_EPS


# ══════════════════════════════════════════════════════════════════════════════
# Energy value/grad functions (mirrors EBM_param_fields.make_grad_energy, but
# also exposes the raw energy VALUE, needed for the barrier function H(x)).
# ══════════════════════════════════════════════════════════════════════════════

def make_energy_fns(layers: tuple):
    """Returns (grad_energy_fn, energy_value_fn), both with signature
    fn(x, weights, biases) -> array, consistent with every other call site of
    grad_energy_fn in this codebase (EBM_param_fields/EBM_rollout/EBM_training)."""
    def value_fn(x, weights, biases):
        return energy_EBM(x, weights, biases, layers)
    grad_fn = jax.grad(value_fn, argnums=0)
    return grad_fn, value_fn


# ══════════════════════════════════════════════════════════════════════════════
# Core controller: barrier value, Lie-derivative affine terms, least-squares
# (minimum-norm) CBF-QP safety filter with closed-form solution.
# ══════════════════════════════════════════════════════════════════════════════

def make_controller_fns(grad_energy_fn: Callable, energy_value_fn: Callable, d: int, m: int) -> dict:
    """Builds the barrier/Lie-derivative/safety-filter closures for a given
    (grad_energy_fn, energy_value_fn, d, m) triple -- exactly the ingredients
    already used to build a rollout via EBM_rollout.make_rollout_fn."""

    def barrier_H(params: EBMParams, x: jax.Array, epsilon: jax.Array) -> jax.Array:
        E = energy_value_fn(x, params.ebm_weights, params.ebm_biases)
        return epsilon - E

    def lie_derivative_terms(params: EBMParams, x: jax.Array):
        """Returns (a, drift) such that, along the model's own port-Hamiltonian
        field, Hdot(u) = drift - a . u  (EXACT, uses the skew-symmetry of A)."""
        tp = params.trunk
        raw_gE = grad_energy_fn(x, params.ebm_weights, params.ebm_biases)
        gE = pf._soft_clip_norm(pf._safe(raw_gE), pf.GRAD_E_CLIP)
        M = pf._assemble_M(tp.L_M, d, x=x, G_damp=tp.G_damp, h_damp=tp.h_damp)
        B = pf._safe(tp.B, clip=pf.MATRIX_CLIP)
        vf = pf.VF_SCALE
        drift = vf * jnp.dot(gE, M @ gE)     # = VF_SCALE * gE^T M gE, >= 0 (M is PSD)
        a = vf * (B.T @ gE)                   # m-vector control sensitivity of Hdot
        return a, drift

    def least_squares_safety_filter(params: EBMParams, x: jax.Array, u_nom: jax.Array,
                                    epsilon: jax.Array, gamma: jax.Array):
        """Closed-form minimum-norm projection of u_nom onto the CBF halfspace
        {u : a.u <= c}. Exact solution of min_u ||u-u_nom||^2 s.t. a.u<=c."""
        a, drift = lie_derivative_terms(params, x)
        H = barrier_H(params, x, epsilon)
        c = drift + gamma * H
        a_dot_u = jnp.dot(a, u_nom)
        violation = a_dot_u - c
        denom = jnp.dot(a, a) + pf.SAFE_EPS
        needs_fix = violation > 0.0
        scale = jnp.where(needs_fix, violation / denom, 0.0)
        u_safe = u_nom - scale * a
        info = {
            'H': H,
            'violation': jnp.maximum(violation, 0.0),
            'active': needs_fix,
            'correction_norm': jnp.abs(scale) * jnp.linalg.norm(a),
        }
        return u_safe, info

    batch_safety_filter = jax.jit(jax.vmap(
        least_squares_safety_filter, in_axes=(None, 0, 0, None, None)))

    def verify_condition(params: EBMParams, x: jax.Array, u: jax.Array,
                        epsilon: jax.Array, gamma: jax.Array):
        """Diagnostic/unit-test helper: checks Hdot(x,u) >= -gamma*H(x) for an
        ARBITRARY control u (e.g. the output of least_squares_safety_filter),
        returning (satisfied: bool, slack, H). slack >= 0 <=> condition holds;
        by construction slack should be ~0 (active) or >0 (inactive) whenever
        u == least_squares_safety_filter(...)[0]."""
        a, drift = lie_derivative_terms(params, x)
        H = barrier_H(params, x, epsilon)
        Hdot = drift - jnp.dot(a, u)
        slack = Hdot + gamma * H
        # tolerance accounts for float32 rounding in the closed-form projection,
        # not an approximation of the condition itself (which is enforced exactly).
        return slack >= -1e-3, slack, H

    batch_verify_condition = jax.jit(jax.vmap(
        verify_condition, in_axes=(None, 0, 0, None, None)))

    return {
        'barrier_H': barrier_H,
        'lie_derivative_terms': lie_derivative_terms,
        'least_squares_safety_filter': least_squares_safety_filter,
        'batch_safety_filter': batch_safety_filter,
        'verify_condition': verify_condition,
        'batch_verify_condition': batch_verify_condition,
    }


# ══════════════════════════════════════════════════════════════════════════════
# min_x E(x) estimation (feeds epsilon = min_energy_est + margin)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_min_energy(params: EBMParams, energy_value_fn: Callable, x_samples: jax.Array,
                        n_steps: int = 50, lr: float = 1e-2) -> jax.Array:
    """Approximates min_x E(x) by running a short gradient descent on E from
    every row of x_samples (e.g. states visited during a recent training
    rollout, flattened to [N, d]) and taking the minimum reached value.
    stop_gradient'd: this is a numerical diagnostic used to *set* epsilon, we
    never want to backprop the EBM weights through this inner minimisation.

    NOTE (Jul21 bugfix): the descent is clipped to the SAME state-norm bound
    (pf.STATE_NORM_CLIP) used everywhere else in this codebase (rollout,
    encode_x0, ...). Without this, the unconstrained descent can escape into
    a direction of unbounded energy decrease that a real (always-clipped)
    rollout can never reach, giving a wildly-too-negative min_energy_est
    (observed: -9.3 vs typical visited-state energies far higher) that makes
    epsilon far too tight - the controller then treats almost every normal
    trajectory state as unsafe and intervenes aggressively (activation
    fraction ~0.4-0.5 of test steps), roughly DOUBLING nrmse in the ced_v6
    controller batch (0.37-0.48 -> 0.74-0.83) instead of being "minimally
    invasive". Clipping keeps the estimate representative of the physically
    reachable region the model actually operates in."""
    def descend(x0):
        def body(x, _):
            g = jax.grad(lambda xx: energy_value_fn(xx, params.ebm_weights, params.ebm_biases))(x)
            g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            x_next = x - lr * g
            norm = jnp.linalg.norm(x_next)
            scale = jnp.minimum(1.0, pf.STATE_NORM_CLIP / (norm + 1e-8))
            return x_next * scale, None
        x_final, _ = jax.lax.scan(body, x0, xs=None, length=n_steps)
        return energy_value_fn(x_final, params.ebm_weights, params.ebm_biases)

    E_candidates = jax.vmap(descend)(x_samples)
    E_candidates = jnp.nan_to_num(E_candidates, nan=jnp.inf, posinf=jnp.inf, neginf=0.0)
    return jax.lax.stop_gradient(jnp.min(E_candidates))


_estimate_min_energy_jit = jax.jit(estimate_min_energy, static_argnames=('energy_value_fn', 'n_steps'))


# ══════════════════════════════════════════════════════════════════════════════
# Flag + loss-threshold gating (mirrors the SWATS/local_opt_switch EMA-gate
# pattern already used elsewhere in this codebase for a similar "only switch
# once training is good enough" decision).
# ══════════════════════════════════════════════════════════════════════════════

def update_controller_gate(state: ControllerState, loss_proxy, cfg: ControllerConfig) -> ControllerState:
    """Updates the EMA of `loss_proxy` (any scalar the caller wants to gate on,
    e.g. sqrt(2*obs_loss) train-NRMSE proxy or a held-out test NRMSE) and
    recomputes `active`. `active` can only ever be True if cfg.enable_controller
    is True AND the EMA has dropped below cfg.activation_loss_threshold."""
    loss_proxy = jnp.asarray(loss_proxy, dtype=jnp.float32)
    prev_ema = state.loss_ema
    seeded = jnp.isfinite(prev_ema)
    ema = jnp.where(seeded, cfg.ema_beta * prev_ema + (1.0 - cfg.ema_beta) * loss_proxy, loss_proxy)
    active = jnp.logical_and(bool(cfg.enable_controller), ema < cfg.activation_loss_threshold)
    return state._replace(loss_ema=ema, active=active)


# ══════════════════════════════════════════════════════════════════════════════
# Controller tuning: minimise correction magnitude (minimal invasiveness)
# while keeping the barrier tight (margin regularised toward 0).
# ══════════════════════════════════════════════════════════════════════════════

def make_controller_loss_fn(grad_energy_fn: Callable, energy_value_fn: Callable,
                            d: int, m: int, cfg: ControllerConfig) -> Callable:
    fns = make_controller_fns(grad_energy_fn, energy_value_fn, d, m)
    safety_filter = fns['least_squares_safety_filter']

    def loss_fn(ctrl_params: ControllerParams, params: EBMParams,
                x_batch: jax.Array, u_batch: jax.Array, min_energy_est: jax.Array):
        gamma = gamma_from(ctrl_params)
        margin = margin_from(ctrl_params, cfg.min_margin_floor)
        epsilon = min_energy_est + margin

        def per_sample(x, u_nom):
            u_safe, info = safety_filter(params, x, u_nom, epsilon, gamma)
            corr_sq = jnp.sum((u_safe - u_nom) ** 2)
            return corr_sq, info['active'].astype(jnp.float32)

        corr_sq, active = jax.vmap(per_sample)(x_batch, u_batch)
        correction_loss = jnp.mean(corr_sq)
        activation_rate = jnp.mean(active)
        # Minimal-invasiveness objective: small corrections AND a tight
        # (small) barrier margin -- prevents the tuner from "cheating" by
        # inflating epsilon so the constraint is (almost) never active.
        loss = correction_loss + cfg.w_margin_reg * margin
        aux = {
            'correction_loss': correction_loss,
            'activation_rate': activation_rate,
            'gamma': gamma,
            'margin': margin,
            'epsilon': epsilon,
        }
        return loss, aux
    return loss_fn


def make_controller_train_step(loss_fn: Callable, optimizer: optax.GradientTransformation) -> Callable:
    @jax.jit
    def train_step(ctrl_params: ControllerParams, opt_state, params: EBMParams,
                    x_batch: jax.Array, u_batch: jax.Array, min_energy_est: jax.Array):
        (loss, aux), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
            ctrl_params, params, x_batch, u_batch, min_energy_est)
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, ctrl_params)
        ctrl_params = optax.apply_updates(ctrl_params, updates)
        return ctrl_params, opt_state, loss, aux
    return train_step


def controller_epoch_update(state: ControllerState, cfg: ControllerConfig,
                            params: EBMParams, x_batch: jax.Array, u_batch: jax.Array,
                            loss_proxy, controller_train_step: Callable,
                            energy_value_fn: Optional[Callable] = None,
                            re_estimate_min_energy: bool = True):
    """Top-level per-epoch entry point, meant to be called once per epoch from
    a training loop (e.g. Silverbox_main.run_training), AFTER computing that
    epoch's loss_proxy and a batch of recently-visited (x, u) pairs from the
    normal (uncontrolled) rollout already computed for the EBM loss.

    Gating: if cfg.enable_controller is False, or the loss EMA hasn't dropped
    below cfg.activation_loss_threshold yet, this is a (very cheap) no-op that
    only updates the EMA bookkeeping -- ctrl_params/opt_state/min_energy_est
    are left untouched, so calling this every epoch is always safe/harmless
    even with the controller fully disabled (back-compatible default).

    Returns (new_state, aux_or_None) where aux (when not None) is a dict of
    scalars suitable for logging (e.g. wandb.log({'controller/'+k: v ...})).
    """
    state = update_controller_gate(state, loss_proxy, cfg)
    if not bool(state.active):
        return state, None

    min_energy_est = state.min_energy_est
    if re_estimate_min_energy and energy_value_fn is not None:
        x_flat = x_batch.reshape(-1, x_batch.shape[-1])
        min_energy_est = _estimate_min_energy_jit(
            params, energy_value_fn, x_flat,
            n_steps=cfg.min_energy_n_steps, lr=cfg.min_energy_lr)

    x_flat = x_batch.reshape(-1, x_batch.shape[-1])
    u_flat = u_batch.reshape(-1, u_batch.shape[-1])
    ctrl_params, opt_state = state.ctrl_params, state.opt_state
    aux = None
    for _ in range(cfg.n_ctrl_steps_per_epoch):
        ctrl_params, opt_state, _, aux = controller_train_step(
            ctrl_params, opt_state, params, x_flat, u_flat, min_energy_est)

    new_state = state._replace(
        ctrl_params=ctrl_params, opt_state=opt_state,
        min_energy_est=min_energy_est, step_count=state.step_count + 1,
    )
    return new_state, aux


# ══════════════════════════════════════════════════════════════════════════════
# Controlled rollout (test-time / closed-loop use): applies the least-squares
# safety filter to the nominal input at every integration step, using the same
# Euler/RK4 zero-order-hold treatment of u as EBM_rollout.make_rollout_fn.
# ══════════════════════════════════════════════════════════════════════════════

def make_controlled_rollout_fn(grad_energy_fn: Callable, energy_value_fn: Callable,
                                d: int, m: int, dt: float, integrator: str = 'rk4'):
    fns = make_controller_fns(grad_energy_fn, energy_value_fn, d, m)
    barrier_H = fns['barrier_H']
    safety_filter = fns['least_squares_safety_filter']

    def step_dynamics(params, x, u):
        xdot, _, _ = vector_field_and_output(params, x, u, grad_energy_fn, d, m)
        return xdot

    def integrate_step(params, x, u_nom, epsilon, gamma, apply_control: bool):
        x = pf._soft_clip_norm(pf._safe(x), pf.STATE_NORM_CLIP)
        H = barrier_H(params, x, epsilon)
        if apply_control:
            u, info = safety_filter(params, x, u_nom, epsilon, gamma)
            active = info['active']
        else:
            u = u_nom
            active = jnp.array(False)

        if integrator == 'euler':
            xdot = step_dynamics(params, x, u)
            x_next = pf._soft_clip_norm(pf._safe(x + dt * xdot), pf.STATE_NORM_CLIP)
        elif integrator == 'rk4':
            # Zero-order-hold on u over the RK4 substeps, matching EBM_rollout.
            k1 = step_dynamics(params, x, u)
            k2 = step_dynamics(params, pf._soft_clip_norm(pf._safe(x + 0.5 * dt * k1), pf.STATE_NORM_CLIP), u)
            k3 = step_dynamics(params, pf._soft_clip_norm(pf._safe(x + 0.5 * dt * k2), pf.STATE_NORM_CLIP), u)
            k4 = step_dynamics(params, pf._soft_clip_norm(pf._safe(x + dt * k3), pf.STATE_NORM_CLIP), u)
            x_next = pf._soft_clip_norm(pf._safe(x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)), pf.STATE_NORM_CLIP)
        else:
            raise ValueError(f'Unknown integrator: {integrator}')

        _, y_port, y_obs = vector_field_and_output(params, x_next, u, grad_energy_fn, d, m)
        return x_next, y_port, y_obs, u, H, active

    @partial(jax.jit, static_argnames=('stop_grad', 'apply_control'))
    def rollout(params: EBMParams, x0: jax.Array, u_traj: jax.Array,
                epsilon: jax.Array, gamma: jax.Array,
                stop_grad: bool = False, apply_control: bool = True):
        def step(x, u_k):
            xin = jax.lax.stop_gradient(x) if stop_grad else x
            x_next, yp, yo, u_applied, H, active = integrate_step(params, xin, u_k, epsilon, gamma, apply_control)
            return x_next, (x_next, yp, yo, u_applied, H, active)
        x0c = pf._soft_clip_norm(pf._safe(x0), pf.STATE_NORM_CLIP)
        _, (x_traj, y_port, y_obs, u_applied, H_traj, active_traj) = jax.lax.scan(step, x0c, u_traj)
        return x_traj, y_port, y_obs, u_applied, H_traj, active_traj
    return rollout


def make_controlled_batch_rollout_fn(grad_energy_fn: Callable, energy_value_fn: Callable,
                                    d: int, m: int, dt: float, integrator: str = 'rk4'):
    rollout = make_controlled_rollout_fn(grad_energy_fn, energy_value_fn, d, m, dt, integrator=integrator)

    @partial(jax.jit, static_argnames=('stop_grad', 'apply_control'))
    def batch_rollout(params, x0_batch, u_batch, epsilon, gamma, stop_grad: bool = False, apply_control: bool = True):
        return jax.vmap(
            lambda x0, u_traj: rollout(params, x0, u_traj, epsilon, gamma, stop_grad, apply_control)
        )(x0_batch, u_batch)
    return batch_rollout


# ══════════════════════════════════════════════════════════════════════════════
# Example integration (comment only — NOT executed; illustrates how a training
# script such as Silverbox_main.py would wire this module in, back-compatibly).
# ══════════════════════════════════════════════════════════════════════════════
#
#   import EBM_controller as ctrl_mod
#
#   ctrl_cfg = ctrl_mod.ControllerConfig(
#       enable_controller=config.get('enable_controller', False),
#       activation_loss_threshold=config.get('controller_loss_threshold', 0.30),
#   )
#   ctrl_state = ctrl_mod.init_controller_state(ctrl_cfg)
#   grad_energy_fn, energy_value_fn = ctrl_mod.make_energy_fns(layers)
#   controller_loss_fn = ctrl_mod.make_controller_loss_fn(grad_energy_fn, energy_value_fn, d, m, ctrl_cfg)
#   controller_train_step = ctrl_mod.make_controller_train_step(
#       controller_loss_fn, optax.adam(ctrl_cfg.ctrl_lr))
#
#   # ... inside the per-epoch loop, after computing this epoch's obs_loss and
#   # the (x_traj, u_epoch) already produced by the normal EBM rollout/loss:
#   train_nrmse_proxy = jnp.sqrt(2.0 * obs_loss)   # same proxy used by SWATS
#   ctrl_state, ctrl_aux = ctrl_mod.controller_epoch_update(
#       ctrl_state, ctrl_cfg, params, x_traj, u_epoch,
#       train_nrmse_proxy, controller_train_step, energy_value_fn=energy_value_fn)
#   if ctrl_aux is not None and use_wandb:
#       wandb.log({f'controller/{k}': v for k, v in ctrl_aux.items()})
#
#   # ... at test/eval time, to roll out WITH the safety filter engaged:
#   gamma = ctrl_mod.gamma_from(ctrl_state.ctrl_params)
#   epsilon = ctrl_state.min_energy_est + ctrl_mod.margin_from(
#       ctrl_state.ctrl_params, ctrl_cfg.min_margin_floor)
#   controlled_rollout = ctrl_mod.make_controlled_rollout_fn(
#       grad_energy_fn, energy_value_fn, d, m, dt, integrator='rk4')
#   x_traj, y_port, y_obs, u_applied, H_traj, active_traj = controlled_rollout(
#       params, x0, u_traj, epsilon, gamma, apply_control=True)
