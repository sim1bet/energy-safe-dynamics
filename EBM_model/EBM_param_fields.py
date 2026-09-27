# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

# EBM_param_fields.py — Parameters, matrices, and vector field
# Robust refactor preserving a free EBM storage function.
# Main changes:
#   1. keep the free EBM parametrization untouched
#   2. stabilise the pH vector field with sanitisation + norm clipping
#   3. expose a continuous-time vector_field_and_output used by higher-order rollout integrators
#   4. decouple numerical safeguards from the energy parametrization
# Public release

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable

from EBM_class import energy_EBM

# Safety constants (overridable from Silverbox_main during sweeps)
GRAD_E_CLIP     = 5.0
XDOT_CLIP       = 5.0
X0_NORM_MAX     = 3.0
CHOL_CLIP_EXP   = 1.5
STATE_NORM_CLIP = 10.0
MATRIX_CLIP     = 25.0
OUTPUT_CLIP     = 20.0
DAMPING_SCALE   = 1.0   # global multiplier on dissipation matrix M; <1 forces light damping
B_INIT_SCALE    = 1.0   # scale of random B init; 0 reproduces old zero-B (input-starved) behaviour
VF_SCALE        = 1.0   # multiplier on the vector field; >1 speeds dynamics to match resonance
USE_FEEDTHROUGH = True  # add D_feed @ u direct feedthrough term to the observation readout
USE_IDENTITY_READOUT = False  # y_obs=x exactly; requires d=n_obs and all optional readout heads off
USE_NONLINEAR_READOUT = False  # add a bounded (sigmoid) hidden head to the readout for kink/saturation outputs
READOUT_HIDDEN  = 16    # width of the bounded readout head (only used when USE_NONLINEAR_READOUT)
USE_STATE_DAMPING = False  # gate the dissipation matrix by state: M(x)=L diag(s(x)) L^T (level-dependent loss)
USE_STATE_INTERCONNECTION = False  # J(x) remains skew pointwise via state-dependent upper-triangle entries
USE_QUADRATIC_INTERCONNECTION = False  # augment skew entries with a learned x^2 basis
USE_CUBIC_INTERCONNECTION = False  # augment skew entries with a learned x^3 basis
# Saturating (bounded) output readout: models a PHYSICAL ceiling/floor on y_obs
# (e.g. cascaded-tanks tank overflow at the top / empty-tank floor at the
# bottom) via a LEARNABLE, asymmetric soft-clamp y = cap*tanh(y_lin/cap)
# applied per-side (positive/negative) to the linear(+feedthrough/NLRO)
# readout. Unlike USE_NONLINEAR_READOUT (an unconstrained ADDITIVE kink term,
# which cannot itself prevent overshoot past a ceiling), this directly bounds
# the output. Default off / cap initialised LARGE => near-identical to the
# unclamped linear readout at start (back-compatible warm start); gradient
# descent can tighten the cap toward the true physical saturation level only
# where the data actually calls for it.
USE_SATURATING_READOUT = False
SAT_SCALE_INIT = 6.0   # initial +/- cap magnitude (y-units); generous headroom above typical peaks
SAT_SCALE_MIN  = 0.5    # floor on the learned cap so it can never collapse to ~0 (degenerate)
USE_INPUT_AWARE_ENCODER = False  # x0 encoder also sees the burn-in INPUT window u, not just output y
# State-gated INPUT GAIN: xdot's forced term becomes g(x) * (B @ u) with a
# learnable gate g in (0,2), zero-init => g = 1 exactly => bit-identical to the
# constant-B baseline at start (clean A/B, same pattern as USE_STATE_DAMPING).
# Physically motivated for cascaded-tanks: the pump/valve inflow's EFFECT on
# the tank level depends on the level itself (overflow spills excess inflow;
# near-empty dynamics differ) - a single constant B cannot be simultaneously
# right for sharp large input spikes (where the model overshoots) and moderate
# sustained input (which it under-integrates). Two modes:
#   'scalar' (default, 5 params for d=4): g = 2*sigmoid(w.x + b), one shared
#            scalar gate - minimal-capacity, low overfit risk (the v6 state-
#            damping lesson: unregularised d*d gates overfit CT's 1024 samples).
#   'vector' (d*d+d params): g = 2*sigmoid(G@x + h) per state row, elementwise
#            on (B@u) - capacity-dose contrast probe.
USE_INPUT_GAIN  = False
INPUT_GAIN_MODE = 'scalar'
USE_RICH_INTERCONNECTION = False
USE_RICH_DAMPING = False
USE_RICH_INPUT_MATRIX = False
PH_FIELD_WIDTH = 64
INPUT_MATRIX_STRUCTURE = 'unrestricted'
# Saturating INPUT transform (Hammerstein-style pump/actuator saturation):
# u_eff = cap * tanh(u / cap) with a learnable per-channel cap > 0, applied to
# u BEFORE it enters the vector field (B@u), the feedthrough (D_feed@u) and
# y_port. Models physical pump-flow saturation: the largest/fastest input
# spike in the CT test trace produces an output overshoot, consistent with the
# real actuator saturating where the model's linear input path does not.
# cap initialised LARGE (near-linear warm start), can only be tightened by
# gradient descent where the data calls for it. Default off/back-compatible.
USE_SATURATING_INPUT = False
SAT_U_INIT = 4.0   # initial cap (normalised-u units; CT's |u| max is ~3.5)
SAT_U_MIN  = 0.5   # floor on the learned cap (never collapses to ~0)
SAFE_EPS        = 1e-8


class TrunkParams(NamedTuple):
    # issue 5: constant learnable matrices — no state-dependent trunk network
    L_M    : jax.Array   # [d*(d+1)//2]            Cholesky entries: M = L @ L^T (SPSD)
    e_A    : jax.Array   # [max(d*(d-1)//2, 1)]    skew entries: A = upper_tri - lower_tri
    B      : jax.Array   # [d, m]                  constant input/output port matrix
    # issue 1: direct readout from state — replaces C @ B^T @ grad_H
    C_state: jax.Array   # [n_obs, d]  linear readout from state
    c_bias : jax.Array   # [n_obs]     readout bias
    D_feed : jax.Array   # [n_obs, m]  direct input feedthrough (D @ u) in the readout
    C_ro1  : jax.Array   # [h, d]      bounded readout head: hidden weights
    b_ro1  : jax.Array   # [h]         bounded readout head: hidden bias
    C_ro2  : jax.Array   # [n_obs, h]  bounded readout head: output weights (zero init => linear at start)
    # Saturating readout: y = cap*tanh(y_lin/cap), one learnable cap per side
    # per output channel. Raw (pre-softplus) params; actual cap = SAT_SCALE_MIN
    # + softplus(raw), always > 0. Initialised so cap == SAT_SCALE_INIT at start.
    y_sat_pos_raw: jax.Array  # [n_obs]  raw param for the POSITIVE-side cap (overflow ceiling)
    y_sat_neg_raw: jax.Array  # [n_obs]  raw param for the NEGATIVE-side cap (empty-tank floor)
    # state-dependent damping gate: s(x) = 2*sigmoid(G_damp@x + h_damp) in (0,2)
    G_damp : jax.Array   # [d, d]      damping-gate weights (zero init => s=1 => constant M at start)
    h_damp : jax.Array   # [d]         damping-gate bias    (zero init => s=1)
    # state-gated input gain (config-gated): scalar mode uses w_gain/b_gain,
    # vector mode uses G_gain/h_gain. All zero-init => gate == 1 at start.
    w_gain : jax.Array   # [d]         scalar-gate weights: g = 2*sigmoid(w.x + b)
    b_gain : jax.Array   # []          scalar-gate bias
    G_gain : jax.Array   # [d, d]      vector-gate weights: g = 2*sigmoid(G@x + h)
    h_gain : jax.Array   # [d]         vector-gate bias
    # saturating input transform (config-gated): cap = SAT_U_MIN + softplus(raw)
    u_sat_raw: jax.Array # [m]         raw param; u_eff = cap*tanh(u/cap)
    # initial state encoder
    W_enc  : jax.Array   # [d, n_obs * init_win], or [d, (n_obs+m) * init_win]
                         # when USE_INPUT_AWARE_ENCODER (sees burn-in u too)
    b_enc  : jax.Array   # [d]
    G_A    : jax.Array = None  # [nA, d] state-dependent skew-entry weights
    h_A    : jax.Array = None  # [nA] state-dependent skew-entry bias
    G_A_quadratic: jax.Array = None  # [nA, d] quadratic skew-entry weights
    G_A_cubic: jax.Array = None  # [nA, d] cubic skew-entry weights
    W_field1: jax.Array = None  # [width, d] shared bounded pH-field features
    b_field1: jax.Array = None  # [width]
    W_field2: jax.Array = None  # [width, width]
    b_field2: jax.Array = None  # [width]
    W_rich_A: jax.Array = None  # [nA, width] skew-entry residual
    W_rich_M: jax.Array = None  # [nM, width] Cholesky-entry residual
    W_rich_B: jax.Array = None  # [d*m, width] input-matrix residual


class EBMParams(NamedTuple):
    ebm_weights: list
    ebm_biases : list
    trunk      : TrunkParams


def _fan(shape):
    return jnp.sqrt(2.0 / (shape[0] + shape[1]))


def _safe(x, clip=None):
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
        x = jnp.clip(x, -clip, clip)
    return x


def _clip_norm(x: jax.Array, max_norm: float) -> jax.Array:
    norm = jnp.linalg.norm(x)
    scale = jnp.minimum(1.0, max_norm / (norm + SAFE_EPS))
    return x * scale


def _soft_clip(x: jax.Array, cap: float) -> jax.Array:
    # Smooth elementwise saturation: ~x for |x| << cap, bends smoothly to ±cap.
    # Replaces hard jnp.clip so the field stays bounded (no NaN/blow-up) while
    # keeping a non-zero gradient everywhere (no throttle, no dead zone).
    return cap * jnp.tanh(x / cap)


def _soft_clip_norm(x: jax.Array, max_norm: float) -> jax.Array:
    # Direction-preserving smooth norm saturation: maps a vector of norm n to
    # norm max_norm * tanh(n / max_norm). Linear for small n, saturates softly.
    norm   = jnp.linalg.norm(x)
    target = max_norm * jnp.tanh(norm / (max_norm + SAFE_EPS))
    scale  = target / (norm + SAFE_EPS)
    return x * scale


def init_ebm_params(key, d_in: int, layer_dims: list):
    dims = [d_in] + list(layer_dims)
    weights, biases = [], []
    for k in range(len(layer_dims)):
        key, k1 = jax.random.split(key)
        d_out, d_prev = dims[k + 1], dims[k]
        weights.append(jax.random.normal(k1, (d_out, d_prev)) * _fan((d_out, d_prev)))
        biases.append(jnp.zeros((d_out,)))
    return weights, biases


def init_trunk(key, d: int, m: int, n_obs: int, init_win: int, hidden: int = 64, dt: float = None) -> TrunkParams:
    nM = d * (d + 1) // 2
    nA = max(d * (d - 1) // 2, 1)
    ks = jax.random.split(key, 6)

    # M (damping): initialize SMALL, like the reference's zero-initialized R.
    # Diagonal uses exp(clip(., ±CHOL_CLIP_EXP)); set diagonal entries to the
    # negative clip bound so M starts as light damping instead of identity.
    L_M = jnp.zeros(nM)
    L_M = L_M.at[:d].set(-CHOL_CLIP_EXP)

    # A (gyroscopic/skew): initialize to a symplectic-like oscillatory structure,
    # mirroring the reference's fixed SymplecticMatrix. Couple dimension i with
    # i + d//2 so the system oscillates from step 1. A entries are tanh(e_A), so
    # a raw value of 1.0 gives ~0.76 coupling with healthy (non-saturated) gradient.
    e_A = jnp.zeros(nA)
    half = d // 2
    if d >= 2:
        def _triu_pos(i, j):
            # linear index of (i, j) within triu_indices(d, k=1) row-major order
            return sum(d - 1 - r for r in range(i)) + (j - i - 1)
        for i in range(half):
            e_A = e_A.at[_triu_pos(i, i + half)].set(1.0)
    G_A = jnp.zeros((nA, d))
    h_A = jnp.zeros((nA,))
    G_A_quadratic = jnp.zeros((nA, d))
    G_A_cubic = jnp.zeros((nA, d))

    field_width = max(int(PH_FIELD_WIDTH), 1)
    W_field1 = jax.random.normal(ks[4], (field_width, d)) * _fan((field_width, d))
    b_field1 = jnp.zeros((field_width,))
    W_field2 = jax.random.normal(ks[5], (field_width, field_width)) * _fan(
        (field_width, field_width)
    )
    b_field2 = jnp.zeros((field_width,))
    W_rich_A = jnp.zeros((nA, field_width))
    W_rich_M = jnp.zeros((nM, field_width))
    W_rich_B = jnp.zeros((d * m, field_width))

    # B: small NONZERO init so the input actually drives the state from step 1.
    # Zero init starved the input path (xdot = (A-M)gE + B u), so the model could
    # only produce a free response and collapsed to the mean predictor. B_INIT_SCALE=0
    # reproduces the old behaviour for A/B comparisons.
    B = jax.random.normal(ks[3], (d, m)) * (_fan((d, m)) * B_INIT_SCALE)

    # C_state: small random readout from state
    C_state = jax.random.normal(ks[1], (n_obs, d)) * _fan((n_obs, d))
    c_bias  = jnp.zeros((n_obs,))
    # D_feed: direct input feedthrough; zero init gives the input an immediate,
    # strong gradient path to the loss without biasing the initial output.
    D_feed  = jnp.zeros((n_obs, m))
    # Bounded nonlinear readout head (config-gated): y += C_ro2 @ sigmoid(C_ro1@x + b_ro1).
    # C_ro2 zero-init => readout starts IDENTICAL to the linear baseline; the head
    # only adds the overflow/saturation kink as it trains. Always allocated for a
    # fixed pytree; unused (zero) when USE_NONLINEAR_READOUT is False.
    h_ro    = max(int(READOUT_HIDDEN), 1)
    C_ro1   = jax.random.normal(ks[0], (h_ro, d)) * _fan((h_ro, d))
    b_ro1   = jnp.zeros((h_ro,))
    C_ro2   = jnp.zeros((n_obs, h_ro))

    # Saturating readout caps (config-gated): cap = SAT_SCALE_MIN + softplus(raw).
    # Solve softplus(raw) = SAT_SCALE_INIT - SAT_SCALE_MIN for raw (inverse
    # softplus) so BOTH sides start at exactly SAT_SCALE_INIT (symmetric, gentle
    # warm start); training can then pull each side independently. Always
    # allocated for a fixed pytree; unused when USE_SATURATING_READOUT is False.
    _sat_target = max(float(SAT_SCALE_INIT) - float(SAT_SCALE_MIN), 1e-3)
    _sat_raw_init = jnp.log(jnp.expm1(jnp.asarray(_sat_target)))
    y_sat_pos_raw = jnp.full((n_obs,), _sat_raw_init)
    y_sat_neg_raw = jnp.full((n_obs,), _sat_raw_init)

    # State-dependent damping gate (config-gated): M(x) = L diag(s(x)) L^T with
    # s(x) = 2*sigmoid(G_damp@x + h_damp). Zero init => s = 1 exactly => M is
    # IDENTICAL to the constant-damping baseline at start (clean A/B). Physically
    # models level-dependent dissipation (tank drain/overflow). Always allocated
    # for a fixed pytree; unused (identity gate) when USE_STATE_DAMPING is False.
    G_damp  = jnp.zeros((d, d))
    h_damp  = jnp.zeros((d,))

    # State-gated input gain (config-gated): zero init => g = 1 exactly =>
    # forced term identical to constant B@u at start (clean A/B warm start).
    w_gain  = jnp.zeros((d,))
    b_gain  = jnp.zeros(())
    G_gain  = jnp.zeros((d, d))
    h_gain  = jnp.zeros((d,))

    # Saturating input cap (config-gated): cap = SAT_U_MIN + softplus(raw),
    # initialised so cap == SAT_U_INIT exactly (inverse softplus), mirroring
    # the saturating-readout warm-start pattern. Always allocated (fixed pytree).
    _usat_target  = max(float(SAT_U_INIT) - float(SAT_U_MIN), 1e-3)
    u_sat_raw = jnp.full((m,), jnp.log(jnp.expm1(jnp.asarray(_usat_target))))

    # issue 9: physics-informed encoder init
    # x[0] <- last observed output  (position proxy)
    # x[1] <- unscaled finite difference of last two outputs (velocity proxy)
    # x[2:] <- small random readout from recent window for extra context
    #
    # USE_INPUT_AWARE_ENCODER (config-gated, default off): the caller-supplied
    # "y_window" passed to encode_x0 is treated as an OPAQUE per-timestep
    # feature block of width n_obs_eff = n_obs + m (y-channels first, then the
    # burn-in u-channels, concatenated along the last axis by the caller -
    # encode_x0 itself is unchanged, it only ever flattens whatever window it
    # is given). This lets the encoder see the input driving the system
    # (previously completely invisible to x0 estimation - a poor guess for an
    # input-FORCED system) with NO changes anywhere else in the training loop,
    # since y_win_batch/yw_ep stay single opaque arrays throughout. All the
    # physics-informed index math below only ever targets the Y portion (the
    # first n_obs columns of each per-timestep block), so it is unaffected in
    # substance - only the per-timestep STRIDE grows from n_obs to n_obs_eff
    # when the extra u-columns are inserted. (n_obs is always 1 for every
    # dataset in this repo, so "channel 0 of each timestep" == the sole
    # observed output, exactly as before.)
    n_obs_eff = n_obs + (m if USE_INPUT_AWARE_ENCODER else 0)
    W_enc = jnp.zeros((d, n_obs_eff * init_win))
    b_enc = jnp.zeros((d,))
    if d >= 1:
        W_enc = W_enc.at[0, n_obs_eff * (init_win - 1)].set(1.0)
    if d >= 2:
        W_enc = W_enc.at[1, n_obs_eff * (init_win - 1)].set( 1.0)
        W_enc = W_enc.at[1, n_obs_eff * (init_win - 2)].set(-1.0)
    if d > 2:
        W_extra = jax.random.normal(ks[2], (d - 2, n_obs_eff * init_win)) * _fan((d - 2, n_obs_eff * init_win))
        W_enc = W_enc.at[2:, :].set(W_extra)

    return TrunkParams(
        L_M    = L_M,
        e_A    = e_A,
        B      = B,
        C_state= C_state,
        c_bias = c_bias,
        D_feed = D_feed,
        C_ro1  = C_ro1,
        b_ro1  = b_ro1,
        C_ro2  = C_ro2,
        y_sat_pos_raw = y_sat_pos_raw,
        y_sat_neg_raw = y_sat_neg_raw,
        G_damp = G_damp,
        h_damp = h_damp,
        w_gain = w_gain,
        b_gain = b_gain,
        G_gain = G_gain,
        h_gain = h_gain,
        u_sat_raw = u_sat_raw,
        W_enc  = W_enc,
        b_enc  = b_enc,
        G_A    = G_A,
        h_A    = h_A,
        G_A_quadratic = G_A_quadratic,
        G_A_cubic = G_A_cubic,
        W_field1 = W_field1,
        b_field1 = b_field1,
        W_field2 = W_field2,
        b_field2 = b_field2,
        W_rich_A = W_rich_A,
        W_rich_M = W_rich_M,
        W_rich_B = W_rich_B,
    )


def init_params(key, d: int, m: int, n_obs: int, init_win: int, layer_dims: list, hidden: int = 64, dt: float = None) -> EBMParams:
    k1, k2 = jax.random.split(key)
    ebm_w, ebm_b = init_ebm_params(k1, d_in=d, layer_dims=layer_dims)
    trunk = init_trunk(k2, d=d, m=m, n_obs=n_obs, init_win=init_win, hidden=hidden, dt=dt)
    return EBMParams(ebm_weights=ebm_w, ebm_biases=ebm_b, trunk=trunk)


def upgrade_legacy_params(params: EBMParams, initialized: EBMParams) -> EBMParams:
    """Fill fields absent from an older checkpoint without changing learned arrays."""
    replacements = {
        name: getattr(initialized.trunk, name)
        for name in TrunkParams._fields
        if getattr(params.trunk, name, None) is None
    }
    return params._replace(trunk=params.trunk._replace(**replacements))


# _trunk_heads removed: trunk is now constant; matrices assembled directly from stored entries.


def _cholesky_bounded(entries: jax.Array, d: int) -> jax.Array:
    diag = jnp.arange(d)
    L    = jnp.zeros((d, d))
    clipped_diag = jnp.clip(entries[:d], -CHOL_CLIP_EXP, CHOL_CLIP_EXP)
    L = L.at[diag, diag].set(jnp.exp(clipped_diag))
    if d > 1:
        L = L.at[jnp.tril_indices(d, k=-1)].set(jnp.tanh(entries[d:]))
    return _safe(L, clip=MATRIX_CLIP)


def _field_features(tp: TrunkParams, x: jax.Array) -> jax.Array:
    if tp.W_field1 is None or tp.W_field2 is None:
        return jnp.zeros((0,), dtype=x.dtype)
    hidden = jnp.tanh(_safe(tp.W_field1 @ x + tp.b_field1))
    return jnp.tanh(_safe(tp.W_field2 @ hidden + tp.b_field2))


def _assemble_M(e_M: jax.Array, d: int, x: jax.Array = None,
                G_damp: jax.Array = None, h_damp: jax.Array = None,
                rich_features: jax.Array = None,
                W_rich_M: jax.Array = None) -> jax.Array:
    entries = e_M
    if USE_RICH_DAMPING and rich_features is not None and W_rich_M is not None:
        entries = entries + jnp.tanh(W_rich_M @ rich_features)
    L = _cholesky_bounded(entries, d)
    if USE_STATE_DAMPING and x is not None and G_damp is not None:
        # State-dependent gate: M(x) = L diag(s(x)) L^T with s(x) in (0,2).
        # Congruence by a positive diagonal keeps M positive-semidefinite (passive)
        # for any state; s=1 (zero-init gate) reduces exactly to the constant M.
        s = 2.0 * jax.nn.sigmoid(_safe(G_damp @ x + h_damp))
        M = L @ (s[:, None] * L.T)
    else:
        M = L @ L.T
    # DAMPING_SCALE multiplies the whole dissipation matrix; small values force
    # light damping (resonant systems) while keeping M positive-semidefinite.
    return _safe(DAMPING_SCALE * M, clip=MATRIX_CLIP)


def _assemble_A(e_A: jax.Array, d: int, x: jax.Array = None,
                G_A: jax.Array = None, h_A: jax.Array = None,
                G_A_quadratic: jax.Array = None,
                G_A_cubic: jax.Array = None,
                rich_features: jax.Array = None,
                W_rich_A: jax.Array = None) -> jax.Array:
    if d <= 1:
        return jnp.zeros((d, d))
    triu = jnp.triu_indices(d, k=1)
    entries = e_A
    if USE_STATE_INTERCONNECTION and x is not None and G_A is not None:
        entries = entries + G_A @ x + h_A
        if USE_QUADRATIC_INTERCONNECTION and G_A_quadratic is not None:
            entries = entries + G_A_quadratic @ jnp.square(x)
        if USE_CUBIC_INTERCONNECTION and G_A_cubic is not None:
            entries = entries + G_A_cubic @ jnp.power(x, 3)
    if USE_RICH_INTERCONNECTION and rich_features is not None and W_rich_A is not None:
        entries = entries + jnp.tanh(W_rich_A @ rich_features)
    A = jnp.zeros((d, d))
    A = A.at[triu].set(jnp.tanh(entries))
    return _safe(A - A.T, clip=MATRIX_CLIP)


def _assemble_input_matrix(tp: TrunkParams, x: jax.Array,
                           rich_features: jax.Array = None) -> jax.Array:
    """Return the exact matrix G(x) in the raw-input affine term G(x)u."""
    B = _safe(tp.B, clip=MATRIX_CLIP)
    if USE_INPUT_GAIN and INPUT_GAIN_MODE == 'vector':
        gain = 2.0 * jax.nn.sigmoid(_safe(tp.G_gain @ x + tp.h_gain))
        B = gain[:, None] * B
    elif USE_INPUT_GAIN:
        gain = 2.0 * jax.nn.sigmoid(_safe(jnp.dot(tp.w_gain, x) + tp.b_gain))
        B = gain * B
    if USE_RICH_INPUT_MATRIX and rich_features is not None and tp.W_rich_B is not None:
        residual = jnp.tanh(tp.W_rich_B @ rich_features).reshape(B.shape)
        B = B + residual
    if INPUT_MATRIX_STRUCTURE == 'mechanical_velocity_rows':
        row_mask = (jnp.arange(B.shape[0]) >= 8).astype(B.dtype)[:, None]
        B = B * row_mask
    return _safe(B, clip=MATRIX_CLIP)


def _assemble_B(e_B: jax.Array, d: int, m: int) -> jax.Array:
    return _safe(e_B.reshape(d, m), clip=MATRIX_CLIP)


def make_grad_energy(layers: tuple) -> Callable:
    def _energy(x, weights, biases):
        return energy_EBM(x, weights, biases, layers)
    return jax.grad(_energy, argnums=0)


def vector_field_and_output(params: EBMParams, x: jax.Array, u: jax.Array,
                            grad_energy_fn: Callable, d: int, m: int):
    tp = params.trunk
    observed_state = _safe(x)
    x  = _soft_clip_norm(_safe(x), STATE_NORM_CLIP)
    u  = _safe(u)
    if USE_SATURATING_INPUT:
        # Hammerstein-style learnable actuator saturation: u_eff=cap*tanh(u/cap)
        # per input channel. Applied ONCE here so the dynamics (B@u), the
        # feedthrough (D_feed@u) and y_port all see the same saturated input.
        u_cap = SAT_U_MIN + jax.nn.softplus(tp.u_sat_raw)
        u = u_cap * jnp.tanh(u / u_cap)

    raw_gE = grad_energy_fn(x, params.ebm_weights, params.ebm_biases)
    gE     = _soft_clip_norm(_safe(raw_gE), GRAD_E_CLIP)
    rich_features = _field_features(tp, x)

    # issue 5: matrices assembled directly from stored entries. M may be gated by
    # the state when USE_STATE_DAMPING is on (level-dependent dissipation).
    M = _assemble_M(
        tp.L_M, d, x=x, G_damp=tp.G_damp, h_damp=tp.h_damp,
        rich_features=rich_features, W_rich_M=tp.W_rich_M,
    )
    A = _assemble_A(
        tp.e_A, d, x=x, G_A=tp.G_A, h_A=tp.h_A,
        G_A_quadratic=tp.G_A_quadratic,
        G_A_cubic=tp.G_A_cubic,
        rich_features=rich_features,
        W_rich_A=tp.W_rich_A,
    )
    B = _assemble_input_matrix(tp, x, rich_features=rich_features)

    # VF_SCALE speeds up the dynamics so dt * xdot produces meaningful per-step motion
    # (Silverbox resonance is fast relative to dt); 1.0 leaves the field unchanged.
    forced = B @ u
    xdot   = _soft_clip(_safe(VF_SCALE * ((A - M) @ gE + forced)), XDOT_CLIP)
    y_port = _soft_clip(_safe(B.T @ gE), OUTPUT_CLIP)
    if USE_IDENTITY_READOUT:
        if tp.C_state.shape[0] != d:
            raise ValueError(
                f"Identity readout requires d == n_obs, got d={d}, n_obs={tp.C_state.shape[0]}"
            )
        incompatible = USE_FEEDTHROUGH or USE_NONLINEAR_READOUT or USE_SATURATING_READOUT
        if incompatible:
            raise ValueError(
                "Identity readout requires feedthrough, nonlinear readout, and saturating readout off"
            )
        return xdot, y_port, observed_state
    # issue 1: direct linear readout from state — decouples observation from energy gradient.
    # Optional D_feed @ u feedthrough reconnects the input directly to the output.
    y_obs_lin = tp.C_state @ x + tp.c_bias
    if USE_FEEDTHROUGH:
        y_obs_lin = y_obs_lin + tp.D_feed @ u
    if USE_NONLINEAR_READOUT:
        # bounded (sigmoid in [0,1]) hidden head adds output kink/saturation;
        # residual on top of the linear map, zero-init => no change at start.
        y_obs_lin = y_obs_lin + tp.C_ro2 @ jax.nn.sigmoid(tp.C_ro1 @ x + tp.b_ro1)
    if USE_SATURATING_READOUT:
        # Learnable, asymmetric soft-clamp: y = cap*tanh(y_lin/cap) per side,
        # models a genuine physical output ceiling/floor (tank overflow / empty
        # tank) instead of the unbounded linear(+D_feed+NLRO) map above, which
        # has no way to stop y_obs from overshooting past a saturation point.
        # Continuous & C1 at 0 regardless of the (possibly different) pos/neg
        # cap values. cap = SAT_SCALE_MIN + softplus(raw) keeps caps > 0 always.
        pos_cap = SAT_SCALE_MIN + jax.nn.softplus(tp.y_sat_pos_raw)
        neg_cap = SAT_SCALE_MIN + jax.nn.softplus(tp.y_sat_neg_raw)
        y_obs_lin = jnp.where(y_obs_lin >= 0,
                               pos_cap * jnp.tanh(y_obs_lin / pos_cap),
                               neg_cap * jnp.tanh(y_obs_lin / neg_cap))
    y_obs  = _soft_clip(_safe(y_obs_lin), OUTPUT_CLIP)
    return xdot, y_port, y_obs


def ph_step(params: EBMParams, x: jax.Array, u: jax.Array,
            grad_energy_fn: Callable, d: int, m: int, dt: float):
    xdot, _, _ = vector_field_and_output(params, x, u, grad_energy_fn, d, m)
    x_next = _soft_clip_norm(_safe(x + dt * xdot), STATE_NORM_CLIP)
    _, y_port, y_obs = vector_field_and_output(params, x_next, u, grad_energy_fn, d, m)
    return x_next, y_port, y_obs


def encode_x0(params: EBMParams, y_window: jax.Array) -> jax.Array:
    """y_window is an OPAQUE per-timestep feature window, flattened and
    linearly mapped to x0 via W_enc. Normally shape [init_win, n_obs] (output
    burn-in only). When USE_INPUT_AWARE_ENCODER is enabled, callers instead
    pass the caller-built concatenation [y_window, u_window] along the last
    axis (shape [init_win, n_obs+m]) - W_enc is sized accordingly by
    init_trunk, so this function needs no change either way."""
    tp = params.trunk
    flat = _soft_clip(_safe(y_window.reshape(-1)), OUTPUT_CLIP)
    x0 = _soft_clip_norm(_safe(tp.W_enc @ flat + tp.b_enc), STATE_NORM_CLIP)
    return _soft_clip_norm(x0, X0_NORM_MAX)


def encode_x0_batch(params: EBMParams, y_windows: jax.Array) -> jax.Array:
    return jax.vmap(lambda yw: encode_x0(params, yw))(y_windows)


def model_vector_field_and_output(ebm_params, x, u, layers, grad_energy_fn, d, m):
    xdot, y_port, _ = vector_field_and_output(ebm_params, x, u, grad_energy_fn, d, m)
    return xdot, y_port
