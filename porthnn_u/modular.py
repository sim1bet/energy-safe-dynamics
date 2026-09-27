"""Modular $(H, J, R, G)$ port-Hamiltonian latent dynamics.

Migration note: legacy PortHNN-u uses ``[dH/dp, -dH/dq + N*dH/dp + F(u)]``.
In ``legacy_porthnn_u`` mode this module delegates exactly to that implementation.
Generalized modes use ``(J(z) - R(z)) grad(H(z)) + G(z) u`` and have a distinct
parameter schema; old checkpoints are therefore supported only in legacy mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .model import ModelSpec, _init_branch, _init_dense, activation_fn, damping_vector, decode, encode_initial_state, forward_branch, init_params, inverse_softplus, vector_field as legacy_vector_field


@dataclass(frozen=True)
class ModularSpec:
    state_dim: int
    input_dim: int
    output_dim: int
    storage_hidden_dims: tuple[int, ...] = (200, 200, 200)
    encoder_hidden_dims: tuple[int, ...] = (128, 64)
    matrix_hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    interconnection: str = "canonical_plus_learned_skew"
    dissipation: str = "learned_full_psd"
    input_map: str = "learned_full_matrix"
    legacy_damping: str = "scalar_negative_softplus"
    damping_floor: float = 1e-4
    r_diagonal: float = 0.05
    readout: str = "affine_normalized"
    encoder_type: str = "mlp"
    encoder_features: str = "raw"

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.state_dim % 2:
            raise ValueError("modular PortHNN-u requires an even state_dim")
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if self.interconnection not in {"canonical", "learned_skew", "canonical_plus_learned_skew"}:
            raise ValueError("unsupported interconnection module")
        if self.dissipation not in {"legacy_scalar", "learned_full_psd", "momentum_block_psd", "full_psd_constant", "full_psd_state_dependent"}:
            raise ValueError("unsupported dissipation module")
        if self.input_map not in {"legacy_force", "learned_full_matrix", "full_matrix_constant", "full_matrix_state_dependent"}:
            raise ValueError("unsupported input map module")
        if self.r_diagonal <= 0:
            raise ValueError("r_diagonal must be positive")
        if self.readout not in {"affine_normalized", "softabs_linear", "norm2"}:
            raise ValueError("unsupported modular readout")
        if self.encoder_type not in {"mlp", "gru", "tcn"}:
            raise ValueError("unsupported causal encoder")
        if self.encoder_features not in {"raw", "first_difference", "rich_causal"}:
            raise ValueError("unsupported encoder feature set")
        activation_fn(self.activation)

    @property
    def packed_skew_size(self) -> int:
        return self.state_dim * (self.state_dim - 1) // 2

    @property
    def packed_lower_size(self) -> int:
        return self.state_dim * (self.state_dim + 1) // 2

    @property
    def momentum_packed_lower_size(self) -> int:
        half = self.state_dim // 2
        return half * (half + 1) // 2


def canonical_matrix(spec: ModularSpec) -> jax.Array:
    half = spec.state_dim // 2
    zero = jnp.zeros((half, half), dtype=jnp.float32)
    identity = jnp.eye(half, dtype=jnp.float32)
    return jnp.block([[zero, identity], [-identity, zero]])


def _legacy_spec(spec: ModularSpec) -> ModelSpec:
    return ModelSpec(
        spec.state_dim, spec.input_dim, spec.output_dim, spec.storage_hidden_dims,
        spec.activation, spec.legacy_damping, spec.encoder_hidden_dims,
        damping_floor=spec.damping_floor,
    )


def init_modular_params(key: jax.Array, spec: ModularSpec, init_window: int, *, legacy_porthnn_u: bool = False):
    """Initialize either explicit legacy-compatible or generalized parameters."""
    if legacy_porthnn_u:
        return init_params(key, _legacy_spec(spec), init_window)
    keys = jax.random.split(key, 8)
    feature_width = {"raw": spec.input_dim + spec.output_dim, "first_difference": 2 * (spec.input_dim + spec.output_dim), "rich_causal": 2 * (spec.input_dim + spec.output_dim) + 5}[spec.encoder_features]
    encoder_input = init_window * feature_width
    r_dimension = spec.state_dim // 2 if spec.dissipation == "momentum_block_psd" else spec.state_dim
    r_outputs = r_dimension * (r_dimension + 1) // 2
    dissipation = _init_branch(keys[2], spec.state_dim, spec.matrix_hidden_dims, r_outputs, output_bias=True)
    rows, columns = jnp.tril_indices(r_dimension)
    diagonal = rows == columns
    diagonal_raw = jnp.log(jnp.expm1(jnp.sqrt(spec.r_diagonal) - 1e-5))
    dissipation_last = dict(dissipation[-1])
    dissipation_last["bias"] = dissipation_last["bias"].at[diagonal].set(diagonal_raw)
    interconnection = _init_branch(keys[1], spec.state_dim, spec.matrix_hidden_dims, spec.packed_skew_size, output_bias=True)
    if spec.interconnection == "canonical_plus_learned_skew":
        interconnection_last = dict(interconnection[-1])
        interconnection_last["weight"] = 1e-3 * interconnection_last["weight"]
        interconnection_last["bias"] = 1e-3 * interconnection_last["bias"]
        interconnection = (*interconnection[:-1], interconnection_last)
    parameters = {
        "hamiltonian": _init_branch(keys[0], spec.state_dim, spec.storage_hidden_dims, 1, output_bias=False, activation=spec.activation),
        "J": interconnection,
        "G": _init_branch(keys[3], spec.state_dim, spec.matrix_hidden_dims, spec.state_dim * spec.input_dim, output_bias=True),
        "encoder": _init_branch(keys[4], encoder_input, spec.encoder_hidden_dims, spec.state_dim, output_bias=True),
        "readout": _init_dense(
            jax.random.fold_in(keys[4], 1), spec.state_dim,
            2 if spec.readout == "norm2" else spec.output_dim, bias=True,
        ),
    }
    if spec.encoder_type == "gru":
        hidden = spec.encoder_hidden_dims[0]
        parameters["encoder_gru"] = {
            "input": _init_dense(keys[5], feature_width, 3 * hidden, bias=True),
            "hidden": _init_dense(keys[6], hidden, 3 * hidden, bias=False),
            "output": _init_dense(keys[7], hidden, spec.state_dim, bias=True),
        }
    elif spec.encoder_type == "tcn":
        width = spec.encoder_hidden_dims[0]
        parameters["encoder_tcn"] = {
            "input": _init_dense(keys[5], feature_width, width, bias=True),
            "layers": tuple(_init_dense(jax.random.fold_in(keys[6], index), 2 * width, width, bias=True) for index in range(3)),
            "output": _init_dense(keys[7], width, spec.state_dim, bias=True),
        }
    if spec.dissipation in {"learned_full_psd", "full_psd_state_dependent", "momentum_block_psd"}:
        parameters["R"] = (*dissipation[:-1], dissipation_last)
    elif spec.dissipation == "full_psd_constant":
        parameters["R_constant"] = dissipation_last["bias"]
    if spec.input_map == "full_matrix_constant":
        parameters["G_constant"] = 0.05 * jax.random.normal(keys[3], (spec.state_dim, spec.input_dim))
    if spec.dissipation == "legacy_scalar":
        parameters["damping"] = inverse_softplus(jnp.asarray(0.10 - spec.damping_floor, dtype=jnp.float32))
    return parameters


def causal_encoder_features(inputs: jax.Array, outputs: jax.Array, spec: ModularSpec, dt: float, statistics=None) -> jax.Array:
    """Build causal burn-in features; statistics are fit from training-fold histories."""
    raw = jnp.concatenate((inputs, outputs), axis=-1)
    if spec.encoder_features == "raw":
        return raw
    first = jnp.concatenate((jnp.zeros_like(raw[:1]), raw[1:] - raw[:-1]), axis=0)
    if spec.encoder_features == "first_difference":
        return jnp.concatenate((raw, first), axis=-1)
    second = jnp.concatenate((jnp.zeros_like(raw[:1]), first[1:] - first[:-1]), axis=0)
    cumulative_input = jnp.cumsum(inputs, axis=0) * dt
    cumulative_work = jnp.cumsum(inputs * outputs, axis=0) * dt
    elapsed = jnp.arange(raw.shape[0], dtype=raw.dtype)[:, None] / jnp.maximum(raw.shape[0] - 1, 1)
    features = jnp.concatenate((raw, first, second, cumulative_input, cumulative_work, elapsed), axis=-1)
    if statistics is None:
        return features
    return (features - statistics[0]) / statistics[1]


def modular_encode_initial_state(params, inputs: jax.Array, outputs: jax.Array, spec: ModularSpec, dt: float = 1.0, feature_statistics=None) -> jax.Array:
    if spec.encoder_type == "mlp":
        features = causal_encoder_features(inputs, outputs, spec, dt, feature_statistics)
        return forward_branch(params["encoder"], features.reshape(-1), jnp.tanh)
    features = causal_encoder_features(inputs, outputs, spec, dt, feature_statistics)
    if spec.encoder_type == "gru":
        branch = params["encoder_gru"]
        hidden_size = branch["output"]["weight"].shape[0]
        def step(hidden, feature):
            input_parts = feature @ branch["input"]["weight"] + branch["input"]["bias"]
            hidden_parts = hidden @ branch["hidden"]["weight"]
            input_z, input_r, input_n = jnp.split(input_parts, 3)
            hidden_z, hidden_r, hidden_n = jnp.split(hidden_parts, 3)
            z, r = jax.nn.sigmoid(input_z + hidden_z), jax.nn.sigmoid(input_r + hidden_r)
            candidate = jnp.tanh(input_n + r * hidden_n)
            return (1.0 - z) * hidden + z * candidate, None
        hidden, _ = jax.lax.scan(step, jnp.zeros((hidden_size,), dtype=features.dtype), features)
        return hidden @ branch["output"]["weight"] + branch["output"]["bias"]
    branch = params["encoder_tcn"]
    states = jnp.tanh(features @ branch["input"]["weight"] + branch["input"]["bias"])
    for layer, dilation in zip(branch["layers"], (1, 2, 4)):
        delayed = jnp.concatenate((jnp.zeros_like(states[:dilation]), states[:-dilation]), axis=0)
        stacked = jnp.concatenate((states, delayed), axis=-1)
        states = jnp.tanh(stacked @ layer["weight"] + layer["bias"] + states)
    return states[-1] @ branch["output"]["weight"] + branch["output"]["bias"]


def storage_apply(params, state: jax.Array, spec: ModularSpec) -> jax.Array:
    raw = forward_branch(params["hamiltonian"], state, activation_fn(spec.activation))[0]
    origin = forward_branch(params["hamiltonian"], jnp.zeros_like(state), activation_fn(spec.activation))[0]
    return raw - origin


def modular_decode(params, states: jax.Array, spec: ModularSpec, output_mean=None, output_scale=None) -> jax.Array:
    """Decode in normalized space, with optional physical nonnegative readouts."""
    raw = states @ params["readout"]["weight"] + params["readout"]["bias"]
    if spec.readout == "affine_normalized":
        return raw
    if output_mean is None or output_scale is None:
        raise ValueError("physical magnitude readouts require output normalization statistics")
    epsilon = 1e-3 * output_scale
    if spec.readout == "softabs_linear":
        physical = jnp.sqrt(raw ** 2 + epsilon ** 2) - epsilon
    else:
        physical = jnp.sqrt(jnp.sum(raw ** 2, axis=-1, keepdims=True) + epsilon ** 2) - epsilon
    return (physical - output_mean) / output_scale


def _packed_skew(values: jax.Array, dimension: int) -> jax.Array:
    rows, columns = jnp.triu_indices(dimension, 1)
    matrix = jnp.zeros((dimension, dimension), dtype=values.dtype)
    matrix = matrix.at[rows, columns].set(values)
    return matrix - matrix.T


def _packed_cholesky(values: jax.Array, dimension: int) -> jax.Array:
    rows, columns = jnp.tril_indices(dimension)
    diagonal = rows == columns
    adjusted = jnp.where(diagonal, jax.nn.softplus(values) + 1e-5, values)
    lower = jnp.zeros((dimension, dimension), dtype=values.dtype)
    lower = lower.at[rows, columns].set(adjusted)
    return lower


def interconnection_apply(params, state: jax.Array, spec: ModularSpec) -> jax.Array:
    base = canonical_matrix(spec) if spec.interconnection in {"canonical", "canonical_plus_learned_skew"} else jnp.zeros((spec.state_dim, spec.state_dim), dtype=state.dtype)
    if spec.interconnection == "canonical":
        return base
    return base + _packed_skew(forward_branch(params["J"], state, jnp.tanh), spec.state_dim)


def dissipation_apply(params, state: jax.Array, spec: ModularSpec) -> jax.Array:
    if spec.dissipation == "legacy_scalar":
        damping = damping_vector(params, _legacy_spec(spec))
        half = spec.state_dim // 2
        return jnp.diag(jnp.concatenate((jnp.zeros(half), -damping)))
    if spec.dissipation == "full_psd_constant":
        lower = _packed_cholesky(params["R_constant"], spec.state_dim)
    elif spec.dissipation == "momentum_block_psd":
        lower = _packed_cholesky(forward_branch(params["R"], state, jnp.tanh), spec.state_dim // 2)
        zero = jnp.zeros((spec.state_dim // 2, spec.state_dim // 2), dtype=state.dtype)
        return jnp.block([[zero, zero], [zero, lower @ lower.T]])
    else:
        lower = _packed_cholesky(forward_branch(params["R"], state, jnp.tanh), spec.state_dim)
    return lower @ lower.T


def input_map_apply(params, state: jax.Array, spec: ModularSpec) -> jax.Array:
    if spec.input_map == "legacy_force":
        raise ValueError("legacy_force is only valid through legacy_porthnn_u compatibility mode")
    if spec.input_map == "full_matrix_constant":
        return params["G_constant"]
    return forward_branch(params["G"], state, jnp.tanh).reshape((spec.state_dim, spec.input_dim))


def modular_vector_field(params, state: jax.Array, measured_input: jax.Array, spec: ModularSpec, *, legacy_porthnn_u: bool = False, diagnostics: bool = False):
    if legacy_porthnn_u:
        result = legacy_vector_field(params, state, measured_input, _legacy_spec(spec))
        return (result, {}) if diagnostics else result
    gradient = jax.grad(storage_apply, argnums=1)(params, state, spec)
    interconnection = interconnection_apply(params, state, spec)
    dissipation = dissipation_apply(params, state, spec)
    input_map = input_map_apply(params, state, spec)
    conservative = interconnection @ gradient
    dissipative = -(dissipation @ gradient)
    controlled = input_map @ measured_input
    result = conservative + dissipative + controlled
    if diagnostics:
        return result, {"gradient": gradient, "J": interconnection, "R": dissipation, "G": input_map, "conservative": conservative, "dissipative": dissipative, "controlled": controlled}
    return result


def modular_rk4_step(params, state: jax.Array, measured_input: jax.Array, dt: float, spec: ModularSpec, *, legacy_porthnn_u: bool = False) -> jax.Array:
    field = lambda value: modular_vector_field(params, value, measured_input, spec, legacy_porthnn_u=legacy_porthnn_u)
    k1 = field(state)
    k2 = field(state + 0.5 * dt * k1)
    k3 = field(state + 0.5 * dt * k2)
    k4 = field(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def modular_rollout(params, initial_state: jax.Array, inputs: jax.Array, dt: float, spec: ModularSpec, *, legacy_porthnn_u: bool = False) -> jax.Array:
    def step(state, measured_input):
        next_state = modular_rk4_step(params, state, measured_input, dt, spec, legacy_porthnn_u=legacy_porthnn_u)
        return next_state, next_state
    return jax.lax.scan(step, initial_state, inputs)[1]


def power_diagnostics(params, state: jax.Array, measured_input: jax.Array, spec: ModularSpec) -> dict:
    """Evaluate pH power components and the algebraic gradient-flow residual."""
    vector, details = modular_vector_field(params, state, measured_input, spec, diagnostics=True)
    gradient = details["gradient"]
    p_j = gradient @ details["conservative"]
    p_r = gradient @ details["dissipative"]
    p_g = gradient @ details["controlled"]
    derivative = jax.grad(storage_apply, argnums=1)(params, state, spec) @ vector
    return {"P_J": p_j, "P_R": p_r, "P_G": p_g, "dot_H": derivative, "power_balance_residual": derivative - (p_j + p_r + p_g)}