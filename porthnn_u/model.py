"""Canonical measured-input PortHNN-u dynamics and latent observation model."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ModelSpec:
    state_dim: int
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...] = (200, 200, 200)
    activation: str = "tanh"
    damping: str = "scalar_unconstrained"
    encoder_hidden_dims: tuple[int, ...] = (128, 64)
    forcing_hidden_dims: tuple[int, ...] | None = None
    forcing_parameterization: str = "direct"
    force_cap: float | None = None
    damping_floor: float = 1e-6
    sin_omega_0: float = 10.0

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.state_dim % 2:
            raise ValueError("canonical PortHNN-u requires an even state dimension")
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if any(width <= 0 for width in self.encoder_hidden_dims):
            raise ValueError("encoder_hidden_dims must contain positive widths")
        if self.forcing_hidden_dims is not None and any(width <= 0 for width in self.forcing_hidden_dims):
            raise ValueError("forcing_hidden_dims must contain positive widths")
        activation_fn(self.activation)
        if self.damping not in {
            "scalar_unconstrained",
            "vector_unconstrained",
            "scalar_negative_softplus",
            "vector_negative_softplus",
        }:
            raise ValueError(f"unsupported damping mode: {self.damping}")
        if self.forcing_parameterization not in {"direct", "bounded"}:
            raise ValueError(f"unsupported forcing parameterization: {self.forcing_parameterization}")
        if self.forcing_parameterization == "bounded" and (self.force_cap is None or self.force_cap <= 0):
            raise ValueError("bounded forcing requires a positive force_cap")
        if self.damping_floor <= 0:
            raise ValueError("damping_floor must be positive")

    @property
    def momentum_dim(self) -> int:
        return self.state_dim // 2


def activation_fn(name: str):
    if name == "tanh":
        return jnp.tanh
    if name == "sin":
        return jnp.sin
    raise ValueError(f"unsupported activation: {name}")


def _init_dense(key: jax.Array, input_dim: int, output_dim: int, *, bias: bool, initializer="xavier", omega=1.0):
    if initializer == "xavier":
        weight = jax.nn.initializers.glorot_uniform()(key, (input_dim, output_dim), jnp.float32)
    elif initializer == "siren":
        limit = (1.0 / input_dim) if omega != 1.0 else (6.0 / input_dim) ** 0.5 / omega
        weight = jax.random.uniform(key, (input_dim, output_dim), minval=-limit, maxval=limit, dtype=jnp.float32)
    else:
        raise ValueError(f"unsupported initializer: {initializer}")
    layer = {"weight": weight}
    if bias:
        layer["bias"] = jnp.zeros((output_dim,), dtype=jnp.float32)
    return layer


def _init_branch(
    key: jax.Array,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    *,
    output_bias: bool,
    activation: str = "tanh",
    sin_omega_0: float = 10.0,
):
    widths = (*hidden_dims, output_dim)
    keys = jax.random.split(key, len(widths))
    layers = []
    previous = input_dim
    for index, (layer_key, width) in enumerate(zip(keys, widths)):
        hidden = index < len(widths) - 1
        initializer = "siren" if activation == "sin" and hidden else "xavier"
        omega = sin_omega_0 if activation == "sin" and index == 0 else 1.0
        layers.append(
            _init_dense(layer_key, previous, width, bias=hidden or output_bias, initializer=initializer, omega=omega)
        )
        previous = width
    return tuple(layers)


def init_params(key: jax.Array, spec: ModelSpec, init_window: int):
    """Initialize all model components from independent PRNG streams."""
    if init_window <= 0:
        raise ValueError("init_window must be positive")
    hamiltonian_key, forcing_key, damping_key, encoder_key, readout_key = jax.random.split(key, 5)
    if spec.damping == "scalar_unconstrained":
        damping = 0.1 * jax.random.normal(damping_key, ())
    elif spec.damping == "vector_unconstrained":
        damping = 0.1 * jax.random.normal(damping_key, (spec.momentum_dim,))
    else:
        initial_raw = inverse_softplus(jnp.asarray(0.10 - spec.damping_floor, dtype=jnp.float32))
        damping = initial_raw if spec.damping == "scalar_negative_softplus" else jnp.full((spec.momentum_dim,), initial_raw)
    encoder_input_dim = init_window * (spec.input_dim + spec.output_dim)
    return {
        "hamiltonian": _init_branch(
            hamiltonian_key, spec.state_dim, spec.hidden_dims, 1, output_bias=False,
            activation=spec.activation, sin_omega_0=spec.sin_omega_0,
        ),
        "forcing": _init_branch(
            forcing_key, spec.input_dim, spec.forcing_hidden_dims or spec.hidden_dims, spec.momentum_dim, output_bias=False
        ),
        "damping": damping,
        "encoder": _init_branch(
            encoder_key,
            encoder_input_dim,
            spec.encoder_hidden_dims,
            spec.state_dim,
            output_bias=True,
        ),
        "readout": _init_dense(readout_key, spec.state_dim, spec.output_dim, bias=True),
    }


def forward_branch(layers, values: jax.Array, activation):
    hidden = values
    for layer in layers[:-1]:
        hidden = activation(hidden @ layer["weight"] + layer["bias"])
    return hidden @ layers[-1]["weight"] + layers[-1].get("bias", 0.0)


def hamiltonian(params, state: jax.Array, spec: ModelSpec) -> jax.Array:
    raw = forward_branch(params["hamiltonian"], state, activation_fn(spec.activation))[0]
    origin = forward_branch(params["hamiltonian"], jnp.zeros_like(state), activation_fn(spec.activation))[0]
    return raw - origin


def forcing(params, measured_input: jax.Array, spec: ModelSpec) -> jax.Array:
    raw = forward_branch(params["forcing"], measured_input, jnp.tanh)
    if spec.forcing_parameterization == "bounded":
        return spec.force_cap * jnp.tanh(raw / spec.force_cap)
    return raw


def inverse_softplus(value: jax.Array) -> jax.Array:
    return jnp.where(value > 20.0, value, jnp.log(jnp.expm1(value)))


def damping_vector(params, spec: ModelSpec) -> jax.Array:
    raw = params["damping"]
    if spec.damping == "scalar_unconstrained":
        return jnp.broadcast_to(raw, (spec.momentum_dim,))
    if spec.damping == "vector_unconstrained":
        return raw
    damping = -jax.nn.softplus(raw) - spec.damping_floor
    if spec.damping == "scalar_negative_softplus":
        return jnp.broadcast_to(damping, (spec.momentum_dim,))
    return damping


def vector_field(params, state: jax.Array, measured_input: jax.Array, spec: ModelSpec) -> jax.Array:
    """Evaluate [dH/dp, -dH/dq + N*dH/dp + F(u)]."""
    gradient = jax.grad(hamiltonian, argnums=1)(params, state, spec)
    gradient_q, gradient_p = jnp.split(gradient, 2)
    momentum_dot = -gradient_q + damping_vector(params, spec) * gradient_p
    momentum_dot = momentum_dot + forcing(params, measured_input, spec)
    return jnp.concatenate((gradient_p, momentum_dot))


def rk4_step(params, state: jax.Array, measured_input: jax.Array, dt: float, spec: ModelSpec) -> jax.Array:
    field = lambda value: vector_field(params, value, measured_input, spec)
    k1 = field(state)
    k2 = field(state + 0.5 * dt * k1)
    k3 = field(state + 0.5 * dt * k2)
    k4 = field(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(params, initial_state: jax.Array, inputs: jax.Array, dt: float, spec: ModelSpec) -> jax.Array:
    """Return one predicted next state for each zero-order-held input sample."""
    def step(state, value):
        next_state = rk4_step(params, state, value, dt, spec)
        return next_state, next_state

    _, states = jax.lax.scan(
        step,
        initial_state,
        inputs,
    )
    return states


def encode_initial_state(params, inputs: jax.Array, outputs: jax.Array, spec: ModelSpec) -> jax.Array:
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("encoder input and output windows must have equal length")
    features = jnp.concatenate((outputs, inputs), axis=-1).reshape(-1)
    return forward_branch(params["encoder"], features, activation_fn(spec.activation))


def decode(params, states: jax.Array) -> jax.Array:
    return states @ params["readout"]["weight"] + params["readout"]["bias"]