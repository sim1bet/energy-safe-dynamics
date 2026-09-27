"""JAX reimplementation of the Desai et al. PortHNN with measured input.

The only architectural adaptation is F(t) -> F(u). The Hamiltonian branch,
canonical interconnection, scalar damping, and three-layer forcing branch are
kept separate as in the published model.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


HIDDEN_DIMS = (200, 200, 200)


def _dense(key: jax.Array, input_dim: int, output_dim: int, *, bias: bool = True) -> dict:
    weight = jax.nn.initializers.orthogonal()(key, (input_dim, output_dim))
    return {
        "weight": weight,
        "bias": jnp.zeros((output_dim,)) if bias else None,
    }


def _branch(keys, input_dim: int, hidden_dims: tuple[int, ...]) -> tuple[dict, ...]:
    dimensions = (input_dim, *hidden_dims, 1)
    return tuple(
        _dense(keys[index], dimensions[index], dimensions[index + 1],
               bias=index < len(dimensions) - 2)
        for index in range(len(dimensions) - 1)
    )


def initialize(
    seed: int = 0,
    width: int | None = None,
    hidden_dims: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if width is not None and hidden_dims is not None:
        raise ValueError("specify width or hidden_dims, not both")
    hidden_dims = (width,) * 3 if width is not None else (hidden_dims or HIDDEN_DIMS)
    if not hidden_dims or any(dimension <= 0 for dimension in hidden_dims):
        raise ValueError("hidden_dims must contain positive integers")
    branch_key_count = len(hidden_dims) + 1
    keys = jax.random.split(jax.random.PRNGKey(seed), 2 * branch_key_count + 1)
    return {
        "hamiltonian": _branch(keys[:branch_key_count], 2, hidden_dims),
        "forcing": _branch(keys[branch_key_count:2 * branch_key_count], 1, hidden_dims),
        "damping": 0.1 * jax.random.normal(keys[-1], ()),
    }


def _forward(layers: tuple[dict, ...], values: jax.Array) -> jax.Array:
    hidden = values
    for layer in layers[:-1]:
        hidden = jnp.tanh(hidden @ layer["weight"] + layer["bias"])
    return (hidden @ layers[-1]["weight"])[0]


def hamiltonian(params: dict, state: jax.Array) -> jax.Array:
    return _forward(params["hamiltonian"], state)


def forcing(params: dict, measured_input: jax.Array) -> jax.Array:
    return _forward(params["forcing"], jnp.reshape(measured_input, (1,)))


def vector_field(params: dict, state: jax.Array, measured_input: jax.Array) -> jax.Array:
    gradient = jax.grad(hamiltonian, argnums=1)(params, state)
    q_dot = gradient[1]
    p_dot = -gradient[0] + params["damping"] * q_dot + forcing(params, measured_input)
    return jnp.asarray([q_dot, p_dot])


def rk4_step(params: dict, state: jax.Array, measured_input: jax.Array, dt: float) -> jax.Array:
    field = lambda value: vector_field(params, value, measured_input)
    k1 = field(state)
    k2 = field(state + 0.5 * dt * k1)
    k3 = field(state + 0.5 * dt * k2)
    k4 = field(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(params: dict, initial_state: jax.Array, inputs: jax.Array, dt: float) -> jax.Array:
    def step(state, measured_input):
        next_state = rk4_step(params, state, measured_input, dt)
        return next_state, next_state

    _, states = jax.lax.scan(step, initial_state, inputs)
    return jnp.concatenate([initial_state[None, :], states], axis=0)


def parameter_count(params: dict) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def save_parameters(path, params: dict) -> None:
    arrays = {"damping": np.asarray(params["damping"])}
    for branch_name in ("hamiltonian", "forcing"):
        for index, layer in enumerate(params[branch_name]):
            arrays[f"{branch_name}_weight_{index}"] = np.asarray(layer["weight"])
            if layer["bias"] is not None:
                arrays[f"{branch_name}_bias_{index}"] = np.asarray(layer["bias"])
    np.savez_compressed(path, **arrays)


def load_parameters(path) -> dict:
    arrays = np.load(path)
    # Checkpoints may be stored as float64, while release data is float32.
    # Use one runtime dtype throughout a lax.scan rollout.
    params = {"damping": jnp.asarray(arrays["damping"], dtype=jnp.float32)}
    for branch_name in ("hamiltonian", "forcing"):
        layers = []
        layer_count = sum(key.startswith(f"{branch_name}_weight_") for key in arrays.files)
        for index in range(layer_count):
            bias_key = f"{branch_name}_bias_{index}"
            layers.append({
                "weight": jnp.asarray(arrays[f"{branch_name}_weight_{index}"], dtype=jnp.float32),
                "bias": jnp.asarray(arrays[bias_key], dtype=jnp.float32) if bias_key in arrays.files else None,
            })
        params[branch_name] = tuple(layers)
    return params
