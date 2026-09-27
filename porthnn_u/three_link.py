"""Full-matrix, measured-input PortHNN-u for canonical six-state systems.

This is deliberately separate from ``porthnn_u_jax.py``: that file is the
frozen two-state Duffing comparison.  H retains its three-layer tanh scalar
storage architecture; J, R and G are state-dependent matrix heads.
"""
from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jnp

HIDDEN_DIMS = (200, 200, 200)

def _dense(key, n_in, n_out, bias=True):
    return {"weight": jax.nn.initializers.orthogonal()(key, (n_in, n_out)),
            "bias": jnp.zeros((n_out,)) if bias else None}

def _mlp(keys, n_in, widths, n_out, output_bias=True):
    sizes = (n_in, *widths, n_out)
    return tuple(_dense(keys[i], sizes[i], sizes[i+1], i < len(sizes)-2 or output_bias)
                 for i in range(len(sizes)-1))

def _forward(layers, x):
    h = x
    for layer in layers[:-1]: h = jnp.tanh(h @ layer["weight"] + layer["bias"])
    last = layers[-1]
    return h @ last["weight"] + (last["bias"] if last["bias"] is not None else 0.)

def initialize(seed=0, state_dim=6, input_dim=1, h_hidden_dims=HIDDEN_DIMS,
               matrix_hidden_dims=(128, 128)) -> dict[str, Any]:
    if state_dim != 6: raise ValueError("this implementation is intentionally the 3-link, six-state model")
    k = iter(jax.random.split(jax.random.PRNGKey(seed), 4))
    # Make R weak at initialization.  Only Cholesky diagonal entries receive
    # the softplus preimage; off-diagonals start at zero rather than -2.35.
    r = _mlp(jax.random.split(next(k), len(matrix_hidden_dims)+1), state_dim, matrix_hidden_dims, 21)
    diagonal_packed = jnp.asarray((0, 2, 5, 9, 14, 20))
    r_bias = jnp.zeros((21,)).at[diagonal_packed].set(-2.35)
    r = r[:-1] + ({**r[-1], "weight": .05 * r[-1]["weight"], "bias": r_bias},)
    return {"hamiltonian": _mlp(jax.random.split(next(k), len(h_hidden_dims)+1), state_dim, h_hidden_dims, 1, False),
            "j_head": _mlp(jax.random.split(next(k), len(matrix_hidden_dims)+1), state_dim, matrix_hidden_dims, 36),
            "r_head": r,
            "g_head": _mlp(jax.random.split(next(k), len(matrix_hidden_dims)+1), state_dim, matrix_hidden_dims, state_dim*input_dim)}

def hamiltonian(params, state): return jnp.squeeze(_forward(params["hamiltonian"], state), -1)
def matrices(params, state):
    n = 6
    a = _forward(params["j_head"], state).reshape(n,n); j = .5*(a-a.T)
    raw = _forward(params["r_head"], state); idx = jnp.tril_indices(n)
    # Do not allow a globally enabled JAX x64 mode to promote a float32
    # checkpoint/state during lax.scan.
    l = jnp.zeros((n, n), dtype=state.dtype).at[idx].set(raw)
    l = l.at[jnp.diag_indices(n)].set(
        jax.nn.softplus(jnp.diag(l)) + jnp.asarray(1e-4, dtype=state.dtype)
    )
    g_flat = _forward(params["g_head"], state)
    return j, l @ l.T, g_flat.reshape(n, g_flat.size // n)
def vector_field(params, state, measured_input):
    grad_h = jax.grad(hamiltonian, argnums=1)(params, state)
    j, r, g = matrices(params, state)
    return (j-r) @ grad_h + g @ jnp.asarray(measured_input)
def rk4_step(params, state, measured_input, dt):
    f=lambda x: vector_field(params,x,measured_input); k1=f(state); k2=f(state+.5*dt*k1); k3=f(state+.5*dt*k2); k4=f(state+dt*k3)
    return state + dt*(k1+2*k2+2*k3+k4)/6
def rollout(params, initial_state, inputs, dt):
    def step(state, measured_input):
        next_state = rk4_step(params, state, measured_input, dt)
        return next_state, next_state
    _, values=jax.lax.scan(step, initial_state, inputs)
    return jnp.concatenate((initial_state[None],values))
def parameter_counts(params):
    return {name:int(sum(x.size for x in jax.tree_util.tree_leaves(params[name]))) for name in ("hamiltonian","j_head","r_head","g_head")}
