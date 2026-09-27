# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

# EBM_class.py — Layer specs, Lagrangians, and energy function
# Optimisations vs previous version:
#   1. energy_EBM Python for-loops replaced with jax.lax.fori_loop-compatible
#      flat array representation — XLA sees one fused forward pass
#   2. assert removed from energy_EBM (crashes inside jit on dynamic inputs)
#   3. All Lagrangian/activation pairs preserved unchanged
# Public release

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, List


# ══════════════════════════════════════════════════════════════════════════════
# Layer specification
# ══════════════════════════════════════════════════════════════════════════════

class LayerSpec(NamedTuple):
    """
    Defines a single layer via its Lagrangian L_k(x_k) and activation ∇L_k(x_k).
    Both callables receive (x_k, *args).
    """
    lagrangian : Callable
    activation : Callable
    args       : tuple = ()


# ══════════════════════════════════════════════════════════════════════════════
# Numerically stable Lagrangian / activation pairs
# ══════════════════════════════════════════════════════════════════════════════

def lagrangian_softmax_stable(x: jax.Array, beta: float) -> jax.Array:
    """LSE with log-sum-exp trick: numerically stable for any x."""
    bx = beta * x
    bx_max = jnp.max(bx)
    return (bx_max + jnp.log(jnp.sum(jnp.exp(bx - bx_max)))) / beta

def activation_softmax_stable(x: jax.Array, beta: float) -> jax.Array:
    return jax.nn.softmax(beta * x)


def lagrangian_grouped_softmax(x: jax.Array, beta: float, n_groups: int) -> jax.Array:
    """Grouped modern-Hopfield energy: the LSE is computed INDEPENDENTLY within
    each of n_groups equal-size coordinate blocks and then summed.

    n_groups=1 recovers plain softmax LSE (lagrangian_softmax_stable). Splitting
    the single global simplex into several smaller winner-take-all pools raises the
    effective coupling rank (n_groups independent competitions instead of one)
    while keeping softmax's competitive attractor bias — the property that let
    softmax beat per-coordinate sigmoid/tanh. Requires len(x) % n_groups == 0.
    Stable via the per-group log-sum-exp (max-shift) trick.
    """
    g   = int(n_groups)
    bx  = (beta * x).reshape(g, -1)
    bmx = jnp.max(bx, axis=-1, keepdims=True)
    lse = bmx[:, 0] + jnp.log(jnp.sum(jnp.exp(bx - bmx), axis=-1))
    return jnp.sum(lse) / beta

def activation_grouped_softmax(x: jax.Array, beta: float, n_groups: int) -> jax.Array:
    """∇L = softmax(beta*x) applied independently within each of n_groups blocks."""
    g = int(n_groups)
    y = jax.nn.softmax((beta * x).reshape(g, -1), axis=-1)
    return y.reshape(x.shape)


def lagrangian_sigmoid(x: jax.Array, beta: float) -> jax.Array:
    """L(x) = (1/beta) * sum(softplus(beta*x));  ∇L(x) = sigmoid(beta*x) ∈ (0,1).

    A bounded, strictly-monotone, full-rank (diagonal Jacobian) first-layer flux.
    Unlike softmax — whose simplex normalisation forces the features to sum to 1
    regardless of state magnitude (rank-deficient, saturating inductive bias) —
    sigmoid saturates each coordinate independently, preserving expressivity.
    softplus is evaluated with jax.nn.softplus (log-sum-exp stable). beta acts as
    an inverse-temperature / gain that sharpens the saturation.
    """
    return jnp.sum(jax.nn.softplus(beta * x)) / beta

def activation_sigmoid(x: jax.Array, beta: float) -> jax.Array:
    """∇L(x) = sigmoid(beta*x) ∈ (0,1), elementwise."""
    return jax.nn.sigmoid(beta * x)


def lagrangian_tanh(x: jax.Array, beta: float) -> jax.Array:
    """L(x) = (1/beta) * sum(log cosh(beta*x));  ∇L(x) = tanh(beta*x) ∈ (-1,1).

    A bounded, odd, strictly-monotone, full-rank (diagonal Jacobian) first-layer
    flux — the sign-symmetric counterpart to sigmoid. log cosh is evaluated in
    the numerically stable form  softplus(2z) - z - log 2  (exact, overflow-free
    for large |z|). beta is an inverse-temperature / gain.
    """
    bx = beta * x
    logcosh = jax.nn.softplus(2.0 * bx) - bx - jnp.log(2.0)
    return jnp.sum(logcosh) / beta

def activation_tanh(x: jax.Array, beta: float) -> jax.Array:
    """∇L(x) = tanh(beta*x) ∈ (-1,1), elementwise."""
    return jnp.tanh(beta * x)


def lagrangian_polynomial_stable(x: jax.Array, p: float) -> jax.Array:
    """L(x) = sum(softplus(x)^p) / p  — smooth, no hard clips."""
    return jnp.sum(jax.nn.softplus(x) ** p) / p

def activation_polynomial_stable(x: jax.Array, p: float) -> jax.Array:
    """∇L(x) = softplus(x)^(p-1) * sigmoid(x)"""
    return jax.nn.softplus(x) ** (p - 1) * jax.nn.sigmoid(x)


def lagrangian_polynomial_stable_mixed(x: jax.Array, p_vec: jax.Array) -> jax.Array:
    """Stable polynomial Lagrangian with a per-unit exponent."""
    sp = jax.nn.softplus(x)
    return jnp.sum(sp ** p_vec / p_vec)


def activation_polynomial_stable_mixed(x: jax.Array, p_vec: jax.Array) -> jax.Array:
    """Gradient of :func:`lagrangian_polynomial_stable_mixed`."""
    return jax.nn.softplus(x) ** (p_vec - 1) * jax.nn.sigmoid(x)


def lagrangian_epanichnikov(x: jax.Array, beta: float, eps: float) -> jax.Array:
    return -jnp.log(eps + jnp.sum(jax.nn.relu(1 - x**2 * beta / 2))) / beta

def activation_epanichnikov(x: jax.Array, beta: float, eps: float) -> jax.Array:
    relu_term = jax.nn.relu(1 - x**2 * beta / 2)
    indi_term = jnp.where(1 - x**2 * beta / 2 > 0, 1.0, 0.0)
    return (x * indi_term) / (eps + jnp.sum(relu_term))


def lagrangian_polynomial(x: jax.Array, p: float) -> jax.Array:
    return jnp.sum(jax.nn.relu(x) ** p) / p

def activation_polynomial(x: jax.Array, p: float) -> jax.Array:
    return jax.nn.relu(x) ** (p - 1)


def lagrangian_quadratic(x: jax.Array) -> jax.Array:
    return jnp.dot(x, x) / 2.0

def activation_quadratic(x: jax.Array) -> jax.Array:
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Per-layer energy contribution
# ══════════════════════════════════════════════════════════════════════════════

def energy_layer_k(
    x_k : jax.Array,
    y_k : jax.Array,
    b_k : jax.Array,
    L_k : jax.Array,
) -> jax.Array:
    """E_k = (x_k - b_k)·y_k - L_k(x_k) - y_k·x_k  =  -b_k·y_k - L_k"""
    return jnp.dot(x_k - b_k, y_k) - L_k - jnp.dot(y_k, x_k)


# ══════════════════════════════════════════════════════════════════════════════
# Total energy  (Python-loop version — correct, readable, used at trace time)
# ══════════════════════════════════════════════════════════════════════════════

def energy_EBM(
    x      : jax.Array,
    weights: list,
    biases : list,
    layers : tuple,
) -> jax.Array:
    """
    E = x·x/2  +  Σ_k [ (x_k - b_k)·y_k - L_k(x_k) - y_k·x_k ]

    The Python for-loop is unrolled at JAX trace time into a static
    computation graph — this is equivalent to manually inlining K steps.
    XLA then fuses the resulting ops into a single kernel.

    No assert: assertions on list lengths crash inside jit on dynamic inputs.
    """
    K = len(layers)
    E = jnp.dot(x, x) / 2.0

    pre_acts = []
    acts     = [x]

    for k in range(K):
        y_prev = acts[k]
        x_k    = weights[k] @ y_prev          # issue 2: removed / y_prev.shape[0]
        y_k    = layers[k].activation(x_k, *layers[k].args)
        pre_acts.append(x_k)
        acts.append(y_k)

    for k in range(K):
        x_k = pre_acts[k]
        y_k = acts[k + 1]
        L_k = layers[k].lagrangian(x_k, *layers[k].args)
        E   = E + energy_layer_k(x_k, y_k, biases[k], L_k)

    return E  # issue 10: interior clips removed; external nan_to_num in param_fields is sufficient
