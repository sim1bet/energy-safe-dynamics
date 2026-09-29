"""Rotational diagnostics and differentiable losses for NanoDrone."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.signal import butter, sosfiltfilt, welch



DT = 0.01
SAMPLE_RATE_HZ = 100.0
INERTIA = np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5])


def lowpass(values: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Apply the benchmark's zero-phase fourth-order Butterworth convention."""
    sos = butter(4, cutoff_hz, btype="low", fs=SAMPLE_RATE_HZ, output="sos")
    return sosfiltfilt(sos, np.asarray(values, dtype=np.float64), axis=0)


def quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_vector, left_scalar = left[..., :3], left[..., 3:]
    right_vector, right_scalar = right[..., :3], right[..., 3:]
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + np.cross(left_vector, right_vector)
    )
    scalar = left_scalar * right_scalar - np.sum(
        left_vector * right_vector, axis=-1, keepdims=True
    )
    return np.concatenate([vector, scalar], axis=-1)


def relative_rotation_log(quaternion: np.ndarray) -> np.ndarray:
    """Return Log(R_t.T R_t+1) from scalar-last body-to-world quaternions."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    inverse = quaternion[:-1].copy()
    inverse[..., :3] *= -1.0
    relative = quaternion_multiply_xyzw(inverse, quaternion[1:])
    relative = np.where((relative[..., 3] < 0.0)[..., None], -relative, relative)
    vector_norm = np.linalg.norm(relative[..., :3], axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.maximum(relative[..., 3], 0.0))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1e-12,
    )
    return relative[..., :3] * scale[..., None]


def five_point_derivative(values: np.ndarray, dt: float = DT) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (
        -values[4:] + 8.0 * values[3:-1] - 8.0 * values[1:-3] + values[:-4]
    ) / (12.0 * dt)


def measured_torque_proxy(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return interior sample indices and the 12-Hz measured torque proxy."""
    omega = np.asarray(omega, dtype=np.float64)
    omega_dot = five_point_derivative(omega)
    interior = omega[2:-2]
    angular_momentum = interior @ INERTIA.T
    torque = omega_dot @ INERTIA.T + np.cross(interior, angular_momentum)
    return np.arange(2, len(omega) - 2), lowpass(torque, 12.0)


def torque_targets_for_windows(trajectories, windows, seq_len: int) -> np.ndarray:
    """Return `[window,horizon,torque_xyz+valid]` targets aligned at x_t."""
    targets = np.zeros((len(windows.x0), seq_len, 4), dtype=np.float32)
    by_name = {trajectory.name: trajectory for trajectory in trajectories}
    proxy = {}
    for name, trajectory in by_name.items():
        indices, torque = measured_torque_proxy(trajectory.y[:, 9:12])
        values = np.zeros((len(trajectory.y), 3), dtype=np.float64)
        valid = np.zeros(len(trajectory.y), dtype=bool)
        values[indices] = torque
        valid[indices] = True
        proxy[name] = (values, valid)
    for window_index, (name, start) in enumerate(
        zip(windows.trajectory_name, windows.start_index)
    ):
        values, valid = proxy[str(name)]
        selected = slice(int(start), int(start) + seq_len)
        targets[window_index, :, :3] = values[selected]
        targets[window_index, :, 3] = valid[selected]
    return targets


def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(values - median), axis=0)
    rms_floor = 0.05 * np.sqrt(np.mean(values ** 2, axis=0))
    return np.maximum(mad_scale, np.maximum(rms_floor, 1e-12))


def spectral_noise_audit(values: np.ndarray) -> dict:
    frequencies, power = welch(
        np.asarray(values, dtype=np.float64), fs=SAMPLE_RATE_HZ, axis=0, nperseg=512
    )
    signal_band = frequencies <= 12.0
    noise_band = (frequencies >= 18.0) & (frequencies <= 45.0)
    noise_floor = np.median(power[noise_band], axis=0)
    signal_power = np.sum(power[signal_band], axis=0)
    excess_power = np.sum(
        np.maximum(power[signal_band] - noise_floor[None, :], 0.0), axis=0
    )
    return {
        "below_12hz_power": signal_power.tolist(),
        "estimated_noise_floor_psd": noise_floor.tolist(),
        "fraction_above_noise_floor": np.divide(
            excess_power, signal_power, out=np.zeros_like(excess_power), where=signal_power > 0
        ).tolist(),
    }


def rotvec_to_quaternion_jax(rotvec: jax.Array) -> jax.Array:
    angle_squared = jnp.sum(rotvec ** 2, axis=-1, keepdims=True)
    angle = jnp.sqrt(angle_squared + 1e-16)
    half = 0.5 * angle
    regular = jnp.sin(half) / angle
    series = 0.5 - angle_squared / 48.0 + angle_squared ** 2 / 3840.0
    scale = jnp.where(angle_squared > 1e-8, regular, series)
    quaternion = jnp.concatenate([rotvec * scale, jnp.cos(half)], axis=-1)
    return quaternion / jnp.sqrt(jnp.sum(quaternion ** 2, axis=-1, keepdims=True) + 1e-16)


def quaternion_multiply_jax(left: jax.Array, right: jax.Array) -> jax.Array:
    left_vector, left_scalar = left[..., :3], left[..., 3:]
    right_vector, right_scalar = right[..., :3], right[..., 3:]
    return jnp.concatenate(
        [
            left_scalar * right_vector
            + right_scalar * left_vector
            + jnp.cross(left_vector, right_vector),
            left_scalar * right_scalar
            - jnp.sum(left_vector * right_vector, axis=-1, keepdims=True),
        ],
        axis=-1,
    )


def quaternion_log_jax(quaternion: jax.Array) -> jax.Array:
    quaternion = quaternion / jnp.sqrt(
        jnp.sum(quaternion ** 2, axis=-1, keepdims=True) + 1e-16
    )
    quaternion = jnp.where((quaternion[..., 3] < 0.0)[..., None], -quaternion, quaternion)
    vector = quaternion[..., :3]
    vector_squared = jnp.sum(vector ** 2, axis=-1, keepdims=True)
    vector_norm = jnp.sqrt(vector_squared + 1e-16)
    angle = 2.0 * jnp.arctan2(vector_norm, jnp.maximum(quaternion[..., 3:], 0.0))
    regular = angle / vector_norm
    series = 2.0 + vector_squared / 3.0 + 3.0 * vector_squared ** 2 / 20.0
    return vector * jnp.where(vector_squared > 1e-8, regular, series)


def relative_rotation_log_jax(previous_rotvec: jax.Array, next_rotvec: jax.Array) -> jax.Array:
    previous = rotvec_to_quaternion_jax(previous_rotvec)
    inverse = jnp.concatenate([-previous[..., :3], previous[..., 3:]], axis=-1)
    return quaternion_log_jax(
        quaternion_multiply_jax(inverse, rotvec_to_quaternion_jax(next_rotvec))
    )


def clipped_power_weights(power: jax.Array, maximum: float = 0.20) -> jax.Array:
    """Normalize per-axis frequency weights while respecting a hard bin cap."""
    weights = power / (jnp.sum(power, axis=-1, keepdims=True) + 1e-12)

    def redistribute(_, current):
        clipped = jnp.minimum(current, maximum)
        remaining = jnp.maximum(1.0 - jnp.sum(clipped, axis=-1, keepdims=True), 0.0)
        headroom = jnp.maximum(maximum - clipped, 0.0)
        return clipped + remaining * headroom / (jnp.sum(headroom, axis=-1, keepdims=True) + 1e-12)

    return jax.lax.fori_loop(0, 8, redistribute, weights)


def complex_alignment_terms(
    predicted_spectrum: jax.Array, true_spectrum: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Aggregate batch spectra into gain, coherence, phase, and truth power."""
    cross = jnp.sum(predicted_spectrum * jnp.conj(true_spectrum), axis=0).T
    predicted_power = jnp.sum(jnp.abs(predicted_spectrum) ** 2, axis=0).T
    true_power = jnp.sum(jnp.abs(true_spectrum) ** 2, axis=0).T
    gain = jnp.sqrt((predicted_power + 1e-12) / (true_power + 1e-12))
    coherence_squared = jnp.abs(cross) ** 2 / (
        (predicted_power + 1e-12) * (true_power + 1e-12)
    )
    return gain, coherence_squared, jnp.angle(cross), true_power


def omega_sign_mask(omega: jax.Array, omega_scale: jax.Array) -> jax.Array:
    threshold = jnp.maximum(0.1, 0.25 * omega_scale)
    return jnp.abs(omega) > threshold


def make_chirp_frequency_loss_fn(
    grad_energy_fn,
    state_mean,
    state_std,
    *,
    d: int = 12,
    m: int = 4,
    dt: float = DT,
    integrator: str = "rk4",
):
    from EBM_rollout import make_batch_rollout_fn

    rollout = make_batch_rollout_fn(grad_energy_fn, d, m, dt, integrator=integrator)
    state_mean = jnp.asarray(state_mean)
    state_std = jnp.asarray(state_std)
    hann = jnp.hanning(128)[None, :, None]
    frequencies = jnp.fft.rfftfreq(128, dt)
    selected = (frequencies >= 0.8) & (frequencies <= 15.0)

    def loss_fn(params, x0_batch, u_batch, true_batch):
        predicted, _, _ = rollout(params, x0_batch, u_batch, stop_grad=False)
        true_omega = true_batch[..., 9:12] * state_std[9:12] + state_mean[9:12]
        predicted_omega = predicted[..., 9:12] * state_std[9:12] + state_mean[9:12]
        true_spectrum = jnp.fft.rfft(true_omega * hann, axis=1)[:, selected, :]
        predicted_spectrum = jnp.fft.rfft(predicted_omega * hann, axis=1)[:, selected, :]
        gain, coherence_squared, phase, true_power = complex_alignment_terms(
            predicted_spectrum, true_spectrum
        )
        weights = clipped_power_weights(true_power)
        terms = (
            jnp.log(gain) ** 2
            + 0.5 * (1.0 - coherence_squared)
            + 0.25 * (1.0 - jnp.cos(phase))
        )
        loss = jnp.mean(jnp.sum(weights * terms, axis=-1))
        return loss, {
            "frequency_gain_loss": jnp.mean(jnp.sum(weights * jnp.log(gain) ** 2, axis=-1)),
            "frequency_coherence_loss": jnp.mean(jnp.sum(weights * (1.0 - coherence_squared), axis=-1)),
            "frequency_phase_loss": jnp.mean(jnp.sum(weights * (1.0 - jnp.cos(phase)), axis=-1)),
        }

    return loss_fn