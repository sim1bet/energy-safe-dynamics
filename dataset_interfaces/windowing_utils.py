"""windowing_utils.py — shared multi-trajectory BPTT window builders.

Both new experiments (Duffing double-well, Deep-Dissipative n-link) generate
*many independent short trajectories* rather than one long benchmark series,
so the existing ``Interface_code/*_main.py`` pattern (``make_dataset`` slicing
overlapping windows out of a single series) does not apply directly: sliding a
window across a trajectory boundary would silently mix two unrelated episodes.
These helpers slide only *within* each trajectory.
"""
from __future__ import annotations

import numpy as np


def make_state_windows(z_traj: np.ndarray, u_traj: np.ndarray, seq_len: int, stride: int) -> tuple:
    """State-observed windows: x0 is the TRUE state at the window start.

    ``z_traj``: [n_traj, T+1, d] (index 0 is the initial condition).
    ``u_traj``: [n_traj, T, m].
    Returns ``(x0, u_seg, z_seg_true)`` with ``z_seg_true[k]`` the state AFTER
    applying ``u_seg[k]`` (matches ``EBM_rollout.make_rollout_fn``'s indexing).
    """
    n_traj, T_plus_1, d = z_traj.shape
    T = T_plus_1 - 1
    x0s, u_segs, z_segs = [], [], []
    for i in range(n_traj):
        for s in range(0, T - seq_len + 1, stride):
            x0s.append(z_traj[i, s])
            u_segs.append(u_traj[i, s:s + seq_len])
            z_segs.append(z_traj[i, s + 1:s + 1 + seq_len])
    return (np.stack(x0s).astype(np.float32),
            np.stack(u_segs).astype(np.float32),
            np.stack(z_segs).astype(np.float32))


def make_encoder_windows(y_traj: np.ndarray, u_traj: np.ndarray,
                         init_win: int, seq_len: int, stride: int) -> tuple:
    """Output-only (latent) windows: burn-in + rollout segment, per trajectory.

    Mirrors ``Silverbox_main.make_dataset``'s ``(y_win, u_win, u_seg, y_seg)``
    convention exactly, but never slides a window across a trajectory
    boundary. ``y_traj``/``u_traj``: [n_traj, T, n_obs]/[n_traj, T, m].
    """
    n_traj, T, n_obs = y_traj.shape
    seg_len = init_win + seq_len
    y_win, u_win, u_seg, y_seg = [], [], [], []
    for i in range(n_traj):
        for s in range(0, T - seg_len + 1, stride):
            y_win.append(y_traj[i, s:s + init_win])
            u_win.append(u_traj[i, s:s + init_win])
            u_seg.append(u_traj[i, s + init_win:s + seg_len])
            y_seg.append(y_traj[i, s + init_win:s + seg_len])
    return (np.stack(y_win).astype(np.float32),
            np.stack(u_win).astype(np.float32),
            np.stack(u_seg).astype(np.float32),
            np.stack(y_seg).astype(np.float32))
