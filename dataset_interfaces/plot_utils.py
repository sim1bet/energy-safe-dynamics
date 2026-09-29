# plot_utils.py — dataset-agnostic visualisation helpers for the EBM benchmark suite.
# Pure array-in / file-out; no model or dataset assumptions so any entry point
# (Silverbox, Cascaded_Tanks, EMPS, WienerHammerstein, CED, ...) can reuse it.
# Public release

import os
import matplotlib
matplotlib.use("Agg")  # headless: safe on login/compute nodes without a display
import matplotlib.pyplot as plt
import numpy as np


def _as_1d(a: np.ndarray) -> np.ndarray:
    """Flatten a [T] or [T, 1] array to [T]; leave [T, C] (C>1) untouched."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    return a


def plot_true_vs_pred(
    y_true,
    y_pred,
    save_path: str,
    dt: float = None,
    unit_label: str = "",
    title: str = None,
    rmse: float = None,
    nrmse: float = None,
    max_points: int = None,
    u_input=None,
    u_label: str = "input u (normalised)",
) -> str:
    """Plot true vs reconstructed test signal (+ per-sample error) and save a PNG.

    Dataset-agnostic: pass the ground-truth and predicted output trajectories in
    whatever physical unit you want on the axis. Handles single-channel ([T] or
    [T,1]) and multi-channel ([T,C]) outputs (one row of panels per channel).

    Parameters
    ----------
    y_true, y_pred : array-like, shape [T], [T,1] or [T,C]
        Ground-truth and predicted output trajectories (already in ``unit_label``).
    save_path : str
        Destination PNG path; parent directories are created if needed.
    dt : float, optional
        Sample time (s). If given, the x-axis is time in seconds, else sample index.
    unit_label : str
        Physical unit shown on the y-axis (e.g. 'mV', 'V').
    title : str, optional
        Figure title; a default is built from ``rmse``/``nrmse`` when omitted.
    rmse, nrmse : float, optional
        Metrics annotated in the title/legend.
    max_points : int, optional
        If set and T exceeds it, the series are uniformly subsampled for plotting
        (keeps very long trajectories, e.g. WienerHammerstein, light to render).
    u_input : array-like, shape [T] or [T,1], optional
        Control input trajectory aligned sample-for-sample with ``y_true``/
        ``y_pred``. When given, an extra panel is prepended showing u(t) on
        its own axis, directly below the shared time axis - lets you visually
        correlate reconstruction-error onset with rises/falls of the driving
        input (e.g. whether errors concentrate at input direction reversals)
        without having to cross-reference a separate figure. None (default)
        reproduces the exact old behaviour (no input panel).
    u_label : str
        Y-axis label for the optional input panel.

    Returns
    -------
    str
        The ``save_path`` written.
    """
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    yt = yt.reshape(-1, 1) if yt.ndim == 1 else yt
    yp = yp.reshape(-1, 1) if yp.ndim == 1 else yp
    T = min(yt.shape[0], yp.shape[0])
    yt, yp = yt[:T], yp[:T]
    n_ch = yt.shape[1]

    # Optional uniform subsampling for very long trajectories.
    step = 1
    if max_points is not None and T > int(max_points) > 0:
        step = int(np.ceil(T / int(max_points)))
    sl = slice(None, None, step)

    if dt is not None and dt > 0:
        t = np.arange(T)[sl] * float(dt)
        x_label = "time [s]"
    else:
        t = np.arange(T)[sl]
        x_label = "sample index"

    u = f" [{unit_label}]" if unit_label else ""
    has_u_panel = u_input is not None
    n_rows = (1 if has_u_panel else 0) + 2 * n_ch
    row_heights = ([1.5] if has_u_panel else []) + [3, 1] * n_ch
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(11, 3.2 * n_ch + 1.4 * n_ch + (1.6 if has_u_panel else 0)),
        sharex=True, gridspec_kw={"height_ratios": row_heights},
    )
    axes = np.atleast_1d(axes).ravel()

    if has_u_panel:
        u_arr = _as_1d(np.asarray(u_input))[:T][sl]
        ax_u = axes[0]
        ax_u.plot(t, u_arr, color="#7f7f7f", lw=1.0)
        ax_u.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax_u.set_ylabel(u_label, fontsize=8)
        ax_u.grid(True, alpha=0.3)

    off = 1 if has_u_panel else 0
    for c in range(n_ch):
        ax_sig = axes[off + 2 * c]
        ax_err = axes[off + 2 * c + 1]
        true_c = yt[sl, c]
        pred_c = yp[sl, c]
        ax_sig.plot(t, true_c, color="#1f77b4", lw=1.3, label="true")
        ax_sig.plot(t, pred_c, color="#d62728", lw=1.1, ls="--", label="reconstructed")
        ch_tag = "" if n_ch == 1 else f" (channel {c})"
        ax_sig.set_ylabel(f"output{ch_tag}{u}")
        ax_sig.grid(True, alpha=0.3)
        ax_sig.legend(loc="upper right", fontsize=9)

        ax_err.plot(t, pred_c - true_c, color="#2ca02c", lw=0.9)
        ax_err.axhline(0.0, color="k", lw=0.6, alpha=0.5)
        ax_err.set_ylabel(f"error{u}")
        ax_err.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)

    if title is None:
        bits = []
        if rmse is not None:
            bits.append(f"RMSE={rmse:.4g} {unit_label}".strip())
        if nrmse is not None:
            bits.append(f"NRMSE={nrmse:.4g}")
        title = "True vs reconstructed test signal"
        if bits:
            title += "  (" + ", ".join(bits) + ")"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    save_path = os.fspath(save_path)
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path
