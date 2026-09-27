# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Publication-oriented figures for Hamiltonian wells and energy-CBF safety sets.

The plotting functions intentionally label *slice* and *profile* views
separately.  A profiled landscape is a projection/lower envelope, not a literal
2-D section of the high-dimensional Hamiltonian, and the figure should never
blur that distinction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import PowerNorm

from .geometry import MinimaResult, Plane


def _paper_rc():
    return {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def _levels(E: np.ndarray, epsilon: float) -> tuple[np.ndarray, float, float]:
    finite = E[np.isfinite(E)]
    e0 = float(np.min(finite))
    # Keep high-energy corners from washing out the wells.  The safety boundary
    # is force-included in the range even when it lies beyond the 98.5% quantile.
    ehi = max(float(np.quantile(finite, 0.985)), float(epsilon))
    if ehi <= e0 + 1e-10:
        ehi = e0 + 1.0
    return np.linspace(e0, ehi, 36), e0, ehi


def _draw_landscape(ax, data: dict, epsilon: float, title: str, plane: Plane,
                    minima: MinimaResult, reference_states: Optional[np.ndarray]):
    E = np.asarray(data["energy"])
    levels, e0, ehi = _levels(E, epsilon)
    # A square-root power law devotes more dynamic range to the bottom of wells
    # while retaining the actual energy values on the colorbar.
    cf = ax.contourf(
        data["xx"], data["yy"], E,
        levels=levels, cmap="viridis", norm=PowerNorm(gamma=0.55, vmin=e0, vmax=ehi),
        extend="max",
    )
    # Thin iso-energy contours make nonconvex valleys/rings legible in print.
    ax.contour(data["xx"], data["yy"], E, levels=levels[::4], colors="k", linewidths=0.35, alpha=0.35)

    # The barrier boundary is the visual protagonist.  A white line with a thin
    # dark halo survives both dark and light parts of the colormap and remains
    # legible after PDF compression.
    try:
        # Matplotlib defaults negative contour levels to dashed lines.  epsilon
        # is often negative in this model, so force a solid CBF boundary.
        if e0 <= epsilon <= ehi:
            ax.contourf(data["xx"], data["yy"], E, levels=[e0, epsilon], colors=["white"], alpha=0.07)
        cs = ax.contour(data["xx"], data["yy"], E, levels=[epsilon], colors="white", linewidths=2.4, linestyles="solid")
        for coll in cs.collections:
            coll.set_path_effects([pe.Stroke(linewidth=4.0, foreground="black", alpha=0.65), pe.Normal()])
    except Exception:
        pass

    if reference_states is not None and len(reference_states):
        z = plane.project(reference_states)
        # Downsample only for rendering; the projection calculation uses all data.
        stride = max(len(z) // 5000, 1)
        ax.scatter(z[::stride, 0], z[::stride, 1], s=3, c="white", alpha=0.16, linewidths=0, rasterized=True,
                   label="visited states")

    if len(minima.states):
        zw = plane.project(minima.states)
        ax.scatter(zw[:, 0], zw[:, 1], marker="*", s=115, c="white", edgecolors="black", linewidths=0.9,
                   zorder=8, label="discovered wells")
        for i, (zx, zy) in enumerate(zw):
            if i >= 8:
                break
            ax.annotate(f"W{i+1}", (zx, zy), xytext=(5, 5), textcoords="offset points",
                        fontsize=8, weight="bold", color="white",
                        path_effects=[pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()])

    ax.set_title(title)
    ax.set_xlabel(r"plane coordinate $z_1$")
    ax.set_ylabel(r"plane coordinate $z_2$")
    ax.set_xlim(float(data["gx"][0]), float(data["gx"][-1]))
    ax.set_ylim(float(data["gy"][0]), float(data["gy"][-1]))
    ax.set_aspect("equal", adjustable="box")
    return cf


def plot_landscape_comparison(
    plane: Plane,
    minima: MinimaResult,
    epsilon: float,
    slice_data: Optional[dict],
    profile_data: Optional[dict],
    reference_states: Optional[np.ndarray],
    output_path: str | Path,
):
    """Save a slice/profile comparison suitable for a main-paper figure."""
    panels = []
    if slice_data is not None: panels.append((slice_data, "Affine slice: $H(c+Uz)$"))
    if profile_data is not None: panels.append((profile_data, r"Profiled Hamiltonian: $\min_w H(c+Uz+Vw)$"))
    if not panels:
        raise ValueError("At least one of slice_data/profile_data must be provided")

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, len(panels), figsize=(6.1 * len(panels), 5.2), constrained_layout=True)
        if len(panels) == 1: axes = [axes]
        cf = None
        for ax, (data, title) in zip(axes, panels):
            cf = _draw_landscape(ax, data, epsilon, title, plane, minima, reference_states)
        cbar = fig.colorbar(cf, ax=axes, shrink=0.88, pad=0.02)
        cbar.set_label(r"learned Hamiltonian $H(x)$")
        fig.suptitle(rf"Hamiltonian geometry and energy-CBF boundary $H(x)=\epsilon$;  $\epsilon={epsilon:.4g}$", y=1.02)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_profile_surface(profile_data: dict, epsilon: float, output_path: str | Path):
    """Optional 3-D profiled-energy view; useful as supplement/teaser figure."""
    with plt.rc_context(_paper_rc()):
        fig = plt.figure(figsize=(7.2, 5.8))
        ax = fig.add_subplot(111, projection="3d")
        E = np.asarray(profile_data["energy"])
        surf = ax.plot_surface(profile_data["xx"], profile_data["yy"], E, cmap="viridis",
                               linewidth=0, antialiased=True, rcount=100, ccount=100, alpha=0.96)
        try:
            ax.contour(profile_data["xx"], profile_data["yy"], E, levels=[epsilon], zdir="z",
                       offset=float(np.nanmin(E)), colors="black", linewidths=2.0)
        except Exception:
            pass
        ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$"); ax.set_zlabel(r"$\widetilde H(z)$")
        ax.set_title(r"Profiled learned Hamiltonian and projected safe-set boundary")
        fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=r"$\widetilde H(z)$")
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)



def plot_profile_stationarity(profile_data: dict, output_path: str | Path):
    """Plot the optimization residual behind the profiled Hamiltonian.

    The profile at each planar coordinate minimizes H over the orthogonal
    complement V.  A necessary first-order condition is

        || V^T grad H(x_star(z)) ||_2 = 0.

    This diagnostic should be inspected before using the profiled landscape in
    a paper.  Large residuals mean the apparent projected wells or safety
    boundary may still reflect an under-converged inner optimization.
    """
    if "profile_stationarity" not in profile_data:
        raise ValueError("profile_data does not contain profile_stationarity")
    R = np.asarray(profile_data["profile_stationarity"], dtype=float)
    # A logarithmic display makes both well-converged and problematic regions
    # visible on the same panel.  The floor is display-only.
    logR = np.log10(np.maximum(R, 1e-12))
    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(6.1, 5.0), constrained_layout=True)
        im = ax.pcolormesh(
            profile_data["xx"], profile_data["yy"], logR,
            shading="auto", cmap="magma",
        )
        ax.set_xlabel(r"plane coordinate $z_1$")
        ax.set_ylabel(r"plane coordinate $z_2$")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Profile optimization residual")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r"$\log_{10} \|V^T \nabla H(x_\star(z))\|_2$")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

def plot_adversarial_rollouts(
    plane: Plane,
    profile_data: dict,
    epsilon: float,
    rollout_states: np.ndarray,
    rollout_h: np.ndarray,
    output_path: str | Path,
    max_trajectories: int = 24,
):
    """Overlay worst-case trajectories on the projected safe set + h(t)."""
    B, T, _ = rollout_states.shape
    choose = np.linspace(0, B - 1, min(B, max_trajectories), dtype=int)
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
        _draw_landscape(axes[0], profile_data, epsilon, "Worst-case forcing in projected safe set",
                        plane, MinimaResult(np.empty((0, plane.center.size)), np.empty(0), np.empty(0, int),
                                            np.empty((0, plane.center.size)), np.empty(0)), None)
        for i in choose:
            z = plane.project(rollout_states[i])
            violated = np.min(rollout_h[i]) < 0
            axes[0].plot(z[:, 0], z[:, 1], lw=1.0, alpha=0.75,
                         color=("crimson" if violated else "white"))
            axes[0].scatter(z[0, 0], z[0, 1], s=10, c="black", zorder=9)

        t = np.arange(T)
        for i in choose:
            axes[1].plot(t, rollout_h[i], lw=0.9, alpha=0.55)
        axes[1].axhline(0.0, color="black", lw=1.4, ls="--")
        axes[1].set_xlabel("integration step")
        axes[1].set_ylabel(r"barrier $h(x)=\epsilon-H(x)$")
        axes[1].set_title("Barrier margin under greedy worst-case forcing")

        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_robustness_frontier(frontier: dict, output_path: str | Path):
    """Plot worst boundary CBF slack versus input/disturbance-set scaling.

    The zero contour is the empirical boundary (over sampled H=epsilon points)
    between positive and negative robust CBF margin for the *analytic affine
    certificate*.  The nominal claimed uncertainty set is marked at (1,1).
    """
    from matplotlib.colors import TwoSlopeNorm

    X, Y = np.meshgrid(frontier["input_scales"], frontier["disturbance_scales"], indexing="xy")
    S = np.asarray(frontier["worst_slack"])
    finite = S[np.isfinite(S)]
    vmax = float(np.quantile(np.abs(finite), 0.98)) if len(finite) else 1.0
    vmax = max(vmax, 1e-8)

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        im = ax.pcolormesh(
            X, Y, S, shading="auto", cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        )
        try:
            cs = ax.contour(X, Y, S, levels=[0.0], colors="black", linewidths=2.0, linestyles="solid")
            ax.clabel(cs, fmt={0.0: "zero robust slack"}, fontsize=8, inline=True)
        except Exception:
            pass
        ax.scatter([1.0], [1.0], marker="*", s=180, c="gold", edgecolors="black", linewidths=1.0,
                   zorder=8, label="claimed set")
        ax.set_xlabel("input-set scale")
        ax.set_ylabel("state-disturbance-set scale")
        ax.set_title(r"Worst CBF margin on $H(x)=\epsilon$")
        ax.legend(loc="best", frameon=True)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r"$\min_{x\in\partial\mathcal{C}_\epsilon}\;[\dot h+\gamma h]$")
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
