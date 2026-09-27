# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Dataset-agnostic publication figures for sampled energy certificates.

The functions in this module only render prepared numerical views. They do not
load checkpoints, choose projection planes, discover equilibria, or infer
whether sampled evidence satisfies a theorem. Those experiment-specific and
verification-sensitive decisions belong to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LightSource, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

DUFFING_PANEL_HEIGHT_TO_WIDTH = 0.46017578
SADDLE_SHELL_LOG_PROMINENCE = 0.05
# Below this many successful boundary rays, the traced shell is too sparse to
# be representative geometry evidence; callers should eliminate the
# certificate-boundary panel rather than render it (see
# nlink_certificate_figure.py::render_nlink_theory_figure).
MIN_CERTIFICATE_BOUNDARY_RAYS = 100

ENERGY_CAMERA_READY = LinearSegmentedColormap.from_list(
    "energy_camera_ready",
    ["#047D7A", "#2FA7A2", "#8FD4CE", "#DCEFEA", "#F3E5B8", "#EBAE6A", "#D96C55"],
    N=256,
)
ENERGY_LIGHT_SOURCE = LightSource(azdeg=315, altdeg=50)


@dataclass(frozen=True)
class ProjectedEnergyView:
    xx: np.ndarray
    yy: np.ndarray
    display_energy: np.ndarray
    shell_level: float
    boundary_xy: np.ndarray
    pointwise_radius: np.ndarray
    title: str
    x_label: str
    y_label: str
    minima_xy: np.ndarray | None = None
    critical_label: str | None = None
    critical_label_xy: tuple[float, float] | None = None
    critical_xy: tuple[float, float] | None = None
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None


@dataclass(frozen=True)
class RadiusCurveView:
    parameter: np.ndarray
    pointwise_radius: np.ndarray
    component_indices: np.ndarray
    component_summaries: Sequence[Mapping[str, object]]
    parameter_label: str
    component_labels: Sequence[str] | None = None
    connect_points: bool = True
    title: str = "Largest uniform radius on each component"


@dataclass(frozen=True)
class EnergySurfaceView:
    xx: np.ndarray
    yy: np.ndarray
    energy: np.ndarray
    cmap: str
    title: str
    energy_label: str
    x_label: str
    y_label: str
    surface_x_label: str | None = None
    surface_y_label: str | None = None


def paper_rc() -> dict:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3f3f3f",
        "axes.labelcolor": "#242424",
        "xtick.color": "#3f3f3f",
        "ytick.color": "#3f3f3f",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _radius_limits(radius: np.ndarray) -> tuple[float, float, np.ndarray]:
    finite = np.asarray(radius, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if not len(finite):
        raise ValueError("At least one finite positive pointwise radius is required")
    floor = max(float(np.min(finite)), 1e-8)
    ceiling = max(float(np.quantile(finite, 0.96)), floor * 1.01)
    return floor, ceiling, finite


def _save_figure(fig, output_base: str | Path) -> dict:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def saddle_shell_minimum_indices(
    epsilon_minus_min_h: np.ndarray,
    regular_radius: np.ndarray,
    *,
    minimum_log_prominence: float = SADDLE_SHELL_LOG_PROMINENCE,
) -> np.ndarray:
    """Return prominent strict interior minima at positive relative energy."""
    energy = np.asarray(epsilon_minus_min_h, dtype=float)
    radius = np.asarray(regular_radius, dtype=float)
    if energy.ndim != 1 or radius.shape != energy.shape:
        raise ValueError("relative energy and regular radius must be one-dimensional and equal-length")
    if minimum_log_prominence < 0.0:
        raise ValueError("minimum_log_prominence must be nonnegative")
    if len(energy) < 3:
        return np.asarray([], dtype=int)
    valid = np.isfinite(energy) & np.isfinite(radius) & (radius > 0.0)
    log_radius = np.full(radius.shape, np.nan, dtype=float)
    log_radius[valid] = np.log(radius[valid])
    finite_triplet = valid[:-2] & valid[1:-1] & valid[2:]
    local = (
        finite_triplet
        & (energy[1:-1] > 0.0)
        & (radius[1:-1] < radius[:-2])
        & (radius[1:-1] < radius[2:])
    )
    candidates = np.flatnonzero(local) + 1
    selected = []
    for index in candidates:
        center = log_radius[index]
        segment_start = index - 1
        while segment_start > 0 and valid[segment_start - 1]:
            if log_radius[segment_start - 1] < center:
                break
            segment_start -= 1
        segment_end = index + 1
        while segment_end < len(radius) - 1 and valid[segment_end + 1]:
            if log_radius[segment_end + 1] < center:
                break
            segment_end += 1
        left_relief = np.max(log_radius[segment_start:index]) - center
        right_relief = np.max(log_radius[index + 1:segment_end + 1]) - center
        if min(left_relief, right_relief) >= minimum_log_prominence:
            selected.append(index)
    return np.asarray(selected, dtype=int)


def eps_sweep_journal_rc() -> dict:
    return {
        **paper_rc(),
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.65,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def draw_epsilon_radius_sweep(
    axis,
    epsilon_minus_min_h: np.ndarray,
    regular_radius: np.ndarray,
    sampled_radius: np.ndarray,
    *,
    nominal_epsilon_minus_min_h: float | None = None,
    nominal_label: str = r"nominal $\epsilon$",
    comparison_epsilon_minus_min_h: float | None = None,
    comparison_label: str = r"comparison-figure $\epsilon$",
    panel_label: str | None = None,
) -> None:
    """Draw the regular/sampled radius-vs-shell-energy sweep onto ``axis``.

    Assumes ``epsilon_minus_min_h`` is already sorted and validated (see
    ``plot_epsilon_radius_sweep``, the figure-producing wrapper around this
    function). Factored out so other figures (e.g. a merged multi-source
    plate) can embed this exact panel natively instead of stitching a
    pre-rendered image.
    """
    axis.plot(
        epsilon_minus_min_h, sampled_radius, color="#0072B2",
        linewidth=1.45, label=r"Sampled $\rho^\ast$", zorder=3,
    )
    axis.plot(
        epsilon_minus_min_h, regular_radius, color="#D55E00",
        linewidth=1.3, linestyle=(0, (4.0, 2.0)),
        label=r"Regular-shell estimate $\rho$", zorder=3,
    )
    saddle_indices = saddle_shell_minimum_indices(epsilon_minus_min_h, regular_radius)
    if len(saddle_indices):
        axis.plot(
        epsilon_minus_min_h[saddle_indices], regular_radius[saddle_indices],
            linestyle="none", marker="D", markersize=4.6,
            markerfacecolor="#8BD646", markeredgecolor="#34452B",
            markeredgewidth=0.5, label="Saddle energy shell", zorder=5,
        )
    if nominal_epsilon_minus_min_h is not None and np.isfinite(nominal_epsilon_minus_min_h):
        nominal = float(nominal_epsilon_minus_min_h)
        right = float(np.max(epsilon_minus_min_h))
        if nominal < right:
            axis.axvspan(nominal, right, color="#f2f2f2", linewidth=0, zorder=0)
            axis.text(
                0.5 * (nominal + right), 0.97, "beyond nominal",
                transform=axis.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.5, color="#666666",
            )
        axis.axvline(
            nominal, color="#555555", linewidth=0.85,
            linestyle=(0, (1.5, 2.0)), zorder=2,
        )
        axis.annotate(
            nominal_label, xy=(nominal, 1.0),
            xycoords=("data", "axes fraction"), xytext=(-4, -3),
            textcoords="offset points", ha="right", va="top",
            fontsize=6.5, color="#555555",
        )
    if comparison_epsilon_minus_min_h is not None and np.isfinite(comparison_epsilon_minus_min_h):
        comparison = float(comparison_epsilon_minus_min_h)
        axis.axvline(
            comparison, color="#009E73", linewidth=1.0,
            linestyle=(0, (1.0, 1.5)), zorder=2,
        )
        axis.annotate(
            comparison_label, xy=(comparison, 1.0),
            xycoords=("data", "axes fraction"), xytext=(4, -3),
            textcoords="offset points", ha="left", va="top",
            fontsize=6.5, color="#009E73",
        )
    finite = np.concatenate([regular_radius, sampled_radius])
    finite = finite[np.isfinite(finite)]
    if len(finite) and np.all(finite > 0):
        axis.set_yscale("log")
    else:
        axis.set_ylim(bottom=0.0)
    right_limit = 1.025 * float(np.max(epsilon_minus_min_h))
    axis.set_xlim(0.0, right_limit)
    axis.set_xlabel(r"Relative shell energy, $\epsilon-H_{\min}$", labelpad=4.0)
    axis.set_ylabel("Admissible input radius", labelpad=4.0)
    axis.set_axisbelow(True)
    axis.grid(axis="y", which="major", color="#d9d9d9", linewidth=0.45)
    axis.tick_params(which="minor", width=0.5, length=2.0)
    if panel_label:
        axis.text(
            -0.045, 1.11, panel_label, transform=axis.transAxes,
            fontsize=10.0, fontweight="bold", ha="left", va="top",
        )
    axis.legend(
        frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0, ncol=3, handlelength=2.15,
        handletextpad=0.5, columnspacing=1.0,
    )


def plot_epsilon_radius_sweep(
    epsilon_minus_min_h: np.ndarray,
    regular_radius: np.ndarray,
    sampled_radius: np.ndarray,
    output_base: str | Path,
    *,
    nominal_epsilon_minus_min_h: float | None = None,
    nominal_label: str = r"nominal $\epsilon$",
    comparison_epsilon_minus_min_h: float | None = None,
    comparison_label: str = r"comparison-figure $\epsilon$",
    figure_size: tuple[float, float] = (4.8, 3.15),
        panel_label: str | None = None,
) -> dict:
    """Plot regular-shell and sampled input-radius estimates over energy shells."""
    epsilon_minus_min_h = np.asarray(epsilon_minus_min_h, dtype=float)
    regular_radius = np.asarray(regular_radius, dtype=float)
    sampled_radius = np.asarray(sampled_radius, dtype=float)
    if (
        epsilon_minus_min_h.ndim != 1
        or regular_radius.shape != epsilon_minus_min_h.shape
        or sampled_radius.shape != epsilon_minus_min_h.shape
    ):
        raise ValueError("relative energy and both radius arrays must be one-dimensional and equal-length")
    if np.any(epsilon_minus_min_h < 0.0):
        raise ValueError("epsilon_minus_min_h must be nonnegative")
    order = np.argsort(epsilon_minus_min_h)
    epsilon_minus_min_h = epsilon_minus_min_h[order]
    regular_radius = regular_radius[order]
    sampled_radius = sampled_radius[order]

    journal_rc = eps_sweep_journal_rc()
    with plt.rc_context(journal_rc):
        fig, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
        draw_epsilon_radius_sweep(
            axis, epsilon_minus_min_h, regular_radius, sampled_radius,
            nominal_epsilon_minus_min_h=nominal_epsilon_minus_min_h, nominal_label=nominal_label,
            comparison_epsilon_minus_min_h=comparison_epsilon_minus_min_h, comparison_label=comparison_label,
            panel_label=panel_label,
        )
        return _save_figure(fig, output_base)


def map_points_to_level_contour(
    xx: np.ndarray,
    yy: np.ndarray,
    field: np.ndarray,
    level: float,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map 2-D points to their nearest segments on a displayed level contour.

    This is a display-coordinate operation. It does not move the source states
    at which certificate quantities were evaluated. The returned distances make
    that distinction measurable and suitable for artifact metadata.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [n_points, 2]")
    temporary, axis = plt.subplots()
    contour = axis.contour(xx, yy, field, levels=[float(level)])
    segments = [
        np.asarray(vertices, dtype=float)
        for vertices in contour.allsegs[0]
        if len(vertices) >= 2
    ]
    plt.close(temporary)
    if not segments:
        raise ValueError(f"No contour found at level {level}")

    starts = np.concatenate([vertices[:-1] for vertices in segments], axis=0)
    ends = np.concatenate([vertices[1:] for vertices in segments], axis=0)
    directions = ends - starts
    squared_lengths = np.sum(directions * directions, axis=1)
    valid = squared_lengths > 1e-20
    starts = starts[valid]
    directions = directions[valid]
    squared_lengths = squared_lengths[valid]
    if not len(starts):
        raise ValueError("Level contour contains no nondegenerate segments")

    mapped = np.empty_like(points)
    distances = np.empty(len(points), dtype=float)
    for index, point in enumerate(points):
        fraction = np.sum((point - starts) * directions, axis=1) / squared_lengths
        fraction = np.clip(fraction, 0.0, 1.0)
        candidates = starts + fraction[:, None] * directions
        squared_distance = np.sum((candidates - point) ** 2, axis=1)
        nearest = int(np.argmin(squared_distance))
        mapped[index] = candidates[nearest]
        distances[index] = np.sqrt(squared_distance[nearest])
    return mapped, distances


def select_contour_coverage_samples(
    xx: np.ndarray,
    yy: np.ndarray,
    field: np.ndarray,
    level: float,
    source_points: np.ndarray,
    n_display: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair evenly spaced contour targets with unique directional samples.

    Returned points are display coordinates on the longest contour component.
    ``source_indices`` identifies the full-state samples supplying their radius
    values. Pairing minimizes projected polar-angle mismatch greedily while
    using every source at most once.
    """
    source_points = np.asarray(source_points, dtype=float)
    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise ValueError("source_points must have shape [n_points, 2]")
    if n_display < 2:
        raise ValueError("n_display must be at least 2")

    temporary, axis = plt.subplots()
    contour = axis.contour(xx, yy, field, levels=[float(level)])
    segments = [
        np.asarray(vertices, dtype=float)
        for vertices in contour.allsegs[0]
        if len(vertices) >= 2
    ]
    plt.close(temporary)
    if not segments:
        raise ValueError(f"No contour found at level {level}")
    vertices = max(segments, key=lambda item: np.sum(np.linalg.norm(np.diff(item, axis=0), axis=1)))
    if np.linalg.norm(vertices[0] - vertices[-1]) > 1e-10:
        vertices = np.vstack([vertices, vertices[0]])
    segment_length = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_length)])
    perimeter = float(cumulative[-1])
    if perimeter <= 0:
        raise ValueError("Level contour has zero perimeter")

    count = min(int(n_display), len(source_points))
    target_distance = np.linspace(0.0, perimeter, count, endpoint=False)
    targets = np.column_stack([
        np.interp(target_distance, cumulative, vertices[:, coordinate])
        for coordinate in range(2)
    ])
    center = np.mean(vertices[:-1], axis=0)
    target_angle = np.arctan2(targets[:, 1] - center[1], targets[:, 0] - center[0])
    source_angle = np.arctan2(
        source_points[:, 1] - center[1], source_points[:, 0] - center[0]
    )

    available = np.ones(len(source_points), dtype=bool)
    source_indices = np.empty(count, dtype=int)
    angular_error = np.empty(count, dtype=float)
    for target_index, angle in enumerate(target_angle):
        difference = np.abs(np.angle(np.exp(1j * (source_angle - angle))))
        difference[~available] = np.inf
        source_index = int(np.argmin(difference))
        source_indices[target_index] = source_index
        angular_error[target_index] = difference[source_index]
        available[source_index] = False
    displacement = np.linalg.norm(targets - source_points[source_indices], axis=1)
    return targets, source_indices, displacement, angular_error


def sublevel_view_limits(
    xx: np.ndarray,
    yy: np.ndarray,
    field: np.ndarray,
    max_level: float,
    padding_fraction: float = 0.02,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return tight padded limits around a finite displayed sublevel region."""
    mask = np.isfinite(field) & (np.asarray(field) <= float(max_level))
    if not np.any(mask):
        raise ValueError(f"No finite field values at or below level {max_level}")
    x_values = np.asarray(xx)[mask]
    y_values = np.asarray(yy)[mask]
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_pad = max(float(padding_fraction) * (x_max - x_min), 1e-12)
    y_pad = max(float(padding_fraction) * (y_max - y_min), 1e-12)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def draw_projected_energy_geometry(
    axis,
    fig,
    energy: ProjectedEnergyView,
    boundary: np.ndarray,
    radius_display: np.ndarray,
    radius_floor: float,
    radius_ceiling: float,
    *,
    curve_pointwise: np.ndarray | None = None,
    components: np.ndarray | None = None,
    component_summaries: Sequence[Mapping[str, object]] = (),
    panel_height_to_width: float | None = DUFFING_PANEL_HEIGHT_TO_WIDTH,
    colorbar_location: str = "bottom",
    colorbar_axis=None,
    show_boundary: bool = True,
) -> None:
    """Draw the projected energy-shell geometry (contour + boundary scatter +
    colorbar) onto ``axis``. Factored out of ``plot_projected_radius_certificate``
    so other figures can embed this exact panel natively. ``fig`` is needed
    only for the inset colorbar. Pass ``colorbar_axis`` (a real subplot) to
    place the colorbar there instead of an inset that may overflow ``axis``'s
    own footprint (orientation still follows ``colorbar_location``).

    ``show_boundary=False`` keeps the projected-Hamiltonian background
    (contourf) but omits the shell-level contour, boundary scatter, critical
    marker/label, and colorbar -- for use when too few boundary rays
    succeeded for that overlay to be representative evidence (see
    ``MIN_CERTIFICATE_BOUNDARY_RAYS``), while still showing the Hamiltonian
    itself.
    """
    display = np.asarray(energy.display_energy, dtype=float)
    finite_energy = display[np.isfinite(display)]
    upper = max(float(np.quantile(finite_energy, 0.94)), float(energy.shell_level) * 1.15)
    lower = float(np.min(finite_energy))
    if upper <= lower:
        upper = lower + 1.0
    levels = np.linspace(lower, upper, 24)
    axis.contourf(
        energy.xx, energy.yy, display, levels=levels, cmap="Blues_r", extend="max"
    )
    points = None
    if show_boundary:
        axis.contour(
            energy.xx, energy.yy, display, levels=[float(energy.shell_level)],
            colors="#202020", linewidths=1.0,
        )
        points = axis.scatter(
            boundary[:, 0], boundary[:, 1], c=radius_display, cmap="viridis",
            norm=LogNorm(vmin=radius_floor, vmax=radius_ceiling),
            s=16, linewidths=0, zorder=5, rasterized=True,
        )
    if energy.minima_xy is not None and len(energy.minima_xy):
        minima = np.asarray(energy.minima_xy)
        axis.scatter(
            minima[:, 0], minima[:, 1], marker="o", s=30, color="white",
            edgecolor="#101010", linewidth=0.8, zorder=7,
        )

    if show_boundary:
        if energy.critical_xy is not None:
            axis.scatter(
                *energy.critical_xy, marker="x", s=42,
                color="#ff2a00", linewidth=1.8, zorder=8,
            )
        else:
            for summary in component_summaries:
                component = int(summary["component_index"])
                mask = components == component
                local = np.where(mask, curve_pointwise, np.inf)
                critical = int(np.argmin(local))
                axis.scatter(
                    boundary[critical, 0], boundary[critical, 1], marker="x", s=42,
                    color="#ff2a00", linewidth=1.8, zorder=8,
                )
        if energy.critical_label and energy.critical_label_xy is not None:
            axis.text(
                *energy.critical_label_xy, energy.critical_label,
                color="#ff2a00", fontsize=8.0, fontweight="bold",
                ha="center", va="center", zorder=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            )

    axis.set(
        xlabel=energy.x_label, ylabel=energy.y_label, title=energy.title
    )
    if panel_height_to_width is not None:
        axis.set_box_aspect(float(panel_height_to_width))
    if energy.x_limits is not None and energy.y_limits is not None:
        axis.set_xlim(*energy.x_limits)
        axis.set_ylim(*energy.y_limits)
        axis.set_aspect("auto")
    else:
        axis.set_aspect("equal", adjustable="datalim")
    axis.set_anchor("N")
    if colorbar_axis is not None and points is not None:
        orientation = "horizontal" if colorbar_location == "bottom" else "vertical"
        colorbar = fig.colorbar(points, cax=colorbar_axis, orientation=orientation)
        ticks = np.geomspace(radius_floor, radius_ceiling, 4)
        axis_kind = colorbar.ax.xaxis if orientation == "horizontal" else colorbar.ax.yaxis
        axis_kind.set_major_locator(FixedLocator(ticks))
        axis_kind.set_minor_locator(NullLocator())
        if orientation == "horizontal":
            colorbar.ax.set_xticklabels([f"{value:.2g}" for value in ticks])
        else:
            colorbar.ax.set_yticklabels([f"{value:.2g}" for value in ticks])
            if colorbar_location == "left":
                colorbar.ax.yaxis.set_ticks_position("left")
                colorbar.ax.yaxis.set_label_position("left")
    elif colorbar_location == "left":
        colorbar_axis = axis.inset_axes([-0.22, 0.0, 0.045, 1.0])
        colorbar = fig.colorbar(points, cax=colorbar_axis, orientation="vertical")
        ticks = np.geomspace(radius_floor, radius_ceiling, 4)
        colorbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
        colorbar.ax.yaxis.set_minor_locator(NullLocator())
        colorbar.ax.set_yticklabels([f"{value:.2g}" for value in ticks])
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
    elif colorbar_location == "right":
        colorbar_axis = axis.inset_axes([1.06, 0.0, 0.045, 1.0])
        colorbar = fig.colorbar(points, cax=colorbar_axis, orientation="vertical")
        ticks = np.geomspace(radius_floor, radius_ceiling, 4)
        colorbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
        colorbar.ax.yaxis.set_minor_locator(NullLocator())
        colorbar.ax.set_yticklabels([f"{value:.2g}" for value in ticks])
    elif colorbar_location == "bottom":
        colorbar_axis = axis.inset_axes([0.0, -0.30, 1.0, 0.045])
        colorbar = fig.colorbar(points, cax=colorbar_axis, orientation="horizontal")
        ticks = np.geomspace(radius_floor, radius_ceiling, 4)
        colorbar.ax.xaxis.set_major_locator(FixedLocator(ticks))
        colorbar.ax.xaxis.set_minor_locator(NullLocator())
        colorbar.ax.set_xticklabels([f"{value:.2g}" for value in ticks])
    else:
        raise ValueError(f"Unknown colorbar_location={colorbar_location!r}")
    colorbar.set_label(r"pointwise admissible radius $\rho_{\mathrm{CBF}}$")


def plot_projected_radius_certificate(
    energy: ProjectedEnergyView,
    radius: RadiusCurveView,
    output_base: str | Path,
    *,
    width_ratio: tuple[float, float] = (0.6, 0.4),
    figure_size: tuple[float, float] = (12.0, 4.45),
    legend_fontsize: float = 9.0,
    panel_height_to_width: float = DUFFING_PANEL_HEIGHT_TO_WIDTH,
    comparison_radius: Mapping[str, object] | None = None,
) -> dict:
    """Plot projected energy-shell geometry and sampled admissible radii.

    The left panel is a visualization of a caller-defined 2-D energy view. For
    a high-dimensional model this may be a slice or a profiled lower envelope;
    the title and coordinate labels must state which. The right panel displays
    sampled pointwise radii and component-wise extrema. No formal continuous-
    shell guarantee is implied by this rendering function.
    """
    boundary = np.asarray(energy.boundary_xy, dtype=float)
    geometry_pointwise = np.asarray(energy.pointwise_radius, dtype=float)
    parameter = np.asarray(radius.parameter, dtype=float)
    curve_pointwise = np.asarray(radius.pointwise_radius, dtype=float)
    components = np.asarray(radius.component_indices, dtype=int)
    if boundary.shape != (len(geometry_pointwise), 2):
        raise ValueError("boundary_xy must have shape [n_boundary, 2]")
    if len(parameter) != len(curve_pointwise) or len(components) != len(curve_pointwise):
        raise ValueError("radius-curve arrays must have equal lengths")
    if energy.critical_xy is None and len(geometry_pointwise) != len(curve_pointwise):
        raise ValueError("critical_xy is required when geometry and curve samples differ")

    radius_floor, radius_ceiling, finite_positive = _radius_limits(curve_pointwise)
    radius_display = np.clip(geometry_pointwise, radius_floor, radius_ceiling)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]

    with plt.rc_context(paper_rc()):
        fig = plt.figure(figsize=figure_size, constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=width_ratio)
        geometry_axis = fig.add_subplot(grid[0, 0])
        radius_axis = fig.add_subplot(grid[0, 1])

        draw_projected_energy_geometry(
            geometry_axis, fig, energy, boundary, radius_display, radius_floor, radius_ceiling,
            curve_pointwise=curve_pointwise, components=components,
            component_summaries=radius.component_summaries,
            panel_height_to_width=panel_height_to_width,
        )

        well_handles = []
        summaries = list(radius.component_summaries)
        labels = list(radius.component_labels or [])
        for summary_index, summary in enumerate(summaries):
            component = int(summary["component_index"])
            mask = components == component
            order = np.argsort(parameter[mask])
            color = colors[summary_index % len(colors)]
            local_parameter = parameter[mask][order]
            local_radius = curve_pointwise[mask][order]
            finite = np.isfinite(local_radius) & (local_radius > 0)
            component_label = labels[summary_index] if summary_index < len(labels) else f"component {component + 1}"
            if radius.connect_points:
                handle, = radius_axis.plot(
                    local_parameter[finite], local_radius[finite], color=color,
                    linewidth=1.35, label=rf"{component_label}: $\rho_{{\mathrm{{CBF}}}}$",
                )
            else:
                handle = radius_axis.scatter(
                    local_parameter[finite], local_radius[finite], color=color,
                    s=9, alpha=0.72, linewidths=0,
                    label=rf"{component_label}: $\rho_{{\mathrm{{CBF}}}}$",
                    rasterized=True,
                )
            well_handles.append(handle)
            sampled = float(summary["sampled_exact_radius"])
            regular = float(summary["regular_boundary_lower_estimate"])
            radius_axis.axhline(sampled, color=color, linewidth=1.25, linestyle=":")
            if np.isfinite(regular) and regular > 0:
                radius_axis.axhline(regular, color=color, linewidth=1.1, linestyle="--", alpha=0.85)
            critical = int(np.argmin(local_radius))
            radius_axis.scatter(
                local_parameter[critical], local_radius[critical], s=24,
                color=color, edgecolor="white", linewidth=0.5, zorder=6,
            )

        radius_axis.set_yscale("log")
        lower_values = [
            float(summary["regular_boundary_lower_estimate"])
            for summary in summaries
            if np.isfinite(float(summary["regular_boundary_lower_estimate"]))
            and float(summary["regular_boundary_lower_estimate"]) > 0
        ]
        y_min = 0.75 * min(lower_values or [radius_floor])
        y_max = 1.25 * float(np.quantile(finite_positive, 0.99))
        radius_axis.set_ylim(y_min, max(y_max, y_min * 2.0))
        radius_axis.set(
            xlabel=radius.parameter_label,
            ylabel=r"admissible input radius $\rho$",
            title=radius.title,
        )
        radius_axis.grid(axis="y", which="both", color="0.9", linewidth=0.6)
        style_handles = [
            Line2D([0], [0], color="0.25", lw=1.25, ls=":", label=r"sampled $\rho^\star$"),
            Line2D([0], [0], color="0.25", lw=1.1, ls="--", label="regular-shell estimate"),
        ]
        if comparison_radius is not None:
            comparison_color = "#CC79A7"
            comparison_label = str(comparison_radius.get("label", "comparison-figure $\\epsilon$"))
            comparison_sampled = comparison_radius.get("sampled")
            comparison_regular = comparison_radius.get("regular")
            if comparison_sampled is not None and np.isfinite(comparison_sampled):
                radius_axis.axhline(
                    float(comparison_sampled), color=comparison_color, linewidth=1.4, linestyle=":", zorder=4,
                )
            if comparison_regular is not None and np.isfinite(comparison_regular) and float(comparison_regular) > 0:
                radius_axis.axhline(
                    float(comparison_regular), color=comparison_color, linewidth=1.25, linestyle="--", zorder=4,
                )
            style_handles.append(
                Line2D([0], [0], color=comparison_color, lw=1.4, ls="-", label=comparison_label)
            )
        legend_axis = radius_axis.inset_axes([0.0, -0.30, 1.0, 0.15])
        legend_axis.axis("off")
        legend_axis.legend(
            handles=well_handles + style_handles, frameon=False,
            loc="center", fontsize=legend_fontsize, ncol=2, columnspacing=1.35,
        )
        geometry_axis.text(-0.11, 1.03, "a", transform=geometry_axis.transAxes,
                           weight="bold", fontsize=12)
        radius_axis.text(-0.11, 1.03, "b", transform=radius_axis.transAxes,
                         weight="bold", fontsize=12)
        return _save_figure(fig, output_base)


def plot_energy_surface_comparison(
    views: Sequence[EnergySurfaceView],
    output_base: str | Path,
    *,
    figure_size: tuple[float, float] = (10.2, 6.45),
    row_heights: tuple[float, float] = (0.40, 0.60),
) -> dict:
    """Plot harmonized 2-D maps and 3-D surfaces for prepared energies.

    A shared normalisation is deliberately used within each comparison plate:
    it makes differences in energy calibration visible rather than assigning
    each panel an independently flattering colour range.  The upper limit is
    robust to the sparsely sampled outermost tails; values beyond it remain
    visibly saturated in both the map and the surface.
    """
    if not views:
        raise ValueError("At least one energy surface view is required")
    finite_values = [np.asarray(view.energy, dtype=float)[np.isfinite(view.energy)] for view in views]
    if any(values.size == 0 for values in finite_values):
        raise ValueError("Energy surface views must contain finite values")
    comparison_values = np.concatenate(finite_values)
    lower = float(np.min(comparison_values))
    upper = float(np.quantile(comparison_values, 0.995))
    if not upper > lower:
        upper = float(np.max(comparison_values))
    if not upper > lower:
        upper = lower + 1.0
    norm = Normalize(vmin=lower, vmax=upper, clip=True)
    # One normalisation across a plate makes differences in energy calibration
    # visible rather than assigning each panel an independently flattering range.
    cmap = ENERGY_CAMERA_READY
    fill_levels = np.linspace(lower, upper, 81)
    line_levels = np.linspace(lower, upper, 7)[1:-1]
    with plt.rc_context(paper_rc()):
        fig = plt.figure(figsize=figure_size, constrained_layout=True)
        grid = fig.add_gridspec(
            2, len(views), height_ratios=row_heights, hspace=0.0, wspace=0.28
        )
        panel_axes = []
        for column, view in enumerate(views):
            axis = fig.add_subplot(grid[0, column])
            values = np.asarray(view.energy, dtype=float)
            contour = axis.contourf(
                view.xx, view.yy, values, levels=fill_levels, cmap=cmap,
                norm=norm, extend="max", antialiased=True,
            )
            axis.contour(
                view.xx, view.yy, values, levels=line_levels,
                colors="#ffffff", linewidths=0.40, alpha=0.48,
            )
            colorbar_axis = axis.inset_axes([1.025, 0.03, 0.030, 0.94])
            colorbar = fig.colorbar(contour, cax=colorbar_axis)
            colorbar.set_ticks(np.linspace(lower, upper, 5))
            colorbar.set_label(view.energy_label)
            axis.set(
                xlabel=view.x_label, ylabel=view.y_label,
                xlim=(float(np.nanmin(view.xx)), float(np.nanmax(view.xx))),
                ylim=(float(np.nanmin(view.yy)), float(np.nanmax(view.yy))),
            )
            axis.set_title(view.title, fontweight="bold")
            axis.set_aspect("equal", adjustable="box")
            axis.set_anchor("S")
            axis.tick_params(width=0.75)

            surface_axis = fig.add_subplot(grid[1, column], projection="3d")
            clipped = np.minimum(values, upper)
            # Shade normalized heights so the lighting is consistent with the
            # common comparison colour scale, rather than renormalising every
            # surface independently.
            face = ENERGY_LIGHT_SOURCE.shade(
                norm(clipped), cmap=cmap, vert_exag=1.34, blend_mode="soft"
            )
            face[..., -1] = 1.0
            # Render every available grid sample.  This eliminates the visible
            # rstride/cstride mesh of the previous figure while retaining a
            # vector surface in PDF output (no rasterised artist).
            surface_axis.plot_surface(
                view.xx, view.yy, clipped, facecolors=face,
                rcount=values.shape[0], ccount=values.shape[1],
                linewidth=0, edgecolor="none", antialiased=False,
                shade=False, rasterized=False,
            )
            surface_axis.set(
                xlabel=view.surface_x_label or r"$z_1$",
                ylabel=view.surface_y_label or r"$z_2$",
                zlabel=view.energy_label,
            )
            if "affine slice" in view.title.lower():
                surface_title = "Affine slice · 3D"
            elif "profiled envelope" in view.title.lower():
                surface_title = "Profiled envelope · 3D"
            elif view.title.lower().startswith("true"):
                surface_title = "True Hamiltonian · 3D"
            elif view.title.lower().startswith("reconstructed"):
                surface_title = "Reconstructed Hamiltonian · 3D"
            else:
                surface_title = f"{view.title} · 3D"
            surface_axis.tick_params(labelsize=7.5, pad=0)
            surface_axis.xaxis.labelpad = 4
            surface_axis.yaxis.labelpad = 4
            surface_axis.zaxis.labelpad = 5
            surface_axis.view_init(elev=31, azim=-57)
            surface_axis.set_box_aspect((1.30, 1.0, 0.72))
            surface_axis.xaxis.pane.set_facecolor((0.96, 0.96, 0.95, 0.32))
            surface_axis.yaxis.pane.set_facecolor((0.96, 0.96, 0.95, 0.20))
            surface_axis.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            surface_axis.xaxis.pane.set_edgecolor("none")
            surface_axis.yaxis.pane.set_edgecolor("none")
            surface_axis.zaxis.pane.set_edgecolor("none")
            surface_axis.grid(False)
            surface_axis.text2D(
                0.5, 0.985, surface_title, transform=surface_axis.transAxes,
                ha="center", va="top", weight="bold", fontsize=10,
            )
            panel_axes.append((axis, surface_axis))
        # The 3-D axes extend farther left than the 2-D maps.  Position panel
        # letters in figure coordinates after constrained layout so a/c and
        # b/d are exactly vertically aligned despite that differing footprint.
        fig.canvas.draw()
        inverse_figure = fig.transFigure.inverted()
        for column, (map_axis, surface_axis) in enumerate(panel_axes):
            label_x, lower_y = inverse_figure.transform(
                surface_axis.transAxes.transform((-0.11, 0.985))
            )
            _, upper_y = inverse_figure.transform(
                map_axis.transAxes.transform((0.0, 1.03))
            )
            fig.text(label_x, upper_y, chr(ord("a") + column),
                     ha="left", va="bottom", weight="bold", fontsize=12)
            fig.text(label_x, lower_y, chr(ord("a") + len(views) + column),
                     ha="left", va="top", weight="bold", fontsize=12)
        return _save_figure(fig, output_base)
