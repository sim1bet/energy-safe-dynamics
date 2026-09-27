# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network

"""Configuration objects for the EBM stress-testing suite.

The stress suite is intentionally configured independently from training.  The
training configuration describes how a model was fitted; these dataclasses
describe how aggressively the *finished* model is interrogated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence


NormKind = Literal["l2", "linf"]
PlaneMethod = Literal["auto", "pca", "wells", "safety_pca", "readout", "hessian"]
LandscapeMode = Literal["slice", "profile", "both"]


@dataclass
class UncertaintySet:
    """Admissible external forcing and additive state disturbance.

    The input set is a norm ball around ``input_center``.  For ``linf`` this is
    an axis-aligned box; ``input_radius`` may therefore be either one scalar or
    one radius per input channel.  For ``l2`` it must be scalar.

    ``state_disturbance_radius`` models an additive disturbance ``w`` in
    ``xdot = f(x,u) + w``.  This is deliberately distinct from input-port
    uncertainty because the two have different support functions in Hdot.

    ``disturbance_map`` generalizes the disturbance to ``w = E d``,
    ``|d| <= state_disturbance_radius`` (support function
    ``sigma_W(q) = rho_d * |E^T q|_*``), e.g. ``E = [[0],[1]]`` for a Duffing
    disturbance that only enters the momentum channel.  ``None`` (default)
    reproduces the original identity-map behaviour (``w`` directly in the
    state space) bit-for-bit — fully backward compatible.
    """

    input_norm: NormKind = "linf"
    input_center: Optional[Sequence[float]] = None
    input_radius: float | Sequence[float] = 1.0
    state_disturbance_norm: NormKind = "l2"
    state_disturbance_radius: float = 0.0
    disturbance_map: Optional[Sequence[Sequence[float]]] = None


@dataclass
class ProjectionConfig:
    """Controls Hamiltonian-well discovery and two-dimensional visualization."""

    plane_method: PlaneMethod = "auto"
    landscape_mode: LandscapeMode = "both"
    grid_size: int = 141
    padding_fraction: float = 0.18

    # Multi-start well search.  The initial states are a mixture of observed
    # states and random perturbations spanning the same empirical scale.
    n_minima_starts: int = 192
    minima_steps: int = 300
    minima_lr: float = 2e-2
    minima_cluster_tol: float = 0.12
    minima_grad_norm_tol: float = 1.0
    max_wells_to_show: int = 12

    # Profiled Hamiltonian H_tilde(z) = min_w H(c + U z + V w).
    # A 141x141 grid with 100-140 profile steps is a good publication-quality
    # starting point for d <= ~12.  Increase only for the final camera-ready.
    profile_steps: int = 120
    profile_lr: float = 3e-2
    profile_restarts: int = 4
    profile_chunk_size: int = 2048
    # None profiles over the full orthogonal complement. Set a finite radius
    # only when the theorem/certificate itself is restricted to an operational ball.
    profile_state_radius: Optional[float] = None

    # Plane selection / plot support.  Quantiles make the plot resistant to a
    # few extreme latent-state excursions.
    extent_quantile_low: float = 0.01
    extent_quantile_high: float = 0.99

    # For a radially-unbounded Hamiltonian the projected epsilon-sublevel set
    # should be compact. Expand the plotting plane until H_tilde=epsilon closes
    # inside the canvas (or until the cap below is reached).
    ensure_closed_safety_contour: bool = True
    max_extent_expansions: int = 2
    extent_expansion_factor: float = 1.45


@dataclass
class StressTestConfig:
    """Top-level stress-test configuration."""

    seed: int = 0

    # Keep only Hamiltonian geometry and the analytic continuous-time boundary
    # certificate. Disable for paper pipelines whose theory does not contain
    # implemented-field PGD, random/greedy rollout, timestep, fidelity, or
    # ad-hoc frontier claims.
    run_non_theoretical_diagnostics: bool = True
    relative_energy_alpha: Optional[float] = None

    # Boundary audit: directions are shot from each discovered well and the
    # first crossing H(x)=epsilon is located by bracketing+bisection.
    n_boundary_directions: int = 768
    boundary_bisection_steps: int = 60
    boundary_initial_radius: float = 0.05
    boundary_max_radius: float = 50.0
    boundary_energy_tolerance: float = 1e-4

    # Monte-Carlo / adversarial trajectory tests.
    n_rollouts: int = 96
    horizon_steps: int = 2000
    boundary_inset: float = 5e-3
    random_piecewise_hold: int = 25

    # A continuous-time CBF certificate can appear violated solely because the
    # numerical solver is too coarse.  The suite repeats selected trajectories
    # at these dt multipliers to diagnose that effect.
    timestep_multipliers: tuple[float, ...] = (1.0, 0.5, 0.25)

    # Numerical tolerances used only when *reporting* violations.  They do not
    # modify the model or the certificate.
    cbf_slack_tolerance: float = 1e-5
    barrier_tolerance: float = 1e-5

    # Random-state certificate-fidelity audit.
    n_fidelity_samples: int = 4096

    # Projected-gradient attack on the *implemented* vector field.  This is
    # complementary to the analytic worst input for the ideal affine pH formula.
    implemented_attack_steps: int = 40
    implemented_attack_restarts: int = 4
    implemented_attack_lr: float = 0.08

    # Robustness-frontier figure: scales the claimed input/disturbance sets from
    # zero to these multiples and plots the worst boundary CBF slack.
    frontier_grid_size: int = 81
    frontier_max_input_scale: float = 2.0
    frontier_max_disturbance_scale: float = 2.0

    uncertainty: UncertaintySet = field(default_factory=UncertaintySet)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
