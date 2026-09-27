# Theory-aligned visualization contract

## Scope

The shared visualization layer renders only quantities appearing in, or
directly supporting, the paper's energy-CBF argument:

| Visual quantity | Paper role | Epistemic status of the rendered value |
|---|---|---|
| Learned energy landscape | Defines `H`, `h=epsilon-H`, and the safe sublevel set | Exact evaluation on the displayed grid |
| `H=epsilon` contour | Boundary entering T6/T8/T9/T10 | Exact contour of the displayed 2-D view; not necessarily the full high-dimensional boundary |
| Sampled shell points | Numerical support for T6/T8/T9 | Finite boundary sample after normal reprojection |
| Pointwise radius `delta/||a_H||_*` | Exact local formula behind T6/T9 | Exact formula at sampled states, using raw theory gradients |
| Sampled `rho*` | Discrete approximation to T8/T9 | Minimum over sampled states; not a continuous-shell bound |
| Regular-shell estimate | Formula using T10/T11 ingredients | Sampled estimate unless all extrema are independently bounded |
| Affine slice `H(c+Uz)` | Literal learned-energy cross-section | Exact on the displayed plane |
| Profile `min_w H(c+Uz+Vw)` | Projection of the energy sublevel geometry | Optimization-derived; valid only to the reported stationarity residual |

The figure intentionally excludes implemented-field PGD, random-state fidelity,
ad-hoc fixed-radius frontiers, random/greedy rollouts, and timestep studies.
Those may be useful engineering diagnostics, but they are not represented as
paper-theorem certificate evidence here.

## Shared API

`visualization.py` provides:

- `ProjectedEnergyView`: a prepared 2-D energy field, its displayed shell
  level, projected boundary samples, and coordinate semantics.
- `RadiusCurveView`: pointwise radii, component IDs, sampled component
  summaries, and a caller-defined boundary parameter.
- `EnergySurfaceView`: one learned or analytic energy field for matching 2-D
  and 3-D rendering.
- `plot_projected_radius_certificate`: the common energy/shell/radius figure.
- `plot_energy_surface_comparison`: matching 2-D projections and 3-D surfaces.
- `plot_epsilon_radius_sweep`: standalone regular-shell and sampled-radius
  curves over caller-supplied energy thresholds.

These functions do not load checkpoints, select planes, discover minima, infer
components, or assign theorem status. This is deliberate: those operations
change the scientific meaning of a plot and must remain explicit in each
experiment adapter.

## Standalone epsilon sweep

`eps_sweep_input.py` traces a fresh full-state `H=epsilon` boundary at each
horizontal coordinate and recomputes the raw-theory radius quantities. The
displayed coordinate is the nonnegative relative shell energy
`epsilon - min(H)`; absolute epsilon and `min(H)` remain in the artifacts. The
figure is deliberately separate from `*_certificate_figure` and is saved as
`eps_sweep_input.png` and `eps_sweep_input.pdf`. Its two curves are:

- sampled `rho*`: the minimum sampled `delta/||a_H||_*`;
- regular-shell `rho`: `(r_min*kappa-w_perp_max)/g_perp_max`, using sampled
  shell extrema.

All traced points at one energy are treated as a single global sampled level
set. This remains conservative when disconnected components merge as epsilon
crosses a saddle. Failed shells remain gaps and are recorded explicitly; they are never
interpolated. The companion NPZ and JSON retain epsilon values, both radii,
sample counts, failed-ray fractions, component summaries, and selection policy.
Both curves remain sampled evidence unless the continuous-shell extrema are
independently verified.

The standalone plot uses compact journal typography, embedded TrueType fonts,
colorblind-safe blue/vermillion curves, and no in-panel title. The regular-shell
curve alone carries lime diamonds at strict interior minima with positive
relative energy and at least `0.05` prominence in log-radius, labelled
`Saddle energy shell`. The prominence rule rejects discretization-scale wiggles
without smoothing the curve and permits multiple markers for nonconvex energy
landscapes. These deterministic curve-derived designations are not independent
saddle certifications. A lightly shaded region beyond nominal epsilon denotes
sensitivity analysis, not an expanded safety claim.

## Duffing semantics

The Duffing state is physically two-dimensional, so its displayed plane is the
actual `(q,p)` state space. Connected shell components are assigned to verified
low-gradient well candidates. The analytic Hamiltonian exists and can be shown
beside the learned Hamiltonian.

## Deep-dissipative n-link semantics

The n-link model uses an eight-dimensional learned latent state and has no
available true Hamiltonian. Therefore:

1. The energy comparison shows a literal affine slice and a profiled lower
   envelope, not "true" versus "reconstructed" energy.
2. Raw projections of full-state `H=epsilon` samples can lie inside the
  profile's `H_tilde=epsilon` contour. Panel a uses evenly spaced arc-length
  targets on the longest contour component and pairs each target with a unique
  source sample of nearest projected polar direction. Colors retain the radius
  evaluated at that original full state. Panel b still displays all samples.
  The artifact stores target positions, source indices, mapping distances, and
  angular mismatches.
3. Projected polar angle is a deterministic display coordinate, not an
   intrinsic parameterization of the high-dimensional shell. Points are not
   connected by a line.
4. Existing n-link minima are nonstationary. The renderer uses one sampled
   shell and makes no equilibrium-component claim.
5. The JSON artifact records failed-ray fraction, runtime-clip excursions,
   minimum gradient norm, profile stationarity residuals, and shell projection
   residuals. These diagnostics must accompany paper interpretation.

All projected-radius figures use the Duffing reference panel height-to-width
ratio `0.46017578`. Physical state-plane views retain equal coordinate scaling.
The horizontal radius colorbar is an inset with exactly 100% of the panel-a
data-plane width.

For high-dimensional profiled views, an oversized plane-search canvas must not
become empty figure margin. Panel limits are cropped to the finite region with
normalized profiled energy at most `2.0`, plus 2% padding. The fixed panel box
ratio is retained; its projected coordinate scales may therefore differ. Exact
limits are persisted in the JSON artifact.

## Current n-link validation

The generalized renderer was exercised on `copilot_B3_combo_lr_low`:

- 768/768 traced rays;
- zero-input feasible at every reprojected sample;
- sampled radius `rho*=0.9139833`;
- sampled regular-shell estimate `0.2833482`;
- maximum post-projection energy residual `6.87e-4`;
- 0.65% of source shell samples beyond the runtime state clip;
- stored minimum gradient norm `54.9` and profile p99 orthogonal-gradient
  residual `75.6`, so T5/T10 and profile convergence are not established.

The permitted conclusion is that this checkpoint has positive sampled
pointwise input tolerance on the traced shell under the ideal theory formula.
It is not a formal maximal invariant-radius certificate for the continuous
shell or the physical plant.