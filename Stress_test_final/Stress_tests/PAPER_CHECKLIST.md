# Paper-facing stress-test checklist

This checklist is deliberately stricter than a normal model-fit evaluation.  It
separates mathematical claims from diagnostics and empirical attacks so that a
paper does not accidentally present a numerical observation as a certificate.

## A. Freeze the claim before running attacks

Record the exact quantities that define the theorem-level claim:

- Hamiltonian level `epsilon` and class-K rate `gamma`;
- input-set center, norm and radius;
- additive state-perturbation norm and radius;
- whether the claim is global or restricted to an operational state domain;
- which continuous vector field the proof refers to.

Do not tune the uncertainty radii after looking at the stress-test plots.  A
robustness frontier can be shown around the claimed point, but `(1, 1)` must
mean the uncertainty set stated in the theorem/experiment protocol.

## B. Check theorem-code fidelity first

Before interpreting any rollout, inspect:

1. `compatibility.warnings` in `summary.json`;
2. `certificate_fidelity.max_abs_hdot_gap` and its p99/median;
3. `certificate_boundary_audit.min_robust_implemented_slack`;
4. `boundary.fraction_beyond_runtime_state_clip`;
5. the implemented-input PGD result versus the analytic affine adversary.

The repository contains smooth clipping/saturation safeguards.  If they are
active in the region claimed by the theorem, either include them in the formal
model or restrict the claim to a region where their effect is negligible.

## C. Validate the Hamiltonian geometry figure

For the main Hamiltonian figure:

- prefer `landscape_mode="both"`;
- state the plane-selection rule in the caption;
- describe the left panel as an affine *slice*;
- describe the right panel as the orthogonally *profiled Hamiltonian*;
- state that the profile sublevel set is the planar projection of the full
  energy sublevel set only up to numerical inner-optimization accuracy;
- inspect `profile_stationarity.pdf` and report the p99 residual from
  `summary.json` in the supplement if the profile is central to the paper;
- increase `profile_steps` and/or `profile_restarts` until the geometry is
  visually stable and the orthogonal-gradient residual is satisfactorily small;
- verify `plane.safety_contour_touches_grid_edge == false` before using the
  figure to argue that a connected/disconnected safety region is fully shown.

For latent input-output models, `plane_method="readout"` is a useful secondary
view: its first direction maximizes local output sensitivity at the lowest
found well.  The default `auto` view instead prioritizes well and safety-set
geometry and is generally better for revealing multiple minima.

## D. Stress the boundary, not only nominal trajectories

A certificate is most informative near `H(x)=epsilon`.  The suite therefore
traces that boundary directly.  For a paper run, verify:

- boundary residuals are numerically small;
- the failed-ray fraction is reported rather than silently discarded;
- boundary directions are numerous enough to cover the state dimension;
- adversarial rollouts start just inside the boundary;
- both greedy state-dependent and random piecewise forcing are shown/reported.

If the learned Hamiltonian has multiple wells, make sure every well below
`epsilon` contributes boundary rays; otherwise a weak component can be missed.

## E. Show a robustness frontier

`robustness_frontier.pdf` plots the worst sampled-boundary CBF margin while
scaling the claimed input set and additive state-disturbance set.  A compelling
paper panel should mark `(1, 1)` explicitly and include the zero-slack contour.

Interpretation:

- positive side: sampled boundary retains formula-side robust CBF margin;
- zero contour: empirical transition on the sampled boundary;
- negative side: at least one sampled boundary state loses margin.

This frontier is a visualization/audit of the analytic support-function
certificate over sampled boundary points.  It is not a replacement for the
formal proof over the full boundary.

## F. Separate three kinds of evidence

Use unambiguous wording in captions/tables:

1. **Formal**: follows from the theorem under its assumptions.
2. **Numerical audit**: e.g. formula-vs-implemented derivative agreement,
   boundary tracing, profiled-landscape optimization residual.
3. **Empirical attack**: long-horizon greedy/PGD/random forcing found no
   violation; this does not prove absence of all finite-horizon attacks.

This distinction will make the experimental section stronger, not weaker.

## G. Numerical-integration audit

Report the same physical-duration stress test at multiple step sizes.  Check
that minimum barrier margin and violation fraction converge as `dt` decreases.
If the result changes qualitatively with step size, do not attribute the effect
to continuous-time stability without resolving the discretization issue.

## H. Input-output / latent-state scope

The CBF is evaluated on latent state `x`, whereas experiments begin from input-
output data and an estimated initial state.  If the paper claims resilience to
measurement/observer error as well as process disturbances, that uncertainty
must be modeled explicitly.  The current default stress suite certifies/tests
bounded external inputs and additive state-dynamics perturbations; it does not
silently convert measurement noise into a latent-state uncertainty guarantee.

A simple extra empirical test, when relevant, is to perturb the burn-in output
window, re-encode `x0`, and launch the same boundary/rollout analysis.  For a
formal observer-error guarantee, use the uncertainty model from the theorem
rather than this heuristic.

## I. Recommended paper artifacts

Main text candidates:

- `hamiltonian_landscape.pdf`: slice + profiled projection of H and H=epsilon;
- `robustness_frontier.pdf`: resilience phase diagram;
- `adversarial_rollouts.pdf`: projected trajectories + barrier margin.

Supplement / audit candidates:

- `profile_stationarity.pdf`;
- `hamiltonian_profile_3d.pdf`;
- `summary.json` tables for formula/implementation gap and time-step study.

The code also writes PNG previews for rapid inspection and PDF/vector-oriented
figures for the manuscript workflow.
