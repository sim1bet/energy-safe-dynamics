# Stress_tests: certificate and Hamiltonian-geometry evaluation

This folder is a **post-training** evaluation package for the free-EBM
port-Hamiltonian Neural ODE.  It does not modify training and it does not tune
parameters.  Its purpose is to answer the questions a control / verification
reviewer is likely to ask once the paper claims a set-valued energy-CBF
certificate

\[
    h(x)=\epsilon-H(x), \qquad \mathcal C_\epsilon=\{x:H(x)\le \epsilon\}.
\]

The suite was designed around the actual repository equations, including the
numerical soft-clips present in `EBM_param_fields.py`.

## Directory contract

The package is split by epistemic role. Do not mix these roles when copying,
extending, or reporting results.

```text
Stress_tests/
  boundary.py, certificate.py, config.py   # energy-shell and CBF theory
  geometry.py, frontier.py                 # Hamiltonian projection and margins
  plotting.py, rollouts.py, suite.py       # plots and post-training orchestration
  integration.py, model_adapter.py         # repository-specific pH-EBM bridge
  experiments/                             # experiment-specific theory adapters
  theory_aligned/                          # theorem registry and paper figures
  Additional_tests/                        # noncanonical material and regressions
```

`Stress_tests.*` is the production namespace. Production modules must not
import from `Additional_tests`. Canonical theorem checks stay under
`theory_aligned/tests`; general regression tests stay under
`Additional_tests/tests`.

## LLM-agent integration contract

This section is the authoritative procedure for plugging this folder into a
port-Hamiltonian EBM repository that does not yet contain it. An agent should
complete every step and must not infer missing physical or theorem parameters
from favorable test outcomes.

### 1. Copy without flattening

Copy the entire `Stress_tests/` directory into the target repository and keep
the package name and subdirectory structure unchanged. Add the target
repository root and its model package to Python's import path. Install the
target model's JAX version plus NumPy, SciPy, Matplotlib, PyYAML, and pytest.

### 2. Implement the model boundary

Only `model_adapter.py`, `integration.py`, and `pathing.py` know this
repository's flat `EBM_model` modules. Adapt those files to the target model;
do not insert target-model logic into geometry, certificate, or plotting code.
The adapter consumed by the theory pipeline must expose:

```text
d, m, dt, params, energy_value_fn
energy(x), energy_batch(X)
grad_energy(x), grad_energy_batch(X)
barrier(x, epsilon)
theory_energy_terms(x) -> (a_H, d_H, grad_H, R, G)
affine_terms(x) -> (a, drift)
vector_field(x, u), output(x, u)
implemented_hdot(x, u), controller_hdot(x, u)
compatibility_report(), certificate_fidelity(...)
```

The sign convention is fixed:

\[
h(x)=\epsilon-H(x),\qquad
\dot h=d_H-a_H^T u,\qquad
d_H=\nabla H^T R\nabla H.
\]

Verify the target repository's convention for `R`, `J`, `G`, and input signs
analytically before running a checkpoint. Do not make plots until a synthetic
finite-difference check confirms `implemented_hdot = -grad(H)^T f`.

### 3. Restore the trained runtime exactly

Extend `apply_runtime_config` so every architecture switch, scaling constant,
clipping threshold, input gate, and integrator choice used during training is
restored before constructing the adapter. A loaded parameter tree with default
module globals is not a faithful checkpoint reconstruction.

Collect representative latent states from a held-out ordinary rollout. These
states choose and scale the visualization plane; they do not certify safety.
If the target model has no encoder, provide its actual evaluated initial latent
state and bypass `encode_x0_from_burnin`.

### 4. Declare theory inputs before evaluation

Provide `d`, `m`, `dt`, trained parameters, Hamiltonian layer metadata,
`epsilon`, `gamma`, and the admissible input/disturbance set. Set these from the
model definition and theorem, never by searching for values that make the
certificate pass. For a theory-only paper run, set
`run_non_theoretical_diagnostics=False`; empirical rollout and implementation
fidelity diagnostics are useful evidence but are not theorem obligations.

For physical systems, add an adapter under `experiments/` that owns analytic
Hamiltonians, physical coordinate labels, equilibrium rules, and disturbance
maps. For latent systems without a known true Hamiltonian, label the learned
affine slice and profiled lower envelope explicitly and never present either as
ground truth.

#### Autonomous parameter-selection policy

An agent may choose the stress parameters autonomously only with the following
leakage-free policy. It must use training/validation information and physical
metadata available **before** stress, OOD, or test outcomes are inspected. It
must save the selected values, method, data split, random seed, and all policy
hyperparameters alongside the run.

**Choose `epsilon` by the first applicable rule:**

1. **Specified physical safe set:** if the application supplies an energy
   limit, convert that limit into the learned Hamiltonian convention and use it.
   This takes precedence over every empirical rule.
2. **Known physical multi-well topology (Duffing rule):** verify a local
   minimum and the separating saddle by small gradient norms, predeclare
   `alpha=0.8`, and set

   \[
   \epsilon=H_{\min}+\alpha(H_{\mathrm{saddle}}-H_{\min}).
   \]

   Require `0 < alpha < 1`. Do not tune `alpha` using certificate violations.
  The reference implementation is
  `experiments/duffing_doublewell.py::alpha_to_epsilon`.
3. **Unknown/latent topology (n-link rule):** use only ordinary validation
   rollout states, predeclare `quantile=0.95` and `margin_fraction=0.10`, and set

   \[
   \epsilon=H_{\mathrm{ref}}+
   (1+0.10)Q_{0.95}\!\left(H(x_{\mathrm{val}})-H_{\mathrm{ref}}\right).
   \]

   Use a converged minimum as `H_ref` when one exists. If no candidate meets
   the configured gradient-norm tolerance, use the minimum validation energy
   only to define an **operational sampled shell**; do not claim an equilibrium
   component or a theorem-level invariant set. Freeze `epsilon` before any
   boundary attack or OOD evaluation.
  The reference implementation is
  `experiments/nlink_dissipative.py::select_epsilon_shell`.

Reject an automatically selected shell if it is empty, nonfinite, cannot be
traced reliably, contains a critical point on its boundary, or its profiled
contour does not close on the plotting support. Expanding the plotting canvas
is allowed; changing `epsilon` after observing certificate failure is not.

**Choose `gamma` without optimizing boundary results:**

- Reuse the controller's trained or formally specified positive `gamma` when
  one exists.
- If a valid PL constant `mu` and dissipation lower bound `r` have been
  established on the component, choose the smallest saturating linear rate
  `gamma = 2*r*mu`; the interior bound
  `epsilon*min(2*r*mu, gamma)` cannot improve by increasing it further.
- Otherwise nondimensionalize time by a predeclared characteristic time
  `T_ref` and use `gamma=1/T_ref` (`gamma=1.0` when time is already
  nondimensionalized). Report a sensitivity sweep separately.

On `H=epsilon`, `h=0`, so `gamma*h=0`. Consequently an agent must never tune
`gamma` to repair boundary infeasibility or increase the boundary radius; it
cannot do either.

**Choose the input set from actuator geometry, then compute its radius:**

- Use `linf` with one radius per channel for independent actuator bounds or
  normalized controls whose declared support is a box. Set `input_center` to
  the physical/normalized trim input. This is the Duffing/n-link convention;
  scalar radius `1.0` means a unit box in model-input coordinates.
- Use `l2` only when the specification gives a joint Euclidean power/magnitude
  budget. Do not choose between `linf` and `l2` according to which passes.
- Compute the sampled maximal radius with
  `compute_uniform_radius_certificate` under the selected norm. Compare it to
  the declared actuator set. A sampled value is an estimate of
  `inf_Gamma delta/||a_H||_*`, not a certified continuous-shell lower bound.
  Claim a radius only when continuous-shell extrema are independently bounded;
  otherwise label it "largest sampled admissible radius."
- If no actuator bounds exist, a unit box may be used only as a clearly labeled
  normalized diagnostic. It is not an autonomous physical safety claim.

**Choose the disturbance set from the disturbance model:**

- Set `disturbance_map=E` for `xdot=f(x,u)+E*d`; use identity only when the
  disturbance truly enters every state coordinate directly.
- Use `l2` for direction-independent energy/noise budgets and `linf` for
  independently bounded disturbance channels. Set the radius from a physical
  bound or a predeclared identification-error bound.
- If no such bound exists, use radius `0.0` and state that no disturbance
  robustness is claimed. Values such as Walker2d's `0.05` may be included in a
  separately labeled sensitivity sweep, but must not be promoted to a theorem
  parameter merely because they pass.

In pseudocode, the decision is:

```python
epsilon = physical_limit if physical_limit is not None else (
    H_min + 0.8 * (H_saddle - H_min)
    if verified_physical_minimum_and_saddle
    else validation_quantile_shell(q=0.95, margin=0.10)
)
gamma = trained_gamma or verified_pl_rate or (1.0 / characteristic_time)
input_norm, declared_input_radius = actuator_specification()
disturbance_norm, disturbance_radius, disturbance_map = disturbance_specification()
sampled_radius = compute_uniform_radius_certificate(...)
```

The agent may optimize model-independent hyperparameters only on a dedicated
calibration split with an objective written down before evaluation. It must
then rerun once on untouched test/OOD data. It may not jointly optimize
`epsilon`, uncertainty radii, or norms to obtain a passing certificate; those
quantities define the claim being tested.

### 5. Run and inspect

Call `run_stress_suite` as shown in the quick start below. Inspect, in order:

1. adapter compatibility warnings and formula/implemented derivative gaps;
2. minimum-search gradient norms and boundary energy residuals;
3. profiled-Hamiltonian stationarity residuals and contour closure;
4. sampled certificate margins and uncertainty-set provenance;
5. generated PDF/PNG figures for labels, support, and nonblank content.

Use `theory_aligned/visualization.py` only with prepared views. Radius colors
must remain indexed to the original full-state evaluation samples even when
their panel coordinates are mapped to a projected contour.

### 6. Port-validation gate

Before accepting an integration, run:

```bash
python -m pytest -q Stress_tests/Additional_tests/tests
python -m pytest -q Stress_tests/theory_aligned/tests
```

Then execute a small deterministic checkpoint smoke run twice and compare its
`summary.json` and NPZ arrays within declared numerical tolerances. Preserve
existing result directories: integration and regression tests must write to a
temporary output directory. Record dependency versions, random seeds, and the
exact checkpoint/config paths in the resulting report.

An agent must stop and report a blocked integration if it cannot identify the
actual Hamiltonian, vector field, trained runtime settings, or theorem-defined
uncertainty set. Guessing any of these can produce attractive but invalid
certificate figures.

## Three-panel publication figures

The exact Duffing and n-link workflow for generating certificate panels a-b,
computing the 300-shell panel c, and composing the width-matched final PNG/PDF
plate is documented in `theory_aligned/README.md` under **Generate the
three-panel paper figures**. Run those commands from the repository root after
the checkpoint, config, stress summary, and stress arrays have been written.

In brief, use the matching experiment certificate renderer, run
`theory_aligned/eps_sweep_input.py <run_dir> --count 300`, then run
`theory_aligned/combined_figure.py <run_dir>`. The final artifact is
`<run_dir>/stress_tests/theory_aligned/combined_theory_aligned_figure.{png,pdf}`.
The composition command requires `pypdf` and preserves the certificate width.

## What is tested

### 1. Geometry of the learned Hamiltonian

`geometry.py` performs multi-start minimization of the learned Hamiltonian and
clusters the terminal states into distinct wells.  It then chooses a 2-D plane
using the wells and/or the latent states visited by ordinary rollouts.

**Do not use a coordinate slice such as**

```text
H(x1, x2, 0, ..., 0)
```

as the main paper visualization unless those coordinates have a physical
meaning.  In a latent state space, that slice can completely miss a well that
is displaced in one of the hidden coordinates.

The package therefore produces two views:

#### Affine slice

\[
H_{\rm slice}(z)=H(c+Uz), \qquad U^TU=I_2.
\]

This is a literal 2-D cross-section of the learned Hamiltonian.  It is honest
and easy to interpret, but a well can be absent simply because the chosen plane
does not pass through it.

#### Profiled Hamiltonian

\[
\widetilde H(z)=\min_w H(c+Uz+Vw),
\qquad [U\;V]^T[U\;V]=I.
\]

This is the lower envelope obtained by minimizing over directions orthogonal to
the plotting plane.  Most importantly,

\[
\{z:\widetilde H(z)\le\epsilon\}
\]

is the planar projection of the energy sublevel set (up to numerical
optimization error and the configured operational state-radius restriction).
That makes the `H=epsilon` contour a meaningful picture of the **projected
safety region**, not an arbitrary latent-coordinate section.

For a main paper figure, the recommended presentation is the side-by-side
`slice | profile` plot generated by `plot_landscape_comparison`.  Caption the
second panel explicitly as a *profiled/projection view*, not as a literal slice.

### 2. Boundary-focused certificate audit

`boundary.py` does not sample random states and hope they happen to lie near the
CBF boundary.  Instead, it starts from discovered wells, shoots rays in many
high-dimensional directions, brackets the first crossing of

\[
H(x)=\epsilon,
\]

and resolves each crossing by bisection.  This targets the region where a CBF
certificate is most fragile and most informative.

`certificate.py` then evaluates the pointwise worst admissible input.  For the
affine pH formula

\[
\dot h = \mathrm{drift}(x)-a(x)^T u,
\]

the worst input is obtained analytically from the support function of the
configured input set:

- `linf`: an axis-aligned input box, with the adversary at the appropriate
  corner;
- `l2`: a Euclidean input ball, with the adversary aligned with `a(x)`.

An additive state disturbance

\[
\dot x=f(x,u)+w
\]

is handled separately.  Since its contribution to `hdot` is
`-grad(H)^T w`, the worst `w` is also available in closed form for L2/Linf
balls.

### 3. Formula-vs-implementation certificate fidelity

This is an important diagnostic for this repository.

`EBM_controller.py` derives an affine CBF expression from the intended pH
field.  `EBM_param_fields.vector_field_and_output`, however, also contains
smooth numerical safeguards such as:

- direction-preserving gradient norm saturation;
- state norm saturation;
- elementwise `xdot` saturation;
- optional nonlinear input gain / input saturation.

The suite therefore reports both

\[
\dot h_{\rm formula}
\]

and

\[
\dot h_{\rm implemented}
  =-\nabla H(x)^T f_{\rm implemented}(x,u),
\]

plus absolute and relative gaps.  A strong result is not merely "no empirical
violations"; it is showing that the formula/implementation gap is negligible
throughout the claimed certified region, or modifying the theorem path if it is
not.

If `USE_INPUT_GAIN=True` or `USE_SATURATING_INPUT=True`, the suite emits an
explicit warning because the existing affine controller formula is no longer
the exact raw-input derivative.

### 4. Long-horizon adversarial tests

`rollouts.py` starts slightly *inside* the high-dimensional energy boundary and
uses a state-dependent greedy adversary at every integration step:

- input chosen to minimize the ideal CBF derivative;
- additive disturbance aligned against the barrier gradient.

This is not claimed to be a globally optimal finite-horizon attack.  For that
reason the suite also launches high-amplitude, piecewise-constant random input
and disturbance sequences to search for temporally non-myopic failures.

### 5. Robustness frontier

`frontier.py` converts the support-function form of the affine pH certificate
into a two-dimensional resilience phase diagram.  It scales the declared input
set and additive state-disturbance set independently and computes

```text
min over sampled H(x)=epsilon boundary of [ hdot + gamma h ].
```

The claimed uncertainty set is always the point `(1, 1)`.  The zero-slack
contour gives a visually immediate stress margin around that claim.  This plot
is an audit over the sampled boundary, not a substitute for a proof over the
continuous boundary.

### 6. Profile-optimization fidelity

The profiled Hamiltonian is itself the result of an inner optimization.  The
suite therefore saves `profile_stationarity.pdf`, a heat map of

```text
log10 || V^T grad H(x_star(z)) ||_2,
```

plus median/p99/max residuals in `summary.json`.  Inspect this diagnostic before
using projected wells or projected safety topology in the manuscript.  If the
residual is too large, increase `profile_steps`, adjust `profile_lr`, or add
restarts.

### 7. Time-step convergence audit

A continuous-time CBF guarantee can appear violated by a coarse numerical
integrator.  Conversely, a saturating/clipping integrator can conceal a problem
in the continuous field.  The suite therefore repeats the greedy attack at
several time steps while keeping the **physical test duration fixed**.

Report whether the minimum barrier margin and violation rate converge as `dt`
is reduced.  Also report the maximum displacement caused by the state soft
clip.  If the result changes qualitatively with time step, that belongs in the
analysis before claiming a continuous-time certificate.

---

## Quick start after training

The stress suite needs the trained `params`, the same `layers` tuple used to
construct the Hamiltonian, the state/input dimensions, `dt`, and the final
`epsilon` / `gamma` certificate parameters.

Providing ordinary latent rollout states is strongly recommended because they
make the 2-D plane data-aware and provide sensible starts for well discovery.

```python
from Stress_tests import StressTestConfig, UncertaintySet, run_stress_suite
from Stress_tests.integration import (
    apply_runtime_config, collect_reference_rollout, encode_x0_from_burnin,
)
from Stress_tests.model_adapter import EBMStressAdapter

# Objects returned by the existing training code:
# params, layers, grad_E, stats, controller_info = run_training(...)

D = int(cfg["d"])
M = int(cfg["m_ports"])

# Required when evaluation is launched in a fresh process: training writes
# several model choices into EBM_param_fields module-level globals.
apply_runtime_config(cfg)
adapter = EBMStressAdapter(params, layers, d=D, m=M, dt=dt)

# Use a normal held-out trajectory only to obtain representative latent states.
x0 = encode_x0_from_burnin(
    params,
    y_te[:init_len],
    u_te[:init_len],  # ignored unless USE_INPUT_AWARE_ENCODER=True
)
reference_states = collect_reference_rollout(
    adapter,
    x0,
    u_te[init_len:],
    integrator=cfg["rollout_integrator"],
)

stress_cfg = StressTestConfig(
    n_rollouts=96,
    horizon_steps=2000,
    uncertainty=UncertaintySet(
        input_norm="linf",
        input_center=[0.0] * M,
        input_radius=2.0,          # choose from the theorem / claimed input set
        state_disturbance_norm="l2",
        state_disturbance_radius=0.05,
    ),
)

summary = run_stress_suite(
    params=params,
    layers=layers,
    d=D,
    m=M,
    dt=dt,
    epsilon=controller_info["epsilon"],
    gamma=controller_info["gamma"],
    reference_states=reference_states,
    reference_inputs=u_te[init_len:],
    output_dir="results/stress/cascaded_tanks",
    config=stress_cfg,
)
```

### Important: set the uncertainty from the theorem, not from the dataset

Do **not** choose `input_radius` after seeing which value produces zero
violations.  The uncertainty set is part of the claim.  Define it first from
physical/domain assumptions or from the formal proposition, then run this code
as an attack on that claim.

For normalized benchmark data, remember that `input_radius` is in the same
normalized coordinates seen by the model unless you convert it explicitly.

---

## Recommended main-paper figures

### Figure A — learned Hamiltonian geometry

Use `figures/hamiltonian_landscape.pdf`.

Recommended caption content:

1. how the plane was chosen (`auto/wells/safety_pca/readout/PCA/Hessian`);
2. left panel is a literal affine slice;
3. right panel is the orthogonally profiled Hamiltonian;
4. thick contour is `H=epsilon`;
5. white points are ordinary visited states;
6. stars are minima discovered independently by multi-start Hamiltonian descent.

The profile can reveal disconnected wells, rings, channels and non-star-convex
sublevel sets that are invisible in a coordinate slice.  If the latent state is
abstract and the figure should connect directly to input-output behavior,
`plane_method="readout"` provides a useful secondary view whose first axis is
the strongest local output-sensitive state direction.

### Figure B — robustness frontier

Use `figures/robustness_frontier.pdf`.  It visualizes the minimum sampled-boundary
CBF margin as the declared input and state-disturbance sets are scaled.  The
nominal theorem/experiment claim is the starred point `(1,1)`; do not move that
point by retuning the uncertainty after seeing the result.

### Figure C — adversarial invariance

Use `figures/adversarial_rollouts.pdf`.

The left panel overlays worst-case trajectories on the projected safety region;
the right panel shows the actual barrier margin `h(t)`.  Start trajectories just
inside the boundary so the test does not waste most of its horizon deep inside
the safe set.

### Supplement — 3-D profiled Hamiltonian

`figures/hamiltonian_profile_3d.pdf` is visually useful, but the 2-D contour
figure is usually more quantitative and easier to read in a conference paper.

---

## Outputs

A run creates:

```text
stress_results/
  summary.json
  arrays/
    stress_arrays.npz
  figures/
    hamiltonian_landscape.pdf
    hamiltonian_profile_3d.pdf
    profile_stationarity.pdf
    robustness_frontier.pdf
    adversarial_rollouts.pdf
    # PNG previews of all figures are also written.
```

`summary.json` contains:

- discovered well energies/multiplicities and gradient norms;
- profiled-Hamiltonian stationarity residuals;
- visualization-plane method, limits and safety-contour edge check;
- boundary-tracing success rate;
- robust formula-side CBF slack;
- robust implemented-dynamics slack;
- formula/implementation derivative gap;
- greedy adversarial violation rate;
- random temporal stress violation rate;
- time-step convergence statistics;
- maximum state-softclip displacement;
- compatibility warnings;
- nominal point on the input/disturbance robustness frontier.

The NPZ file retains the states/inputs/slacks needed for additional plots without
rerunning the expensive profile or adversarial simulation.

---

## Suggested ablation table for the paper

A useful certificate table is structurally different from an RMSE leaderboard:

| model / setting | fit metric | min robust CBF slack | empirical violation rate | max formula-field gap | max clip displacement |
|---|---:|---:|---:|---:|---:|
| learned pH model | ... | ... | ... | ... | ... |
| reduced epsilon | ... | ... | ... | ... | ... |
| larger input set | ... | ... | ... | ... | ... |
| larger disturbance set | ... | ... | ... | ... | ... |
| coarse dt | ... | ... | ... | ... | ... |

For the claimed certified regime, the most important entries are the worst-case
slack, violation rate, and formula/implementation gap.  RMSE remains useful to
show that certification did not destroy identification quality, but it should
not dominate this section.

---

## Performance notes

The profiled Hamiltonian is intentionally the expensive part.  It solves a
small optimization problem at every grid point and uses several warm starts.
For iteration/debugging use, for example:

```python
cfg.projection.grid_size = 81
cfg.projection.profile_steps = 60
cfg.projection.profile_restarts = 2
```

For final figures, `141-201` grid points per axis and multiple restarts are
reasonable for the small latent dimensions used by this repository.

The boundary and trajectory tests are JAX-vectorized.  If GPU memory is tight,
reduce `n_rollouts` first; do not reduce the number of boundary directions too
aggressively, because boundary coverage is more informative than repeating many
nearly identical trajectories.


## Before a submission run

Read `PAPER_CHECKLIST.md`.  It separates theorem-level claims, numerical audits
and empirical attacks, and lists the checks that should pass before using the
profiled Hamiltonian or robustness plots in a paper.  A concrete integration
template is in `Additional_tests/examples/after_training.py`.
