# `Stress_tests/theory_aligned/` — application-agnostic theory-verification suite

This directory is a **new, separate** suite that sits alongside the existing
`Stress_tests/` production pipeline (`suite.py`, `boundary.py`, `certificate.py`,
`geometry.py`, `frontier.py`, `rollouts.py`, `model_adapter.py` — all
**unchanged, unmodified** by this work). It exists to answer one question the
legacy pipeline was never structured to answer cleanly:

> **Which specific theorem, in which specific theory document, does each
> stress-test check actually establish — and which checks establish nothing
> more than useful engineering evidence?**

## Why this is application-agnostic

Every experiment in this repository (`deep_dissipative_nlink`,
`duffing_doublewell`, and any future task built against
`nonlinearbenchmark.org`-style benchmarks) that uses the shared `EBM_model`
port-Hamiltonian architecture (`EBM_param_fields.py`, `EBM_class.py`,
`EBM_rollout.py`, `EBM_training.py`) shares the same structural claims:
skew-symmetric interconnection, positive-(semi)definite dissipation, an
energy-CBF safety certificate, etc. This suite is therefore organized as:

* **`metadata/`** — theory-agnostic-*schema*, but experiment-specific
  *content*: `theorem_registry.yaml` and `proof_obligations.yaml` support
  multiple `theory_source` documents and multiple `applies_to_experiments`
  values. Today only `iclr2027_conference_robust_extensions_red.tex`
  (governing `deep_dissipative_nlink`) is populated; a `duffing_doublewell`
  theory document (if one exists or is written) can add its own `claims:`
  entries without restructuring anything.
* **`certificates/`** — the executable check *code* (`checkpoint_io.py`,
  `structural_checks.py`, `unforced_dissipativity_probe.py`,
  `equilibrium_hessian_probe.py`) is 100% experiment-agnostic: every function
  takes a checkpoint path and a config dict as arguments. Pointing them at a
  `duffing_doublewell` checkpoint (which uses the identical `EBMParams`
  pytree layout) requires no code changes, only a new
  `configs/certificates/<experiment>_<run>.yaml`.
* **`fitting/`** — likewise generic: `fitting_report.py` reads any run
  directory following the shared `experiment_paths.py`/`post_training_eval.py`
  layout (`<run>/config/config.json`, `<run>/audits/audit.json`).
* **`configs/`** — the only genuinely **per-experiment, per-run** content:
  `configs/certificates/nlink_copilotA1.yaml` points the generic code at one
  specific checkpoint. A `configs/certificates/duffing_<run>.yaml` would work
  identically for that experiment.

## What "canonical" means here

A test is **canonical** only if it is listed under
`metadata/test_traceability.yaml`'s `canonical_certificate_tests` (suite:
`certificate`) or is the single fitting-suite entry (suite: `fitting`). Every
canonical certificate entry maps to exactly one `theorem_claim_id` (in
`metadata/theorem_registry.yaml`) and one `obligation_id` (in
`metadata/proof_obligations.yaml`). Nothing else — no legacy `suite.py`
sub-analysis, no `Stress_tests/Additional_tests/tests/` pytest file — is part of the
certificate pass/fail decision. See `THEORY_TRACEABILITY.md` for the full
mapping and `../Additional_tests/archive_noncanonical/README.md` for what was reclassified
and why.

## Directory contents

```
theory_aligned/
    README.md                          (this file)
    THEORY_TRACEABILITY.md             (theorem <-> obligation <-> test map, human-readable)
    metadata/
        theorem_registry.yaml          (18 claims, from iclr2027...tex)
        proof_obligations.yaml         (18 atomic obligations, one per claim)
        test_traceability.yaml         (canonical + legacy-reclassified test map)
        removal_manifest.yaml          (every removed/reclassified check, with reasons)
    certificates/
        checkpoint_io.py               (JAX-free checkpoint loader)
        structural_checks.py           (T1/T2/T11/T16: skew, PD, output-decoupling)
        unforced_dissipativity_probe.py (T3: sampled, finite-difference)
        equilibrium_hessian_probe.py   (T5/T12: diagnostic only, does not certify)
    fitting/
        fitting_report.py              (reads stored audit.json; no retraining)
    configs/
        fitting/nlink_copilotA1.yaml
        certificates/nlink_copilotA1.yaml
    scripts/
        run_fitting_suite.py
        run_certificate_suite.py
        generate_verification_report.py
    tests/
        test_traceability_completeness.py
        test_no_unmapped_certificate_tests.py
        test_no_supplementary_certificate_checks.py
    reports/
        copilotA1_fitting_suite_output.json       (executed 2026-08-20)
        copilotA1_certificate_suite_output.json   (executed 2026-08-20)
        copilotA1_verification_summary.json       (executed 2026-08-20)
```

## What this suite does NOT do

* It does not retrain, fine-tune, or otherwise modify any checkpoint.
* It does not modify `Stress_tests/suite.py` or any other legacy production
  module — those remain the shared figure/diagnostic pipeline for every
  experiment.
* It does not claim a "formal proof" for any sampled or finite-difference
  check. See `FORMAL_GUARANTEE_STATUS.md` for the exact, per-obligation
  epistemic status.
* It is not, by itself, a replacement for the legacy `Stress_tests/suite.py`
  pipeline's figures/diagnostics, which remain valuable and continue to run
  unchanged as part of the normal training pipeline (`post_training_eval.py`).

## Dataset-agnostic theory visualization

`visualization.py` defines prepared-data view objects and publication renderers
for projected energy geometry, sampled pointwise input radii, component-wise
radius estimates, and paired 2-D/3-D energy views. It contains no experiment
names, checkpoint loading, equilibrium discovery, or assumptions about physical
state coordinates. Experiment adapters are responsible for declaring whether a
plane is a literal slice, a profiled lower envelope, or a physical state plane.

`duffing_certificate_figure.py` prepares physical `(q,p)` views and, where
available, compares the analytic and learned Hamiltonians.
`nlink_certificate_figure.py` prepares latent-plane views. Because no true
n-link Hamiltonian is available, it compares the learned affine slice with the
learned profiled lower envelope. It treats the traced boundary as one sampled
shell and does not infer equilibrium components from nonstationary candidates.

See `VISUALIZATION_CONTRACT.md` for theorem traceability, required labels, and
the distinction between sampled evidence and formal guarantees.

## Primary Duffing certificate-validation experiment

The dedicated known-system experiment compares the exact mechanical Duffing
certificate against the selected learned checkpoint in physical `(q,p)`
coordinates. It does not retrain the model. From the repository root, run:

```bash
JAX_PLATFORMS=cpu apptainer exec -B /compute:/compute \
  <REDACTED_LEGACY_PATH> python \
  Stress_tests/theory_aligned/duffing_certificate_experiment.py \
  --max-resolution 4096
```

Use `--max-resolution 128 --output-dir /tmp/duffing_certificate_smoke` for a
fast end-to-end check. Canonical outputs are written under
`results/duffing_doublewell/certificate_experiment/`, including the frozen
manifest, exact and learned critical points, per-shell boundary arrays,
resolution convergence, derivative-fidelity and damping-mismatch audits, and
eight figures in PNG/PDF/NPZ form. `FINAL_SUMMARY.md` states which conclusions
are exact, sampled, or invalidated by implemented numerical safeguards.

## Generate the three-panel paper figures

Run every command from the repository root. A completed run must contain
`config/config.json`, `checkpoints/params.pkl`, `stress_tests/summary.json`, and
`stress_tests/arrays/stress_arrays.npz`. The vector-PDF composition step uses
`pypdf`; install it once in the same Python environment used for rendering:

```bash
python -m pip install --user pypdf
```

When using the repository's Apptainer image on the login node, define this
helper once (add `--nv` after `exec` on a GPU node if required):

```bash
CONTAINER=<REDACTED_LEGACY_PATH>
run_python() {
  JAX_PLATFORMS=cpu apptainer exec -B /compute:/compute "$CONTAINER" python "$@"
}
run_python -m pip install --user pypdf
```

Set the two completed run directories:

```bash
DUFFING_RUN=results/duffing_doublewell/runs/duffing_v5_champion_exact_seed4
NLINK_RUN=results/deep_dissipative_nlink/runs/copilot_B3_combo_lr_low
```

### 1. Generate panels a-b

The experiment-specific certificate renderer creates the two-panel upper
figure in both PNG and PDF form:

```bash
run_python Stress_tests/theory_aligned/duffing_certificate_figure.py "$DUFFING_RUN"
run_python Stress_tests/theory_aligned/nlink_certificate_figure.py "$NLINK_RUN"
```

Expected upper-figure files are:

```text
$DUFFING_RUN/stress_tests/theory_aligned/duffing_certificate_figure.{png,pdf}
$NLINK_RUN/stress_tests/theory_aligned/nlink_projected_radius_figure.{png,pdf}
```

The Duffing renderer may create `radius_certificate.{npz,json}` if absent. The
n-link renderer requires its completed stress arrays and reconstructs the
projected certificate evidence from the saved checkpoint.

### 2. Compute panel c

Compute the 300-shell input-radius sweep once for each run:

```bash
run_python Stress_tests/theory_aligned/eps_sweep_input.py "$DUFFING_RUN" --count 300
run_python Stress_tests/theory_aligned/eps_sweep_input.py "$NLINK_RUN" --count 300
```

The default range uses 300 relative shell levels from 0.10 to 5/3 times
`nominal_epsilon - min(H)`. Thus it extends exactly two-thirds of the nominal
energy gap beyond nominal epsilon. For Duffing this crosses the inferred saddle,
so each level set is evaluated globally rather than partitioned by source-well
anchor. Explicit bounds are available through `--epsilon-min` and
`--epsilon-max`. Outputs are
`eps_sweep_input.{png,pdf,npz,json}` under the run's `stress_tests/theory_aligned/`
directory. This command does not modify either certificate figure.
The horizontal coordinate is `epsilon - min(H)` and begins at zero; absolute
epsilon and the stored minimum remain available in the NPZ/JSON artifacts.
Because the sweep extends beyond the nominal shell, its ray-search cap is 1.5
times the original stress-test cap; both values and every failed-ray fraction
are recorded in JSON.

The figure marks every strict interior minimum of the regular-shell curve at
positive relative energy whose log-radius prominence is at least `0.05`. Lime
diamonds labelled `Saddle energy shell` therefore permit multiple events in a
nonconvex landscape while rejecting discretization-scale wiggles. The
sampled-radius curve remains unmarked, and neither curve is smoothed.

This is the expensive stage: it retraces all 300 shells. Do not rerun it merely
to rebuild the combined plate when valid `eps_sweep_input.{npz,json}` artifacts
already exist. To choose explicit absolute epsilon bounds, supply both
`--epsilon-min VALUE` and `--epsilon-max VALUE`; never supply only one.

### 3. Compose the final plate

Stack panels a-b above panel c for both experiments:

```bash
run_python Stress_tests/theory_aligned/combined_figure.py "$DUFFING_RUN"
run_python Stress_tests/theory_aligned/combined_figure.py "$NLINK_RUN"
```

This rerenders `eps_sweep_input` at the certificate's 12-inch width, preserves
the certificate figure unchanged as panels a-b, and appends the sweep as panel
c. The resulting `combined_theory_aligned_figure.{png,pdf}` retains 400 dpi PNG
output and vector PDF content. The command intentionally overwrites only the
standalone `eps_sweep_input.{png,pdf}` with its wide panel-c rendering; it does
not alter `eps_sweep_input.{npz,json}` or either certificate figure.

Final outputs:

```text
$DUFFING_RUN/stress_tests/theory_aligned/combined_theory_aligned_figure.{png,pdf}
$NLINK_RUN/stress_tests/theory_aligned/combined_theory_aligned_figure.{png,pdf}
```

### 4. Validate

Confirm the focused renderers and compositor pass:

```bash
run_python -m pytest -q \
  Stress_tests/Additional_tests/tests/test_eps_sweep_input.py \
  Stress_tests/Additional_tests/tests/test_theory_aligned_visualization.py
```

The combined PNG must have exactly the certificate PNG width. The certificate
source hashes must remain unchanged across step 3; inspect both final PNG and
PDF files before publication for panel labels, clipping, and font embedding.

## Test-input operating-regime audit

Compare held-out applied inputs with the interpolated shell-wide `rho` and
`rho*` margins using learned pre-step energy as the operating coordinate:

```bash
python Stress_tests/theory_aligned/operating_regime_margin_audit.py <run_dir>
```

For output-only n-link runs, pass the exact source dataset stem with
`--nlink-dataset-stem`. Results are split at prominent saddle shells and the
nominal shell and written as `operating_regime_margin_audit.{json,csv,md}`.
Interpolation never crosses a failed shell or extrapolates beyond sweep support.
An above-margin input is outside the sampled shell-wide guarantee; it is not by
itself evidence of an unsafe observed transition.

## Automatic execution (wired into the training pipeline)

As of this update, `post_training_eval.py::run_post_training_evaluation`
(the shared post-training entry point called by every experiment's
`train()`) automatically invokes the canonical certificate checks and the
fitting-evidence snapshot **for every future run**, right after the legacy
`Stress_tests.suite.run_stress_suite` call. No config flag is needed to
enable this — it runs unconditionally, guarded the same way every other
pipeline stage is (a failure here is recorded in `errors` and
`run_status.json["theory_aligned_certificates_complete"]`, and never
aborts reconstruction/plots/audit/legacy-stress).

Per-run outputs land inside the **existing** `stress_tests/` folder, not a
separate top-level directory, so `runs/<run_id>/stress_tests/` now
contains both the legacy sub-folders (`arrays/`, `figures/`, `summary.json`)
and a new `theory_aligned/` sub-folder:

```
runs/<run_id>/stress_tests/
    arrays/                       (legacy, unchanged)
    figures/                      (legacy, unchanged)
    summary.json                  (legacy, unchanged)
    theory_aligned/
        certificate_results.json  (T1/T2/T3/T11/T16 structural+sampled checks,
                                   plus the T5/T12 equilibrium/Hessian diagnostic)
        fitting_evidence.json     (test NRMSE snapshot, read from audit.json)
```

`run_status.json` gains one new key, `theory_aligned_certificates_complete`;
`audits/audit.json` gains two new `metrics` keys,
`theory_aligned_certificates` and `theory_aligned_fitting_evidence`. No
existing key in either file is removed or renamed.

**Runs completed before this wiring change** (including the first
`copilotA1` rerun analyzed in
`copilotA1_theory_aligned_stress_suite_report.md`) do **not** have this
`theory_aligned/` sub-folder unless it was backfilled manually — see that
report for the standalone invocation commands
(`scripts/run_certificate_suite.py` / `scripts/run_fitting_suite.py`),
which remain available for exactly this purpose.

## Running the suite manually (standalone, e.g. to backfill an older run)

```bash
# Fitting evidence (stdlib json only, works with plain system python3):
python3 fitting/fitting_report.py <run_dir> [threshold]

# Certificate checks (numpy + pyyaml; this repo's .venv_deep_dissipative works):
<venv>/bin/python3 scripts/run_certificate_suite.py configs/certificates/nlink_copilotA1.yaml

# Suite-integrity meta-tests:
<venv>/bin/python3 tests/test_traceability_completeness.py
<venv>/bin/python3 tests/test_no_unmapped_certificate_tests.py
<venv>/bin/python3 tests/test_no_supplementary_certificate_checks.py
```
