# Theory Traceability — human-readable summary

Full machine-readable detail: `metadata/theorem_registry.yaml`,
`metadata/proof_obligations.yaml`, `metadata/test_traceability.yaml`.
Executed results (2026-08-20): `reports/copilotA1_certificate_suite_output.json`,
`reports/copilotA1_fitting_suite_output.json`.

| Claim | Name | Obligation(s) | Canonical test | Status |
|---|---|---|---|---|
| T1 | J skew-symmetry | OB-T1-1 | `certificates/structural_checks.py::check_skew_symmetry` | **PROVED_BY_CONSTRUCTION** (executed: max defect = 0.0) |
| T2 | R positive-(semi)definiteness | OB-T2-1 | `certificates/structural_checks.py::check_dissipation_pd` | **PROVED_BY_CONSTRUCTION** (executed: λ_min=0.004419, strictly PD) |
| T3 | Unforced dissipativity | OB-T3-1 | `certificates/unforced_dissipativity_probe.py` | **STRUCTURAL_REGRESSION_CHECKED** (sampled, finite-difference; 8/8 samples pass) |
| T4 | Coercivity | OB-T4-1 | none (proof-extension needed) | **NOT_VERIFIED** |
| T5 | Isolated equilibrium z* | OB-T5-1 | `certificates/equilibrium_hessian_probe.py` (diagnostic only) | **NOT_VERIFIED** |
| T6 | Exact pointwise CBF set | OB-T6-1 | legacy `suite.py` certificate_boundary_audit (reclassified, SAMPLED_ONLY) | **CONTRADICTED** (7.8% violation) |
| T7 | Implemented-dynamics slack | OB-T7-1 | legacy `suite.py` (reclassified, out-of-theory-scope diagnostic) | **EMPIRICALLY_SUPPORTED** (16.7% violation, engineering metric) |
| T8 | Maximal uniform robust input set | OB-T8-1 | none (blocked on T5) | **NOT_VERIFIED** |
| T9 | Exact certified radius | OB-T9-1 | none (blocked on T5; never computed) | **NOT_VERIFIED** |
| T10 | Regular boundary | OB-T10-1 | none (missing instrumentation) | **NOT_VERIFIED** |
| T11 | Boundary dissipation lower bound | OB-T11-1 | `certificates/structural_checks.py::check_dissipation_pd` (shared) | **PROVED_BY_CONSTRUCTION** (r_{ε,⋆} ≥ 0.004419, automatic) |
| T12 | Local PL / Hessian margin | OB-T12-1 | `certificates/equilibrium_hessian_probe.py` (diagnostic only) | **NOT_VERIFIED** (Hessian PD at non-stationary point, does not certify PL) |
| T13 | Input-to-energy tube | OB-T13-1 | none (blocked on T5, T12) | **NOT_VERIFIED** |
| T14 | State-disturbance robustness | OB-T14-1 | legacy unit test (formula only; radius=0 for copilotA1) | **NOT_VERIFIED** (never exercised at radius>0) |
| T15 | Plant-level transfer | OB-T15-1 | none | **NOT_APPLICABLE** (output_only mode, by design) |
| T16 | Output/port decoupling | OB-T16-1 | `certificates/structural_checks.py::check_output_port_decoupling` | **PROVED_BY_CONSTRUCTION** (executed: confirmed distinct code paths) |
| T17 | No asymptotic-stability claim | OB-T17-1 | none needed | **NOT_APPLICABLE** (both theory and code agree) |
| T18 | Continuous-time forward invariance | OB-T18-1 | legacy `suite.py` rollout tests (reclassified, empirical-on-surrogate) | **EMPIRICALLY_SUPPORTED** (0% violation, greedy adversary, discretized+projected system) |

**Totals (18 obligations):** 4 `PROVED_BY_CONSTRUCTION`, 1
`STRUCTURAL_REGRESSION_CHECKED`, 2 `EMPIRICALLY_SUPPORTED`, 1 `CONTRADICTED`,
8 `NOT_VERIFIED`, 2 `NOT_APPLICABLE`.

## Reading this table correctly

* A `PROVED_BY_CONSTRUCTION` obligation means the property follows
  algebraically from the parameterization for **any** parameter values, and
  this suite additionally confirmed it numerically for the **actual trained**
  checkpoint. It is not a statement about any other obligation.
* `CONTRADICTED` (T6) means the specific, already-tested operating point
  (ε, γ, input radius) fails the theorem's pointwise inequality on a
  material fraction of a finite sample — it does not mean the theorem is
  false, only that this configuration does not satisfy it as tested.
* `NOT_VERIFIED` is not a euphemism for "probably fine" — see
  `FORMAL_GUARANTEE_STATUS.md` for exactly what evidence (if any) exists for
  each `NOT_VERIFIED` entry and what a genuine verification would require.
