# Formal-Guarantee Status — `copilotA1` (`deep_dissipative_nlink`)

Instance document for run `copilot_A1_full_stageA_recipe`, checkpoint
`results/deep_dissipative_nlink/runs/copilot_A1_full_stageA_recipe/checkpoints/params.pkl`,
theory source `iclr2027_conference_robust_extensions_red.tex`. Generated
against `metadata/theorem_registry.yaml` / `proof_obligations.yaml`
(2026-08-20). This document's *structure* is application-agnostic (any
other run/experiment gets its own instance of this file); the *content*
below is specific to this one checkpoint.

Status vocabulary used: `FORMALLY_ESTABLISHED_WITHIN_STATED_SCOPE`,
`ESTABLISHED_CONDITIONAL_ON_EXTERNAL_ASSUMPTIONS`, `PARTIALLY_ESTABLISHED`,
`EMPIRICALLY_SUPPORTED_ONLY`, `INCONCLUSIVE`, `NOT_ESTABLISHED`,
`CONTRADICTED`.

---

## T1 — Interconnection skew-symmetry

- **Claim:** J_θJ(z) = -J_θJ(z)^T for all z.
- **Theoretical source:** eq. (ph_dynamics).
- **Scope:** global, all z ∈ R^8.
- **Assumptions:** none beyond the parameterization itself.
- **Implementation mapping:** `EBM_param_fields.py::_assemble_A`.
- **Verification status:** PROVED_BY_CONSTRUCTION.
- **Evidence type:** exact algebraic identity, numerically confirmed on the trained checkpoint (float64 recompute, max defect = 0.0).
- **Passing evidence:** `reports/copilotA1_certificate_suite_output.json` → `OB-T1-1`.
- **Unresolved obligations:** none.
- **Permitted conclusion:** J_θJ is exactly skew-symmetric for `copilotA1`, for every z, unconditionally.
- **Prohibited overstatement:** this says nothing about R, the CBF certificate, or fitting quality.

## T2 — Dissipation matrix positive-(semi)definiteness

- **Claim:** 0 ⪯ R_θR(z) = R_θR(z)^T.
- **Theoretical source:** eq. (ph_dynamics).
- **Scope:** global, all z (state-independent for `copilotA1`).
- **Assumptions:** none.
- **Implementation mapping:** `EBM_param_fields.py::_assemble_M`, `_cholesky_bounded`.
- **Verification status:** FORMALLY_ESTABLISHED_WITHIN_STATED_SCOPE (exceeds the theory's requirement — strict PD, not merely PSD).
- **Evidence type:** exact algebraic construction + numerical eigendecomposition of the actual trained matrix.
- **Passing evidence:** `OB-T2-1`: λ_min=0.004419, λ_max=0.447725, condition number ≈101.3.
- **Unresolved obligations:** none for this claim in isolation.
- **Permitted conclusion:** R_θR ≻ 0 strictly, for every z, for `copilotA1`.
- **Prohibited overstatement:** strict positive-definiteness of R alone does not imply the forced/robust CBF certificate holds (see T6).

## T3 — Unforced dissipativity identity

- **Claim:** dH/dt = -∇H^T R ∇H + y_p^T u ≤ y_p^T u.
- **Theoretical source:** eq. (energy_balance).
- **Scope:** global, idealized (unclipped) formula.
- **Assumptions:** T1, T2.
- **Implementation mapping:** `EBM_param_fields.py::vector_field_and_output` (uses a GRAD_E_CLIP-clipped gradient, not the exact ∇H).
- **Verification status:** ESTABLISHED_CONDITIONAL_ON_EXTERNAL_ASSUMPTIONS (holds exactly given T1+T2; the *clipped* surrogate used elsewhere in the pipeline is a distinct, approximately-equal-but-not-identical quantity).
- **Evidence type:** algebraic consequence of T1/T2, confirmed numerically at 8 finite-difference-sampled states.
- **Passing evidence:** `OB-T3-1`: drift ≥ 0 at all 8 samples; GRAD_E_CLIP found materially active (clip scale factors 0.54–0.98).
- **Unresolved obligations:** exact (unclipped) verification at the specific points used by the legacy certificate audit was not re-run with this suite.
- **Permitted conclusion:** the idealized unforced dissipativity identity holds for `copilotA1`, exactly at u=0, subject to T1/T2.
- **Prohibited overstatement:** this does not establish anything about u≠0 (see T6/T7) or about the fully-clipped, RK4-integrated system (see T18).

## T4 — Coercivity

- **Claim:** H_θH(z) ≥ (1/2)‖z‖² - c1‖z‖ - c0.
- **Theoretical source:** eq. (coercive_lower_bound), App. app:hybrid.
- **Scope:** global, all z ∈ R^8.
- **Assumptions:** first hidden activation (tanh, bounded) has bounded image/derivative; deeper layers continuous; **theory's proof omits the per-layer bias term that `EBM_class.py::energy_layer_k` actually includes.**
- **Implementation mapping:** `EBM_class.py::energy_EBM`.
- **Verification status:** INCONCLUSIVE (the activation-boundedness precondition is satisfied; the bias-term extension is plausible but not written anywhere as a proof).
- **Evidence type:** none executed; this is a proof-obligation for the theory document.
- **Passing evidence:** none.
- **Unresolved obligations:** OB-T4-1 (extend App. app:hybrid's proof to the biased Hamiltonian).
- **Permitted conclusion:** coercivity is plausible for `copilotA1`'s architecture but not formally established as stated.
- **Prohibited overstatement:** do not cite T4 as "proved" for the exact implemented Hamiltonian.

## T5 — Isolated local minimizer

- **Claim:** ∃ z⋆ with ∇H_θH(z⋆) = 0.
- **Theoretical source:** definitions preceding eq. (component_safe_set).
- **Scope:** operational ball ‖z‖ ≤ 50 (implementation restriction); R^8 (theory, unrestricted).
- **Assumptions:** T4 (for existence somewhere in R^8, via coercivity + continuity).
- **Implementation mapping:** `Stress_tests/geometry.py::discover_minima`, `filter_converged_wells`.
- **Verification status:** NOT_ESTABLISHED.
- **Evidence type:** search failure (192 multi-start descents, 300 steps each), independently reproduced (finite-difference gradient norm 104.5497 vs. audit.json's 104.5496, relative difference 8.4e-7).
- **Passing evidence:** none — this is a documented failure, not passing evidence.
- **Unresolved obligations:** OB-T5-1 (interval Newton / branch-and-bound, or a much larger search budget).
- **Permitted conclusion:** no equilibrium was found within the operational ball for `copilotA1` at the tested search budget; existence elsewhere in R^8 is plausible (not contradicted) by T4's coercivity argument.
- **Prohibited overstatement:** do not say "no equilibrium exists" — that is a stronger, unestablished claim.

## T6 — Exact pointwise CBF admissible-input set

- **Claim:** U_CBF(z) = {u ∈ U : a_H(z)^T u ≤ d_H(z) + α(h_ε(z))} for all z ∈ D.
- **Theoretical source:** Proposition exact_cbf_set.
- **Scope:** global pointwise, any z in an open neighborhood of C_ε; no equilibrium required by the theory itself.
- **Assumptions:** T1, T2, H ∈ C¹ with locally Lipschitz gradient.
- **Implementation mapping:** `Stress_tests/certificate.py::audit_certificate` (formula path — uses the CLIPPED gradient, not exact ∇H).
- **Verification status:** CONTRADICTED, at the tested operating point (ε=292.49, γ=1.0, input radius=1.0).
- **Evidence type:** sampled numerical check, 526 boundary points.
- **Passing evidence:** none — 7.8% (41/526) of sampled points violate the inequality beyond the 1e-5 tolerance.
- **Unresolved obligations:** OB-T9-1 (compute the theory's own certified radius and compare to the ad hoc radius=1.0 actually tested — the violation may simply mean radius=1.0 exceeds the true certified radius, not that no radius works).
- **Permitted conclusion:** the specific, ad hoc operating point tested is NOT certified by this sample; a smaller input radius or larger ε margin may be.
- **Prohibited overstatement:** do not claim "the CBF certificate is proven" or "disproven in general" — only this specific (ε, γ, radius) combination, on this finite sample, is shown to be violated.

## T7 — Implemented-dynamics CBF slack (engineering diagnostic)

- **Claim:** none in the theory document — this is an engineering diagnostic comparing the deployed (clipped/RK4) system to T6.
- **Theoretical source:** n/a.
- **Scope:** sampled boundary states.
- **Assumptions:** n/a.
- **Implementation mapping:** `Stress_tests/model_adapter.py::implemented_hdot`.
- **Verification status:** EMPIRICALLY_SUPPORTED_ONLY (as an engineering metric, not a theorem).
- **Evidence type:** sampled + PGD-refined worst-case input.
- **Passing evidence:** none — 16.7% (88/526) violation.
- **Unresolved obligations:** n/a (out of theory scope).
- **Permitted conclusion:** the deployed, safety-clipped system has a larger observed violation rate than the idealized formula on this sample.
- **Prohibited overstatement:** do not describe this as testing any numbered theorem.

## T8 — Maximal uniform robust input set

- **Verification status:** NOT_ESTABLISHED (blocked on T5). No code computes this.
- **Permitted conclusion:** none can be drawn.
- **Prohibited overstatement:** do not claim any input set has been shown maximal or forward-invariant for `copilotA1`.

## T9 — Exact certified radius ρ*

- **Verification status:** NOT_ESTABLISHED (blocked on T5; never computed by any existing code path).
- **Permitted conclusion:** the ad hoc `input_radius=1.0` used throughout the legacy stress suite has never been compared to its theoretically-justified certified value.
- **Prohibited overstatement:** do not treat `input_radius=1.0`'s pass/fail rate as informative about whether a smaller, actually-certifiable radius exists.

## T10 — Regular energy boundary

- **Verification status:** NOT_ESTABLISHED (plausible but unmeasured; per-point boundary gradient norms are not logged by the existing pipeline).
- **Permitted conclusion:** none.
- **Prohibited overstatement:** do not assume boundary regularity because ray-tracing "succeeded" for 526/768 rays — that measures something else (energy-level crossing), not gradient nonvanishing.

## T11 — Boundary dissipation lower bound

- **Verification status:** FORMALLY_ESTABLISHED_WITHIN_STATED_SCOPE.
- **Evidence type:** structural (shares evidence with T2, since M_θ is state-independent).
- **Passing evidence:** r_{ε,⋆} ≥ λ_min(M_θ) = 0.004419, holding automatically for any z.
- **Permitted conclusion:** this specific precondition of the regular-boundary/PL radius formulas is satisfied for `copilotA1`, independent of whether T5/T10 are resolved.
- **Prohibited overstatement:** this alone does not yield a usable numeric radius (that also needs κ_{ε,⋆} or μ⋆, both blocked on T5).

## T12 — Local Clarke-Hessian PL certificate

- **Verification status:** NOT_ESTABLISHED (blocked on T5).
- **Evidence type:** diagnostic only — finite-difference Hessian at the current non-stationary candidate is positive definite (eigenvalues 9.15–732.09), a favorable but non-conclusive sign.
- **Permitted conclusion:** a Newton step from the current candidate is a promising, cheap next search direction; nothing about the PL condition itself is established.
- **Prohibited overstatement:** do not cite the positive Hessian eigenvalues as evidence for Proposition hessian_pl — that proposition is defined only at a genuine z⋆.

## T13 — Input-to-energy tube

- **Verification status:** NOT_ESTABLISHED (blocked on T5, T12). No code computes this.

## T14 — State-disturbance robustness

- **Verification status:** NOT_ESTABLISHED for `copilotA1` specifically (formula verified correct by infrastructure unit tests; `state_disturbance_radius=0` for this run, so it was never numerically exercised).
- **Permitted conclusion:** the disturbance-robustness *machinery* is implemented correctly (per `Stress_tests/Additional_tests/tests/test_mapped_disturbance.py`); `copilotA1` itself has no evidence either way.

## T15 — Plant-level transfer

- **Verification status:** NOT_APPLICABLE, by design (`output_only` mode has no certified latent↔physical correspondence; both the theory document and the experimental audit independently agree).

## T16 — Fitted-output / port-output decoupling

- **Verification status:** FORMALLY_ESTABLISHED_WITHIN_STATED_SCOPE.
- **Evidence type:** static architectural inspection.
- **Passing evidence:** confirmed distinct code paths; `w_passivity=0.0` (port output never entered training).
- **Permitted conclusion:** for `copilotA1`, fitting quality of y_obs provides zero direct evidence about y_p's dissipativity-relevant behavior — this is intentional, by the theory's own design.

## T17 — No asymptotic-stability claim

- **Verification status:** NOT_APPLICABLE (no such claim exists in either the theory or the implementation; both agree).

## T18 — Continuous-time forward invariance (Nagumo)

- **Verification status:** EMPIRICALLY_SUPPORTED_ONLY, and only for the discretized + hard-clip-projected surrogate system, not the exact ODE the proof concerns.
- **Evidence type:** empirical stress test, 96 rollouts × 3 step sizes, greedy (non-optimal) adversary.
- **Passing evidence:** 0% violation, min h ≈ 91–94.
- **Unresolved obligations:** the exact continuous-time claim, and any claim using a globally-optimal (not greedy) adversary.
- **Permitted conclusion:** the deployed surrogate system did not leave the safe set in this specific finite sample of rollouts.
- **Prohibited overstatement:** do not describe this as proving or even directly testing Theorem thm:uniform_boundary_set, which concerns the exact, unclipped, unprojected ODE.

---

## Overall formal-guarantee conclusion for `copilotA1`

Under assumptions T1 (skew J) and T2 (PD R) — both **formally established
within stated scope** for the actual trained checkpoint — `copilotA1`
**does** satisfy the theory's unforced dissipativity identity (T3) exactly,
in the idealized formula sense. It **cannot currently be shown** to satisfy
the theory's forced/robust CBF invariance claims (T6, T8, T9, T13) over any
domain, because the load-bearing precondition (an isolated, locatable
equilibrium, T5) is not established, and the one concrete pointwise sample
that was tested (T6) is **contradicted** at the specific operating point
used. The conclusion is supported by structural/algebraic evidence for T1,
T2, T3, T11, T16, and by sampled/empirical evidence only for T6, T7, T18 —
none of which rises to a formal, domain-complete guarantee.
