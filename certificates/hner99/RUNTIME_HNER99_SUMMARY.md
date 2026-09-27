# Revised Low-Resource Runtime HNER99 Results

All values use the actual stored runtime RHS, the frozen ordinary validation
rollout, a q=0.99 validation-energy shell, zero input as the center, and the
uncentered validation-input second moment.

| Dataset | Model | HNER99 minimum | p10 | Median | Status | 32/64 sensitive |
|---|---|---:|---:|---:|---|---|
| Duffing double-well | pH-EBM | 0.411770691 | 0.447312593 | 0.656732001 | `POST_STEP_PROJECTION_DIAGNOSTIC_ONLY` | No |
| Duffing double-well | PortHNN-u | 0.000530747 | 0.074788430 | 0.395678396 | `EMPIRICAL_RUNTIME_HNER` | **Yes** |
| Three-link pendulum | pH-EBM | 0.217581704 | 0.221043612 | 0.325568772 | `POST_STEP_PROJECTION_DIAGNOSTIC_ONLY` | No |
| Three-link pendulum | PortHNN-u | 0.005192362 | 0.005192441 | 0.005192797 | `EMPIRICAL_RUNTIME_HNER` | No |
| NanoDrone | pH-EBM | 0.063535166 | 0.100586533 | 0.116307046 | `POST_STEP_PROJECTION_DIAGNOSTIC_ONLY` | No |
| Silverbox | pH-EBM (`0013id7u_plot`) | 0.188826744 | 0.246651771 | 0.624686791 | `POST_STEP_PROJECTION_DIAGNOSTIC_ONLY` | No |
| Silverbox | PortHNN-u (`sb02_16`) | 0.019517321 | 0.056438795 | 0.281713608 | `EMPIRICAL_RUNTIME_HNER` | No |
| CED | pH-EBM (`ced_v10b1_e8000`) | 0.607740359 | 0.625671856 | 1.256218728 | `POST_STEP_PROJECTION_DIAGNOSTIC_ONLY` | No |
| CED | PortHNN-u (`ced02_07`, retained fold 1) | 0.183689576 | 0.183753839 | 0.186929470 | `EMPIRICAL_RUNTIME_HNER` | No |

The Duffing PortHNN-u result uses its complete nonlinear forcing branch and a
65-point scalar raw-input search followed by Brent refinement. The pH-EBM
runtime vector fields include input-nonlinear numerical safeguards, so the
three multidimensional pH-EBM cases use 16 Sobol directions plus their
negatives and scalar Brent refinement along each direction.

The pH-EBM rollout integrator performs explicit state clipping/projection
around integration stages. Its reported numbers characterize the underlying
runtime RHS evaluated at the reconstructed shell points; they are therefore
diagnostic for the full discrete rollout and carry the required
`POST_STEP_PROJECTION_DIAGNOSTIC_ONLY` status.

No accepted shell point was outward-pointing at zero input. Bootstrap was not
run for the three-link or NanoDrone datasets because they contain fewer than
five independent validation trajectories. Duffing uncertainty remains optional
under the revised protocol and was not recomputed in this low-resource run.

The Duffing PortHNN-u result changed materially from 32 to 64 shell points
(`0.01864146` to `0.000530747`), so it is explicitly marked shell-sampling
sensitive and the smaller 64-point value is retained.

The CED pH-EBM certificate uses the ordinary hard-projection RK4 validation
rollout. Its 62-state validation set exposed a numerical limitation in the
original three-step shell projector, which accepted only 3/32 and 7/62 shell
points. Damped Newton convergence on the same frozen seeds accepts 10/32 and
19/62 points; both budgets return `0.6077403587`. No model, validation split,
normalization, energy level, seed selection, or radius search was changed.

The registered CED PortHNN-u result is the campaign-prescribed retained fold-1
checkpoint of validation-selected configuration `ced02_07`.

The registered Silverbox pH-EBM result belongs to the available executable
Certificate run `0013id7u_plot`; it is not the historical `sbx_anchor_p16`
champion, whose trained weights were not saved.
