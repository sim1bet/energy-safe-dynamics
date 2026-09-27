# Runtime HNER99 library

This importable package integrates the supplied runtime Hamiltonian-Normalized
Empirical Radius at 99% validation occupancy protocol. It preserves the fixed
shell budgets, runtime-RHS evaluation, affine and nonlinear input searches,
32/64-shell sensitivity check, and report fields while replacing machine-bound
loaders with release-relative adapters.

The calculation is an empirical numerical diagnostic on a finite validation
energy shell, not a formally verified global invariance certificate.

| File | Purpose |
| --- | --- |
| `runtime.py` | model registry, validation-state construction, shell reconstruction, radius computation, and report writer |
| `audit.py` | fixed-budget loader and shell audit with checkpoint hashes and finite-data checks |
| `artifacts.py` | dependency-light access to frozen reports and provenance |
| `ARTIFACT_INDEX.json` | selector-to-artifact mapping for all nine frozen reports |

Frozen reports remain stored once under `certificates/hner99/`; the index avoids
a second mutable copy.

```bash
python scripts/validate_hner99_artifacts.py
python scripts/compute_robustness_radii.py --list
python scripts/compute_robustness_radii.py duffing_phebm duffing_porthnn
python scripts/audit_hner99_runtime.py ced_phebm ced_porthnn
```

Fresh computations write to `results/hner99_runtime/`; audits write below
`results/hner99_runtime/audits/`. Neither overwrites supplied frozen reports.

The historical Silverbox pHEBM checkpoint was not supplied. The historical
NanoDrone report used seed 23 while the included champion is seed 47. Both
limitations are explicit in the provenance file.

The artifact API works without JAX:

```python
from runtime_hner99 import artifact_index, load_frozen_report, provenance
report = load_frozen_report("ced_porthnn_u")
```

Numerical users can import `Model`, `LOADERS`, `compute`, `reconstruct_shell`,
and `write_report` from `runtime_hner99.runtime` after installing the complete
environment.

