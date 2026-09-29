# Release Audit

## Executive summary

The repository was renamed from its conference-named directory to `EBM_framework` and
non-README conference terminology was removed from source, text metadata, and
artifact filenames. No README bytes, raw datasets, checkpoints, or learned
parameters were modified. Structural, artifact, inference, model-runtime, and
HNER99 smoke validations pass in the repository's JAX Singularity container.

## Initial repository inventory

The initial tree contained 344 files, 14 immutable README files, six public
`scripts/` launchers, 24 directly executable project-owned Python workflows,
scientific datasets/checkpoints, and one vendored Deep Dissipative Model tree.
The target directory is not a Git worktree; `git status` therefore could not
be recorded. Vendored code and third-party notices were not attributed or
rewritten.

## Files renamed or removed

- The conference-named root directory was renamed to `EBM_framework/`.
- Four NanoDrone presentation files were renamed to `nanodrone_main.png` and
  `nanodrone_spotlight_*.png`.
- Compiler-generated `__pycache__` directories were removed after validation.

## Conference-specific references removed

All non-README conference-token matches were removed. Theory metadata now uses the neutral identifier
`reference_robust_extensions`; corresponding references in certificate source,
configuration, formal-status text, removal metadata, and existing JSON audit
records were updated. The Duffing module description now calls the benchmark a
reference experiment.

## Intentionally preserved README references

README files were excluded from remediation by requirement. The final
case-insensitive audit found no conference token in them, so there are zero
preserved README references.

## Attribution changes

Added exactly one `# Author: Simone Betteti` declaration to 24 project-owned
direct workflow files that lacked one, including all six public scripts,
top-level release validation, baseline launchers, and dataset workflow
launchers. The direct-workflow attribution check reports zero missing or
duplicate declarations. Vendored Deep Dissipative Model sources were excluded.

## Code simplifications and comments

No numerical or API refactor was made: preserving scientific behavior took
priority over speculative simplification. One conference-specific Duffing
module sentence and the relevant theory-reference comments/docstrings were
made release-neutral. Broader comment/docstring rewriting was intentionally
deferred because it cannot be validated independently of scientific behavior.

## Smoke-test matrix

| Script | Workflow | Command | Result | Output checked | Notes |
|---|---|---|---|---|---|
| `validate_porthnn_u_artifacts.py` | Stored-result audit | `singularity exec … python scripts/validate_porthnn_u_artifacts.py` | Passed | Four checkpoint hashes; stored metrics | No output files created |
| `validate_porthnn_u_runtime.py` | Runtime rollout | `singularity exec … python scripts/validate_porthnn_u_runtime.py` | Passed | Four finite model rollouts and structure checks | CPU execution |
| `validate_inference.py` | Bundled inference | `singularity exec … python scripts/validate_inference.py` | Passed | Six datasets; finite four-step outputs | CPU execution |
| `validate_hner99_artifacts.py` | Frozen-report integrity | `PYTHONDONTWRITEBYTECODE=1 python scripts/validate_hner99_artifacts.py` | Passed | 9 reports, provenance hashes | No output files created |
| `compute_robustness_radii.py` | HNER99 runtime radius | `singularity exec … python scripts/compute_robustness_radii.py duffing_porthnn --output /tmp/ebm_release_runtime_radius` | Passed | JSON and text report; HNER99 minimum `0.0005307992290985785` | Temporary output only |
| `audit_hner99_runtime.py` | HNER99 audit | `singularity exec … python scripts/audit_hner99_runtime.py duffing_porthnn --output /tmp/ebm_release_runtime_audit` | Passed | `duffing_porthnn_audit.json` | Temporary output only |

## Commands executed

```bash
python -m compileall EBM_framework
PYTHONDONTWRITEBYTECODE=1 python validate_release.py --structural
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_hner99_artifacts.py
singularity exec -B /hpc:/hpc /hpc/home/rias/sbetteti/containers/anaconda-jax-CUDA13.sif python scripts/validate_porthnn_u_artifacts.py
singularity exec -B /hpc:/hpc /hpc/home/rias/sbetteti/containers/anaconda-jax-CUDA13.sif python scripts/validate_porthnn_u_runtime.py
singularity exec -B /hpc:/hpc /hpc/home/rias/sbetteti/containers/anaconda-jax-CUDA13.sif python scripts/validate_inference.py
singularity exec -B /hpc:/hpc /hpc/home/rias/sbetteti/containers/anaconda-jax-CUDA13.sif python scripts/compute_robustness_radii.py duffing_porthnn --output /tmp/ebm_release_runtime_radius
singularity exec -B /hpc:/hpc /hpc/home/rias/sbetteti/containers/anaconda-jax-CUDA13.sif python scripts/audit_hner99_runtime.py duffing_porthnn --output /tmp/ebm_release_runtime_audit
singularity exec --env PYTHONDONTWRITEBYTECODE=1 -B /hpc:/hpc /hpc/home/rias/sbetteti/containers/anaconda-jax-CUDA13.sif python validate_release.py
# Case-insensitive conference-token content and filename searches (no matches)
```

## Checks and blockers

- Passed: `compileall`; aggregate release validation; all six public script
  workflows; non-README content and filename conference-token searches;
  attribution audit; README checksum comparison.
- Failed: none. Skipped: none.
- Runtime validation used JAX 0.9.2 and NumPy 2.5.2 on CPU. The CUDA-enabled
  image reports a non-fatal CUPTI initialization warning without GPU
  passthrough, then correctly selects `CpuDevice(id=0)`.
- Remaining artifact limitations are already explicit: no Silverbox pH-EBM
  checkpoint is supplied, and the historical NanoDrone report used seed 23
  while the contained champion uses seed 47.
- No test runner configuration was found; only a standalone
  `test_full_matrix_jrg.py` workflow exists.

## README integrity verification

| README file | Initial SHA-256 | Final SHA-256 | Unchanged |
|---|---|---|---|
| `README.md` | `21f544672b6df5ee15b1785355ad22cc6fabf91babdc3752134f892a8a112dcf` | `21f544672b6df5ee15b1785355ad22cc6fabf91babdc3752134f892a8a112dcf` | Yes |
| `Stress_test_final/Stress_tests/README.md` | `ee285ca2a3e56179c560cd1a7e5a611073ff02d7865c08b64614726a4b6241f1` | same | Yes |
| `Stress_test_final/Stress_tests/theory_aligned/README.md` | `3f98b35be75748928cadceb37c8f79106a0f3286e55180b10ae348c69a1d3a21` | same | Yes |
| `baselines/porthnn_u/README.md` | `b208598b5377f27ff8bf7970e30c46db36bbfadd5f76e1b562ce084c02263ac6` | same | Yes |
| `dataset_interfaces/baseline_interfaces/README.md` | `014ef3642bf1e4330562ae527132479c71944206ff9f5b4ef699a453f3ff697b` | same | Yes |
| `dataset_interfaces/baseline_interfaces/deep_dissipative_model/README.md` | `83ee98c9f9c87df8b50c774ef355a863c0823d796594d6509c88e1ea7a53c059` | same | Yes |
| `datasets/CED/README.md` | `be792e011ae813702a45be62dda9a99c0c9a46be849d54dccbfbe98754c2d52f` | same | Yes |
| `datasets/README.md` | `31ae1f9b4d01b421020355d121da3e0b467fdea8187cf35379cf3c19575c8a0a` | same | Yes |
| `datasets/Silverbox/README.m` | `90c365916d60afc3cb208f772be40918175370cbcf43d1ada3df19b8a2c58ae8` | same | Yes |
| `datasets/Silverbox/README.md` | `33c3793c45e143cfb453403556067620cbd60d8ef0c0140965801c3fe5016452` | same | Yes |
| `datasets/Silverbox/README.txt` | `90c365916d60afc3cb208f772be40918175370cbcf43d1ada3df19b8a2c58ae8` | same | Yes |
| `datasets/duffing_doublewell/README.md` | `f03348d37d8bdef6147180baa6e2dfc332c90b47475be0e0eb00ded2b0be58c1` | same | Yes |
| `datasets/nanodrone_S3/parameters/raw_data/README.md` | `5e974fe04d05ed563a4a7a0b68600de1bdc3e4c6b7b0a967ec51db4d4104524a` | same | Yes |
| `runtime_hner99/README.md` | `66a66352dafa15b9e4c8db6299fb8ee4d98ef2b4854fec03104f0278d56e6e81` | same | Yes |

## Conclusion

- Files inspected: 344
- Non-README conference references removed: 47 textual occurrences and 5 paths
- Intentionally preserved README references: 0
- Files/directories renamed: 5
- Executable files attributed: 24
- Source files simplified: 0 (no behavior-changing refactor)
- Public scripts tested: 6
- Public script workflows passed: 6; failed: 0; skipped: 0
- Aggregate release validation: passed
- All README files remain byte-for-byte unchanged: yes
- Scientific behavior and public APIs preserved: yes, supported by artifact,
  inference, runtime, structural, and aggregate validation.
- Ready for GitHub publication: yes, with the two documented historical
  artifact limitations above.
