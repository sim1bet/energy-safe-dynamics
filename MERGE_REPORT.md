# Consolidation report

## Outcome

The primary tree remains authoritative. CED and Silverbox were integrated
as benchmark adapters and dataset assets without introducing a second active
EBM implementation. The later PortHNN-u adversarial comparison and HNER99
runtime-radius bundles were then integrated around that same core. The release
contains six pHEBM dataset selectors, one EBM core, one four-benchmark
PortHNN-u package, one stress-test package, and aggregate validators.

## Input inventory summary

| Input | Files | Main content |
|---|---:|---|
| Primary input | 232 | 98 Python files; four benchmark datasets; champion and baseline checkpoints; stress tests |
| CED/Silverbox (secondary) | 339 | 253 Python files, mostly vendored dependencies; two benchmark datasets; one champion checkpoint; raw logs/results |

Only `README.md` and `requirements.txt` occupied the same relative path. The
core files corresponded after mapping primary `EBM_model/*.py` to secondary
root `EBM_*.py`; the stress package corresponded after mapping
`Stress_test_final/Stress_tests/` to `Stress_tests/`.

## Conflict decisions

| Component | Decision | Secondary change incorporated | Validation |
|---|---|---|---|
| `EBM_class.py` | Primary | Per-unit mixed stable-polynomial activation | Compilation; runtime suite prepared |
| `EBM_controller.py` | Either copy was byte-identical; primary retained | None needed | Hash comparison |
| `EBM_param_fields.py` | Primary 31-field modular J/R/G schema | None of the inactive 39-field tank schema | Checkpoint migration design and compilation |
| `EBM_rollout.py` | Primary plus compatible extension | Validated positive integer integration substeps | Compilation; runtime suite prepared |
| `EBM_training.py` | Primary optimizer/gradient diagnostics retained | Mixed Huber-to-MSE loss, sample weights, masks, rollout substeps; experimental tank regularizers reject nonzero use explicitly | Compilation |
| CED/Silverbox interfaces | Secondary, relocated to `dataset_interfaces/` | Dataset loaders, configurations, entry points | Configuration and asset audit; runtime suite prepared |
| CED checkpoint | Secondary bytes retained | Explicit 20-field-to-current-schema migration | Runtime suite prepared |
| Stress tests | Primary | No secondary overwrite; generated reports with private paths excluded | Compilation and structural audit |
| Duffing PortHNN-u | Primary adversarial bundle | Canonicalized its byte-identical implementation at `porthnn_u/duffing.py`; retained frozen seed-0 champion | Hash/schema and stored-metric audit; runtime validator prepared |
| Three-link PortHNN-u | Primary adversarial bundle | Replaced the older comparator with the corrected primary full-J/R/G implementation and frozen champion | Stored-prediction/metric audit; runtime validator prepared |
| CED/Silverbox PortHNN-u | Secondary adversarial bundle | Modular benchmark-specific implementation, configs, selected validation champions, and artifacts | Checkpoint hashes, schemas, finite arrays, selection records; runtime validator prepared |
| Runtime HNER99 | Supplied runtime bundle | Integrated as the importable `runtime_hner99` library; numerical method and audit workflow retained; machine-bound loaders replaced by release-relative model and artifact registries | Package/artifact imports, nine-report schema audit, checkpoint hashes where recorded, shell/error/status checks |

The secondary feature-conditioned tank schema was not activated: it conflicts
with the newer primary modular schema, neither supplied champion configuration
enables its feature modes, and no evidence justified replacing primary J/R/G
capabilities. This alternative is documented here rather than retained as an
ambiguous duplicate implementation.

## Exclusions

- vendored third-party Python packages;
- raw W&B/training logs and generated ranking outputs;
- generated theory reports containing machine paths;
- duplicate Silverbox extracted/archive copies and duplicate top-level baseline mirrors;
- a NanoDrone parent baseline checkpoint whose embedded metadata contained private machine paths (the final baseline checkpoint is retained);
- cluster submission assumptions and machine-specific paths;
- caches and compiled Python files.

The historical Silverbox pHEBM HNER99 checkpoint was not supplied. Its frozen
report is retained, marked non-recomputable, and never associated with another
checkpoint. The historical NanoDrone report used seed 23 while the release
contains seed 47; both facts are explicit in the provenance registry. Older
Duffing and three-link reports did not record checkpoint hashes, so their exact
historical checkpoint identity cannot be proven cryptographically.

## Portability changes

Dataset locations derive from the release file location, with `EBM_DATA_ROOT`
as an explicit override. The batch entry point is named `train_from_config.py`;
it contains no scheduler-specific requirement. Stored legacy provenance paths
were replaced with `<REDACTED_LEGACY_PATH>` while hashes and scientific values
were preserved.

## Reviewer entry points

```bash
python validate_release.py --structural
python validate_release.py
python scripts/validate_porthnn_u_runtime.py
python scripts/compute_robustness_radii.py --help
python dataset_interfaces/run_champion.py <dataset> --dry-run
```

The full validator performs actual four-step inference for every dataset. It
uses trained checkpoints for Duffing, both N-link cases, NanoDrone, and CED.
The supplied Silverbox branch contained no historical trained checkpoint, so
that case validates initialized-model inference and reports that provenance.
